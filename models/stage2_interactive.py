import pdb
import os
import random
import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2Model

from models.utils import init_biased_mask, enc_dec_mask, enc_dec_mask_simple
from base import BaseModel


class AudioAggregator(nn.Module):
    """
    1) De-interleave [B, 200, 1024] into left/right => [B, 100, 1024] each
    2) Pass each through a small Transformer (num_layers, nhead, ff_dim, dropout)
    3) Do average pooling over time => [B,1024] for each
    4) Output shape: [2B, 1024]
    """
    def __init__(self, hidden_dim=768, num_layers=2, nhead=4, ff_dim=2048, dropout=0.1):
        super().__init__()
        
        # (A) Define the TransformerEncoderLayer that will process each [T, B, C] sequence
        encoder_layer_left = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=ff_dim, 
            dropout=dropout,
            activation='relu',
            batch_first=False  # default for nn.Transformer is [T,B,C]
        )
        
        # (B) We'll stack 'num_layers' of these layers
        self.transformer_encoder_left = nn.TransformerEncoder(encoder_layer_left, num_layers=num_layers)

        # (A) Define the TransformerEncoderLayer that will process each [T, B, C] sequence
        encoder_layer_right = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=ff_dim, 
            dropout=dropout,
            activation='relu',
            batch_first=False  # default for nn.Transformer is [T,B,C]
        )
        
        # (B) We'll stack 'num_layers' of these layers
        self.transformer_encoder_right = nn.TransformerEncoder(encoder_layer_right, num_layers=num_layers)

    def _transformer_process(self, x_seq, is_left):
        """
        A helper method that:
          - expects x_seq of shape [B, 100, feature_dim]
          - transforms it with the TransformerEncoder
          - returns [B, feature_dim] by averaging over time
        """
        B, T, C = x_seq.shape  # e.g. [B,100,feature_dim]
        
        # 1) Transpose to [T, B, C] for the Transformer
        x_seq = x_seq.transpose(0, 1)  # => [T=100, B, C=feature_dim]

        # 2) Forward through the stacked transformer
        if is_left:
            x_seq = self.transformer_encoder_left(x_seq)  # => still [T, B, C]
        else:
            x_seq = self.transformer_encoder_right(x_seq)

        # 3) Average over time => [B, C]
        x_seq = x_seq.mean(dim=0)  # => [B, feature_dim]

        return x_seq

    def forward(self, x):
        """
        x: [B, 200, feature_dim], with strictly interleaved frames (0->left, 1->right, 2->left, 3->right, ...)
        Returns: [2B, feature_dim]
        """
        B, T, C = x.shape

        # 1) De-interleave => left: x[:, 0::2, :], right: x[:, 1::2, :]
        x_left  = x[:, 0::2, :]  # => [B, 100, feature_dim]
        x_right = x[:, 1::2, :]  # => [B, 100, feature_dim]

        # 2) Transform each side
        left_ctx  = self._transformer_process(x_left, is_left=True)    # => [B, feature_dim]
        right_ctx = self._transformer_process(x_right, is_left=False)  # => [B, feature_dim]

        # 3) Stack => [B, 2, feature_dim]
        combined = torch.stack([left_ctx, right_ctx], dim=1)

        # 4) Flatten => [2B, feature_dim]
        out = combined.view(B * 2, C)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim] ← this is the key change
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Adaptive Instance Normalization
class AdaLayerNorm(nn.Module):
    """
    LayerNorm in which the affine scale (γ) and shift (β) are produced
    on-the-fly from a per-sample style vector.
    """
    def __init__(self, d_model: int, style_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.style2scale = nn.Linear(style_dim, 2 * d_model)

        # initialise so that γ≈1, β≈0 at start ⇒ behaves like vanilla LN
        nn.init.zeros_(self.style2scale.weight)
        nn.init.constant_(self.style2scale.bias[:d_model], 1.0)   # γ
        nn.init.constant_(self.style2scale.bias[d_model:], 0.0)   # β

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        x     : [B, T, D]
        style : [B, style_dim]
        """
        h = self.norm(x)
        gamma, beta = self.style2scale(style).chunk(2, dim=-1)  # [B, D] each
        return gamma.unsqueeze(1) * h + beta.unsqueeze(1)       # broadcast over T


class FastTalkTransformerDecoderLayerWithADAIN(nn.Module):
    """
    Transformer-decoder block with AdaIN conditioning (speaker / style).
    Drop-in replacement for `nn.TransformerDecoderLayer` except that
    `forward` takes an extra `style` tensor.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        style_dim: int,                   
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True,
        norm_first: bool = False,         # True = Pre-LN
    ):
        super().__init__()

        # --- Attention -----------------------------------------------------
        self.self_attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # --- Feed-forward ---------------------------------------------------
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # --- AdaIN LayerNorms ----------------------------------------------
        self.adanorm1 = AdaLayerNorm(d_model, style_dim)
        self.adanorm2 = AdaLayerNorm(d_model, style_dim)
        self.adanorm3 = AdaLayerNorm(d_model, style_dim)

        # --- Dropout & misc -------------------------------------------------
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu
        self.norm_first = norm_first


    # ---- helper blocks -----------------------------------------------------
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x_sa, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout1(x_sa)

    def _ca_block(self, x, mem, attn_mask, key_padding_mask):
        x_ca, _ = self.cross_attn(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout2(x_ca)

    def _ff_block(self, x):
        return self.linear2(self.dropout3(self.activation(self.linear1(x))))


    # ---- forward ----------------------------------------------------------
    def forward(
        self,
        tgt: torch.Tensor,                       # [B, T, D]
        memory: torch.Tensor,                    # [B, S, D]
        style: torch.Tensor,                     # [B, style_dim]
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.norm_first:          # -------- Pre-LN path
            tgt = tgt + self._sa_block(
                self.adanorm1(tgt, style),
                tgt_mask, tgt_key_padding_mask)
            tgt = tgt + self._ca_block(
                self.adanorm2(tgt, style), memory,
                memory_mask, memory_key_padding_mask)
            tgt = tgt + self._ff_block(self.adanorm3(tgt, style))
            return tgt

        # ------------------------------ Post-LN (default)
        tgt = tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask)
        tgt = self.adanorm1(tgt, style)

        tgt = tgt + self._ca_block(tgt, memory,
                                   memory_mask, memory_key_padding_mask)
        tgt = self.adanorm2(tgt, style)

        tgt = tgt + self._ff_block(tgt)
        tgt = self.adanorm3(tgt, style)
        return tgt


# ------------------------------------------------------------
# FastTalk decoder (stack of layers)
# ------------------------------------------------------------
class FastTalkTransformerDecoder(nn.Module):
    """
    Stacks `num_layers` copies of FastTalkTransformerDecoderLayerWithADAIN.
    API = nn.TransformerDecoder  ➜  plus an extra *style* tensor.
    """
    def __init__(
        self,
        decoder_layer: FastTalkTransformerDecoderLayerWithADAIN,
        num_layers: int,
        norm: Optional[nn.LayerNorm] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm

    def forward(
        self,
        tgt: torch.Tensor,                       # [B, T, D]
        memory: torch.Tensor,                    # [B, S, D]
        style: torch.Tensor,                     # [B, style_dim]  ← NEW
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                style,                           # ← pass through
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)
        return output


class CodeTalker(BaseModel):
    def __init__(self, args):
        super(CodeTalker, self).__init__()
        self.args    = args
        self.dataset = args.dataset

        # ======== Audio feature extraction ========
        self.aggregator = AudioAggregator(hidden_dim=768)
        num_params      = sum(p.numel() for p in self.aggregator.parameters() if p.requires_grad)
        print("-----> Number of trainable parameters in AudioAggregator:", num_params)

        self.context_window = args.interactive_window
        self.hop_length     = args.hop_length

        self.audio_encoder = Wav2Vec2Model.from_pretrained(args.wav2vec2model_path)
        print("Loading pretrained: {}".format(args.wav2vec2model_path))

        # wav2vec 2.0 weights initialization
        #self.audio_encoder.feature_extractor._freeze_parameters()   
        

        self.audio_feature_map = nn.Linear(768, args.feature_dim)

        # motion encoder
        self.blendshapes_map = nn.Linear(args.blendshapes_dim, args.feature_dim)

        # periodic positional encoding
        self.pos_enc = PositionalEncoding(dim=args.feature_dim)

        # temporal bias
        self.biased_mask = init_biased_mask(n_head = 1, max_seq_len = 600, period=args.period)
        

        style_dim = args.feature_dim        # or whatever size your style vector is
        decoder_layer = FastTalkTransformerDecoderLayerWithADAIN(
                                                                    d_model=args.feature_dim,
                                                                    nhead=args.n_head,
                                                                    style_dim=style_dim,
                                                                    dim_feedforward=args.feature_dim,
                                                                    dropout=0.1,
                                                                    activation="gelu",
                                                                    batch_first=True,
                                                                    norm_first=False
                                                                )

        self.transformer_decoder = FastTalkTransformerDecoder(
                                                                decoder_layer,
                                                                num_layers=args.num_layers,
                                                                norm=nn.LayerNorm(args.feature_dim)
                                                            )

        # style
        self.style_proj = nn.Linear(512, 1024)  
        self.style_frame_encoder = nn.TransformerEncoder(
                                                            nn.TransformerEncoderLayer(d_model=1024, nhead=4, batch_first=True),
                                                            num_layers=2
                                                        )

        # motion decoder
        # self.feat_map = nn.Linear(args.feature_dim, 512, bias=False)
        self.feat_map = nn.Linear(args.feature_dim, 512)

        self.device = args.device
        #nn.init.constant_(self.feat_map.weight, 0)

        # VQ-AUTOENCODER
        from models.stage1 import VQAutoEncoder
        self.autoencoder = VQAutoEncoder(args)
        self.autoencoder.load_state_dict(torch.load(args.vqvae_pretrained_path)['state_dict'])
        print("Loading pretrained vq: {}".format(args.vqvae_pretrained_path))

        for param in self.autoencoder.parameters():
            param.requires_grad = False

        #for param in self.audio_feature_map.parameters():
        #    param.requires_grad = False
        
        for param in self.blendshapes_map.parameters():
            param.requires_grad = False
        
        for param in self.transformer_decoder.parameters():
            param.requires_grad = False
        
        for param in self.style_proj.parameters():
            param.requires_grad = False

        for param in self.style_frame_encoder.parameters():
            param.requires_grad = False
        
        for param in self.feat_map.parameters():
            param.requires_grad = False


    def _style_view(self, blend, mask):

         # Project input to transformer model dimension
        feats = self.style_proj(blend)  # [B, T, 1024]

        # Positional encoding
        feats = self.pos_enc(feats)     # [B, T, 1024]

        # Prepare key_padding_mask for the Transformer
        key_padding_mask = ~mask        # [B, T], where True = ignore

        # Pass through Transformer
        feats = self.style_frame_encoder(feats, src_key_padding_mask=key_padding_mask)  # [B, T, 1024]

        # Pool only valid frames
        feats = feats * mask.unsqueeze(-1)
        style_vec = feats.sum(1) / (mask.sum(1, keepdim=True) + 1e-6)  # [B, 1024]

        return F.normalize(style_vec, dim=-1)
    

    def _make_two_views(self, blend, mask, crop_ratio=0.8):
        """
        Randomly crops ≈`crop_ratio` of each sequence twice.
        Gives two tensors with *different* valid-frame masks.
        """
        B, T, _ = blend.shape
        keep = int(T * crop_ratio)

        idx1 = torch.randint(0, T - keep + 1, (B,), device=blend.device)
        idx2 = torch.randint(0, T - keep + 1, (B,), device=blend.device)

        def crop(start_idx):
            rng = torch.arange(T, device=blend.device).unsqueeze(0)
            new_mask = (rng >= start_idx.unsqueeze(1)) & (rng < start_idx.unsqueeze(1) + keep)
            new_blend = blend * new_mask.unsqueeze(-1)
            return new_blend, new_mask & mask          # preserve original padding

        return crop(idx1), crop(idx2)


    def nt_xent_unsup(self, z1, z2, temperature=0.5):
        """
        Numerically-stable SimCLR loss.
        z1, z2  : [B, D]  (MUST be l2-normalised, float32)
        returns : scalar loss
        """
        B, _ = z1.shape
        z = torch.cat([z1, z2], dim=0)              # [2B, D]

        # Step 1: cosine-similarity matrix
        sim = torch.mm(z, z.t()).div_(temperature)  # float32, [2B, 2B]

        # Step 2: mask self-similarities
        sim.fill_diagonal_(-float('inf'))

        # Step 3: subtract per-row max  (log-sum-exp trick)
        sim = sim - sim.max(dim=1, keepdim=True).values

        # Step 4: build positive indices  k ↔ k±B
        pos_idx = (torch.arange(2*B, device=z.device) + B) % (2*B)

        # Step 5: cross-entropy
        loss = F.cross_entropy(sim, pos_idx)
        return loss


    def forward(self, padded_blendshapes, blendshapes_mask, padded_audios, audio_mask, criterion, target_style=None):
        B, T, D = padded_blendshapes.shape  # D = 58
        
        # Remove unnecessary singleton dimension: [B, 1, N] -> [B, N]
        padded_audios = padded_audios.squeeze(1)

        # ========== AUDIO FEATURES EXTRACTION ==========
        # Compute the total padded length
        hop_length_int = padded_audios.shape[-1] // T  # Compute hop length for frames
        padding_length = (self.context_window - 1) * hop_length_int

        # Interactive audio preparation
        # Pad the start of audio with zeros for sliding window effect
        padded_audios_init_zeros = torch.nn.functional.pad(
            padded_audios, (padding_length,0), "constant", 0
        )

        # Generate overlapping sliding windows
        sliding_chunks_audio_features = torch.cat(
            [
                padded_audios_init_zeros[:, i * hop_length_int : i * hop_length_int + self.context_window * hop_length_int]
                for i in range(T)
            ],
            dim=0,
        )

        # 2) Compute audio features per chunk -> aggregate them
        hidden_states_wav2vec = self.audio_encoder(sliding_chunks_audio_features).last_hidden_state #Wav2Vec

        hidden_states = self.aggregator(hidden_states_wav2vec).unsqueeze(0)

        hidden_states = self.audio_feature_map(hidden_states)
        _, T_audio, _ = hidden_states.shape

         # Compute padding mask for memory
        frame_num        = blendshapes_mask.sum(dim=1)  # [B], e.g., tensor([501, 501, 499, ...])
        valid_audio_lens = torch.clamp(frame_num * 2, max=T_audio)  # [B], per-sample audio length
        time_range_audio = torch.arange(T_audio, device=hidden_states.device).unsqueeze(0)  # [1, T_audio]
        memory_key_padding_mask = time_range_audio >= valid_audio_lens.unsqueeze(1)  # [B, T_audio]

        # ========== AUDIO FEATURES EXTRACTION ==========


        # ── Style ──────────────────────────────────────────────────────
        feat_q_gt, _, encoded = self.autoencoder.get_quant(padded_blendshapes, blendshapes_mask) # Autoencoder embedded features

        # --- build two cropped views ------------------------------------------------
        (view1, m1), (view2, m2) = self._make_two_views(feat_q_gt.detach(), blendshapes_mask)
        style_a = self._style_view(view1, m1)          # [B, D]
        style_b = self._style_view(view2, m2)          # [B, D]

        # pick one view (or their average) for conditioning
        style_vec = (style_a + style_b) * 0.5          
       
        # ========== AUTOREGRESSIVE ==========
        # Create one frame of 0s and shift right the blendshapes by one frame, insert the 0s
        zero_frame        = torch.zeros(B, 1, D, device=padded_blendshapes.device)        
        padded_blendshapes_shifted = torch.cat((zero_frame, padded_blendshapes[:, :-1, :]), dim=1) 

        emb_blendshapes_tgt = self.blendshapes_map(padded_blendshapes_shifted)  
        emb_blendshapes_tgt = self.pos_enc(emb_blendshapes_tgt)

        # Prepare autoregressive masks
        tgt_mask = self.biased_mask[0, :emb_blendshapes_tgt.shape[1], :emb_blendshapes_tgt.shape[1]].clone().detach().to(device=self.device).squeeze(0)
        memory_mask = enc_dec_mask(self.device, self.dataset, emb_blendshapes_tgt.shape[1], hidden_states.shape[1]) 
        # ========== AUTOREGRESSIVE ==========

     
        # ========== DECODER FOR AUTOREGRESSIVE PREDICTION ==========
        # During forward / predict:
        feat_out = self.transformer_decoder(
                                                tgt=emb_blendshapes_tgt,
                                                memory=hidden_states,
                                                style=style_vec,               #  pass style once
                                                tgt_mask=tgt_mask,
                                                memory_mask=memory_mask,
                                                tgt_key_padding_mask=~blendshapes_mask,
                                                memory_key_padding_mask=memory_key_padding_mask
                                                )

        feat_out = self.feat_map(feat_out)  # [B, T, 58]
    
        # I want the output of the transformer decoder to match the output of the encoder in the VQAutoencoder (after quantization)
        #loss_reg     = criterion(feat_out, feat_q_gt.detach())  # L2 loss between predicted and quantized ground truth features
        loss_reg     = criterion(feat_out, encoded.detach())  # L2 loss between predicted and quantized ground truth features

        # Also make the deocoded blendshapes match the input blendshapes
        feat_out_q, _, _ = self.autoencoder.vq(feat_out) # Quantize the embedding
        blendshapes_out  = self.autoencoder.decode(feat_out_q, blendshapes_mask) # Decode the quantized embedding to get the blendshapes outputfeat_q_gt
        loss_blendshapes = criterion(blendshapes_out, padded_blendshapes)  # L2 loss
      
        # --- add self-supervised style compactness ---------------------------------
        loss_style = self.nt_xent_unsup(style_a, style_b)

        return loss_blendshapes + loss_reg + 0.001 * loss_style,  [loss_blendshapes, loss_reg, loss_style, loss_reg]#, blendshapes_out #+  0.01 * nt_xent_loss,
    

    def predict_no_quantizer(self, audio, target_style=None):
        audio = audio.squeeze(1)

        # Compute the total padded length
        frame_num = audio.shape[-1] // self.hop_length  # Compute hop length for frames
        padding_length = (self.context_window - 1) * self.hop_length 

        # Interactive audio preparation
        # Pad the start of audio with zeros for sliding window effect
        padded_audios_init_zeros = torch.nn.functional.pad(
            audio, (padding_length,0), "constant", 0
        )

        # Generate overlapping sliding windows
        sliding_chunks_audio_features = torch.cat(
            [
                padded_audios_init_zeros[:, i * self.hop_length : i * self.hop_length + self.context_window * self.hop_length]
                for i in range(frame_num)
            ],
            dim=0,
        )


        # Extract audio features using Wav2Vec2
        hidden_states_wav2vec = self.audio_encoder(sliding_chunks_audio_features).last_hidden_state #Wav2Vec

        hidden_states = self.aggregator(hidden_states_wav2vec).unsqueeze(0)

        hidden_states = self.audio_feature_map(hidden_states)


        #  Build style_vec once per batch ───────────────────────────────
        if target_style is not None:
            feat_q_gt, _, encoded = self.autoencoder.get_quant(target_style, torch.ones_like(target_style[...,0], dtype=torch.bool))
            style_feats = self.style_proj(feat_q_gt) 
            style_feats = self.pos_enc(style_feats) 
            style_feats = self.style_frame_encoder(style_feats)    
            style_vec   = style_feats.mean(dim=1)       
            style_vec   = F.normalize(style_vec, dim=-1)   
        else:
            # fallback: a learnable "neutral" token (could also be zeros)
            style_vec = torch.zeros(1, self.args.feature_dim, device=self.device, dtype=hidden_states.dtype)

        # autoregressive facial motion prediction
        for i in range(frame_num):
            if i==0:
                blendshapes_emb   = torch.zeros((1,1,self.args.feature_dim)).to(self.device) # (1,1,feature_dim)
                blendshapes_input = self.pos_enc(blendshapes_emb)
            else:
                blendshapes_input = self.pos_enc(blendshapes_emb)

            tgt_mask    = self.biased_mask[:, :blendshapes_input.shape[1], :blendshapes_input.shape[1]].clone().detach().to(device=self.device).squeeze(0)
            memory_mask = enc_dec_mask(self.device, self.dataset, blendshapes_input.shape[1], hidden_states.shape[1])

            feat_out = self.transformer_decoder(
                                                tgt=blendshapes_input,
                                                memory=hidden_states,
                                                tgt_mask=tgt_mask, 
                                                memory_mask=memory_mask,
                                                style   = style_vec,
                                                )

            feat_out         = self.feat_map(feat_out) # Map the output features to the final feature space (VQAutoencoder 'embedding') 
            #feat_out_q, _, _ = self.autoencoder.vq(feat_out) # Quantize the embedding

            # Quantized features to blendshapes
            if i == 0:
                blendshapes_out_q = self.autoencoder.decode(torch.cat([feat_out, feat_out], dim=1))
                blendshapes_out_q = blendshapes_out_q[:,0].unsqueeze(1)
            else:
                blendshapes_out_q = self.autoencoder.decode(feat_out)

            if i != frame_num - 1:
                new_output        = self.blendshapes_map(blendshapes_out_q[:,-1,:]).unsqueeze(1)
                blendshapes_emb   = torch.cat((blendshapes_emb, new_output), 1)
                
        # quantization and decoding
        #feat_out_q, _, _ = self.autoencoder.vq(feat_out)
        blendshapes_out  = self.autoencoder.decode(feat_out)

        return blendshapes_out




def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        print(f"[{tag}] VRAM used: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB "
              f"| reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")



class CodeTalkerOLD(BaseModel):
    def __init__(self, args):
        super(CodeTalker, self).__init__()
        """
        audio: (batch_size, raw_wav)
        vertice: (batch_size, seq_len, V*3)
        """
        self.args = args
        self.dataset = args.dataset
        self.context_window = args.interactive_window
        self.hop_length = args.hop_length

        # ======== Audio feature extraction ========
        self.aggregator = AudioAggregator(hidden_dim=1024)
        num_params      = sum(p.numel() for p in self.aggregator.parameters() if p.requires_grad)
        print("-----> Number of trainable parameters in AudioAggregator:", num_params)

        self.audio_encoder = AutoModel.from_pretrained(args.wav2vec2model_path)

        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(1024, args.feature_dim)

        # motion encoder
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        # periodic positional encoding
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=args.n_head, dim_feedforward=2*args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.num_layers)
        # motion decoder
        self.feat_map = nn.Linear(args.feature_dim, args.face_quan_num*args.zquant_dim, bias=False)
        # style embedding
        self.learnable_style_emb = nn.Embedding(len(args.train_subjects.split()), args.feature_dim)

        self.device = args.device
        nn.init.constant_(self.feat_map.weight, 0)
        # nn.init.constant_(self.feat_map.bias, 0)

        # VQ-AUTOENCODER
        from models.stage1_vocaset import VQAutoEncoder

        self.autoencoder = VQAutoEncoder(args)
        temp = torch.load(args.vqvae_pretrained_path)['state_dict']

        self.autoencoder.load_state_dict(torch.load(args.vqvae_pretrained_path)['state_dict'])
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        # Put all to sleep ===========
        #for param in self.audio_encoder.parameters():
        #    param.requires_grad = False

        for param in self.audio_feature_map.parameters():
            param.requires_grad = False

        for param in self.vertice_map.parameters():
            param.requires_grad = False
        
        for param in self.transformer_decoder.parameters(): # Blendshape Transformer
            param.requires_grad = False
        
        for param in self.feat_map.parameters():
            param.requires_grad = False

    def computeAudioFeatures(self, sliding_chunks_audio_features):

        hidden_states_wav2vec = self.audio_encoder(sliding_chunks_audio_features).last_hidden_state #Wav2Vec

        aggregated_states = self.aggregator(hidden_states_wav2vec)

        return aggregated_states.unsqueeze(0)


    def forward(self, audio_name, audio, audio_features, vertice, blendshapes, template, criterion):

        frame_num = vertice.shape[1]
        template  = template.unsqueeze(1) # (1,1,V*3)
        
        # ======== Audio feature extraction ========
        # 1) Create sliding window, padding with 0s at the initial frames  ==============================
        # Compute the total padded length
        hop_length_int = audio_features.shape[-1] // frame_num  # Compute hop length for frames
        padding_length = (self.context_window - 1) * hop_length_int

        # Pad the start of audio with zeros for sliding window effect
        padded_audio_features = torch.nn.functional.pad(
            audio_features, (padding_length,0), "constant", 0
        )

        # Generate overlapping sliding windows
        sliding_chunks_audio_features = torch.cat(
            [
                padded_audio_features[:, i * hop_length_int : i * hop_length_int + self.context_window * hop_length_int]
                for i in range(frame_num)
            ],
            dim=0,
        )

        # 2) Compute audio features [frames, n_mfcc] ======================================
        hidden_states_agg = self.computeAudioFeatures(sliding_chunks_audio_features)

        if hidden_states_agg.shape[1]<frame_num*2:
            vertice      = vertice[:, :hidden_states_agg.shape[1]//2]
            blendshapes  = blendshapes[:, :hidden_states_agg.shape[1]//2]
            frame_num    = hidden_states_agg.shape[1]//2

        hidden_states = self.audio_feature_map(hidden_states_agg)

        # gt motion feature extraction
        feat_q_gt, _ = self.autoencoder.get_quant(vertice - template, blendshapes)
        feat_q_gt    = feat_q_gt.permute(0,2,1)

        # prepare vertices
        vertice_input = torch.cat((template,vertice[:,:-1]), 1) # shift one position
        vertice_input = vertice_input - template

        # prepare blendshapes
        blendshapes_input  = torch.cat((torch.zeros(1,1,blendshapes.shape[2]).cuda(self.args.gpu),blendshapes[:,:-1]),1)

        # cat both (for teacher forcing)
        vertice_blendshapes_input = torch.cat((vertice_input, blendshapes_input), dim=-1)

        vertice_input = self.vertice_map(vertice_blendshapes_input)

        vertice_input = self.PPE(vertice_input)
        tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
        memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
        feat_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
        feat_out = self.feat_map(feat_out)
        feat_out = feat_out.reshape(feat_out.shape[0], feat_out.shape[1]*self.args.face_quan_num, -1)
        # feature quantization
        feat_out_q, _, _ = self.autoencoder.quantize(feat_out)
        # feature decoding
        vertice_blendshapes_out = self.autoencoder.decode(feat_out_q)

        vertice_dec, blendshapes_out = torch.split(vertice_blendshapes_out, [15069,56], dim=2)
        vertice_out = vertice_dec + template

        # loss
        loss_vert = criterion(vertice_out, vertice) # (batch, seq_len, V*3)
        loss_blendshapes = criterion(blendshapes_out, blendshapes) #criterion(blendshapes.clone(), blendshapes)#
        loss_reg  = criterion(feat_out, feat_q_gt.detach())

        return self.args.motion_weight*loss_vert + 0.001*loss_blendshapes + self.args.reg_weight*loss_reg, [loss_vert, loss_blendshapes, loss_reg, loss_vert]


    def predict(self, audio_features, template):
        template = template.unsqueeze(1) # (1,1, V*3)

        # Compute the total padded length
        frame_num      = audio_features.shape[-1] // self.hop_length
        padding_length = (self.context_window - 1) * self.hop_length

        # Pad the start and end of audio with zeros for sliding window effect
        padded_audio_features = torch.nn.functional.pad(
            audio_features, (padding_length, 0), "constant", 0
        )

        # Generate overlapping sliding windows
        sliding_chunks_audio_features = torch.cat(
            [
                padded_audio_features[:, i * self.hop_length : i * self.hop_length + self.context_window * self.hop_length]
                for i in range(frame_num)
            ],
            dim=0,
        )

        # ->> 3) Compute audio features [frames, n_mfcc] ======================================
        hidden_states_agg = self.computeAudioFeatures(sliding_chunks_audio_features)

        hidden_states = self.audio_feature_map(hidden_states_agg)
        
        # autoregressive facial motion prediction
        for i in range(frame_num):
            if i==0:
                vertice_emb = torch.zeros((1,1,self.args.feature_dim)).to(self.device) # (1,1,feature_dim)
                vertice_input = self.PPE(vertice_emb)
            else:
                vertice_input = self.PPE(vertice_emb)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            feat_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            feat_out = self.feat_map(feat_out)

            feat_out = feat_out.reshape(feat_out.shape[0], feat_out.shape[1]*self.args.face_quan_num, -1)
            # predicted feature to quantized one
            feat_out_q, _, _ = self.autoencoder.quantize(feat_out)
            # quantized feature to vertice
            if i == 0:
                vertice_blendshapes_out_q = self.autoencoder.decode(torch.cat([feat_out_q, feat_out_q], dim=-1))
                vertice_out_q, blendshapes_out_q = torch.split(vertice_blendshapes_out_q, [15069,56], dim=2)
                vertice_out_q = vertice_out_q[:,0].unsqueeze(1)
                blendshapes_out_q    = blendshapes_out_q[:,0].unsqueeze(1)
            else:
                vertice_blendshapes_out_q = self.autoencoder.decode(feat_out_q)
                vertice_out_q, blendshapes_out_q = torch.split(vertice_blendshapes_out_q, [15069,56], dim=2)

            if i != frame_num - 1:
                vertice_blendshapes_input = torch.cat((vertice_out_q, blendshapes_out_q), dim=-1)
                new_output = self.vertice_map(vertice_blendshapes_input[:,-1,:]).unsqueeze(1)
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

        # quantization and decoding
        feat_out_q, _, _ = self.autoencoder.quantize(feat_out)

        vertice_blendshapes_out = self.autoencoder.decode(feat_out_q)
        vertice_out, blendshapes_out= torch.split(vertice_blendshapes_out, [15069,56], dim=2)

        vertice_out = vertice_out + template

        return vertice_out, blendshapes_out
