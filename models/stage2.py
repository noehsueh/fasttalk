import torch
import torch.nn as nn
from models.lib.wav2vec import Wav2Vec2Model 
from transformers import WavLMModel, AutoModel
from models.utils import init_biased_mask, enc_dec_mask, PeriodicPositionalEncoding
from base import BaseModel
import pdb
import os
import random
import torch.nn.functional as F
#os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        print(f"[{tag}] VRAM used: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB "
              f"| reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


class AudioAggregator(nn.Module):
    """
    1) De-interleave [B, 200, 1024] into left/right => [B, 100, 1024] each
    2) Pass each through a small Transformer (num_layers, nhead, ff_dim, dropout)
    3) Do average pooling over time => [B,1024] for each
    4) Output shape: [2B, 1024]
    """
    def __init__(self, hidden_dim=1024, num_layers=2, nhead=4, ff_dim=2048, dropout=0.1):
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

class CodeTalker(BaseModel):
    def __init__(self, args):
        super(CodeTalker, self).__init__()
        """
        audio: (batch_size, raw_wav)
        vertice: (batch_size, seq_len, V*3)
        """
        self.args = args
        self.dataset = args.dataset

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
        for param in self.audio_feature_map.parameters():
            param.requires_grad = False

        for param in self.vertice_map.parameters():
            param.requires_grad = False
        
        for param in self.transformer_decoder.parameters(): # Blendshape Transformer
            param.requires_grad = False
        
        for param in self.feat_map.parameters():
            param.requires_grad = False

    def computeAudioFeatures(self, sliding_chunks_audio_features, context_window):

        hidden_states_wav2vec = self.audio_encoder(sliding_chunks_audio_features).last_hidden_state #Wav2Vec

        aggregated_states = self.aggregator(hidden_states_wav2vec)

        return aggregated_states.unsqueeze(0)


    def forward(self, audio_name, audio, audio_features, vertice, blendshapes, template, criterion):

        frame_num = vertice.shape[1]
        template = template.unsqueeze(1) # (1,1,V*3)
        
        # ======== Audio feature extraction ========
        # ->> 1) Parameters for chunking (input audio window) ================================
        context_window = 50

        # ->> 2) Create sliding window, padding with 0s at the initial frames  ==============================
        # Compute the total padded length
        hop_length_int = audio_features.shape[-1] // frame_num  # Compute hop length for frames
        padding_length = (context_window - 1) * hop_length_int

        # Pad the start of audio with zeros for sliding window effect
        padded_audio_features = torch.nn.functional.pad(
            audio_features, (padding_length,0), "constant", 0
        )

        # Generate overlapping sliding windows
        sliding_chunks_audio_features = torch.cat(
            [
                padded_audio_features[:, i * hop_length_int : i * hop_length_int + context_window * hop_length_int]
                for i in range(frame_num)
            ],
            dim=0,
        )

        # ->> 3) Compute audio features [frames, n_mfcc] ======================================
        hidden_states_agg = self.computeAudioFeatures(sliding_chunks_audio_features, context_window)

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

        vertice_dec, blendshapes_out = torch.split(vertice_blendshapes_out, [15789,56], dim=2)
        vertice_out = vertice_dec + template

        # loss
        loss_vert = criterion(vertice_out, vertice) # (batch, seq_len, V*3)
        loss_blendshapes = criterion(blendshapes_out, blendshapes) #criterion(blendshapes.clone(), blendshapes)#
        loss_reg  = criterion(feat_out, feat_q_gt.detach())

        return self.args.motion_weight*loss_vert + 0.001*loss_blendshapes + self.args.reg_weight*loss_reg, [loss_vert, loss_blendshapes, loss_reg, loss_vert]


    def predict(self, audio_features, template):
        template = template.unsqueeze(1) # (1,1, V*3)

        context_window = 50

        # ->> 2) Create sliding window with padding on extremes ==============================
        # Compute the total padded length
        hop_length_int = 640 #audio.shape[-1] // frame_num  # Compute hop length for frames
        frame_num      = audio_features.shape[-1] // hop_length_int
        padding_length = (context_window - 1) * hop_length_int

        # Pad the start and end of audio with zeros for sliding window effect
        padded_audio_features = torch.nn.functional.pad(
            audio_features, (padding_length, 0), "constant", 0
        )

        # Generate overlapping sliding windows
        sliding_chunks_audio_features = torch.cat(
            [
                padded_audio_features[:, i * hop_length_int : i * hop_length_int + context_window * hop_length_int]
                for i in range(frame_num)
            ],
            dim=0,
        )

        # ->> 3) Compute audio features [frames, n_mfcc] ======================================
        hidden_states_agg = self.computeAudioFeatures(sliding_chunks_audio_features, context_window)

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
                vertice_out_q, blendshapes_out_q = torch.split(vertice_blendshapes_out_q, [15789,56], dim=2)
                vertice_out_q = vertice_out_q[:,0].unsqueeze(1)
                blendshapes_out_q    = blendshapes_out_q[:,0].unsqueeze(1)
            else:
                vertice_blendshapes_out_q = self.autoencoder.decode(feat_out_q)
                vertice_out_q, blendshapes_out_q = torch.split(vertice_blendshapes_out_q, [15789,56], dim=2)

            if i != frame_num - 1:
                vertice_blendshapes_input = torch.cat((vertice_out_q, blendshapes_out_q), dim=-1)
                new_output = self.vertice_map(vertice_blendshapes_input[:,-1,:]).unsqueeze(1)
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

        # quantization and decoding
        feat_out_q, _, _ = self.autoencoder.quantize(feat_out)

        vertice_blendshapes_out = self.autoencoder.decode(feat_out_q)
        vertice_out, blendshapes_out= torch.split(vertice_blendshapes_out, [15789,56], dim=2)

        vertice_out = vertice_out + template

        return vertice_out, blendshapes_out
