
import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize, GroupedResidualVQ, LFQ
import math
import time

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

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, depth=6):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True, dropout=0.1)
            for _ in range(depth)
        ])
    
    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

class VQAutoEncoder(nn.Module): 
    def __init__(self, args, input_dim=58, hidden_dim=512, codebook_size=2048):
        super().__init__()

        # Input projection
        self.encoder_proj = nn.Linear(input_dim, hidden_dim)

        self.pos_enc = PositionalEncoding(dim=hidden_dim)

        self.encoder_transformer = TransformerBlock(dim=hidden_dim)

        self.vq = GroupedResidualVQ(
                                dim=hidden_dim,               # Total latent dimension (e.g., 512); input and output will match this shape
                                codebook_size=codebook_size,  # Number of entries in each codebook (per quantizer, per group); higher = more expressive
                                groups=32,                     # Split the input vector into 16 equal parts (e.g., 512 → eight 32D sub-vectors)
                                num_quantizers=2,             # For each sub-vector (group), apply 4 quantizers sequentially (residual refinement)
                                commitment_weight=0.1,        # Weight for commitment loss: encourages encoder outputs to match selected codes
                                decay=0.97,                   # EMA decay for updating codebook entries; higher = slower update (more stable)
                                use_cosine_sim=False,         # Use L2 distance for nearest neighbor search (more stable than cosine for reconstruction)
                                rotation_trick=False          # Disable rotation trick (orthogonal transform); better stability and interpretability
                            )

        self.decoder_transformer = TransformerBlock(dim=hidden_dim)

        # Output projection
        self.decoder_proj = nn.Linear(hidden_dim, input_dim)
    
    # ───────────────────────────────────────────────────────── helpers ── #
    @staticmethod
    def _record(label, start_evt, end_evt, cpu_t0, times, use_gpu):
        """Store elapsed time (ms) for one segment."""
        if use_gpu:
            end_evt.record()
            torch.cuda.synchronize()
            elapsed = start_evt.elapsed_time(end_evt)          # already ms
        else:
            elapsed = (time.perf_counter() - cpu_t0) * 1_000   # s → ms
        times[label] = elapsed

    def _new_timer(self, use_gpu):
        """Return (start_event, end_event, cpu_start_time)."""
        if use_gpu:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            return start_evt, end_evt, None
        else:
            return None, None, time.perf_counter()

    # ───────────────────────────────────────────────────────── forward ── #
    def forward_time(self, blendshapes, mask=None):
        times   = {}
        use_gpu = blendshapes.is_cuda

        # 1. encoder projection
        st, ed, t0 = self._new_timer(use_gpu)
        x = self.encoder_proj(blendshapes)
        self._record('encoder_proj', st, ed, t0, times, use_gpu)

        # 2. positional encoding
        st, ed, t0 = self._new_timer(use_gpu)
        x = self.pos_enc(x)
        self._record('pos_enc', st, ed, t0, times, use_gpu)

        # 3. encoder transformer
        st, ed, t0 = self._new_timer(use_gpu)
        x = self.encoder_transformer(
                x,
                src_key_padding_mask=~mask if mask is not None else None
            )
        self._record('encoder_transformer', st, ed, t0, times, use_gpu)

        # 4. vector quantiser
        st, ed, t0 = self._new_timer(use_gpu)
        quantized, _, vq_loss = self.vq(x)
        self._record('vector_quantize', st, ed, t0, times, use_gpu)

        # 5. decoder transformer
        st, ed, t0 = self._new_timer(use_gpu)
        x = self.decoder_transformer(
                quantized,
                src_key_padding_mask=~mask if mask is not None else None
            )
        self._record('decoder_transformer', st, ed, t0, times, use_gpu)

        # 6. output projection
        st, ed, t0 = self._new_timer(use_gpu)
        decoded = self.decoder_proj(x)
        self._record('decoder_proj', st, ed, t0, times, use_gpu)

        # ─── pretty print ─────────────────────────────────────────────── #
        print("│  step                    │   time (ms)")
        print("├──────────────────────────┼────────────")
        for k, v in times.items():
            print(f"│ {k:<24}│ {v:10.3f}")
        print("└──────────────────────────┴────────────\n")

        return decoded, vq_loss
    

    def forward(self, blendshapes, mask=None):
        # blendshapes: [B, T, 58]
        # mask:        [B, T] with 1 for valid tokens, 0 for padded

        x = self.encoder_proj(blendshapes)

        x = self.pos_enc(x)

        x = self.encoder_transformer(x, src_key_padding_mask=~mask if mask is not None else None)

        quantized, _, vq_loss = self.vq(x)

        x = self.decoder_transformer(quantized, src_key_padding_mask=~mask if mask is not None else None)

        decoded = self.decoder_proj(x)

        return decoded, vq_loss


    def get_quant(self, blendshapes, mask=None):
        x = self.encoder_proj(blendshapes)            # [B, T, hidden_dim]

        x = self.pos_enc(x)
        
        x = self.encoder_transformer(x, src_key_padding_mask=~mask if mask is not None else None)

        quantized, indices, _ = self.vq(x)            # indices: [B, T]

        return quantized, indices, x # quantzed, codebook indices, and encoder output


    def decode(self, quantized, mask=None):

        x = self.decoder_transformer(quantized, src_key_padding_mask=~mask if mask is not None else None)

        decoded = self.decoder_proj(x)

        return decoded
