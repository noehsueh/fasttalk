
import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize, GroupedResidualVQ, LFQ
import math

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
                                groups=32,                    # Split the input vector into 16 equal parts (e.g., 512 → eight 32D sub-vectors)
                                num_quantizers=4,             # For each sub-vector (group), apply 4 quantizers sequentially (residual refinement)
                                commitment_weight=0.1,        # Weight for commitment loss: encourages encoder outputs to match selected codes
                                decay=0.97,                   # EMA decay for updating codebook entries; higher = slower update (more stable)
                                use_cosine_sim=False,         # Use L2 distance for nearest neighbor search (more stable than cosine for reconstruction)
                                rotation_trick=False          # Disable rotation trick (orthogonal transform); better stability and interpretability
                            )

        self.decoder_transformer = TransformerBlock(dim=hidden_dim)

        # Output projection
        self.decoder_proj = nn.Linear(hidden_dim, input_dim)

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
