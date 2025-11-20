import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformerLayer(nn.Module):
    """
    Very small Transformer block with pre-norm, used for encoder, main network, and decoder.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: (B, L, D)
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x_norm2 = self.ln2(x)
        ff_out = self.ff(x_norm2)
        x = x + self.dropout(ff_out)
        return x


class DynamicChunkingLayer(nn.Module):
    """
    One-stage dynamic chunking module in the spirit of H-Net.

    It implements:
      - Routing with cosine similarity between adjacent positions (eq. 4).
      - Downsampling by selecting boundary positions.
      - Smoothing (EMA over chunks) (eq. 5).
      - Confidence-weighted upsampling with a straight-through style weighting (eq. 6-9).
      - Ratio loss to target a desired compression ratio (eq. 10).
    """

    def __init__(self, d_model: int, target_compression_N: int = 6):
        super().__init__()
        self.d_model = d_model
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.N = target_compression_N

    def forward(self, x):
        """
        x: (B, L, D) encoder outputs

        Returns:
          x_down: (B, Lc_max, D) compressed sequence with padding
          P_down: (B, Lc_max) compressed boundary probabilities with padding
          aux: dict with routing info for dechunking
          ratio_loss: scalar tensor
        """
        B, L, D = x.shape
        # Routing: project and compute cosine similarity to previous position
        q = self.Wq(x)
        k = self.Wk(x)
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        k_prev = torch.roll(k_norm, shifts=1, dims=1)  # k_{t-1}
        cos_sim = (q_norm * k_prev).sum(dim=-1)  # (B, L)
        # Boundary probability in [0,1]
        p = 0.5 * (1.0 - cos_sim)
        p[:, 0] = 1.0  # first position always boundary
        # Hard boundary indicator
        b = (p >= 0.5).float()
        b[:, 0] = 1.0

        # Ratio loss (targets average compression of roughly 1/N)
        F_val = b.mean()
        G_val = p.mean()
        N = float(self.N)
        ratio_loss = N / (N - 1.0) * ((N - 1.0) * F_val * G_val + (1.0 - F_val) * (1.0 - G_val))

        # Downsample: keep boundary positions
        indices_list = []
        lengths = []
        max_Lc = 0
        for i in range(B):
            idx = torch.nonzero(b[i] > 0.5, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                idx = torch.tensor([0], device=x.device, dtype=torch.long)
            indices_list.append(idx)
            lengths.append(idx.numel())
            if idx.numel() > max_Lc:
                max_Lc = idx.numel()

        device = x.device
        lengths_tensor = torch.tensor(lengths, device=device, dtype=torch.long)
        x_down = x.new_zeros((B, max_Lc, D))
        P_down = p.new_zeros((B, max_Lc))
        for i in range(B):
            idx = indices_list[i]
            Lc = idx.numel()
            x_down[i, :Lc] = x[i, idx]
            P_down[i, :Lc] = p[i, idx]

        aux = {
            "b": b,                  # (B, L)
            "p": p,                  # (B, L)
            "indices": indices_list, # list of tensors of boundary indices
            "lengths": lengths_tensor,  # (B,)
        }
        return x_down, P_down, aux, ratio_loss

    def dechunk(self, z_down, P_down, aux):
        """
        z_down: (B, Lc_max, D) main network outputs on compressed sequence
        P_down: (B, Lc_max) compressed boundary probabilities
        aux: dict from forward()

        Returns:
          z_full: (B, L, D) sequence upsampled back to original length
        """
        b = aux["b"]
        p = aux["p"]
        indices_list = aux["indices"]
        lengths = aux["lengths"]
        B, L = b.shape
        D = z_down.size(-1)
        device = z_down.device

        z_full = z_down.new_zeros((B, L, D))

        for i in range(B):
            Lc = int(lengths[i].item())
            if Lc == 0:
                continue
            idx = indices_list[i]
            z_chunks = z_down[i, :Lc]   # (Lc, D)
            P_chunks = P_down[i, :Lc]   # (Lc,)

            # Smoothing: EMA over chunk sequence (eq. 5)
            z_smooth = torch.empty_like(z_chunks)
            prev = torch.zeros(D, device=device, dtype=z_down.dtype)
            for t in range(Lc):
                z_curr = P_chunks[t] * z_chunks[t] + (1.0 - P_chunks[t]) * prev
                z_smooth[t] = z_curr
                prev = z_curr

            # Causal expansion: repeat each smoothed chunk until next boundary (eq. 8)
            for j in range(Lc):
                start = int(idx[j].item())
                end = int(idx[j + 1].item()) if j + 1 < Lc else L
                z_full[i, start:end] = z_smooth[j]

        # Confidence weighting with a straight-through style trick (eq. 6-9)
        c = torch.where(b > 0.5, p, 1.0 - p)     # (B, L)
        ste_c = c + (1.0 - c).detach()          # forward uses 1, gradient flows through c
        z_full = z_full * ste_c.unsqueeze(-1)
        return z_full


class HNetTinyLM(nn.Module):
    """
    Byte-level, single-stage H-Net style language model.

    Components:
      - Byte embedding + positional embedding.
      - Encoder: a few Transformer layers on the full sequence.
      - DynamicChunkingLayer: learn boundaries and compress.
      - Main network: several Transformer layers on the compressed sequence.
      - Dechunking: smooth and upsample back to full resolution.
      - Decoder: a few Transformer layers on the full sequence.
      - LM head: next-byte prediction.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        n_layers_encoder: int = 2,
        n_layers_main: int = 4,
        n_layers_decoder: int = 2,
        max_seq_len: int = 512,
        target_compression_N: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.encoder_layers = nn.ModuleList(
            [SimpleTransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers_encoder)]
        )
        self.dc = DynamicChunkingLayer(d_model, target_compression_N=target_compression_N)
        self.main_layers = nn.ModuleList(
            [SimpleTransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers_main)]
        )
        self.decoder_layers = nn.ModuleList(
            [SimpleTransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers_decoder)]
        )

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self._causal_masks = {}

    def causal_mask(self, L, device):
        key = (L, device)
        if key in self._causal_masks:
            return self._causal_masks[key]
        mask = torch.full((L, L), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)  # upper triangle gets -inf, others 0
        self._causal_masks[key] = mask
        return mask

    def forward(self, input_ids):
        """
        input_ids: (B, L) integer tokens in [0, vocab_size)

        Returns:
          logits: (B, L, vocab_size)
          aux_losses: dict with 'ratio_loss'
        """
        B, L = input_ids.shape
        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}")

        device = input_ids.device
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.dropout(x)

        enc_mask = self.causal_mask(L, device)
        for layer in self.encoder_layers:
            x = layer(x, attn_mask=enc_mask)

        x_down, P_down, aux_dc, ratio_loss = self.dc(x)

        _, Lc_max, _ = x_down.shape
        main_mask = self.causal_mask(Lc_max, device)
        for layer in self.main_layers:
            x_down = layer(x_down, attn_mask=main_mask)

        z_full = self.dc.dechunk(x_down, P_down, aux_dc)

        for layer in self.decoder_layers:
            z_full = layer(z_full, attn_mask=enc_mask)

        z_full = self.ln_final(z_full)
        logits = self.lm_head(z_full)
        return logits, {"ratio_loss": ratio_loss}


if __name__ == "__main__":
    # Quick smoke test
    model = HNetTinyLM()
    x = torch.randint(0, 256, (2, 64))
    logits, aux = model(x)
    print("logits:", logits.shape, "ratio_loss:", aux["ratio_loss"].item())
