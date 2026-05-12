"""
SMILES Transformer (Autoregressive Language Model)
====================================================
GPT-2-style decoder-only Transformer for SMILES generation.

Architecture
------------
- Token embedding + sinusoidal positional encoding
- N stacked TransformerBlocks:
    LayerNorm → CausalSelfAttention (multi-head, causal mask) → residual
    LayerNorm → FFN (d_model → 4*d_model → d_model, GELU) → residual
- Final LayerNorm → linear projection to vocab_size (language model head)

Generation
----------
Supports greedy, temperature, top-k, nucleus (top-p) sampling, and beam search.
The main ``generate`` method returns decoded SMILES strings via the attached tokenizer.

References
----------
Radford et al. (2019) Language Models are Unsupervised Multitask Learners.
Vaswani et al. (2017) Attention Is All You Need.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Adds position information to token embeddings:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Parameters
    ----------
    d_model : int
        Embedding dimensionality.
    max_len : int
        Maximum sequence length.
    dropout : float
        Dropout probability applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                      # [L, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [L, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer (moves with .to(device), not a parameter)
        self.register_buffer("pe", pe.unsqueeze(0))             # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to embeddings.

        Parameters
        ----------
        x : Tensor of shape [B, L, D]

        Returns
        -------
        Tensor of shape [B, L, D]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Causal Self-Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (look-ahead) mask.

    Each position can only attend to previous positions and itself, which is
    required for autoregressive sequence generation.

    Parameters
    ----------
    d_model : int
        Model dimensionality (must be divisible by n_heads).
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout on attention weights.
    max_len : int
        Maximum sequence length (used to pre-compute the causal mask).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: upper-triangular boolean mask — positions j > i are masked
        causal_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal_mask)        # [L, L]

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute causal multi-head self-attention.

        Parameters
        ----------
        x : Tensor of shape [B, L, D]
        padding_mask : BoolTensor of shape [B, L], True = padding position

        Returns
        -------
        output : Tensor of shape [B, L, D]
        attn_weights : Tensor of shape [B, n_heads, L, L]
        """
        B, L, D = x.shape

        qkv = self.qkv_proj(x)                                  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)                          # each [B, L, D]

        # Reshape to [B, n_heads, L, d_head]
        def _split_heads(t):
            return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = _split_heads(q), _split_heads(k), _split_heads(v)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, L, L]

        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:L, :L].unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply key padding mask (marks padding positions)
        if padding_mask is not None:
            # padding_mask: [B, L] → [B, 1, 1, L]
            attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)
        # Replace NaN arising from all-masked rows (pure padding)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)                      # [B, H, L, d_head]
        out = out.transpose(1, 2).contiguous().view(B, L, D)     # [B, L, D]
        out = self.resid_dropout(self.out_proj(out))
        return out, attn_weights


# ---------------------------------------------------------------------------
# Feed-Forward Network block
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation.

    FFN(x) = GELU(x W1 + b1) W2 + b2
    The inner dimension is expanded by factor ``expand`` (default 4).
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single GPT-style Transformer block (Pre-Norm variant).

    Pre-Norm: LayerNorm is applied *before* the sublayer (more stable training).
    Residual connections wrap both sublayers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply one Transformer block.

        Parameters
        ----------
        x : Tensor [B, L, D]
        padding_mask : BoolTensor [B, L], optional

        Returns
        -------
        x : Tensor [B, L, D]
        attn_weights : Tensor [B, H, L, L]
        """
        attn_out, attn_weights = self.attn(self.ln1(x), padding_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights


# ---------------------------------------------------------------------------
# SMILES Transformer (main model)
# ---------------------------------------------------------------------------

class SmilesTransformer(nn.Module):
    """GPT-2-style decoder-only Transformer for autoregressive SMILES generation.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size (number of unique tokens).
    max_len : int
        Maximum token sequence length.
    d_model : int
        Embedding and hidden dimensionality (default 256).
    n_heads : int
        Number of attention heads (default 8).
    n_layers : int
        Number of Transformer blocks (default 6).
    d_ff : int
        Inner FFN dimensionality (default 1024, i.e. 4×d_model).
    dropout : float
        Dropout probability throughout the model.
    pad_id : int
        Token ID used for padding (used to create padding masks).
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.pad_id = pad_id

        # Input layers
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len + 2, dropout=dropout)

        # Transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_len + 2)
            for _ in range(n_layers)
        ])

        # Output layers
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding and output projection weights
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply GPT-2-style weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of (trainable) parameters."""
        params = filter(lambda p: p.requires_grad, self.parameters()) if trainable_only else self.parameters()
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the Transformer.

        Parameters
        ----------
        x : LongTensor of shape [B, L]
            Input token IDs.
        targets : LongTensor of shape [B, L], optional
            Shifted target token IDs for computing cross-entropy loss.
            If provided, loss is returned alongside logits.

        Returns
        -------
        logits : FloatTensor of shape [B, L, vocab_size]
        loss : FloatTensor scalar or None
        """
        B, L = x.shape
        padding_mask = (x == self.pad_id)                       # [B, L], True = pad

        h = self.token_embedding(x)                             # [B, L, D]
        h = self.pos_encoding(h)                                # [B, L, D]

        for block in self.blocks:
            h, _ = block(h, padding_mask)

        h = self.ln_final(h)                                    # [B, L, D]
        logits = self.lm_head(h)                                # [B, L, V]

        loss = None
        if targets is not None:
            # Flatten for cross-entropy; ignore PAD positions
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=self.pad_id,
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Embeddings utility
    # ------------------------------------------------------------------

    def get_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return contextual representations for a token sequence.

        Parameters
        ----------
        token_ids : LongTensor of shape [L] or [1, L]

        Returns
        -------
        FloatTensor of shape [L, d_model]
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        h = self.token_embedding(token_ids)
        h = self.pos_encoding(h)
        padding_mask = (token_ids == self.pad_id)
        for block in self.blocks:
            h, _ = block(h, padding_mask)
        h = self.ln_final(h)
        return h.squeeze(0)                                     # [L, D]

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Zero out logits below the top-k threshold."""
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = v[..., -1].unsqueeze(-1)
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def _nucleus_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus (top-p) sampling: keep smallest set of tokens with cumulative
        probability ≥ top_p, then sample from that distribution.

        Parameters
        ----------
        logits : FloatTensor of shape [vocab_size]
        top_p : float
            Cumulative probability threshold (0 < top_p ≤ 1).

        Returns
        -------
        next_token : LongTensor scalar
        """
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens beyond the nucleus (first position where cumsum ≥ top_p)
        remove_mask = cumulative - sorted_probs > top_p
        sorted_probs[remove_mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()

        next_token = sorted_indices[torch.multinomial(sorted_probs, num_samples=1)]
        return next_token

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        bos_token: int,
        max_len: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 0.9,
        n_samples: int = 1,
        tokenizer=None,
        device: Optional[torch.device] = None,
        greedy: bool = False,
    ) -> List[str]:
        """Generate SMILES strings autoregressively.

        Strategy priority: greedy > top_k (if set) > top_p nucleus sampling.

        Parameters
        ----------
        bos_token : int
            BOS token ID to start generation.
        max_len : int
            Maximum number of tokens to generate (excluding BOS).
        temperature : float
            Softmax temperature. Lower = sharper distribution.
        top_k : int, optional
            If set, restrict sampling to the top-k logits.
        top_p : float
            Nucleus sampling cumulative probability threshold.
        n_samples : int
            Number of sequences to generate.
        tokenizer : SmilesTokenizer, optional
            If provided, sequences are decoded to SMILES strings.
            Otherwise raw token ID lists are returned.
        device : torch.device, optional
            Target device; defaults to current model device.
        greedy : bool
            If True, always select the argmax token (temperature ignored).

        Returns
        -------
        List[str] or List[List[int]]
            Generated SMILES strings (if tokenizer provided) or raw token IDs.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        eos_id = tokenizer.eos_id if tokenizer is not None else None
        pad_id = tokenizer.pad_id if tokenizer is not None else self.pad_id

        # Start with BOS token for all samples: [n_samples, 1]
        seqs = torch.full((n_samples, 1), bos_token, dtype=torch.long, device=device)
        finished = torch.zeros(n_samples, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits, _ = self.forward(seqs)          # [n_samples, cur_len, V]
            next_logits = logits[:, -1, :]           # [n_samples, V]

            if greedy:
                next_tokens = next_logits.argmax(dim=-1, keepdim=True)   # [n, 1]
            else:
                next_tokens = []
                for i in range(n_samples):
                    lg = next_logits[i] / max(temperature, 1e-8)
                    if top_k is not None and top_k > 0:
                        lg = self._top_k_filter(lg, top_k)
                    tok = self._nucleus_sample(lg, top_p)
                    next_tokens.append(tok)
                next_tokens = torch.stack(next_tokens).unsqueeze(1)      # [n, 1]

            # Mask finished sequences (replace next token with PAD)
            next_tokens[finished.unsqueeze(1)] = pad_id
            seqs = torch.cat([seqs, next_tokens], dim=1)                 # [n, cur_len+1]

            # Mark newly finished sequences
            if eos_id is not None:
                finished = finished | (next_tokens.squeeze(1) == eos_id)
                if finished.all():
                    break

        # Decode
        if tokenizer is not None:
            results = []
            for seq in seqs.cpu().numpy():
                smi = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
                results.append(smi)
            return results

        return [seq.tolist() for seq in seqs.cpu()]

    # ------------------------------------------------------------------
    # Beam search
    # ------------------------------------------------------------------

    @torch.no_grad()
    def beam_search(
        self,
        bos_token: int,
        beam_width: int = 5,
        max_len: int = 100,
        tokenizer=None,
        device: Optional[torch.device] = None,
        length_penalty: float = 1.0,
    ) -> List[Tuple[str, float]]:
        """Beam search decoding for single-sequence (most likely) generation.

        Parameters
        ----------
        bos_token : int
            Starting token ID.
        beam_width : int
            Number of beams to maintain.
        max_len : int
            Maximum sequence length.
        tokenizer : SmilesTokenizer, optional
            Used for decoding and EOS detection.
        device : torch.device, optional
        length_penalty : float
            Score normalisation: score / length^length_penalty.

        Returns
        -------
        List[Tuple[str, float]]
            List of (smiles, score) pairs sorted by score descending,
            up to beam_width completed sequences.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        eos_id = tokenizer.eos_id if tokenizer else None

        # Each beam: (sequence_tensor [1, L], cumulative_log_prob)
        beams: List[Tuple[torch.Tensor, float]] = [
            (torch.tensor([[bos_token]], device=device), 0.0)
        ]
        completed: List[Tuple[List[int], float]] = []

        for _ in range(max_len):
            if not beams:
                break
            all_candidates: List[Tuple[torch.Tensor, float]] = []

            for seq, cum_logp in beams:
                logits, _ = self.forward(seq)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)  # [V]
                topk_logp, topk_ids = torch.topk(log_probs, beam_width)

                for logp, tid in zip(topk_logp.tolist(), topk_ids.tolist()):
                    new_seq = torch.cat([seq, torch.tensor([[tid]], device=device)], dim=1)
                    new_score = cum_logp + logp
                    if eos_id is not None and tid == eos_id:
                        length = new_seq.size(1)
                        normalised = new_score / (length ** length_penalty)
                        completed.append((new_seq.squeeze(0).tolist(), normalised))
                    else:
                        all_candidates.append((new_seq, new_score))

            # Prune to top beam_width active beams
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

        # Add any unfinished beams as fallback completions
        for seq, score in beams:
            length = seq.size(1)
            normalised = score / max(length ** length_penalty, 1)
            completed.append((seq.squeeze(0).tolist(), normalised))

        completed.sort(key=lambda x: x[1], reverse=True)
        completed = completed[:beam_width]

        if tokenizer is not None:
            return [
                (tokenizer.decode(ids, skip_special_tokens=True), score)
                for ids, score in completed
            ]
        return [(ids, score) for ids, score in completed]

    # ------------------------------------------------------------------
    # Log-probability utility (used by RL)
    # ------------------------------------------------------------------

    def log_prob_sequence(
        self,
        token_ids: torch.Tensor,
        pad_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute per-token log-probabilities for a batch of sequences.

        Used during RL training to retrieve log π_θ(a|s) for REINFORCE.

        Parameters
        ----------
        token_ids : LongTensor of shape [B, L]
            Full sequences including BOS (and EOS if present).
        pad_id : int, optional
            PAD token ID; positions at PAD are given log-prob = 0.

        Returns
        -------
        log_probs : FloatTensor of shape [B]
            Sum of per-token log-probs for each sequence in the batch.
        """
        if pad_id is None:
            pad_id = self.pad_id

        # Input: tokens[:-1], Target: tokens[1:]
        inputs = token_ids[:, :-1]
        targets = token_ids[:, 1:]

        logits, _ = self.forward(inputs)                        # [B, L-1, V]
        log_probs_all = F.log_softmax(logits, dim=-1)           # [B, L-1, V]

        # Gather log-prob of actual next token
        target_log_probs = log_probs_all.gather(
            dim=2, index=targets.unsqueeze(-1)
        ).squeeze(-1)                                           # [B, L-1]

        # Zero out PAD positions
        non_pad_mask = (targets != pad_id).float()
        target_log_probs = target_log_probs * non_pad_mask

        return target_log_probs.sum(dim=-1)                     # [B]

    def __repr__(self) -> str:
        return (
            f"SmilesTransformer("
            f"vocab={self.vocab_size}, d_model={self.d_model}, "
            f"params={self.num_parameters():,})"
        )
