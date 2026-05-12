"""
ESM-2 Protein Language Model for Antibody Engineering
=======================================================
Fine-tunes ESM-2 (650M) embeddings for three simultaneous prediction tasks:
  1. Binding affinity prediction (regression, log pKd)
  2. Aggregation propensity (binary classification)
  3. Thermal stability Tm prediction (regression, °C)

Zero-shot mutant effect prediction: masked language modelling log-likelihood
ratios provide fitness estimates without any fine-tuning labels.

When ESM-2 is not installed, a biophysics-informed mock embedder is used
that computes a 480-dim representation from physicochemical features via
random projection — enough to demonstrate the full downstream pipeline.

Reference
---------
Lin et al. (2023) "Evolutionary-scale prediction of atomic-level protein
structure with a language model" — Science 379, 1123–1130.

Python  : >= 3.10
PyTorch : >= 2.0
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_features import (
    encode_sequence, compute_sequence_features, AA_PROPERTIES, AMINO_ACIDS
)

warnings.filterwarnings("ignore")

ESM_EMBEDDING_DIM = 480   # mock dimension (real esm2_t33 = 1280)


# ── Mock ESM-2 Embedder ───────────────────────────────────────────────────────

class ESMEmbedder:
    """
    Wrapper around ESM-2 that falls back to a biophysics-informed mock
    when the esm package is not installed.

    Mock strategy
    -------------
    1. Compute physicochemical features [L, 30] (combined encoding)
    2. Apply a fixed random projection R ∈ R^{30 × ESM_DIM} to get [L, ESM_DIM]
    3. Mean-pool → sequence representation [ESM_DIM]
    4. Add small Gaussian noise to simulate model stochasticity
    The projection is seeded so results are reproducible.
    """

    def __init__(
        self,
        model_name: str  = "esm2_t33_650M_UR50D",
        device: str      = "cpu",
        mock: bool       = True,
    ):
        self.model_name = model_name
        self.device     = device
        self.mock       = mock
        self._esm_model = None
        self._alphabet  = None

        if not mock:
            try:
                import esm as esm_pkg
                self._esm_model, self._alphabet = esm_pkg.pretrained.load_model_and_alphabet(model_name)
                self._esm_model = self._esm_model.eval().to(device)
                self._batch_converter = self._alphabet.get_batch_converter()
                self.embedding_dim = self._esm_model.embed_dim
                self.mock = False
                print(f"Loaded ESM-2 model: {model_name} ({self.embedding_dim}-dim)")
            except (ImportError, Exception) as e:
                warnings.warn(f"ESM not available ({e}). Using mock embedder.")
                self.mock = True

        if self.mock:
            self.embedding_dim = ESM_EMBEDDING_DIM
            rng = np.random.default_rng(42)
            self._proj = rng.standard_normal((30, ESM_EMBEDDING_DIM)).astype(np.float32)
            # Orthonormalise: Vt from SVD has orthonormal rows, shape (30, ESM_DIM)
            # feats (L, 30) @ Vt (30, ESM_DIM) → (L, ESM_DIM)
            _, _, Vt = np.linalg.svd(self._proj, full_matrices=False)
            self._proj = Vt
            print(f"Using mock ESM embedder ({ESM_EMBEDDING_DIM}-dim)")

    def encode(self, sequences: List[str]) -> np.ndarray:
        """
        Encode sequences to sequence-level embeddings.

        Parameters
        ----------
        sequences : List of amino acid sequences

        Returns
        -------
        np.ndarray [B, embedding_dim]
        """
        if not self.mock and self._esm_model is not None:
            return self._encode_real(sequences)
        return self._encode_mock(sequences)

    def _encode_mock(self, sequences: List[str]) -> np.ndarray:
        embeddings = []
        for seq in sequences:
            # Physicochemical features [L, 30]
            feats = encode_sequence(seq, method="combined")  # [L, 30]
            # Project to high-dim [L, ESM_DIM]: (L, 30) @ (30, ESM_DIM) → (L, ESM_DIM)
            proj = feats @ self._proj                         # [L, ESM_DIM]
            # Mean pool + L2 norm
            emb = proj.mean(axis=0)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            # Add very small noise for realistic variation
            rng = np.random.default_rng(abs(hash(seq)) % (2**31))
            emb += rng.normal(0, 0.01, emb.shape).astype(np.float32)
            embeddings.append(emb)
        return np.stack(embeddings, axis=0)

    def _encode_real(self, sequences: List[str]) -> np.ndarray:
        """Encode using the real ESM-2 model."""
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, tokens = self._batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            results = self._esm_model(tokens, repr_layers=[self._esm_model.num_layers])
        token_reps = results["representations"][self._esm_model.num_layers]  # [B, L, D]
        seq_reps   = token_reps[:, 1:-1, :].mean(dim=1)                      # [B, D]
        return seq_reps.cpu().numpy()

    def encode_residue_level(self, sequence: str) -> np.ndarray:
        """
        Per-residue embeddings for a single sequence.

        Returns
        -------
        np.ndarray [L, embedding_dim]
        """
        if not self.mock and self._esm_model is not None:
            data = [("seq", sequence)]
            _, _, tokens = self._batch_converter(data)
            tokens = tokens.to(self.device)
            with torch.no_grad():
                results = self._esm_model(tokens, repr_layers=[self._esm_model.num_layers])
            return results["representations"][self._esm_model.num_layers][0, 1:-1].cpu().numpy()
        else:
            feats = encode_sequence(sequence, method="combined")
            return feats @ self._proj

    def zero_shot_mutation_score(
        self,
        wt_seq: str,
        position: int,
        mutant_aa: str,
    ) -> float:
        """
        Zero-shot mutation fitness score via masked log-likelihood ratio.
        score(i, m) = log P(m | context) − log P(wt_i | context)

        Positive score = predicted improvement over WT.
        When using mock: estimated from physicochemical property differences.
        """
        if not self.mock and self._esm_model is not None:
            return self._real_mutant_score(wt_seq, position, mutant_aa)
        return self._mock_mutant_score(wt_seq, position, mutant_aa)

    def _mock_mutant_score(
        self, wt_seq: str, position: int, mutant_aa: str
    ) -> float:
        """
        Approximate mutation score from physicochemical properties.
        Heuristic: mutations that maintain charge, size, and hydrophobicity
        profile of the wildtype residue tend to be better tolerated.
        """
        if position >= len(wt_seq):
            return 0.0
        wt_aa = wt_seq[position]
        if wt_aa not in AA_PROPERTIES or mutant_aa not in AA_PROPERTIES:
            return 0.0

        wt_props  = AA_PROPERTIES[wt_aa]
        mut_props = AA_PROPERTIES[mutant_aa]

        # Penalise charge changes (usually deleterious)
        delta_charge = abs(mut_props["charge"] - wt_props["charge"])
        # Reward conservation of hydrophobicity class
        delta_hydro  = abs(mut_props["hydrophobicity"] - wt_props["hydrophobicity"])
        # Penalise dramatic size changes
        delta_mw     = abs(mut_props["mw"] - wt_props["mw"]) / 100.0

        # Mock score: ~ Blosum62 analogue
        score = -0.4 * delta_charge - 0.2 * delta_hydro - 0.1 * delta_mw
        # Add small random component to simulate ESM variance
        rng = np.random.default_rng(abs(hash(f"{wt_seq}_{position}_{mutant_aa}")) % (2**31))
        score += rng.normal(0, 0.1)
        return float(score)

    def _real_mutant_score(self, wt_seq, position, mutant_aa):
        """Real ESM masked language model scoring."""
        # Mask the position
        masked = list(wt_seq)
        masked[position] = self._alphabet.mask_tok
        masked_seq = "".join(masked)
        mutant_seq = list(wt_seq)
        mutant_seq[position] = mutant_aa
        mutant_seq = "".join(mutant_seq)

        with torch.no_grad():
            for seq_label, seq in [("masked", masked_seq)]:
                data = [(seq_label, seq)]
                _, _, tokens = self._batch_converter(data)
                tokens = tokens.to(self.device)
                results = self._esm_model(tokens, repr_layers=[])
                logits = results["logits"][0, position + 1]  # +1 for BOS
                log_probs = F.log_softmax(logits, dim=-1)
                wt_idx  = self._alphabet.get_idx(wt_seq[position])
                mut_idx = self._alphabet.get_idx(mutant_aa)
                return float(log_probs[mut_idx] - log_probs[wt_idx])


# ── Multi-task prediction head ────────────────────────────────────────────────

class MultiTaskAntibodyHead(nn.Module):
    """
    Shared trunk + 3 task-specific heads for:
      1. binding_affinity (regression)
      2. aggregation_prob (binary classification)
      3. thermal_stability (regression, Tm in °C)
    """

    def __init__(
        self,
        input_dim: int   = ESM_EMBEDDING_DIM,
        trunk_hidden: int = 256,
        dropout: float    = 0.3,
    ):
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_hidden * 2),
            nn.LayerNorm(trunk_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden * 2, trunk_hidden),
            nn.LayerNorm(trunk_hidden),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )

        # Task heads
        self.binding_head    = nn.Linear(trunk_hidden, 1)
        self.aggregation_head = nn.Sequential(nn.Linear(trunk_hidden, 1), nn.Sigmoid())
        self.stability_head   = nn.Linear(trunk_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        embeddings : [B, input_dim]

        Returns
        -------
        dict: binding [B,1], aggregation [B,1], stability [B,1]
        """
        h = self.trunk(embeddings)
        return {
            "binding":     self.binding_head(h),
            "aggregation": self.aggregation_head(h),
            "stability":   self.stability_head(h),
        }


# ── Uncertainty-weighted multi-task loss ─────────────────────────────────────

class MultitaskLoss(nn.Module):
    """
    Kendall & Gal (2018) uncertainty-weighted multi-task loss.
    Learnable log variance parameters automatically balance task contributions.
    """

    def __init__(self, n_tasks: int = 3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(
        self,
        preds:   Dict[str, torch.Tensor],
        targets: Dict[str, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        task_losses = []
        task_names  = ["binding", "aggregation", "stability"]
        task_types  = ["regression", "classification", "regression"]

        for i, (name, t_type) in enumerate(zip(task_names, task_types)):
            y = targets.get(name)
            if y is None:
                task_losses.append(torch.tensor(0.0))
                continue
            p = preds[name].squeeze(-1)
            valid = ~torch.isnan(y)
            if valid.sum() == 0:
                task_losses.append(torch.tensor(0.0))
                continue
            p_v, y_v = p[valid], y[valid]
            if t_type == "regression":
                raw = F.mse_loss(p_v, y_v)
            else:
                raw = F.binary_cross_entropy(p_v, y_v)
            task_losses.append(raw)

        # Uncertainty weighting
        total = torch.tensor(0.0)
        for i, raw in enumerate(task_losses):
            precision = torch.exp(-self.log_vars[i])
            total     = total + (precision * raw + self.log_vars[i])

        details = {name: float(l.item())
                   for name, l in zip(task_names, task_losses)}
        return total, details


# ── Full antibody model ───────────────────────────────────────────────────────

class ESMAntibodyModel:
    """
    End-to-end antibody property prediction model.
    Wraps ESMEmbedder + MultiTaskAntibodyHead with sklearn-compatible API.
    """

    def __init__(
        self,
        mock: bool    = True,
        device: str   = "cpu",
        lr: float     = 3e-4,
        epochs: int   = 50,
        batch_size: int = 32,
    ):
        self.device     = device
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size

        self.embedder   = ESMEmbedder(mock=mock, device=device)
        self.head       = MultiTaskAntibodyHead(
            input_dim=self.embedder.embedding_dim
        ).to(device)
        self.loss_fn    = MultitaskLoss(n_tasks=3).to(device)
        self.optimizer  = torch.optim.AdamW(
            list(self.head.parameters()) + list(self.loss_fn.parameters()),
            lr=lr, weight_decay=1e-4
        )
        self._is_fitted = False

    def fit(
        self,
        sequences: List[str],
        binding:    Optional[np.ndarray] = None,
        aggregation: Optional[np.ndarray] = None,
        stability:  Optional[np.ndarray] = None,
    ) -> List[float]:
        """Train the head on available label sets (any can be None)."""
        print(f"Computing ESM embeddings for {len(sequences)} sequences...")
        embeddings = self.embedder.encode(sequences)
        emb_t = torch.tensor(embeddings, dtype=torch.float, device=self.device)

        def _maybe_tensor(arr):
            if arr is None:
                return None
            return torch.tensor(arr, dtype=torch.float, device=self.device)

        bind_t = _maybe_tensor(binding)
        aggr_t = _maybe_tensor(aggregation)
        stab_t = _maybe_tensor(stability)

        losses = []
        N = len(sequences)
        print(f"Training multi-task head for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            self.head.train()
            self.loss_fn.train()
            perm = np.random.permutation(N)
            ep_loss = 0.0
            n_batch = 0

            for start in range(0, N, self.batch_size):
                idx  = perm[start:start + self.batch_size]
                e_b  = emb_t[idx]
                preds = self.head(e_b)
                targets = {
                    "binding":     bind_t[idx] if bind_t is not None else None,
                    "aggregation": aggr_t[idx] if aggr_t is not None else None,
                    "stability":   stab_t[idx] if stab_t is not None else None,
                }
                loss, _ = self.loss_fn(preds, targets)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
                self.optimizer.step()
                ep_loss += loss.item()
                n_batch += 1

            avg = ep_loss / max(n_batch, 1)
            losses.append(avg)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} | loss={avg:.4f}")

        self._is_fitted = True
        return losses

    @torch.no_grad()
    def predict(self, sequences: List[str]) -> "pd.DataFrame":
        """
        Predict all three properties for a list of sequences.

        Returns
        -------
        pd.DataFrame with columns: sequence, binding_pred, aggregation_prob, stability_pred
        """
        import pandas as pd
        self.head.eval()
        embeddings = self.embedder.encode(sequences)
        emb_t = torch.tensor(embeddings, dtype=torch.float, device=self.device)
        preds = self.head(emb_t)
        return pd.DataFrame({
            "sequence":         sequences,
            "binding_pred":     preds["binding"].squeeze(-1).cpu().numpy(),
            "aggregation_prob": preds["aggregation"].squeeze(-1).cpu().numpy(),
            "stability_pred":   preds["stability"].squeeze(-1).cpu().numpy(),
        })
