"""
REINFORCE Policy Gradient for Multi-Objective Molecular Optimisation
======================================================================
Adapts a pre-trained SMILES Transformer as a policy network and
optimises it towards a composite reward using the REINFORCE algorithm.

Reward decomposition
--------------------
r(m) = w₁·QED(m) + w₂·SA_scaled(m) + w₃·target_sim(m)
       + w₄·selectivity(m) + w₅·novelty(m) − w₆·penalty(m)

Reference: Olivecrona et al. (2017) "Molecular de-novo design through
deep reinforcement learning" — J. Cheminformatics 9, 48.

Python  : >= 3.10
PyTorch : >= 2.0
RDKit   : >= 2023.09
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED as RDKitQED
from rdkit.Chem import rdMolDescriptors, DataStructs, FilterCatalog

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Synthetic Accessibility (SA score) approximation ─────────────────────────

def sa_score_approx(mol: Chem.Mol) -> float:
    """
    Approximate SA score using ring complexity and stereocentres.
    Returns value in [1, 10] where 1 = easy to synthesise.
    """
    if mol is None:
        return 10.0
    n_rings  = rdMolDescriptors.CalcNumRings(mol)
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_heavy  = mol.GetNumHeavyAtoms()
    score = 1.0 + 0.25 * n_rings + 0.4 * n_stereo + 0.015 * n_heavy
    return float(min(10.0, score))


def sa_score_scaled(mol: Chem.Mol) -> float:
    """SA score scaled to [0, 1] where 1 = easiest."""
    return 1.0 - (sa_score_approx(mol) - 1.0) / 9.0


# ── PAINS filter ──────────────────────────────────────────────────────────────

def _build_pains_catalog():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog.FilterCatalog(params)

_PAINS = _build_pains_catalog()


def is_pains(mol: Chem.Mol) -> bool:
    return bool(_PAINS.GetFirstMatch(mol))


# ── Molecular reward function ─────────────────────────────────────────────────

class MolecularReward:
    """
    Composite reward for drug-like CDK4/6 inhibitor generation.

    Components
    ----------
    qed         : Drug-likeness quantified by QED [0,1]
    sa          : Synthetic accessibility [0,1] (1 = easy)
    target_sim  : Max Tanimoto to known CDK4/6 actives (scaffold hop encouraged)
    selectivity : 1 − max Tanimoto to CDK2 anti-targets
    novelty     : 1 if molecule not in reference set, 0 otherwise
    validity    : 0 if SMILES invalid (hard constraint)
    pains       : −0.3 if PAINS alerts present

    Weights (default)
    -----------------
    w_qed=0.35, w_sa=0.15, w_sim=0.25, w_sel=0.15, w_novel=0.10
    """

    def __init__(
        self,
        reference_actives: List[str],     # CDK4/6 actives SMILES
        anti_targets: Optional[List[str]] = None,  # CDK2 SMILES (selectivity counter)
        w_qed:   float = 0.35,
        w_sa:    float = 0.15,
        w_sim:   float = 0.25,
        w_sel:   float = 0.15,
        w_novel: float = 0.10,
        sim_threshold: float = 0.3,  # below this: low scaffold similarity → reward
        max_sim_cap:   float = 0.7,  # above this: scaffold is not novel enough
    ):
        self.w_qed   = w_qed
        self.w_sa    = w_sa
        self.w_sim   = w_sim
        self.w_sel   = w_sel
        self.w_novel = w_novel
        self.sim_threshold = sim_threshold
        self.max_sim_cap   = max_sim_cap

        # Pre-compute reference fingerprints
        self.ref_fps = self._compute_fps(reference_actives)
        self.anti_fps = self._compute_fps(anti_targets or [])
        self._seen_smiles: set = set(reference_actives)

    @staticmethod
    def _compute_fps(smiles_list: List[str]) -> List:
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        return fps

    def qed_score(self, mol: Chem.Mol) -> float:
        try:
            return float(RDKitQED.qed(mol))
        except Exception:
            return 0.0

    def target_similarity(self, mol: Chem.Mol) -> float:
        """Max Tanimoto to reference actives."""
        if not self.ref_fps:
            return 0.0
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, self.ref_fps)
        return float(max(sims)) if sims else 0.0

    def selectivity_score(self, mol: Chem.Mol) -> float:
        """1 - max similarity to anti-targets."""
        if not self.anti_fps:
            return 1.0
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, self.anti_fps)
        return 1.0 - float(max(sims)) if sims else 1.0

    def novelty_score(self, mol: Chem.Mol, smiles: str) -> float:
        """1 if canonical SMILES not in training/reference set."""
        canon = Chem.MolToSmiles(mol)
        if canon in self._seen_smiles:
            return 0.0
        self._seen_smiles.add(canon)
        return 1.0

    def diversity_bonus(self, mol: Chem.Mol, batch_fps: List) -> float:
        """
        Bonus for being dissimilar to molecules already generated in this batch.
        Encourages diversity within a generated set.
        """
        if not batch_fps:
            return 0.5
        fp   = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, batch_fps)
        mean_sim = float(np.mean(sims))
        # Diversity bonus: reward low mean similarity (but not completely random)
        return float(np.clip(1.0 - mean_sim, 0, 1))

    def __call__(
        self,
        smiles: str,
        batch_fps: Optional[List] = None,
    ) -> Dict[str, float]:
        """
        Compute all reward components for a generated SMILES.

        Returns
        -------
        dict with 'total' reward and individual component scores.
        """
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return {"total": 0.0, "valid": 0.0,
                    "qed": 0.0, "sa": 0.0, "sim": 0.0, "sel": 0.0, "novel": 0.0}

        qed  = self.qed_score(mol)
        sa   = sa_score_scaled(mol)
        sim  = self.target_similarity(mol)
        sel  = self.selectivity_score(mol)
        nov  = self.novelty_score(mol, smiles)
        mol_is_pains = is_pains(mol)
        pains_pen = -0.3 if mol_is_pains else 0.0

        # Scaffold hop: reward molecules with intermediate similarity
        # (similar enough to be active, dissimilar enough to be novel)
        if sim < self.sim_threshold:
            sim_reward = sim * 0.5   # too dissimilar from known actives
        elif sim > self.max_sim_cap:
            sim_reward = self.max_sim_cap * 0.8  # too similar — not novel
        else:
            sim_reward = sim   # sweet spot

        total = (
            self.w_qed   * qed +
            self.w_sa    * sa +
            self.w_sim   * sim_reward +
            self.w_sel   * sel +
            self.w_novel * nov +
            pains_pen
        )
        total = float(np.clip(total, 0.0, 1.0))

        return {
            "total":    total,
            "valid":    1.0,
            "qed":      qed,
            "sa":       sa,
            "sim":      sim,
            "sel":      sel,
            "novel":    nov,
            "pains_ok": 0.0 if mol_is_pains else 1.0,
        }

    def batch_rewards(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Vectorised reward computation over a list of SMILES."""
        batch_fps = []
        details   = []
        rewards   = []
        for smi in smiles_list:
            r = self(smi, batch_fps)
            rewards.append(r["total"])
            details.append(r)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                batch_fps.append(
                    AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                )
        return np.array(rewards, dtype=np.float32), details


# ── REINFORCE optimiser ───────────────────────────────────────────────────────

class RLOptimizer:
    """
    REINFORCE policy gradient optimiser for the SMILES Transformer.

    Algorithm
    ---------
    1. Sample n_samples SMILES from the policy network (Transformer)
    2. Compute rewards r(m_i) for each molecule
    3. Compute baseline b = exponential moving average of past rewards
    4. Policy gradient: ∇J = E[(r(m) - b) * ∇ log π(m)]
    5. Update policy via Adam

    Parameters
    ----------
    model      : SmilesTransformer (policy network)
    tokenizer  : SmilesTokenizer
    reward_fn  : MolecularReward (or any callable SMILES → reward dict)
    lr         : learning rate
    baseline   : initial baseline (set to ~0.5 for typical rewards)
    ema_decay  : exponential moving average decay for baseline
    """

    def __init__(
        self,
        model,       # SmilesTransformer
        tokenizer,   # SmilesTokenizer
        reward_fn: MolecularReward,
        lr: float         = 5e-5,
        baseline: float   = 0.45,
        ema_decay: float  = 0.95,
        max_grad_norm: float = 0.5,
    ):
        self.model         = model
        self.tokenizer     = tokenizer
        self.reward_fn     = reward_fn
        self.baseline      = baseline
        self.ema_decay     = ema_decay
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.history: List[Dict] = []

    def _moving_average_baseline(self, rewards: np.ndarray) -> float:
        """Update EMA baseline and return current value."""
        batch_mean     = float(rewards.mean())
        self.baseline  = (self.ema_decay * self.baseline +
                          (1 - self.ema_decay) * batch_mean)
        return self.baseline

    def _sample_with_logprobs(
        self,
        n_samples: int,
        max_len: int = 80,
        temperature: float = 1.0,
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Sample n_samples SMILES from the policy and return
        (smiles_list, log_prob_tensors).
        Each log_prob tensor is the sum of log P(token_t | token_<t).
        """
        self.model.train()
        smiles_list  = []
        log_probs    = []
        device       = next(self.model.parameters()).device

        bos_token = self.tokenizer.bos_id
        eos_token = self.tokenizer.eos_id
        pad_token = self.tokenizer.pad_id

        for _ in range(n_samples):
            # Initialise with BOS
            tokens  = torch.tensor([[bos_token]], dtype=torch.long, device=device)
            lp_list = []

            for _ in range(max_len - 1):
                logits, _ = self.model(tokens)     # [1, L, V]
                next_logits = logits[0, -1, :] / temperature   # [V]
                probs = F.softmax(next_logits, dim=-1)
                # Sample
                next_token = torch.multinomial(probs, 1)       # [1]
                log_p = torch.log(probs[next_token] + 1e-9)
                lp_list.append(log_p.squeeze())
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == eos_token:
                    break

            token_seq = tokens[0].cpu().tolist()
            smi = self.tokenizer.decode(token_seq)
            smiles_list.append(smi if smi else "C")
            lp_sum = torch.stack(lp_list).sum() if lp_list else torch.tensor(0.0, device=device)
            log_probs.append(lp_sum)

        return smiles_list, log_probs

    def step(
        self,
        n_samples: int = 64,
        temperature: float = 1.0,
    ) -> Dict:
        """
        One REINFORCE update step.

        Returns
        -------
        dict: mean_reward, valid_frac, unique_frac, baseline, policy_loss
        """
        smiles_list, log_probs = self._sample_with_logprobs(n_samples, temperature=temperature)

        # Validity
        valid_mask = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            valid_mask.append(mol is not None)
        valid_frac = float(sum(valid_mask)) / n_samples

        # Compute rewards
        rewards_arr, details = self.reward_fn.batch_rewards(smiles_list)

        # Update baseline
        baseline_val = self._moving_average_baseline(rewards_arr)

        # REINFORCE loss: -E[(r - b) * log π(m)]
        advantages  = torch.tensor(rewards_arr - baseline_val, dtype=torch.float,
                                   device=next(self.model.parameters()).device)
        log_prob_t  = torch.stack(log_probs)          # [B]
        policy_loss = -(advantages * log_prob_t).mean()

        # Entropy regularisation to prevent premature convergence
        # (added implicitly via temperature; could also add explicit H term)

        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Uniqueness
        valid_smiles = [smi for smi, v in zip(smiles_list, valid_mask) if v]
        canonical    = set()
        for smi in valid_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical.add(Chem.MolToSmiles(mol))
        unique_frac = len(canonical) / max(len(valid_smiles), 1)

        metrics = {
            "mean_reward":    float(rewards_arr.mean()),
            "max_reward":     float(rewards_arr.max()),
            "valid_frac":     valid_frac,
            "unique_frac":    unique_frac,
            "baseline":       baseline_val,
            "policy_loss":    float(policy_loss.item()),
            "mean_qed":       float(np.mean([d["qed"] for d in details])),
            "mean_sa":        float(np.mean([d["sa"]  for d in details])),
            "novel_frac":     float(np.mean([d["novel"] for d in details])),
        }
        self.history.append(metrics)
        return metrics

    def train(
        self,
        n_steps: int         = 500,
        eval_every: int      = 50,
        n_samples: int       = 64,
        temperature: float   = 1.0,
        verbose: bool        = True,
    ) -> List[Dict]:
        """
        Full RL training loop.

        Parameters
        ----------
        n_steps   : number of policy gradient steps
        eval_every: print progress every N steps
        """
        print(f"\nStarting REINFORCE optimisation ({n_steps} steps, {n_samples} samples/step)")
        print(f"  Initial baseline: {self.baseline:.3f}")
        print("-" * 60)

        for step in range(1, n_steps + 1):
            metrics = self.step(n_samples=n_samples, temperature=temperature)
            if verbose and step % eval_every == 0:
                print(f"  Step {step:4d}/{n_steps} | "
                      f"Reward={metrics['mean_reward']:.3f} "
                      f"(max={metrics['max_reward']:.3f}) | "
                      f"Valid={metrics['valid_frac']*100:.0f}% | "
                      f"Unique={metrics['unique_frac']*100:.0f}% | "
                      f"Novel={metrics['novel_frac']*100:.0f}% | "
                      f"QED={metrics['mean_qed']:.3f}")

        print(f"\nFinal baseline: {self.baseline:.3f}")
        return self.history


# ── Pre-training utility ──────────────────────────────────────────────────────

def pretrain_on_smiles(
    model,
    tokenizer,
    smiles_list: List[str],
    epochs: int    = 10,
    batch_size: int = 64,
    lr: float       = 3e-4,
    device: str     = "cpu",
) -> List[float]:
    """
    Language-model pre-training: teacher-forced cross-entropy on SMILES tokens.

    Returns list of per-epoch average cross-entropy losses.
    """
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    # Tokenise all SMILES
    max_len  = min(tokenizer.max_length, 120)
    encoded  = tokenizer.batch_encode(smiles_list, max_len=max_len, pad=True)  # [N, L]
    N        = encoded.shape[0]
    losses   = []

    print(f"\nPre-training on {N} SMILES | {epochs} epochs | batch={batch_size}")

    for epoch in range(epochs):
        model.train()
        perm     = np.random.permutation(N)
        ep_loss  = 0.0
        n_batch  = 0

        for start in range(0, N, batch_size):
            idx  = perm[start:start + batch_size]
            batch = torch.tensor(encoded[idx], dtype=torch.long, device=device)

            # Input = all tokens except last; Target = all tokens except first
            x_in  = batch[:, :-1]
            y_tgt = batch[:, 1:]

            logits, loss = model(x_in, targets=y_tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_batch += 1

        scheduler.step()
        avg = ep_loss / max(n_batch, 1)
        losses.append(avg)
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg:.4f}")

    return losses
