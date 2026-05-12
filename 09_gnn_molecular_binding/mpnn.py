"""
Message Passing Neural Network (MPNN) for Molecular Property Prediction
=========================================================================
Pure-PyTorch implementation of the MPNN architecture from:
  Gilmer et al. (2017) "Neural Message Passing for Quantum Chemistry"
  https://arxiv.org/abs/1704.01212

Key design choices
------------------
* Edge-conditioned messages: the bond features gate the message content
* GRU-based node update: hidden state evolves over T message-passing steps
* Global attention pooling: learned soft-max weights aggregate node → graph repr
* Multi-task head: single shared GNN backbone → task-specific output layers
* Uncertainty weighting (Kendall & Gal 2018) for multi-task loss balancing

Python  : >= 3.10
PyTorch : >= 2.0
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utility: scatter operations (no PyG dependency) ──────────────────────────

def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    dim: int = 0,
) -> torch.Tensor:
    """Scatter-add src into a zero tensor of size dim_size along dim."""
    out = src.new_zeros(dim_size, *src.shape[1:])
    out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)
    return out


# ── Edge network: transforms bond features into message weight matrices ───────

class EdgeNetwork(nn.Module):
    """
    MLP that maps bond features e_ij → weight matrix A_ij ∈ R^{in×out}.
    Message: m_ij = A_ij @ h_j
    """

    def __init__(self, edge_feat_dim: int, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_feat_dim, 128),
            nn.SiLU(),
            nn.Linear(128, in_dim * out_dim),
        )
        self.in_dim  = in_dim
        self.out_dim = out_dim

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        edge_attr : [E, F_bond]

        Returns
        -------
        A : [E, in_dim, out_dim]
        """
        E = edge_attr.size(0)
        return self.net(edge_attr).view(E, self.in_dim, self.out_dim)


# ── Single message-passing layer ──────────────────────────────────────────────

class MPNNLayer(nn.Module):
    """
    One step of edge-conditioned message passing with GRU update.

    h_i^{t+1} = GRU(sum_j A_ij h_j^t, h_i^t)
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_net = EdgeNetwork(edge_dim, node_dim, hidden_dim)
        self.gru       = nn.GRUCell(hidden_dim, node_dim)
        self.norm      = nn.LayerNorm(node_dim)

    def forward(
        self,
        x: torch.Tensor,           # [N, node_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor,   # [E, edge_dim]
    ) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]          # source/target

        # Edge-conditioned messages: A_ij @ h_j
        A = self.edge_net(edge_attr)                      # [E, node_dim, hidden_dim]
        h_j = x[col]                                      # [E, node_dim]
        # batched matmul: [E, 1, node_dim] @ [E, node_dim, hidden_dim] → [E, hidden_dim]
        messages = torch.bmm(h_j.unsqueeze(1), A).squeeze(1)

        # Aggregate messages per target node
        agg = scatter_add(messages, row, dim_size=x.size(0))  # [N, hidden_dim]

        # GRU update
        x_new = self.gru(agg, x)
        return self.norm(x_new)


# ── Global attention pooling ──────────────────────────────────────────────────

class GlobalAttentionPooling(nn.Module):
    """
    Attentive readout:  z = sum_i  softmax(f(h_i)) * g(h_i)
    Returns both the graph representation and per-atom attention weights.
    """

    def __init__(self, node_dim: int, out_dim: int):
        super().__init__()
        self.gate_net  = nn.Linear(node_dim, 1)
        self.value_net = nn.Sequential(
            nn.Linear(node_dim, out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        x: torch.Tensor,           # [sum_N, node_dim]
        batch: torch.Tensor,       # [sum_N]  graph assignment
        n_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z           : [n_graphs, out_dim]
        attn_weights: [sum_N]  (softmax per graph)
        """
        gates  = self.gate_net(x).squeeze(-1)    # [sum_N]
        values = self.value_net(x)               # [sum_N, out_dim]

        # Softmax over nodes within each graph — subtract max per graph for stability
        max_per_graph = gates.new_full((n_graphs,), float('-inf'))
        max_per_graph.scatter_reduce_(0, batch, gates, reduce='amax', include_self=True)
        gates_shifted = gates - max_per_graph[batch]
        exp_gates     = gates_shifted.exp()
        sum_exp       = scatter_add(
            exp_gates.unsqueeze(-1), batch, dim_size=n_graphs
        ).squeeze(-1)                              # [n_graphs]
        attn = exp_gates / (sum_exp[batch] + 1e-8)  # [sum_N]

        # Weighted sum → graph representation
        weighted = values * attn.unsqueeze(-1)        # [sum_N, out_dim]
        z = scatter_add(weighted, batch, dim_size=n_graphs)  # [n_graphs, out_dim]

        return z, attn


# ── Full MPNN model ───────────────────────────────────────────────────────────

class MPNN(nn.Module):
    """
    Multi-step MPNN for molecular property prediction.

    Architecture
    ------------
    Input projection  → T × MPNNLayer → GlobalAttentionPooling → MLP head(s)

    Parameters
    ----------
    node_feat   : dimensionality of input atom features
    edge_feat   : dimensionality of bond features
    hidden      : hidden node dimension throughout message passing
    n_layers    : number of message-passing steps T
    n_tasks     : number of output regression/classification targets
    dropout     : dropout rate in MLP head
    """

    def __init__(
        self,
        node_feat: int,
        edge_feat: int,
        hidden: int   = 128,
        n_layers: int = 3,
        n_tasks: int  = 1,
        dropout: float = 0.15,
    ):
        super().__init__()

        # Project input atom features into hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        # Message-passing layers
        self.mp_layers = nn.ModuleList([
            MPNNLayer(hidden, edge_feat, hidden) for _ in range(n_layers)
        ])

        # Skip connection to combine all layer outputs
        self.skip_proj = nn.Linear(hidden * (n_layers + 1), hidden)

        # Global attention readout
        self.readout = GlobalAttentionPooling(hidden, hidden)

        # MLP prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, n_tasks),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x          : [sum_N, node_feat]
        edge_index : [2, sum_E]
        edge_attr  : [sum_E, edge_feat]
        batch      : [sum_N]  graph assignment per node

        Returns
        -------
        preds        : [n_graphs, n_tasks]
        attn_weights : [sum_N]
        """
        n_graphs = int(batch.max().item()) + 1

        h = self.input_proj(x)

        # Collect layer outputs for skip connection
        all_h = [h]
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr)
            all_h.append(h)

        # Skip: concatenate all layer representations
        h_cat  = torch.cat(all_h, dim=-1)          # [sum_N, hidden*(n_layers+1)]
        h_skip = self.skip_proj(h_cat)              # [sum_N, hidden]

        # Readout
        z, attn_weights = self.readout(h_skip, batch, n_graphs)

        # Prediction
        preds = self.head(z)
        return preds, attn_weights

    def get_atom_importances(
        self,
        graph: dict,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Return per-atom attention weights for a single molecule graph.

        Parameters
        ----------
        graph : output of mol_graph.mol_to_graph()

        Returns
        -------
        np.ndarray [N_atoms]  — attention weights summing to 1
        """
        self.eval()
        with torch.no_grad():
            x   = torch.tensor(graph["x"],          dtype=torch.float,  device=device)
            ei  = torch.tensor(graph["edge_index"],  dtype=torch.long,   device=device)
            ea  = torch.tensor(graph["edge_attr"],   dtype=torch.float,  device=device)
            b   = torch.zeros(x.size(0), dtype=torch.long, device=device)
            _, attn = self.forward(x, ei, ea, b)
        return attn.cpu().numpy()


# ── Multi-task MPNN with uncertainty weighting ────────────────────────────────

class MPNNMultiTask(MPNN):
    """
    Extension of MPNN with per-task uncertainty-weighted loss (Kendall & Gal 2018).

    log p ∝ -1/2σ² * loss_i + log σ
    → minimise: loss_i / (2*exp(log_var)) + 0.5 * log_var
    """

    def __init__(self, n_tasks: int, task_types: Optional[List[str]] = None, **kwargs):
        super().__init__(n_tasks=n_tasks, **kwargs)
        self.log_vars   = nn.Parameter(torch.zeros(n_tasks))
        self.task_types = task_types or ["regression"] * n_tasks

    def compute_loss(
        self,
        preds: torch.Tensor,     # [B, T]
        targets: torch.Tensor,   # [B, T]
        masks: Optional[torch.Tensor] = None,  # [B, T] — 1 if label present
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total_loss   : scalar
        per_task_loss: [T]
        """
        if masks is None:
            masks = torch.ones_like(targets)

        per_task = []
        for t, task_type in enumerate(self.task_types):
            p = preds[:, t][masks[:, t].bool()]
            y = targets[:, t][masks[:, t].bool()]
            if len(y) == 0:
                per_task.append(torch.tensor(0.0, device=preds.device))
                continue
            if task_type == "regression":
                raw_loss = F.mse_loss(p, y)
            else:
                raw_loss = F.binary_cross_entropy_with_logits(p, y)
            per_task.append(raw_loss)

        per_task_tensor = torch.stack(per_task)

        # Uncertainty weighting
        precision = torch.exp(-self.log_vars)            # 1/σ²
        total = (precision * per_task_tensor + self.log_vars).mean()
        return total, per_task_tensor
