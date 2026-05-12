"""
AttentiveFP: Attention-based Graph Neural Network
===================================================
PyTorch implementation of:
  Xiong et al. (2020) "Pushing the Boundaries of Molecular Representation
  for Drug Discovery with the Graph Attention Mechanism"
  https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959

Architecture summary
--------------------
Atom-level attention:
  For each atom i, compute context vector via multi-head attention
  over its neighbours, then update with GRU.

Molecule-level attention:
  A virtual "supernode" attends over all atom representations to
  produce the graph-level fingerprint (multiple rounds).

Python  : >= 3.10
PyTorch : >= 2.0
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mpnn import scatter_add


# ── Atom-level attention ──────────────────────────────────────────────────────

class AtomAttention(nn.Module):
    """
    Compute context vector for atom i by attending over neighbours j ∈ N(i).

    For each directed edge (i→j):
        e_ij = Alignment(cat(h_i, h_j, edge_feat_ij))
    Attention: α_ij = softmax_{j ∈ N(i)} e_ij
    Context:   c_i  = sum_j α_ij * V(h_j)
    """

    def __init__(self, node_dim: int, edge_dim: int, n_heads: int = 4):
        super().__init__()
        assert node_dim % n_heads == 0, "node_dim must be divisible by n_heads"
        self.n_heads   = n_heads
        self.head_dim  = node_dim // n_heads

        # Alignment MLP (query/key)
        self.align = nn.Linear(2 * node_dim + edge_dim, n_heads)
        # Value projection
        self.value = nn.Linear(node_dim, node_dim)
        # Output projection
        self.proj  = nn.Linear(node_dim, node_dim)
        self.norm  = nn.LayerNorm(node_dim)

    def forward(
        self,
        x: torch.Tensor,           # [N, node_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor,   # [E, edge_dim]
    ) -> torch.Tensor:
        """Returns context-augmented node features [N, node_dim]."""
        row, col = edge_index[0], edge_index[1]  # row=src, col=dst
        N = x.size(0)

        h_i = x[row]           # [E, d]
        h_j = x[col]           # [E, d]
        cat_feat = torch.cat([h_i, h_j, edge_attr], dim=-1)  # [E, 2d + edge_dim]

        # Alignment scores: [E, n_heads]
        scores = self.align(cat_feat)

        # Softmax per source node and head — subtract max per node for stability
        max_per_node = scores.new_full((N, self.n_heads), float('-inf'))
        max_per_node.scatter_reduce_(
            0, row.unsqueeze(-1).expand_as(scores), scores, reduce='amax', include_self=True
        )
        scores_exp = (scores - max_per_node[row]).exp()    # [E, n_heads]
        sum_exp    = scatter_add(scores_exp, row, dim_size=N)  # [N, n_heads]
        alpha      = scores_exp / (sum_exp[row] + 1e-8)   # [E, n_heads]

        # Value: [E, n_heads, head_dim]
        V = self.value(h_j).view(-1, self.n_heads, self.head_dim)
        # Weighted sum: [E, n_heads, head_dim]
        weighted = V * alpha.unsqueeze(-1)
        # Flatten heads and aggregate
        weighted_flat = weighted.view(-1, self.n_heads * self.head_dim)  # [E, d]
        context = scatter_add(weighted_flat, row, dim_size=N)            # [N, d]

        return self.norm(x + self.proj(context))


class AttentiveFPLayer(nn.Module):
    """
    Single atom update step:
      context_i = AtomAttention(h_i, neighbours)
      h_i'      = GRU(context_i, h_i)
    """

    def __init__(self, node_dim: int, edge_dim: int, n_heads: int = 4):
        super().__init__()
        self.attention = AtomAttention(node_dim, edge_dim, n_heads)
        self.gru        = nn.GRUCell(node_dim, node_dim)
        self.norm       = nn.LayerNorm(node_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        context = self.attention(x, edge_index, edge_attr)
        x_new   = self.gru(context, x)
        return self.norm(x_new)


# ── Molecule-level attention (supernode) ─────────────────────────────────────

class MoleculeAttention(nn.Module):
    """
    Multiple rounds of supernode attention over all atom representations
    to produce a single molecular fingerprint.

    Each round:
        h_super^t = GRU(sum_i  softmax(f(h_i)) * g(h_i),  h_super^{t-1})

    Returns the final supernode state as the molecular fingerprint.
    """

    def __init__(self, node_dim: int, n_rounds: int = 2):
        super().__init__()
        self.n_rounds = n_rounds
        self.gate_net  = nn.Linear(node_dim * 2, 1)
        self.value_net = nn.Linear(node_dim, node_dim)
        self.gru        = nn.GRUCell(node_dim, node_dim)

    def forward(
        self,
        x: torch.Tensor,       # [sum_N, node_dim]
        batch: torch.Tensor,   # [sum_N]
        n_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z           : [n_graphs, node_dim]
        attn_weights: [sum_N]
        """
        # Initialise supernode as mean of atom representations
        z = scatter_add(x, batch, dim_size=n_graphs)   # [n_graphs, d]
        counts = scatter_add(
            torch.ones(x.size(0), 1, device=x.device), batch, dim_size=n_graphs
        )
        z = z / (counts + 1e-8)                         # mean pooling init

        last_attn = None
        for _ in range(self.n_rounds):
            # Broadcast supernode to each atom
            z_broadcast = z[batch]                      # [sum_N, d]
            # Gate: cat(h_i, z_broadcast) → scalar score
            cat_feat = torch.cat([x, z_broadcast], dim=-1)  # [sum_N, 2d]
            gates    = self.gate_net(cat_feat).squeeze(-1)   # [sum_N]
            # Softmax within each graph — subtract max per graph for stability
            max_per_graph = gates.new_full((n_graphs,), float('-inf'))
            max_per_graph.scatter_reduce_(0, batch, gates, reduce='amax', include_self=True)
            gates_exp = (gates - max_per_graph[batch]).exp()
            sum_exp   = scatter_add(gates_exp.unsqueeze(-1), batch, dim_size=n_graphs).squeeze(-1)
            attn      = gates_exp / (sum_exp[batch] + 1e-8)  # [sum_N]
            last_attn = attn

            # Weighted message to supernode
            values  = self.value_net(x)                  # [sum_N, d]
            message = scatter_add(values * attn.unsqueeze(-1), batch, dim_size=n_graphs)

            # Update supernode
            z = self.gru(message, z)

        return z, last_attn


# ── Full AttentiveFP model ────────────────────────────────────────────────────

class AttentiveFP(nn.Module):
    """
    AttentiveFP for molecular property prediction.

    Parameters
    ----------
    node_feat   : atom feature dimensionality
    edge_feat   : bond feature dimensionality
    hidden      : hidden dimension (must be divisible by n_heads)
    n_atom_layers    : atom-level message passing steps
    n_mol_rounds     : molecule-level attention rounds
    n_tasks          : number of output targets
    dropout          : dropout rate
    n_heads          : number of attention heads
    """

    def __init__(
        self,
        node_feat: int,
        edge_feat: int,
        hidden: int          = 200,
        n_atom_layers: int   = 2,
        n_mol_rounds: int    = 2,
        n_tasks: int         = 1,
        dropout: float       = 0.2,
        n_heads: int         = 4,
    ):
        super().__init__()

        # Ensure hidden is divisible by n_heads
        hidden = (hidden // n_heads) * n_heads

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
        )

        # Atom-level attention layers
        self.atom_layers = nn.ModuleList([
            AttentiveFPLayer(hidden, edge_feat, n_heads)
            for _ in range(n_atom_layers)
        ])

        # Molecule-level attention
        self.mol_attention = MoleculeAttention(hidden, n_mol_rounds)

        # Prediction head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.ELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, n_tasks),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
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
        batch      : [sum_N]

        Returns
        -------
        preds        : [n_graphs, n_tasks]
        attn_weights : [sum_N]  (from final molecule attention round)
        """
        n_graphs = int(batch.max().item()) + 1

        h = self.input_proj(x)

        for layer in self.atom_layers:
            h = layer(h, edge_index, edge_attr)

        z, attn_weights = self.mol_attention(h, batch, n_graphs)
        preds = self.head(z)

        return preds, attn_weights

    def get_atom_importances(
        self,
        graph: dict,
        device: str = "cpu",
    ) -> "np.ndarray":
        """Return per-atom attention weights for a single molecule."""
        import numpy as np
        self.eval()
        with torch.no_grad():
            x  = torch.tensor(graph["x"],          dtype=torch.float,  device=device)
            ei = torch.tensor(graph["edge_index"],  dtype=torch.long,   device=device)
            ea = torch.tensor(graph["edge_attr"],   dtype=torch.float,  device=device)
            b  = torch.zeros(x.size(0), dtype=torch.long, device=device)
            _, attn = self.forward(x, ei, ea, b)
        return attn.cpu().numpy()
