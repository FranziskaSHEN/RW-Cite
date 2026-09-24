"""Two-layer multi-relation GAT ranker with target-edge mask hooks.

Hand-written sparse multi-channel GAT (no PyG) over incoming, outgoing, and
co-citation relations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GatMvpConfig:
    text_dim: int = 768
    hidden_dim: int = 256
    n_layers: int = 2
    n_heads: int = 1  # per-channel single-head for sparsity simplicity
    dropout: float = 0.2
    pc_dim: int = 64
    neighbor_k: int = 32
    struct_dim: int = 10
    edge_dropout: float = 0.4
    query_out_dropout: float = 0.5
    # ablation switches
    use_l1_sem: bool = True
    use_multi_channel: bool = True
    use_text_inject: bool = True
    use_gate: bool = True
    use_l4: bool = True
    use_struct_residual: bool = True
    struct_residual_scale: float = 1.0
    fixed_alpha: bool = False  # GCN-style 1/sqrt(d_i d_j) ablation
    struct_only: bool = False  # structural-channel ablation
    lambda_init: float = 1.0
    use_ce_feat: bool = False  # append standardized CE score to the output MLP


class SparseGATConv(nn.Module):
    """Single-channel sparse GAT with an optional semantic prior or fixed attention."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        text_dim: int = 768,
        pc_dim: int = 64,
        dropout: float = 0.2,
        use_l1_sem: bool = True,
        fixed_alpha: bool = False,
        lambda_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Linear(2 * out_dim, 1, bias=False)
        self.use_l1_sem = use_l1_sem
        self.fixed_alpha = fixed_alpha
        self.dropout = dropout
        self.Pc = (
            nn.Linear(text_dim, pc_dim, bias=False)
            if use_l1_sem and not fixed_alpha
            else None
        )
        self.lambda_sem = (
            nn.Parameter(torch.tensor(float(lambda_init)))
            if use_l1_sem and not fixed_alpha
            else None
        )
        self.leaky = nn.LeakyReLU(0.2)

    def forward(
        self,
        h: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        *,
        text: torch.Tensor | None = None,
        edge_dropout: float = 0.0,
        force_fixed_alpha: bool | None = None,
        force_no_sem: bool = False,
    ) -> torch.Tensor:
        """
        h: [N, in]
        edge_src → edge_dst : messages from src aggregated at dst
        """
        n, _ = h.shape
        device = h.device
        if edge_src.numel() == 0:
            return torch.zeros(n, self.W.out_features, device=device, dtype=h.dtype)

        if self.training and edge_dropout > 0:
            keep = torch.rand(edge_src.shape[0], device=device) >= edge_dropout
            # always keep at least one edge if any
            if keep.any():
                edge_src = edge_src[keep]
                edge_dst = edge_dst[keep]
            else:
                return torch.zeros(n, self.W.out_features, device=device, dtype=h.dtype)

        Wh = self.W(h)  # [N, out]
        out_dim = Wh.shape[1]
        use_fixed = self.fixed_alpha if force_fixed_alpha is None else force_fixed_alpha

        if use_fixed:
            # degree from current edge set
            deg = torch.zeros(n, device=device, dtype=h.dtype)
            deg.scatter_add_(0, edge_dst, torch.ones_like(edge_dst, dtype=h.dtype))
            deg.scatter_add_(0, edge_src, torch.ones_like(edge_src, dtype=h.dtype))
            deg = deg.clamp(min=1.0)
            alpha = 1.0 / (deg[edge_src].sqrt() * deg[edge_dst].sqrt())
        else:
            cat = torch.cat([Wh[edge_src], Wh[edge_dst]], dim=-1)
            e = self.leaky(self.att(cat)).squeeze(-1)
            if (
                self.use_l1_sem
                and not force_no_sem
                and self.Pc is not None
                and text is not None
                and self.lambda_sem is not None
            ):
                pt = self.Pc(text)
                pt = F.normalize(pt, dim=-1)
                cos = (pt[edge_src] * pt[edge_dst]).sum(dim=-1)
                e = e + self.lambda_sem * cos
            # softmax per destination
            e_max = torch.full((n,), -1e9, device=device, dtype=e.dtype)
            e_max.scatter_reduce_(0, edge_dst, e, reduce="amax", include_self=True)
            e_shift = e - e_max[edge_dst]
            exp_e = e_shift.exp()
            denom = torch.zeros(n, device=device, dtype=e.dtype)
            denom.scatter_add_(0, edge_dst, exp_e)
            alpha = exp_e / (denom[edge_dst] + 1e-8)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = Wh[edge_src] * alpha.unsqueeze(-1)
        out = torch.zeros(n, out_dim, device=device, dtype=h.dtype)
        out.scatter_add_(0, edge_dst.unsqueeze(-1).expand_as(msg), msg)
        return out


class GatMvpModel(nn.Module):
    """L1–L5 graph-aware scorer for frozen windows."""

    def __init__(self, cfg: GatMvpConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GatMvpConfig()
        d = self.cfg.hidden_dim
        td = self.cfg.text_dim

        self.text_ln = nn.LayerNorm(td)
        self.P0 = nn.Linear(td, d)

        def make_convs() -> nn.ModuleDict:
            # Keep the full reported-layer shape so existing checkpoints load.
            return nn.ModuleDict(
                {
                    ch: SparseGATConv(
                        d,
                        d,
                        text_dim=td,
                        pc_dim=self.cfg.pc_dim,
                        dropout=self.cfg.dropout,
                        use_l1_sem=True,
                        fixed_alpha=False,
                        lambda_init=self.cfg.lambda_init,
                    )
                    for ch in ("in", "out", "cocite")
                }
            )

        self.layers = nn.ModuleList([make_convs() for _ in range(self.cfg.n_layers)])
        # Keep all three relations and text injection available across ablations.
        inj_in = d * 3 + d
        self.inject = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(inj_in, d),
                    nn.ReLU(),
                    nn.Dropout(self.cfg.dropout),
                    nn.Linear(d, d),
                )
                for _ in range(self.cfg.n_layers)
            ]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.cfg.n_layers)])
        self.P_l = nn.ModuleList([nn.Linear(td, d) for _ in range(self.cfg.n_layers)])

        # Query-conditioned semantic gate.
        self.Wg = nn.Linear(d * 3, d)
        self.Pt = nn.Linear(td, d)
        self.Ps = nn.Linear(d, d)

        # Query-conditioned attention over the masked graph neighborhood.
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)

        # Output MLP. Keep the full input layout for checkpoint compatibility.
        self.struct_w = nn.Parameter(torch.zeros(self.cfg.struct_dim))
        self.struct_b = nn.Parameter(torch.zeros(()))
        self.register_buffer("struct_mu", torch.zeros(self.cfg.struct_dim))
        self.register_buffer("struct_sd", torch.ones(self.cfg.struct_dim))
        mlp_in = d * 5 + self.cfg.struct_dim  # reserve the graph-context slot
        if self.cfg.use_ce_feat:
            mlp_in += 1  # z(s_CE) scalar
            self.ce_ln = nn.LayerNorm(1)
        else:
            self.ce_ln = None
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, d),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(d, 1),
        )

    def load_partial_from_e4(self, state_dict: dict[str, Any]) -> None:
        """Init cefeat MLP from E4: copy overlapping weights; zero the new CE column."""
        own = self.state_dict()
        with torch.no_grad():
            for k, v in state_dict.items():
                if k not in own:
                    continue
                if own[k].shape == v.shape:
                    own[k].copy_(v)
                elif (
                    k == "mlp.0.weight"
                    and own[k].ndim == 2
                    and v.ndim == 2
                    and own[k].shape[0] == v.shape[0]
                    and own[k].shape[1] == v.shape[1] + 1
                ):
                    own[k][:, : v.shape[1]].copy_(v)
                    own[k][:, v.shape[1] :].zero_()
        self.load_state_dict(own)

    def load_struct_init(self, path: str | Any) -> None:
        data = np.load(path, allow_pickle=True)
        with torch.no_grad():
            self.struct_w.copy_(torch.tensor(data["w"], dtype=torch.float32))
            self.struct_b.copy_(torch.tensor(float(data["b"].reshape(-1)[0]), dtype=torch.float32))
            self.struct_mu.copy_(torch.tensor(data["mu"], dtype=torch.float32))
            self.struct_sd.copy_(torch.tensor(data["sd"], dtype=torch.float32).clamp(min=1e-6))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _propagate(
        self,
        h: torch.Tensor,
        text: torch.Tensor,
        edges: dict[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        edge_dropout: float,
        drop_out_channel: bool = False,
    ) -> torch.Tensor:
        cfg = self.cfg
        for li in range(cfg.n_layers):
            convs = self.layers[li]
            gs = []
            for ch in ("in", "out", "cocite"):
                src, dst = edges[ch]
                ed = edge_dropout if self.training else 0.0
                g = convs[ch](
                    h,
                    src,
                    dst,
                    text=text if cfg.use_l1_sem else None,
                    edge_dropout=ed,
                    force_fixed_alpha=True if cfg.fixed_alpha else None,
                    force_no_sem=not cfg.use_l1_sem,
                )
                if not cfg.use_multi_channel and ch != "cocite":
                    g = torch.zeros_like(g)
                gs.append(g)
            cat = torch.cat(gs, dim=-1)
            if cfg.use_text_inject:
                cat = torch.cat([cat, self.P_l[li](text)], dim=-1)
            else:
                cat = torch.cat([cat, torch.zeros_like(self.P_l[li](text))], dim=-1)
            upd = self.inject[li](cat)
            h = self.layer_norms[li](h + upd)
        return h

    def forward(
        self,
        text_emb: torch.Tensor,
        edges: dict[str, tuple[torch.Tensor, torch.Tensor]],
        cand_idx: torch.Tensor,
        query_idx: torch.Tensor,
        phi_struct: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
        *,
        mask_target_edge: bool = True,
        edge_dropout: float | None = None,
        query_out_dropout: float | None = None,
        ce_feat: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        text_emb: [N, 768] subgraph node texts (whitened preferred)
        edges: channel → (src, dst) int64
        cand_idx: [C] local indices
        query_idx: scalar local index
        phi_struct: [C, 10]
        neighbors: [C, K] local indices (-1 pad)
        neighbor_mask: [C, K] bool
        ce_feat: optional [C] or [C,1] per-query z-scored CE-sent scores (L1)
        return_attn: if True, include L4 ``l4_beta`` / ``l4_mask`` tensors
        """
        cfg = self.cfg
        ed = cfg.edge_dropout if edge_dropout is None else edge_dropout
        qod = cfg.query_out_dropout if query_out_dropout is None else query_out_dropout

        if cfg.struct_only:
            xn = (phi_struct - self.struct_mu) / self.struct_sd
            scores = xn @ self.struct_w + self.struct_b
            z0 = torch.tensor(0.0, device=scores.device)
            out_so: dict[str, torch.Tensor] = {
                "scores": scores,
                "g_bar": z0,
                "max_beta": z0,
                "attn_entropy": z0,
                "n_eff_ge4_frac": z0,
                "max_beta_near1_frac": z0,
            }
            if return_attn:
                C = int(phi_struct.shape[0])
                out_so["l4_beta"] = torch.zeros(C, 1, device=scores.device)
                out_so["l4_mask"] = torch.zeros(C, 1, dtype=torch.bool, device=scores.device)
            return out_so

        text = self.text_ln(text_emb)
        h = self.P0(text)

        # query out-edge dropout: randomly drop edges where src==query in out channel
        edges_use = dict(edges)
        if self.training and qod > 0 and "out" in edges_use:
            src, dst = edges_use["out"]
            if src.numel():
                is_q = src == int(query_idx)
                drop = (torch.rand(src.shape[0], device=src.device) < qod) & is_q
                keep = ~drop
                edges_use["out"] = (src[keep], dst[keep])

        h = self._propagate(h, text, edges_use, edge_dropout=ed)

        # Query-conditioned semantic gate.
        t_proj = self.Pt(text)
        if cfg.use_gate:
            g = torch.sigmoid(self.Wg(torch.cat([t_proj, h, t_proj * h], dim=-1)))
            z = g * t_proj + (1.0 - g) * self.Ps(h)
            g_bar = g.mean()
        else:
            z = 0.5 * (t_proj + self.Ps(h))
            g_bar = torch.tensor(0.5, device=z.device)

        z_q = z[query_idx]
        z_v = z[cand_idx]

        max_beta = torch.tensor(0.0, device=z.device)
        attn_entropy = torch.tensor(0.0, device=z.device)
        n_eff_ge4_frac = torch.tensor(0.0, device=z.device)
        max_beta_near1_frac = torch.tensor(0.0, device=z.device)
        c_v = torch.zeros_like(z_v)
        l4_beta: torch.Tensor | None = None
        l4_mask: torch.Tensor | None = None
        if cfg.use_l4:
            nbr = neighbors.clone()
            nmask = neighbor_mask.clone()
            if mask_target_edge:
                # A: drop query from neighbor sets
                q = int(query_idx)
                hit = nbr == q
                nmask = nmask & ~hit
                nbr = nbr.masked_fill(hit, -1)
            # gather neighbor states
            C, K = nbr.shape
            flat = nbr.clamp(min=0)
            z_n = z[flat]  # [C, K, d]
            qk = self.Wq(z_q).unsqueeze(0).unsqueeze(0)  # [1,1,d]
            kk = self.Wk(z_n)  # [C,K,d]
            scale = cfg.hidden_dim**0.5
            logits = (qk * kk).sum(dim=-1) / scale
            vv = self.Wv(z_n)
            logits_m = logits.masked_fill(~nmask, -1e9)
            beta = torch.softmax(logits_m, dim=-1)
            beta = beta.masked_fill(~nmask, 0.0)
            beta = beta / (beta.sum(dim=-1, keepdim=True) + 1e-8)
            c_v = (beta.unsqueeze(-1) * vv).sum(dim=1)

            if return_attn:
                l4_beta = beta.detach()
                l4_mask = nmask.detach()
            if nmask.any():
                beta_m = beta.masked_fill(~nmask, 0.0)
                max_beta = beta_m.max().detach()
                n_eff = nmask.sum(dim=-1).float()  # [C]
                n_eff_ge4_frac = (n_eff >= 4).float().mean().detach()
                row_max = beta_m.max(dim=-1).values
                valid_rows = n_eff > 0
                if valid_rows.any():
                    max_beta_near1_frac = (row_max[valid_rows] > 0.99).float().mean().detach()
                b = beta_m.clamp(min=1e-8)
                row_sum = b.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                b = b / row_sum
                ent = -(b * b.log()).sum(dim=-1)
                ent_ok = n_eff >= 2
                if ent_ok.any():
                    attn_entropy = ent[ent_ok].mean().detach()

        # Candidate scoring head.
        feats = [z_q.unsqueeze(0).expand_as(z_v), z_v, z_q * z_v, (z_q - z_v) ** 2, c_v]
        feats.append(phi_struct)
        if cfg.use_ce_feat:
            if ce_feat is None:
                ce_col = torch.zeros(z_v.shape[0], 1, device=z_v.device, dtype=z_v.dtype)
            else:
                ce_col = ce_feat.reshape(-1, 1).to(device=z_v.device, dtype=z_v.dtype)
            if self.ce_ln is not None:
                ce_col = self.ce_ln(ce_col)
            feats.append(ce_col)
        mlp_in = torch.cat(feats, dim=-1)
        mlp_score = self.mlp(mlp_in).squeeze(-1)
        if cfg.use_struct_residual:
            xn = (phi_struct - self.struct_mu) / self.struct_sd
            struct_score = xn @ self.struct_w + self.struct_b
            scale_r = float(cfg.struct_residual_scale)
            scores = mlp_score + scale_r * struct_score
        else:
            scores = mlp_score

        out: dict[str, torch.Tensor] = {
            "scores": scores,
            "g_bar": g_bar.detach(),
            "max_beta": max_beta if max_beta.ndim == 0 else max_beta.detach(),
            "attn_entropy": attn_entropy,
            "n_eff_ge4_frac": n_eff_ge4_frac,
            "max_beta_near1_frac": max_beta_near1_frac,
            "z": z,
        }
        if return_attn:
            if l4_beta is None:
                C = int(cand_idx.shape[0])
                out["l4_beta"] = torch.zeros(C, 1, device=scores.device)
                out["l4_mask"] = torch.zeros(C, 1, dtype=torch.bool, device=scores.device)
            else:
                out["l4_beta"] = l4_beta
                out["l4_mask"] = l4_mask  # type: ignore[assignment]
        return out


def build_ablation_config(name: str) -> GatMvpConfig:
    """Named E/F configs for §8.3."""
    cfg = GatMvpConfig()
    n = name.upper().replace("-", "_")
    if n == "E3":
        cfg.struct_only = True
    elif n == "E5":
        cfg.fixed_alpha = True
        cfg.use_l1_sem = False
        cfg.use_multi_channel = False
    elif n == "F5":
        cfg.use_l4 = False
    elif n == "F1":
        cfg.use_gate = False
    elif n == "F2":
        cfg.use_multi_channel = False
    elif n == "F3":
        cfg.use_text_inject = False
    elif n == "F4":
        cfg.use_l1_sem = False
    elif n == "F6":
        cfg.use_struct_residual = False
    elif n in ("E4", "FULL", "DEFAULT"):
        pass
    else:
        raise ValueError(f"unknown ablation {name}")
    return cfg
