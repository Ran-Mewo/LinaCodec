import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module.transformer import Transformer

def _as_btc(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 2 else x


def _resample_time(x_btc: torch.Tensor, length: int) -> torch.Tensor:
    if (t := x_btc.size(1)) == length:
        return x_btc
    return F.interpolate(x_btc.transpose(1, 2), size=length, mode="linear", align_corners=False).transpose(1, 2)


class FlowFormerVoicePack(nn.Module):
    """
    Transformer-conditioned normalizing flow + time adapter.
    It's basically a RealNVP-style affine coupling flow with a Transformer conditioner.
    The Transformer predicts coupling scale/shift (s,t) from [masked-x token] + [condition tokens],
    and predicts a per-time-step delta that offsets the base `global_embedding`.

    - **Flow**: samples a base (utterance-level) LinaCodec `global_embedding`.
    - **Adapter**: turns (content_embedding seq, base global_embedding) into a **time-varying**
      conditioning sequence that LinaCodec's decoder can consume.
    """

    def __init__(
        self,
        cond_dim: int,
        emb_dim: int,
        *,
        cond_len: int = 128,
        dim: int = 256,
        tr_layers: int = 2,
        tr_heads: int = 8,
        coupling_layers: int = 8,
        clamp: float = 2.0,
        delta_clamp: float = 1.0,
    ):
        super().__init__()
        if emb_dim % 2:
            raise ValueError("emb_dim must be even for half-split coupling.")
        if dim % tr_heads:
            raise ValueError("dim must be divisible by tr_heads.")

        self.cond_dim, self.emb_dim, self.cond_len = int(cond_dim), int(emb_dim), int(cond_len)
        self.dim, self.tr_layers, self.tr_heads = int(dim), int(tr_layers), int(tr_heads)
        self.coupling_layers, self.clamp, self.delta_clamp = int(coupling_layers), float(clamp), float(delta_clamp)

        self.x_proj = nn.Linear(emb_dim, dim, bias=False)
        self.g_proj = nn.Linear(emb_dim, dim, bias=False)
        self.cond_proj = nn.Linear(cond_dim, dim, bias=False)
        self.tr = Transformer(dim=dim, n_layers=tr_layers, n_heads=tr_heads, dropout=0.0, max_seq_len=cond_len + 1)
        self.heads = nn.ModuleList([nn.Linear(dim, 2 * emb_dim, bias=True) for _ in range(coupling_layers)])
        for h in self.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.delta_out = nn.Linear(dim, emb_dim, bias=True)
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)

        masks = []
        half = emb_dim // 2
        for i in range(coupling_layers):
            m = torch.zeros(emb_dim)
            m[:half] = 1.0
            masks.append(m if i % 2 == 0 else (1.0 - m))
        self.register_buffer("masks", torch.stack(masks), persistent=False)  # (L, D)

    def _cond_tokens(self, cond_seq: torch.Tensor) -> torch.Tensor:
        cond_seq = _resample_time(_as_btc(cond_seq), self.cond_len)
        return self.cond_proj(cond_seq)

    def _st(self, layer: int, x_masked: torch.Tensor, cond_tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        toks = torch.cat((self.x_proj(x_masked).unsqueeze(1), cond_tok), dim=1)
        h = self.tr(toks)
        h0 = h[:, 0]
        s, t = self.heads[layer](h0).chunk(2, dim=-1)
        s = self.clamp * torch.tanh(s)
        return s, t

    def condition(self, cond_seq: torch.Tensor, base_global: torch.Tensor) -> torch.Tensor:
        """Return time-varying conditioning sequence (B, cond_len, emb_dim)."""
        base_global = base_global.unsqueeze(0) if base_global.dim() == 1 else base_global
        cond_tok = self._cond_tokens(cond_seq)
        h = self.tr(torch.cat((self.g_proj(base_global).unsqueeze(1), cond_tok), dim=1))[:, 1:]
        delta = self.delta_clamp * torch.tanh(self.delta_out(h))
        return base_global.unsqueeze(1) + delta

    def forward(self, z: torch.Tensor, cond_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z -> x. Returns (x, logdet)."""
        x, logdet = z, torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        cond_tok = self._cond_tokens(cond_seq)
        for i in range(self.coupling_layers):
            m = self.masks[i].to(x)
            x_m = x * m
            s, t = self._st(i, x_m, cond_tok)
            x = x_m + (1.0 - m) * (x * torch.exp(s) + t)
            logdet = logdet + ((1.0 - m) * s).sum(dim=-1)
        return x, logdet

    def inverse(self, x: torch.Tensor, cond_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x -> z. Returns (z, logdet_inv)."""
        z, logdet = x, torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        cond_tok = self._cond_tokens(cond_seq)
        for i in range(self.coupling_layers - 1, -1, -1):
            m = self.masks[i].to(z)
            z_m = z * m
            s, t = self._st(i, z_m, cond_tok)
            z = z_m + (1.0 - m) * ((z - t) * torch.exp(-s))
            logdet = logdet - ((1.0 - m) * s).sum(dim=-1)
        return z, logdet

    def loss(self, target_emb: torch.Tensor, cond_seq: torch.Tensor) -> torch.Tensor:
        z, logdet = self.inverse(target_emb, cond_seq)
        log_pz = -0.5 * (z.pow(2) + math.log(2.0 * math.pi)).sum(dim=-1)
        return -(log_pz + logdet).mean()

    @torch.no_grad()
    def sample_global(self, cond_seq: torch.Tensor, *, seed: int | None = 0, temperature: float = 1.0) -> torch.Tensor:
        """Sample an utterance-level `global_embedding` (B, emb_dim)."""
        cond_seq = _as_btc(cond_seq)
        gen = None if seed is None else torch.Generator(device=cond_seq.device).manual_seed(int(seed))
        z = torch.randn(cond_seq.size(0), self.emb_dim, device=cond_seq.device, generator=gen) * float(temperature)
        x, _ = self.forward(z, cond_seq)
        return x

    @torch.no_grad()
    def sample(self, cond_seq: torch.Tensor, *, seed: int | None = 0, temperature: float = 1.0) -> torch.Tensor:
        """Sample a time-varying conditioning sequence for LinaCodec's decoder. (B, cond_len, emb_dim)"""
        return self.condition(cond_seq, self.sample_global(cond_seq, seed=seed, temperature=temperature))

    def save(self, path: str):
        torch.save(
            {
                "format": "flowformer_v1",
                "state_dict": self.state_dict(),
                "cond_dim": self.cond_dim,
                "emb_dim": self.emb_dim,
                "cond_len": self.cond_len,
                "dim": self.dim,
                "tr_layers": self.tr_layers,
                "tr_heads": self.tr_heads,
                "coupling_layers": self.coupling_layers,
                "clamp": self.clamp,
                "delta_clamp": self.delta_clamp,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device="cpu") -> "FlowFormerVoicePack":
        ckpt = torch.load(path, map_location=device)
        if ckpt.get("format") != "flowformer_v1":
            raise ValueError("Not a FlowFormerVoicePack checkpoint.")
        m = cls(
            ckpt["cond_dim"],
            ckpt["emb_dim"],
            cond_len=ckpt["cond_len"],
            dim=ckpt["dim"],
            tr_layers=ckpt["tr_layers"],
            tr_heads=ckpt["tr_heads"],
            coupling_layers=ckpt["coupling_layers"],
            clamp=ckpt["clamp"],
            delta_clamp=ckpt.get("delta_clamp", 1.0),
        )
        m.load_state_dict(ckpt["state_dict"])
        return m.to(device).eval()

