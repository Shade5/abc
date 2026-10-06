"""Batch-1-friendly ABC-DiT sampler: same math as DiTPolicy.sample_actions{,_rtc}.

At small batch sizes the sampler is bound by reading weights, and the stock
loop reads every weight on every Euler step. Two of those reads do not depend
on the step's input:

- adaLN modulation depends on (state, task, t) only. The t schedule is fixed,
  so the modulations of every block for every step (plus the t=0 row RTC
  prefix positions use) come out of one GEMM over the stacked adaLN weights,
  instead of num_steps GEMMs that each stream ~1.4 GB of weights.
- Cross-attention K/V depend on the vision tokens only. Every block's
  norm_xattn_kv is the same non-affine LayerNorm, so the K/V of all blocks
  come out of one GEMM, computed once per inference instead of once per step.

What remains per step is the self-attention, the cross-attention query/output
projections and the MLP.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from abc_minimal.dit import DiTPolicy


class FusedDiTSampler(torch.nn.Module):
    """Drop-in replacement for DiTPolicy.sample_actions / sample_actions_rtc."""

    def __init__(self, model: DiTPolicy):
        super().__init__()
        self.model = model
        blocks = model.blocks
        H = model.config.hidden_size
        self.hidden = H
        self.depth = len(blocks)
        self.num_heads = model.config.num_heads
        for block in blocks:
            assert block.norm_xattn_kv.elementwise_affine is False
            assert block.norm_xattn_kv.eps == blocks[0].norm_xattn_kv.eps
        self.kv_eps = blocks[0].norm_xattn_kv.eps

        with torch.no_grad():
            # Rows [9H * j, 9H * (j + 1)) hold block j; the final layer's 2H last.
            ada = [b.adaLN_modulation[1] for b in blocks] + [model.final_layer.adaLN_modulation[1]]
            self.register_buffer("ada_w", torch.cat([m.weight for m in ada]).contiguous(), persistent=False)
            self.register_buffer("ada_b", torch.cat([m.bias for m in ada]).contiguous(), persistent=False)
            # K and V rows of every block's cross-attention in_proj.
            self.register_buffer(
                "kv_w",
                torch.cat([b.cross_attn.in_proj_weight[H:] for b in blocks]).contiguous(),
                persistent=False,
            )
            self.register_buffer(
                "kv_b",
                torch.cat([b.cross_attn.in_proj_bias[H:] for b in blocks]).contiguous(),
                persistent=False,
            )

    def _modulations(self, state, task_vec, num_steps, dtype):
        """(B, num_steps + 1, depth * 9 + 2, H); row 0 is t = 0, row i + 1 is step i."""
        B = state.shape[0]
        dt = -1.0 / num_steps
        # Same bf16-rounded timesteps the stock loop builds with torch.full.
        ts = torch.tensor(
            [0.0] + [1.0 + i * dt for i in range(num_steps)], device=state.device, dtype=dtype
        )
        c = self.model.compute_cond(state, task_vec, ts.expand(B, -1))  # (B, S+1, H)
        mods = F.linear(F.silu(c), self.ada_w, self.ada_b)
        return mods.view(B, num_steps + 1, self.depth * 9 + 2, self.hidden)

    def _cross_kv(self, vision_tokens):
        """(depth, B, heads, Nv, head_dim) K and V for every block."""
        B, Nv, H = vision_tokens.shape
        kvn = F.layer_norm(vision_tokens, (H,), eps=self.kv_eps)
        kv = F.linear(kvn, self.kv_w, self.kv_b)
        kv = kv.view(B, Nv, self.depth, 2, self.num_heads, H // self.num_heads)
        kv = kv.permute(2, 3, 0, 4, 1, 5)  # (depth, 2, B, heads, Nv, hd)
        return kv[:, 0], kv[:, 1]

    def _velocity(self, x_t, mod, k_all, v_all):
        """mod: (B, T|1, depth * 9 + 2, H) modulation rows (block j, component k at 9j + k)."""
        model = self.model
        H = self.hidden
        nh = self.num_heads
        z = model.y_embedder(x_t) + model.pos_embed[:, : x_t.shape[1], :]
        B, T, _ = z.shape
        for j, block in enumerate(model.blocks):
            (
                shift_msa, scale_msa, gate_msa,
                shift_x, scale_x, gate_x,
                shift_mlp, scale_mlp, gate_mlp,
            ) = mod[:, :, 9 * j : 9 * j + 9].unbind(2)

            h = block.norm1(z) * (1 + scale_msa) + shift_msa
            z = z + gate_msa * block.attn(h)

            h = block.norm_xattn(z) * (1 + scale_x) + shift_x
            xa = block.cross_attn
            q = F.linear(h, xa.in_proj_weight[:H], xa.in_proj_bias[:H])
            q = q.view(B, T, nh, H // nh).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k_all[j], v_all[j])
            o = xa.out_proj(o.transpose(1, 2).reshape(B, T, H))
            z = z + gate_x * o

            h = block.norm2(z) * (1 + scale_mlp) + shift_mlp
            z = z + gate_mlp * block.mlp(h)

        fl = model.final_layer
        shift, scale = mod[:, :, 9 * self.depth : 9 * self.depth + 2].unbind(2)
        return fl.linear(fl.norm_final(z) * (1 + scale) + shift)

    @torch.no_grad()
    def prepare(
        self, batch, num_steps, noise, action_prefix=None, prefix_length: int = 0, prefix_mask=None
    ):
        """Everything step-invariant: x_0, modulations, cross-attention K/V, prefix mask.

        ``prefix_mask`` (bool, (B|1, chunk, 1)) replaces ``prefix_length`` so the
        prefix length can be data instead of a constant (see aot_engine.py).
        """
        model = self.model
        state = batch["state"]
        B = state.shape[0]
        dtype = model.y_embedder.weight.dtype
        if noise is None:
            noise = torch.randn(B, model.chunk_length, model.action_dim, device=state.device, dtype=dtype)
        x_t = noise.to(device=state.device, dtype=dtype)
        if action_prefix is None:
            prefix_mask = None
        else:
            action_prefix = action_prefix.to(device=state.device, dtype=dtype)
            if prefix_mask is None:
                prefix_pos = torch.arange(model.chunk_length, device=state.device) < prefix_length
                prefix_mask = prefix_pos.view(1, model.chunk_length, 1)
            x_t = torch.where(prefix_mask, action_prefix, x_t)
        vision_tokens = model.build_vision_tokens(batch["images"])
        mods = self._modulations(state, batch["task_vec_clip"], num_steps, dtype)
        k_all, v_all = self._cross_kv(vision_tokens)
        return x_t, mods, k_all, v_all, prefix_mask, action_prefix

    @torch.no_grad()
    def step(self, x_t, step_mod, zero_mod, k_all, v_all, prefix_mask, action_prefix, dt: float):
        """One Euler step. step_mod/zero_mod: (B, rows, H) modulations at t and t = 0.

        The step index is not an argument, so a compiled step is reused for every
        step instead of specializing on it.
        """
        step_mod = step_mod[:, None]  # (B, 1, rows, H)
        if prefix_mask is not None:
            # Prefix positions condition on t = 0, the rest on step t.
            step_mod = torch.where(prefix_mask[..., None], zero_mod[:, None], step_mod)
        v = self._velocity(x_t, step_mod, k_all, v_all)
        x_t = x_t + v * dt
        if prefix_mask is not None:
            x_t = torch.where(prefix_mask, action_prefix, x_t)
        return x_t

    @torch.no_grad()
    def sample(self, batch, num_steps=10, noise=None, action_prefix=None, prefix_length: int = 0):
        # A new inference: earlier CUDA-graph outputs are no longer read.
        torch.compiler.cudagraph_mark_step_begin()
        x_t, mods, k_all, v_all, prefix_mask, action_prefix = self.prepare(
            batch, num_steps, noise, action_prefix, prefix_length
        )
        dt = -1.0 / num_steps
        for i in range(num_steps):
            x_t = self.step(
                x_t, mods[:, i + 1], mods[:, 0], k_all, v_all, prefix_mask, action_prefix, dt
            )
        return x_t

    def sample_actions(self, batch, num_steps=10, noise=None):
        return self.sample(batch, num_steps=num_steps, noise=noise)

    def sample_actions_rtc(self, batch, action_prefix, prefix_length: int, num_steps=10, noise=None):
        return self.sample(
            batch, num_steps=num_steps, noise=noise,
            action_prefix=action_prefix, prefix_length=prefix_length,
        )
