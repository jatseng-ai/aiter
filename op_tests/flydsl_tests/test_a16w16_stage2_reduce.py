# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""a16w16 stage2 reduce must replace the caller buffer, not atomic-add into it.

Atomic mode adds into the supplied output, so a nonzero prefill is part of the
result. Reduce mode writes one row per expert slot and then stores the sum over
that output, so the same prefill is discarded.
"""

from __future__ import annotations

import pytest
import torch

from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
from aiter.ops.shuffle import shuffle_weight

_PREFILL = 3.0

pytestmark = pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="a16w16 stage2 reduce is asserted on gfx942 and gfx950",
)


def _mismatch_ratio(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return (
        (~torch.isclose(reference, actual, rtol=1e-2, atol=1e-2)).float().mean().item()
    )


def _sorted_stage1(ref1, topk_ids, sorted_ids, sorted_expert_ids, num_valid, block_m):
    token, _topk, inter_dim = ref1.shape
    gathered = torch.zeros(
        (sorted_ids.shape[0], inter_dim), dtype=torch.bfloat16, device=ref1.device
    )
    token_of = sorted_ids[:num_valid] & 0x00FFFFFF
    rows = (token_of < token).nonzero(as_tuple=True)[0]
    for row in rows.tolist():
        token_id = int(token_of[row].item())
        expert = int(sorted_expert_ids[row // block_m].item())
        slot = (topk_ids[token_id] == expert).nonzero(as_tuple=True)[0]
        if slot.numel() == 0:
            continue
        gathered[row] = ref1[token_id, int(slot[0].item())]
    return gathered


def _torch_stage1(hidden, w1, topk_ids):
    token, topk = topk_ids.shape
    inter_dim = w1.shape[1] // 2
    out = torch.empty(
        (token, topk, inter_dim), device=hidden.device, dtype=torch.bfloat16
    )
    hidden_f = hidden.float()
    w1_f = w1.float()
    for token_id in range(token):
        for slot in range(topk):
            expert = int(topk_ids[token_id, slot])
            gate, up = (hidden_f[token_id] @ w1_f[expert].T).split(inter_dim)
            out[token_id, slot] = (torch.nn.functional.silu(gate) * up).to(
                torch.bfloat16
            )
    return out


def _torch_stage2(inter_states, w2, topk_ids, topk_weights):
    token, topk, _inter_dim = inter_states.shape
    out = torch.zeros(
        (token, w2.shape[1]), device=inter_states.device, dtype=torch.float32
    )
    inter_f = inter_states.float()
    w2_f = w2.float()
    for token_id in range(token):
        for slot in range(topk):
            expert = int(topk_ids[token_id, slot])
            routed = inter_f[token_id, slot] @ w2_f[expert].T
            out[token_id] += routed * topk_weights[token_id, slot]
    return out.to(torch.bfloat16)


def _case(tile_m, tile_k):
    token, model_dim, inter_dim, expert, topk = 8, 256, 256, 8, 2
    assert inter_dim % tile_k == 0
    torch.manual_seed(0)
    hidden = torch.randn((token, model_dim), device="cuda", dtype=dtypes.bf16) / 10
    w1 = (
        torch.randn(
            (expert, inter_dim * 2, model_dim), device="cuda", dtype=dtypes.bf16
        )
        / 10
    )
    w2 = torch.randn((expert, model_dim, inter_dim), device="cuda", dtype=dtypes.bf16)
    topk_ids = torch.stack(
        [(torch.arange(token, device="cuda") + slot) % expert for slot in range(topk)],
        dim=1,
    ).to(torch.int32)
    topk_weights = torch.rand((token, topk), device="cuda", dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    ref1 = _torch_stage1(hidden, w1, topk_ids)
    reference = _torch_stage2(ref1, w2, topk_ids, topk_weights)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _moe_buf = (
        moe_sorting(topk_ids, topk_weights, expert, model_dim, dtypes.bf16, tile_m)
    )
    needed = sorted_expert_ids.shape[0] * tile_m
    if sorted_ids.shape[0] < needed:
        pad = needed - sorted_ids.shape[0]
        sorted_ids = torch.cat(
            [
                sorted_ids,
                torch.full(
                    (pad,), token, dtype=sorted_ids.dtype, device=sorted_ids.device
                ),
            ]
        )
        sorted_weights = torch.cat(
            [
                sorted_weights,
                torch.zeros(
                    pad, dtype=sorted_weights.dtype, device=sorted_weights.device
                ),
            ]
        )
    inter_sorted = _sorted_stage1(
        ref1,
        topk_ids,
        sorted_ids,
        sorted_expert_ids,
        int(num_valid_ids.reshape(-1)[0].item()),
        tile_m,
    )
    return {
        "topk": topk,
        "reference": reference,
        "inter_sorted": inter_sorted,
        "w2": shuffle_weight(w2, (16, 16)),
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
    }


@pytest.mark.parametrize(
    ("tile_m", "tile_k"),
    [
        pytest.param(16, 128, id="t16x128x128"),
        pytest.param(32, 256, id="t32x128x256"),
    ],
)
def test_prefilled_output_atomic_fails_reduce_passes(tile_m, tile_k):
    data = _case(tile_m, tile_k)
    reference = data["reference"]
    ratios = {}
    for mode in ("atomic", "reduce"):
        output = torch.full_like(reference, _PREFILL)
        flydsl_moe_stage2(
            inter_states=data["inter_sorted"],
            w2=data["w2"],
            sorted_token_ids=data["sorted_ids"],
            sorted_expert_ids=data["sorted_expert_ids"],
            num_valid_ids=data["num_valid_ids"],
            out=output,
            topk=data["topk"],
            tile_m=tile_m,
            tile_n=128,
            tile_k=tile_k,
            a_dtype="bf16",
            b_dtype="bf16",
            out_dtype="bf16",
            mode=mode,
            sorted_weights=data["sorted_weights"],
        )
        torch.cuda.synchronize()
        ratios[mode] = _mismatch_ratio(reference, output)
        assert torch.isfinite(output).all(), mode

    assert ratios["atomic"] > 0.5, ratios
    assert ratios["reduce"] <= 0.01, ratios
