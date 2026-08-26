"""Probe whether the wqkv/wo memory-config swap is reachable at all.

Builds an Attention2DConfig with two DIFFERENT weight memory configs and asks
resolve_attention2d_config for the materialized placements, under both the
pre-fix (swapped) and post-fix line ordering.
"""
import sys
from dataclasses import replace

sys.path.insert(0, "models/common/tests/modules/attention")

from models.common.modules.attention import attention_2d as A
from models.common.modules.attention.attention_2d import resolve_attention2d_config

import test_attention_2d as T


def build(*, wqkv_mem, wo_mem, weight_mem, wo_weight_mem):
    mesh = T._Mesh()
    return T._config(
        mesh=mesh,
        bias=True,
        wqkv=replace(
            T._weight((5120, 10240), mesh, T._Tensor("WQKV"), "wqkv-map", "wqkv-dtype"), memory_config=wqkv_mem
        ),
        wo=replace(T._weight((8192, 5120), mesh, T._Tensor("WO"), "wo-map", "wo-dtype"), memory_config=wo_mem),
        wqkv_bias=replace(
            T._weight((10240,), mesh, T._Tensor("BIAS"), "bias-map", "bias-dtype"), memory_config=weight_mem
        ),
        weight_memory_config=weight_mem,
        wo_weight_memory_config=wo_weight_mem,
    )


for label, kwargs in (
    (
        "both weights carry their own (matching) config",
        dict(wqkv_mem="WQKV-MEM", wo_mem="WO-MEM", weight_mem="WQKV-MEM", wo_weight_mem="WO-MEM"),
    ),
    (
        "both weights unplaced (memory_config=None)",
        dict(wqkv_mem=None, wo_mem=None, weight_mem="WQKV-MEM", wo_weight_mem="WO-MEM"),
    ),
    ("wo unplaced only", dict(wqkv_mem="WQKV-MEM", wo_mem=None, weight_mem="WQKV-MEM", wo_weight_mem="WO-MEM")),
):
    try:
        r = resolve_attention2d_config(build(**kwargs))
        print(f"{label:52s} -> wqkv={r.wqkv.memory_config!r} wo={r.wo.memory_config!r}")
    except Exception as exc:
        print(f"{label:52s} -> {type(exc).__name__}: {exc}")
