# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_post's feed cells (feed="on"): golden + regression on a mesh.

The feed makes the call a multi-chip op: y also lands where an all-to-all
over cluster_axis (in_dim -2, out_dim -1) would put it, over the fabric, in
the same program. These tests run on the feed mesh (topology.feed_mesh_shape:
a group of 4 chips along axis 1 — the system mesh of a Galaxy 8x4, a
QuietBox 2's chips opened as 1 x 4) under FABRIC_2D_TORUS_XY (the model's)
with the production fabric payload (conftest.mesh_device). Every chip runs
its own inputs; every gate of test_golden.py holds on every chip, and the
feed output must be the bits of the all-to-all of the chips' own y.

- test_op / test_op_loose: the TARGET x INPUTS cells with feed="on" (the
  feed-off cells are test_golden.py's) and the feed perf cases.
- numerics: the feed against the same call without it (X', stats, hc the
  same bits; y / y_output is unspecified with the feed), links 1 / 2,
  FABRIC_2D as well, repeated calls into two buffers with the same
  semaphores, the refusals.

On a cluster without a group of 4 chips along axis 1 no cell is emitted.
"""

from __future__ import annotations

import pytest
import ttnn

from eval.golden_harness import parametrize_cases, parametrize_loose_cases
from eval.golden_tests.mhc_post.feature_spec import (
    FEED_CLUSTER_AXIS,
    INPUTS,
    INVALID,
    LOOSE_CASES,
    TARGET,
    _case,
)
from eval.golden_tests.mhc_post.helpers import (
    CheckOutputError,
    call_mhc_post_mesh,
    check_feed_call,
    check_mesh_case,
    feed_buffer,
    feed_semaphores,
    make_mesh_case,
    mesh_chips,
    run_mhc_post,
    without_norm,
)
from eval.golden_tests.mhc_post.topology import feed_infeasible_reason
import importlib, os, sys  # feed_l1_landing: the golden feed cells against MY copy (MHC_FL_VARIANT=l1|base)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
_m = importlib.import_module("mhc_post_l1" if os.environ.get("MHC_FL_VARIANT", "l1") == "l1" else "mhc_post_base")
EXCLUSIONS, INPUT_TAGGERS, SUPPORTED, mhc_post = _m.EXCLUSIONS, _m.INPUT_TAGGERS, _m.SUPPORTED, _m.mhc_post
import eval.golden_tests.mhc_post.helpers as _H

_H.mhc_post = mhc_post  # run_mhc_post / call_mhc_post_mesh dispatch through helpers.mhc_post
from eval.golden_tests.mhc_post.conftest import _fabric_mesh_cache, mesh_device  # noqa: F401,E402

_SKIP = feed_infeasible_reason()


def _shape_id(inputs):
    return "X" + "x".join(str(d) for d in inputs[1])


def _feed(param):
    return param.values[1]["feed"] == "on"


def _cells():
    if _SKIP is not None:
        return []
    return [
        p
        for p in parametrize_cases(TARGET, INPUTS, INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, INVALID, shape_id=_shape_id)
        if _feed(p)
    ]


def _loose():
    if _SKIP is not None:
        return []
    return [
        p
        for p in parametrize_loose_cases(LOOSE_CASES, INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, INVALID, shape_id=_shape_id)
        if _feed(p)
    ]


@pytest.mark.parametrize("inputs,axes", _cells())
def test_op(inputs, axes, mesh_device):
    run_mhc_post(inputs, mesh_device=mesh_device, **axes)


@pytest.mark.parametrize("inputs,axes,extras", _loose())
def test_op_loose(inputs, axes, extras, mesh_device):
    run_mhc_post(inputs, mesh_device=mesh_device, extras=extras, **axes)


# ---------------------------------------------------------------------------
# Regression — DeepSeek-V4.1's feed call against the call without the feed
# ---------------------------------------------------------------------------

N = 4
BF16 = dict(dtype=ttnn.bfloat16, sublayer_dtype=ttnn.bfloat16)
HIFI2 = ttnn.MathFidelity.HiFi2
AXIS = FEED_CLUSTER_AXIS

FEED_SHAPES = [
    pytest.param(_case((1, 1, 160, 4 * 5120)), id="T160_C5120"),  # Galaxy 8x4 own rows (one unit a core)
    pytest.param(_case((1, 1, 32, 4 * 5120)), id="T32_C5120"),  # the replay tail
    pytest.param(_case((1, 1, 1280, 4 * 5120)), id="T1280_C5120"),  # several units a core
    pytest.param(_case((1, 1, 640, 4 * 1280)), id="T640_C1280"),  # blocks straddle the column units
    pytest.param(_case((1, 1, 64, 4 * 640)), id="T64_C640"),  # small, ragged last unit
]


def _need(norm="on"):
    """Skip until the op declares DeepSeek-V4.1's feed call (or the raw-y feed):
    every value in SUPPORTED and the cell not in EXCLUSIONS."""
    cell = {
        "dtype": ttnn.bfloat16,
        "sublayer_dtype": ttnn.bfloat16,
        "mode": "collapse",
        "coef_source": "packed",
        "norm": norm,
        "feed": "on",
    }
    for axis, value in cell.items():
        if value not in SUPPORTED.get(axis, []):
            pytest.skip(f"{axis}={value} not in SUPPORTED yet")
    for exc in EXCLUSIONS:
        if all(cell.get(k) == v for k, v in exc.items()):
            pytest.skip(f"excluded: {exc}")


def _feed_kw(mesh, fo, links):
    return dict(feed_output=fo, feed_semaphores=feed_semaphores(mesh), feed_num_links=links, feed_cluster_axis=AXIS)


def _against_no_feed(mesh, inputs, *, norm, links, seed):
    cases = make_mesh_case(
        inputs, mesh_chips(mesh), mode="collapse", coef_source="packed", seed=seed, norm=norm, **BF16
    )
    kw = dict(mode="collapse", coef_source="packed", mesh_device=mesh, math_fidelity=HIFI2, op=mhc_post, **BF16)
    plain = call_mhc_post_mesh([without_norm(c) for c in cases], **kw) if norm == "on" else None
    want = call_mhc_post_mesh(cases, **kw)
    check_mesh_case(want, cases, mode="collapse", dtype=ttnn.bfloat16, math_fidelity=HIFI2, plain_ref=plain)
    fo = feed_buffer(mesh, inputs[1], cluster_axis=AXIS)
    got = call_mhc_post_mesh(cases, **kw, **_feed_kw(mesh, fo, links))
    check_feed_call(got, want, fo, mesh_shape=tuple(mesh.shape), cluster_axis=AXIS)


@pytest.mark.numerics
@pytest.mark.parametrize("links", [1, 2], ids=["links1", "links2"])
@pytest.mark.parametrize("inputs", FEED_SHAPES)
def test_feed_matches_no_feed(inputs, links, mesh_device):
    """DeepSeek-V4.1's attn_residual call (norm + feed): X', stats and hc the
    same bits as the call without the feed; the feed is the all-to-all of
    that call's y on the group's chips, at 1 and 2 links."""
    _need()
    _against_no_feed(mesh_device, inputs, norm="on", links=links, seed=7)


@pytest.mark.numerics
@pytest.mark.parametrize("inputs", FEED_SHAPES[:2])
def test_feed_raw_y(inputs, mesh_device):
    """The feed of the collapse's raw y (no norm_weight): the same contract."""
    _need(norm="off")
    _against_no_feed(mesh_device, inputs, norm="off", links=2, seed=8)


@pytest.mark.numerics
@pytest.mark.fabric_config(ttnn.FabricConfig.FABRIC_2D)
@pytest.mark.parametrize("inputs", [FEED_SHAPES[0], FEED_SHAPES[3]])
def test_feed_fabric_2d(inputs, mesh_device):
    """The same under FABRIC_2D (the group a line, no wrap): the op reads the
    active fabric config and must not change it."""
    _need()
    _against_no_feed(mesh_device, inputs, norm="on", links=2, seed=9)


@pytest.mark.numerics
def test_feed_repeated_calls(mesh_device):
    """Five calls, other inputs each, two feed buffers alternating (a program
    cache hit with another address), the same semaphores: every feed matches
    the y of its own inputs. Catches a ready fence or arrival count not
    re-armed, and a peer's buffer overwritten before that peer reached the
    call."""
    _need()
    inputs = FEED_SHAPES[0].values[0]
    bufs = [feed_buffer(mesh_device, inputs[1], cluster_axis=AXIS, fill=f) for f in (1.0e4, -1.0e4)]
    for k in range(5):
        cases = make_mesh_case(
            inputs, mesh_chips(mesh_device), mode="collapse", coef_source="packed", seed=100 + k, norm="on", **BF16
        )
        kw = dict(
            mode="collapse", coef_source="packed", mesh_device=mesh_device, math_fidelity=HIFI2, op=mhc_post, **BF16
        )
        want = call_mhc_post_mesh(cases, **kw)
        got = call_mhc_post_mesh(cases, **kw, **_feed_kw(mesh_device, bufs[k % 2], links=2))
        check_feed_call(got, want, bufs[k % 2], mesh_shape=tuple(mesh_device.shape), cluster_axis=AXIS)


@pytest.mark.numerics
def test_feed_refusals(mesh_device):
    """A feed without semaphores, into a buffer of the wrong shape, or of a C
    that does not split into tile-wide blocks is refused with a clear error."""
    _need()
    kw = dict(mode="collapse", coef_source="packed", mesh_device=mesh_device, math_fidelity=HIFI2, op=mhc_post, **BF16)
    inputs = _case((1, 1, 64, 4 * 640))
    cases = make_mesh_case(
        inputs, mesh_chips(mesh_device), mode="collapse", coef_source="packed", seed=3, norm="on", **BF16
    )
    fo = feed_buffer(mesh_device, inputs[1], cluster_axis=AXIS)
    with pytest.raises(Exception) as e:  # noqa: PT011 — any clear refusal
        call_mhc_post_mesh(cases, **kw, feed_output=fo, feed_cluster_axis=AXIS)
    assert not isinstance(e.value, CheckOutputError)
    bad = feed_buffer(mesh_device, _case((1, 1, 64, 4 * 1280))[1], cluster_axis=AXIS)
    with pytest.raises(Exception):  # noqa: PT011
        call_mhc_post_mesh(cases, **kw, **_feed_kw(mesh_device, bad, links=1))
    odd = _case((1, 1, 64, 4 * 96))  # C = 96: blocks of 24 columns
    ocases = make_mesh_case(
        odd, mesh_chips(mesh_device), mode="collapse", coef_source="packed", seed=4, norm="on", **BF16
    )
    with pytest.raises(Exception):  # noqa: PT011
        call_mhc_post_mesh(
            ocases, **kw, **_feed_kw(mesh_device, feed_buffer(mesh_device, odd[1], cluster_axis=AXIS), links=1)
        )


if os.environ.get("FL_DEBUG_FEED"):  # feed_l1_landing diagnosis: print the wrong D tiles
    _orig_check_feed = _H.check_feed

    def _dbg_check_feed(feed_dev, ys, mesh_shape, cluster_axis):
        chips = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(feed_dev)]
        want = _H.pytorch_feed(ys, mesh_shape, cluster_axis)
        for i, (g, w) in enumerate(zip(chips, want)):
            g = g.bfloat16().reshape(-1, g.shape[-1])
            w = w.bfloat16().reshape(-1, w.shape[-1])
            bad = g != w
            if bad.any():
                tiles = sorted({(int(r) // 32, int(c) // 32) for r, c in bad.nonzero()})
                print(f"DBG chip {i}: {int(bad.sum())} bad, tiles {tiles[:30]}")
                t0 = tiles[0]
                blk = bad[t0[0] * 32 : (t0[0] + 1) * 32, t0[1] * 32 : (t0[1] + 1) * 32]
                print(
                    "DBG  rows",
                    sorted({int(r) for r, c in blk.nonzero()}),
                    "cols",
                    sorted({int(c) for r, c in blk.nonzero()}),
                )
                gg = g[t0[0] * 32 : (t0[0] + 1) * 32, t0[1] * 32 : (t0[1] + 1) * 32]
                ww = w[t0[0] * 32 : (t0[0] + 1) * 32, t0[1] * 32 : (t0[1] + 1) * 32]
                r, c = [int(v) for v in blk.nonzero()[0]]
                print("DBG  got", gg[r, max(0, c - 2) : c + 4].tolist(), "want", ww[r, max(0, c - 2) : c + 4].tolist())
        return _orig_check_feed(feed_dev, ys, mesh_shape, cluster_axis)

    _H.check_feed = _dbg_check_feed
