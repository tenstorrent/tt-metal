# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device driver for the `traffic_shape_ceiling` bare-traffic bench.

See `bench.py` for what the traffic is, why the tensors are ROW_MAJOR and how
each rung is verified. This file only dispatches and gates.

All seven rungs run in ONE process, INTERLEAVED rep by rep
(`for rep: for variant: ...`), so a process-level drift cannot be read as a
variant difference; rep 0 is dropped by the summarizer as the warm rung.

    scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/tilize/perf_experiments/traffic_shape_ceiling/test_traffic_shape_ceiling.py
    python3 ttnn/ttnn/operations/tilize/perf_experiments/traffic_shape_ceiling/summarize.py

Correctness is the only pass/fail. The dependency-free rungs write garbage BY
DESIGN, so their gate is the transfer-marker check, not a value check — see
`bench.py`'s VERIFICATION note.
"""

import json
import sys
from pathlib import Path

import ttnn

HERE = Path(__file__).parent
# `perf_experiments/__init__.py` DELIBERATELY empties its `__path__` (so that
# `import ttnn` does not execute every bench in the tree), which makes
# `from ttnn.operations.tilize.perf_experiments... import bench` fail by design.
# The documented way in is by explicit path, which is all this harness needs.
sys.path.insert(0, str(HERE))
import bench  # noqa: E402

REPS = 7  # rep 0 is the warm rung, dropped by summarize.py


def _make_tensors(device, torch):
    torch.manual_seed(7)
    a = torch.randn(bench.RM_SHAPE, dtype=torch.float32).bfloat16()
    b = torch.randn(bench.PAGED_SHAPE, dtype=torch.float32).bfloat16()
    src_rm = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    src_paged = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # The geometry claim the whole bench rests on: the RM source is 32 pages of
    # 32768 B, and the 2-KB source / the destination are 512 pages of 2048 B —
    # i.e. exactly the op's TILE output's page geometry.
    assert (int(src_rm.buffer_num_pages()), int(src_rm.buffer_aligned_page_size())) == (32, 32768)
    assert (int(src_paged.buffer_num_pages()), int(src_paged.buffer_aligned_page_size())) == (512, 2048)
    return a, b, src_rm, src_paged


def _expected_chained_512(torch_a, torch):
    """Byte permutation the 512-B chained traffic implies.

    Core b's CB holds row r of its 512-B column slice at L1 offset r*512, i.e.
    `torch_a[0,0,r, 256b:256b+256]`. Write i then moves CB bytes
    [i*2048,(i+1)*2048) — rows 4i..4i+3 — into destination page 8b+i.
    """
    out = torch.zeros(bench.PAGED_SHAPE, dtype=torch.bfloat16)
    for b in range(bench.NUM_CORES):
        sl = torch_a[0, 0, :, 256 * b : 256 * (b + 1)]  # [32, 256]
        for i in range(bench.NUM_WRITES):
            out[0, 0, 8 * b + i, :] = sl[4 * i : 4 * i + 4, :].reshape(-1)
    return out


def _assert_markers(got, torch, label):
    """(c) transfer-marker gate for the scratch-sourced rungs."""
    bits = got.contiguous().view(torch.int16)[0, 0]
    for b in range(bench.NUM_CORES):
        for i in range(bench.NUM_WRITES):
            p = 8 * b + i
            head = (int(bits[p, 0]), int(bits[p, 1]))
            tail = (int(bits[p, -2]), int(bits[p, -1]))
            assert head == (0x4000 | b, 0x4100 | i), f"{label}: page {p} head marker {head} (core {b}, transfer {i})"
            assert tail == (0x4200 | b, 0x4300 | i), f"{label}: page {p} tail marker {tail} (core {b}, transfer {i})"


def test_traffic_shape_ceiling(device):
    import torch

    torch_a, torch_b, src_rm, src_paged = _make_tensors(device, torch)

    dsts = {
        v: ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(bench.PAGED_SHAPE)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        for v in bench.VARIANT_ORDER
    }
    for d in dsts.values():
        assert (int(d.buffer_num_pages()), int(d.buffer_aligned_page_size())) == (512, 2048)

    order = []
    for rep in range(REPS):
        for v in bench.VARIANT_ORDER:
            bench.run(v, src_rm, src_paged, dsts[v])
            order.append({"rep": rep, "variant": v})
    ttnn.synchronize_device(device)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "dispatch_order.json").write_text(json.dumps(order, indent=1))

    # --- gates -----------------------------------------------------------
    # (b) value gate on the two chained rungs: the reader's addresses, lengths
    #     and L1 landing offsets and the writer's page ids, pinned bit-exactly.
    got = ttnn.to_torch(dsts["chained"])
    assert torch.equal(got, _expected_chained_512(torch_a, torch)), "chained: traffic is not the 512-B focus pattern"
    got = ttnn.to_torch(dsts["chained_2k"])
    assert torch.equal(got, torch_b), "chained_2k: 2-KB read/write traffic is not the identity page copy"
    # (c) marker gate on the three scratch-sourced rungs.
    for v in ("independent", "independent_2k", "writes_only"):
        _assert_markers(ttnn.to_torch(dsts[v]), torch, v)
    # The reads-only rungs write nothing: their destination must be untouched
    # (allocate_tensor_on_device leaves it whatever it was, so the only claim
    # that can be made is that `num_writes == 0` really did issue no write —
    # asserted by the marker check FAILING to appear).
    for v in ("reads_only_512", "reads_only_2k"):
        bits = ttnn.to_torch(dsts[v]).contiguous().view(torch.int16)[0, 0]
        assert not all(
            int(bits[8 * b, 0]) == (0x4000 | b) for b in range(bench.NUM_CORES)
        ), f"{v}: destination carries write markers, so writes were NOT ablated"

    print(
        f"\n[traffic_shape_ceiling] {len(order)} dispatches, {REPS} reps x {len(bench.VARIANT_ORDER)} rungs, all gated"
    )
