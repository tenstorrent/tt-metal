# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe: dump the DRAM bank endpoint tables, then read the same W bytes through explicit DRAM ports.

PROBE_CANDS="b:x:y:noc;..." lists candidate endpoints (translated coords). Prints per candidate whether the bytes
equal the TensorAccessor read.
"""

import os
from pathlib import Path

import ttnn

HERE = Path(__file__).parent
REC_BYTES = 2176
Q0 = 100


def _cands():
    spec = os.environ.get("PROBE_CANDS", "")
    return [tuple(int(v) for v in c.split(":")) for c in spec.split(";") if c]


def test_probe(device):
    import torch

    torch.manual_seed(0)
    w = (torch.randn(1536, 4096) * 0.02).bfloat16()
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    num_banks = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    cands = _cands()
    num_rec = 1 + num_banks + len(cands)
    out = ttnn.from_torch(
        torch.zeros(num_rec, REC_BYTES // 4, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    ct = [0, REC_BYTES]
    ct += list(ttnn.TensorAccessorArgs(tw).get_compile_time_args())
    ct += list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [tw.buffer_address(), out.buffer_address(), num_banks, Q0, tw.buffer_page_size(), len(cands)] + [
        v for c in cands for v in c
    ]
    k = ttnn.KernelDescriptor(
        kernel_source=str(HERE / "kernels" / "probe_ports.cpp"),
        core_ranges=core,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
    )
    cb = ttnn.CBDescriptor(
        total_size=REC_BYTES,
        core_ranges=core,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=REC_BYTES)],
    )
    res = ttnn.to_torch(ttnn.generic_op([tw, out], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])))
    t = res[0].tolist()
    print(f"\nW addr {tw.buffer_address():#x} page {tw.buffer_page_size()} worker my_x/y {t[100]},{t[101]}")
    for b in range(num_banks):
        n0, n1, sig, off = (t[4 * b + i] & 0xFFFFFFFF for i in range(4))
        print(
            f"bank {b}: noc0 ({n0 & 63},{(n0 >> 6) & 63}) noc1 ({n1 & 63},{(n1 >> 6) & 63}) "
            f"signal ({sig & 63},{(sig >> 6) & 63}) raw_sig {sig:#x} offset {off}"
        )
    for i, (b, x, y, n) in enumerate(cands):
        got = res[1 + num_banks + i]
        ref = res[1 + b]
        print(
            f"cand bank {b} ({x},{y}) noc{n}: {'MATCH' if torch.equal(got, ref) else 'MISMATCH'} first {got[0].item() & 0xFFFFFFFF:#x} ref {ref[0].item() & 0xFFFFFFFF:#x}"
        )
