# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERF 14 — the L1 BUDGET probe for moe_fused_swiglu, reported against the ALLOCATOR's number.

    MOE_SWIGLU_GRID=11x8 MOE_SWIGLU_M_BLOCK=16 scripts/tt-probe.sh moe_fused_swiglu < this_file

WHY THIS FILE EXISTS, AND THE TRAP IT CLOSES. A naive `sum(cb.total_size)` over the descriptor is
NOT the L1 budget, and reading it as one understates the allocator by a constant that is bigger than
the whole headroom. `ProgramImpl::compile_and_allocate` compares

    cb_region_end  =  L1_UNRESERVED_BASE  +  sum(cb sizes)          <=  max_l1_size

against the FULL L1 size (`program.cpp:1711`; the region starts at `base_address` in
`CircularBufferAllocator::mark_address`, `program.cpp:1201`). So the number in

    "Statically allocated circular buffers ... grow to N B which is beyond max L1 size of M B"

is an ADDRESS, not a size. On this box `L1_UNRESERVED_BASE` is ~111 KB of firmware + kernel
binaries + launch/runtime-arg config + semaphores + the profiler region, and it is charged whether
or not profiling is enabled at runtime — so the usable CB budget is
`get_max_worker_l1_unreserved_size()`, NOT `l1_size_per_core()`. This probe prints both the CB sum
and the allocator-equivalent figure, so neither can be mistaken for the other.

It also prints the L1 that shrinking the resident `x` K extent WOULD free, computed analytically
from the CB formulas rather than by resizing a CB — there is no knob for it, because the kernels
cannot honour a shrunk `cb_x_tiles` without the K-block-major x multicast (not built).
"""

import os

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu_program_descriptor as D
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu import default_compute_kernel_config

EMB, CAP, HID = int(os.environ.get("MOE_L1_EMB", 7168)), 5120, 2048


def main():
    import torch  # lazy: ttnn/ forbids a global torch import

    dev = ttnn.open_device(device_id=0)
    try:
        x = ttnn.from_torch(
            torch.zeros((1, 1, CAP, EMB), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gu_mc, dn_mc = D.weight_memory_configs(dev, EMB, HID)
        mk = lambda s, mc: ttnn.from_torch(  # noqa: E731
            torch.zeros(s, dtype=torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mc,
        )
        wg, wu, wd = mk((EMB, HID), gu_mc), mk((EMB, HID), gu_mc), mk((HID, EMB), dn_mc)
        u32 = lambda t: ttnn.from_torch(  # noqa: E731
            t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        counts, idx = u32(torch.zeros(256, dtype=torch.int32)), u32(torch.zeros(8, dtype=torch.int32))
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, CAP, EMB]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        g = dev.compute_with_storage_grid_size()
        mbox = D.make_mailbox(dev, int(g.x) * int(g.y))
        desc = D.create_program_descriptor(
            x,
            wg,
            wu,
            wd,
            counts,
            idx,
            out,
            mbox,
            local_expert_id=3,
            input_m_tiles=CAP // 32,
            compute_kernel_config=default_compute_kernel_config(),
        )

        names = {v: k for k, v in vars(D).items() if k.startswith("CB_") and isinstance(v, int)}
        rows, tot = [], 0
        for cb in desc.cbs:
            i = cb.format_descriptors[0].buffer_index
            rows.append((names.get(i, f"cb{i}"), cb.total_size))
            tot += cb.total_size
        rows.sort(key=lambda r: -r[1])

        hgroups, kgroups = D.worker_grid(dev)
        hid_t, emb_t = HID // D.TILE, EMB // D.TILE
        kr_pad = -(-emb_t // kgroups)
        m_block = D.M_BLOCK
        depth_x = D.DEPTH_X if (CAP // 32 + m_block - 1) // m_block > 1 else 1
        bfp8 = ttnn.tile_size(ttnn.bfloat8_b)
        hn_pad = -(-hid_t // hgroups)

        # THE OFFSET IS CALIBRATED, NOT READ FROM AN ACCESSOR — and that is deliberate.
        # `get_max_worker_l1_unreserved_size()` reports 1,532,032 on this box, which would imply an
        # offset of 40,832; the allocator's own throws say 111,488. They are DIFFERENT quantities: the
        # accessor is the BUFFER allocator's region, while `cb_region_end` starts above THIS PROGRAM's
        # kernel-config / runtime-arg / semaphore region, which scales with the op's arg count. So the
        # offset is program-specific and has to be calibrated against the allocator itself.
        #
        # PROVENANCE — three independent UNPROFILED `TT_THROW`s on this program, all agreeing:
        #   DEPTH_H=4 -> 1,614,528 = 1,503,040 + 111,488   (CB sum + one cb_h slot)
        #   DEPTH_H=5 -> 1,666,752 = 1,555,264 + 111,488   (CB sum + two cb_h slots)
        #   shipped   -> 1,562,304 = 1,450,816 + 111,488
        L1_MAX = 1_572_864  # `max_l1_size` exactly as the allocator reports it in the throw
        BASE = 111_488  # calibrated L1_UNRESERVED_BASE for THIS program
        budget = L1_MAX - BASE  # == 1,461,312: the real budget for a descriptor CB SUM
        # The calibration SELF-CHECKS in the shipped configuration, so it cannot rot silently.
        shipped = m_block == 8 and depth_x == 2 and (hgroups, kgroups) == (11, 8) and EMB == 7168
        calib = "OK" if (not shipped or tot + BASE == 1_562_304) else f"STALE (got {tot + BASE:,d})"

        print(f"### M_BLOCK={m_block} DEPTH_X={depth_x} GRID={hgroups}x{kgroups} emb={EMB} KR_PAD={kr_pad}")
        for n, s in rows:
            if s > 4096:
                print(f"  {n:22s} {s:9,d}")
        print(f"  {'CB SUM':22s} {tot:9,d}")
        print(f"  {'L1_UNRESERVED_BASE':22s} {BASE:9,d}   <- what a naive CB sum MISSES (calib {calib})")
        print(f"  {'allocator region_end':22s} {BASE + tot:9,d}   vs max_l1_size {L1_MAX:,d}")
        print(f"  {'CB BUDGET':22s} {budget:9,d}   -> FREE {budget - tot:,d}")

        # What K-chunking `x` would free, analytically: cb_x_tiles + the two STAGING CBs, which are also
        # sized by the resident K extent (the stick read is only K tile-columns wide).
        x_stick = kr_pad * D.TILE * 2  # bf16 RM slice of one stick
        print("  --- if the resident x K extent were K (needs the K-block-major multicast, NOT built) ---")
        for kb in (kr_pad, 8, 7, 4):
            cb_x = depth_x * m_block * kb * bfp8
            cb_x_in = D.XSTICK_ROWS * D.TILE * (kb * D.TILE * 2)
            cb_x_stage = D.DEPTH_XSTAGE * kb * bfp8
            now = depth_x * m_block * kr_pad * bfp8 + D.XSTICK_ROWS * D.TILE * x_stick + D.DEPTH_XSTAGE * kr_pad * bfp8
            freed = now - (cb_x + cb_x_in + cb_x_stage)
            # the bf16 gate/up accumulator K-blocking MANDATES (measured: bfp8 L1-acc returns inf/nan)
            acc = m_block * hn_pad * ttnn.tile_size(ttnn.bfloat16)
            print(
                f"    K={kb:3d}: cb_x {cb_x:8,d} +in {cb_x_in:7,d} +stage {cb_x_stage:6,d}"
                f"  frees {freed:8,d}  net after +bf16_acc({acc:,d}) = {freed - acc:+9,d}"
                f"  -> FREE {budget - tot + freed - acc:+9,d}"
            )
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
