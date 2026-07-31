# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""gateup_reduce_overlap — isolated bake-off program descriptor.

Reconstructs moe_fused_swiglu's gate/up matmul + cross-column reduce (op_design.md §4.3) for ONE
grid column (KGROUPS = 10 cores), using the EXACT reduce-tree topology the real op's
`_reduce_tree()` builds for column x = 0 (root row0; children 1,2,4,8; row4's children 5,6; row6's
child 7; row2's child 3; row8's child 9). No x/h multicast: within a column, x is NOT reuse-shared
(each row owns a distinct K-slice of x/w_gate/w_up), so this bench reads its own per-row DRAM slice
directly — the mcast machinery is a separate, already-characterised part of the op and is held
trivial/absent here (concept isolation).

Variants (all under one set of kernels, selected by compile-time args):
  S = 1                       baseline — whole-block matmul, then whole-block reduce (verbatim).
  S > 1, pipelined = False    split-serial — split into S stages, matmul-then-reduce PER STAGE with
                              no reordering. Isolates the DEST-shrink cost alone.
  S > 1, pipelined = True     split-pipelined — stage s+1's matmul is issued BEFORE stage s's reduce
                              is consumed. The idea under test.
  split_axis = "hn"           split the hidden axis (HN_BLOCK = HN_PAD / S) — shrinks the matmul's
                              out_subblock_w, so DEST-shrink cost applies.
  split_axis = "m"            split the token-row axis (M_GROUP = M_EFF / S) — out_subblock_w stays
                              HN_PAD; DEST-shrink cost should NOT apply (the option-4 hypothesis).
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32
KGROUPS = 10  # one grid column, exactly moe_fused_swiglu's KGROUPS on this box (blackhole_p150)
MAX_CHILDREN = 4  # real max fan-in of this tree (the root)
SEM_GO = 0
SEM_DATA = 1


def _split(total, groups):
    """`base + (i < rem)` split — same rule as the real op's program descriptor."""
    base, rem = total // groups, total % groups
    sizes = [base + (1 if i < rem else 0) for i in range(groups)]
    starts, acc = [], 0
    for sz in sizes:
        starts.append(acc)
        acc += sz
    return sizes, starts


def kr_sizes_starts(emb_t):
    return _split(emb_t, KGROUPS)


def reduce_tree(kgroups=KGROUPS):
    """The real op's `_reduce_tree`, restricted to one column (root_y = 0). Reproduced here (not
    imported) so this experiment has zero dependency on the real op's files."""
    info = {}
    root_y = 0
    for y in range(kgroups):
        r = (y - root_y) % kgroups
        children = []
        s = 1
        while s < kgroups:
            if r % (2 * s) == 0 and r + s < kgroups:
                children.append((root_y + r + s) % kgroups)
            s *= 2
        parent = None
        if r != 0:
            low = r & (-r)
            parent = (root_y + r - low) % kgroups
        info[y] = {"is_root": r == 0, "parent": parent, "children": children}
    return info


def _virt(device, y):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(0, y))
    return int(c.x), int(c.y)


def _cb(index, core_range, num_pages, page_size, data_format):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_range,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def make_x_tensor(torch_x, device):
    return ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def make_weight_tensor(torch_w, device):
    return ttnn.from_torch(
        torch_w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def create_program_descriptor(
    x,
    w_gate,
    w_up,
    out,
    *,
    device,
    emb_t,
    hn_pad,
    m_eff,
    s_stages,
    split_axis,  # "hn" or "m"
    pipelined,
    kgroups=None,  # override for diagnostics/testing only; production callers leave this at KGROUPS
):
    if kgroups is None:
        kgroups = KGROUPS
    if split_axis not in ("hn", "m"):
        raise ValueError(f"gateup_reduce_overlap: split_axis must be 'hn' or 'm', got {split_axis!r}")
    axis = 0 if split_axis == "hn" else 1
    if axis == 0:
        if hn_pad % s_stages != 0:
            raise ValueError(f"gateup_reduce_overlap: HN split S={s_stages} must divide hn_pad={hn_pad}")
        hn_block, m_group = hn_pad // s_stages, m_eff
    else:
        if m_eff % s_stages != 0:
            raise ValueError(f"gateup_reduce_overlap: M split S={s_stages} must divide m_eff={m_eff}")
        hn_block, m_group = hn_pad, m_eff // s_stages

    stage_rows = m_group if axis == 1 else m_eff
    stage_cols = hn_block if axis == 0 else hn_pad
    # DEST budget check: out_subblock_h (pinned at 1, matching the real op's OUT_SUBBLOCK_H_GU) *
    # out_subblock_w (= stage_cols) must fit DEST_AUTO_LIMIT (8 bf16 tiles). `stage_rows` is
    # in0_num_subblocks (matmul_block loops over that many M-subblocks, draining DEST each time) —
    # NOT part of one subblock's DEST footprint, which is the whole point of the M-split option.
    if stage_cols > 8:
        raise ValueError(f"gateup_reduce_overlap: stage out_subblock_w {stage_cols} exceeds the 8-tile DEST budget")

    kr_sizes, kr_starts = _split(emb_t, kgroups)
    kr_pad = max(kr_sizes)
    tree = reduce_tree(kgroups)

    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, kgroups - 1))])

    bfp4_tile = ttnn.tile_size(ttnn.bfloat4_b)
    bfp8_tile = ttnn.tile_size(ttnn.bfloat8_b)

    # ---- CB index layout: contiguous bases per family, sized per the split axis (see module doc).
    # x / weight are S-wide on the axis they are SPLIT on, 1-wide (shared) on the other; the rest
    # (acc/A/B/reduce-in) are always S-wide (every stage needs its own). CB_A/CB_B are reused for
    # BOTH root's (silu, h_local) role and non-root's (gate_send, up_send) role — a core is only
    # ever one of those, so this halves the CB count vs separate indices (keeps S<=8 under the 64-CB
    # cap: 1 + 8*8 = 65 would blow it with separate indices; this layout is 2 + 7*8 = 58 at worst).
    n_x = s_stages if axis == 1 else 1
    n_wg = s_stages if axis == 0 else 1
    idx = 0
    CB_X_BASE = idx
    idx += n_x
    CB_WG_BASE = idx
    idx += n_wg
    CB_WU_BASE = idx
    idx += n_wg
    CB_GATE_ACC_BASE = idx
    idx += s_stages
    CB_UP_ACC_BASE = idx
    idx += s_stages
    CB_A_BASE = idx
    idx += s_stages
    CB_B_BASE = idx
    idx += s_stages
    CB_REDUCE_GATE_BASE = idx
    idx += s_stages
    CB_REDUCE_UP_BASE = idx
    idx += s_stages
    if idx > 64:
        raise RuntimeError(f"gateup_reduce_overlap: {idx} CB indices exceeds the 64-CB limit")

    stage_tiles = stage_rows * stage_cols
    cbs = [
        _cb(CB_X_BASE + i, core_range, m_group * kr_pad if axis == 1 else m_eff * kr_pad, bfp8_tile, ttnn.bfloat8_b)
        for i in range(n_x)
    ]
    cbs += [_cb(CB_WG_BASE + i, core_range, kr_pad * stage_cols, bfp4_tile, ttnn.bfloat4_b) for i in range(n_wg)]
    cbs += [_cb(CB_WU_BASE + i, core_range, kr_pad * stage_cols, bfp4_tile, ttnn.bfloat4_b) for i in range(n_wg)]
    for base in (CB_GATE_ACC_BASE, CB_UP_ACC_BASE, CB_A_BASE, CB_B_BASE, CB_REDUCE_GATE_BASE, CB_REDUCE_UP_BASE):
        cbs += [_cb(base + i, core_range, stage_tiles, bfp8_tile, ttnn.bfloat8_b) for i in range(s_stages)]

    reader_ct = [
        kr_pad,
        hn_pad,
        m_eff,
        s_stages,
        axis,
        hn_block,
        m_group,
        emb_t,
        MAX_CHILDREN,
        SEM_GO,
        SEM_DATA,
        bfp8_tile,
        bfp4_tile,
        CB_X_BASE,
        CB_WG_BASE,
        CB_REDUCE_GATE_BASE,
        CB_REDUCE_UP_BASE,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(w_gate).get_compile_time_args())

    writer_ct = [
        kr_pad,
        hn_pad,
        m_eff,
        s_stages,
        axis,
        hn_block,
        m_group,
        emb_t,
        SEM_GO,
        SEM_DATA,
        bfp8_tile,
        bfp4_tile,
        CB_WU_BASE,
        CB_A_BASE,
        CB_B_BASE,
        CB_REDUCE_GATE_BASE,
        CB_REDUCE_UP_BASE,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(w_up).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    compute_ct = [
        kr_pad,
        hn_pad,
        m_eff,
        s_stages,
        axis,
        1 if pipelined else 0,
        hn_block,
        m_group,
        CB_X_BASE,
        CB_WG_BASE,
        CB_WU_BASE,
        CB_GATE_ACC_BASE,
        CB_UP_ACC_BASE,
        CB_A_BASE,
        CB_B_BASE,
        CB_REDUCE_GATE_BASE,
        CB_REDUCE_UP_BASE,
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    for y in range(kgroups):
        node = tree[y]
        kr, kstart = kr_sizes[y], kr_starts[y]
        core = ttnn.CoreCoord(0, y)

        r_args = [
            x.buffer_address(),
            w_gate.buffer_address(),
            kr,
            kstart,
            len(node["children"]),
        ]
        for c in range(MAX_CHILDREN):
            if c < len(node["children"]):
                cx, cy = _virt(device, node["children"][c])
            else:
                cx, cy = 0, 0
            r_args.extend([cx, cy])
        reader_rt[0][y] = r_args

        px, py = _virt(device, node["parent"]) if node["parent"] is not None else (0, 0)
        writer_rt[0][y] = [
            w_up.buffer_address(),
            out.buffer_address(),
            kr,
            kstart,
            1 if node["is_root"] else 0,
            px,
            py,
        ]

        compute_rt[0][y] = [kr, 1 if node["is_root"] else 0, len(node["children"])]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gru_reader.cpp"),
            core_ranges=core_range,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gru_writer.cpp"),
            core_ranges=core_range,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gru_compute.cpp"),
            core_ranges=core_range,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            # SAME user precision contract as the real op — fixed, not a lever (see op_design.md §2).
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        ),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_GO, core_ranges=core_range, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_DATA, core_ranges=core_range, initial_value=0),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def gateup_reduce_overlap(x, w_gate, w_up, out, **kwargs):
    device = x.device()
    descriptor = create_program_descriptor(x, w_gate, w_up, out, device=device, **kwargs)
    return ttnn.generic_op([x, w_gate, w_up, out], descriptor)
