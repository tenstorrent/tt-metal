# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""height_rm_flat: HEIGHT_SHARDED ROW_MAJOR shards with few sticks per Tensix core, normalized as FLAT tiles.

Graduation shape: this module's `create_program_descriptor` is a drop-in for
`rms_norm_program_descriptor.create_program_descriptor` — it routes the flat regime (flat_expressible + flat_fits +
flat_preferred cost model, or the tilized plan not fitting L1) to `create_flat_program_descriptor` and everything
else to the op's unchanged descriptor.

Flat regime (kernels/rms_norm_flat_{reader,compute,writer}.cpp): a stick of W datums is ceil(W / 1024) flat tiles
(1024 consecutive datums each). Per stick: sum x^2 = DEST-accumulated x_j * x_j over its flat tiles + one FPU
REDUCE_SCALAR + finalize; out_j = x_j * bcast_scalar(inv) [* gamma_j], gamma_j = flat tile j of the RM gamma page.
No tilize / untilize, no 32x stick padding, gamma resident at W * elem bytes (no streaming).

CB plan (per Tensix core, S = shard height in sticks, nt = ceil(W / 1024), T = 1024 * elem bytes):
  zero-copy (FLAT_MODE "zc"):
    cb_x   backed on the input shard,  page = gcd(stick pitch, T)  (flat tile j of stick s = page s*sp + j*tp)
    cb_out backed on the output shard, same page
    W % 1024 != 0: cb_x_rm / cb_out_rm  S tail tiles each (the stick's last partial flat tile, staged + zero padded)
  staged (FLAT_MODE "staged"):
    cb_x / cb_out  S * nt tiles (each stick copied into nt zero-padded flat tiles and back)
  cb_sumsq S, cb_inv_rms S (accumulator dtype), cb_ones 1 (bf16), cb_gamma nt flat tiles (gamma dtype).
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

FLAT_KERNEL_DIR = Path(__file__).parent / "kernels"

# Flat regime gate (cost model, measured BH p150b, bf16 x / bf16 RM gamma / fp32 DEST off / HiFi4, 64-130 Tensix
# cores; fp32 x and fp32 DEST make the tilized path slower still, so the model is conservative there):
#   flat     ~ FLAT_FIXED_NS + S * (FLAT_STICK_NS + FLAT_TILE_NS * ceil(W / 1024) + FLAT_TAIL_NS * [W % 1024 != 0])
#   tilized  ~ ceil(S / 32) * (TILIZED_ROW_NS + TILIZED_WTILE_NS * ceil(W / 32))
# S = sticks per Tensix core (shard height). Crossover (flat == tilized): W = 256: S ~ 7; W = 512: S ~ 13; W = 1024:
# S ~ 29 (and 33..60); W = 2048: S ~ 37; W >= 4096: S >= 64. Measured: perf_experiments/height_rm_flat/runs (r4-r7).
FLAT_ENABLE = True
FLAT_MODE = "zc"  # "zc" (zero-copy body + staged tail tile) | "staged" (whole stick copied into flat tiles)
FLAT_FORCE = False  # bench hook: route every expressible + fitting HEIGHT RM cell to the flat path
FLAT_FIXED_NS = 1200
FLAT_STICK_NS = 150  # per stick: REDUCE_SCALAR + one-vector finalize + pack
FLAT_TILE_NS = 230  # per flat tile: x*x (pass 1) + x*inv*gamma (pass 2)
FLAT_TAIL_NS = 80  # per stick with a partial last flat tile (tail staging copy + second CB)
TILIZED_ROW_NS = 2300  # per tile-row of the tilized path (tilize / collapse / untilize fixed cost)
TILIZED_WTILE_NS = 310  # per W tile of a tile-row (tilize + pass 1 + pass 2 + untilize)
# Zero-copy cb_x / cb_out pages are gcd(stick pitch, tile bytes) bytes — keep the per-CB page count well inside the
# CB interface's 16-bit page counters.
FLAT_MAX_CB_PAGES = 32768
# REDUCE_SCALAR moves the 32 first-stage partial sums DEST -> srcB (MOVD2B) for the second pool, so they take srcB's
# format: a bf16 1.0 scaler truncates them to bf16. True: the scaler CB follows the accumulator dtype (fp32 with fp32
# DEST), keeping the partials at srcB's 19-bit (TF32) width (fp32 DEST: rms error vs torch 0.0027 -> 0.0015 bf16
# no-gamma, 0.0014 -> 0.00056 fp32 no-gamma; same device time).
FLAT_ONES_ACC = True
FLAT_EXTRA_DEFINES = ()  # bench hook (e.g. RMS_FLAT_FIN_COL0: finalize all of column 0, the op's body)


def _datum_bytes(dtype) -> int:
    return {ttnn.float32: 4, ttnn.bfloat16: 2}[dtype]


def flat_expressible(input_tensor, gamma) -> bool:
    """HEIGHT_SHARDED ROW_MAJOR fp32 / bf16 input whose shard holds whole sticks, gamma absent or ROW_MAJOR."""
    if not FLAT_ENABLE or input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        return False
    mc = input_tensor.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        return False
    if input_tensor.dtype not in (ttnn.float32, ttnn.bfloat16):
        return False
    if gamma is not None and (
        gamma.layout != ttnn.ROW_MAJOR_LAYOUT or gamma.dtype not in (ttnn.float32, ttnn.bfloat16)
    ):
        return False  # TILE gamma: the flat gamma would need a row-0 gather (not built; golden INVALID for RM HEIGHT)
    return int(mc.shard_spec.shape[1]) >= int(input_tensor.shape[-1])


def flat_plan(input_tensor, gamma, fp32_dest: bool) -> dict:
    """Geometry + CB plan of the flat path ({cb: (pages, page_bytes, dtype)}; backed CBs have pages = None)."""
    W = int(input_tensor.shape[-1])
    x_dtype = input_tensor.dtype
    elem = _datum_bytes(x_dtype)
    T = ttnn.tile_size(x_dtype)
    sh = int(input_tensor.memory_config().shard_spec.shape[0])
    pitch = input_tensor.buffer_aligned_page_size()  # stick pitch inside the shard
    nt = pd._div_up(W, 1024)
    t_acc = pd.acc_dtype(fp32_dest)
    staged = FLAT_MODE == "staged"
    if staged:
        n_body, tail_bytes = nt, 0
        page, stick_pages, tile_pages = T, nt, 1
    else:
        n_body = W // 1024
        tail_bytes = (W - 1024 * n_body) * elem
        page = math.gcd(pitch, T)
        stick_pages, tile_pages = pitch // page, T // page
    has_tail_cb = tail_bytes != 0
    cbs = {}
    if staged:
        cbs[pd.CB_X] = (sh * nt, T, x_dtype)
        cbs[pd.CB_OUT] = (sh * nt, T, x_dtype)
    else:
        cbs[pd.CB_X] = (None, page, x_dtype)  # backed on the input shard
        cbs[pd.CB_OUT] = (None, page, x_dtype)  # backed on the output shard
        if has_tail_cb:
            cbs[pd.CB_X_RM] = (sh, T, x_dtype)
            cbs[pd.CB_OUT_RM] = (sh, T, x_dtype)
    cbs[pd.CB_SUMSQ] = (sh, ttnn.tile_size(t_acc), t_acc)
    cbs[pd.CB_INV_RMS] = (sh, ttnn.tile_size(t_acc), t_acc)
    ones_dtype = t_acc if FLAT_ONES_ACC else ttnn.bfloat16
    cbs[pd.CB_ONES] = (1, ttnn.tile_size(ones_dtype), ones_dtype)
    if gamma is not None:
        cbs[pd.CB_GAMMA] = (nt, ttnn.tile_size(gamma.dtype), gamma.dtype)
    return dict(
        W=W,
        elem=elem,
        T=T,
        sh=sh,
        pitch=pitch,
        nt=nt,
        staged=staged,
        n_body=n_body,
        has_tail_cb=has_tail_cb,
        tail_bytes=tail_bytes,
        page=page,
        stick_pages=stick_pages,
        tile_pages=tile_pages,
        cbs=cbs,
        footprint=sum(n * b for (n, b, _) in cbs.values() if n is not None),
    )


def flat_fits(plan, budget: int) -> bool:
    return plan["footprint"] <= budget and plan["sh"] * plan["stick_pages"] < FLAT_MAX_CB_PAGES


def flat_preferred(plan) -> bool:
    """Cost model (see FLAT_* / TILIZED_*): the flat path is estimated no slower than the tilized one."""
    S, W, nt = plan["sh"], plan["W"], plan["nt"]
    tail = 1 if W % 1024 else 0
    flat = FLAT_FIXED_NS + S * (FLAT_STICK_NS + FLAT_TILE_NS * nt + FLAT_TAIL_NS * tail)
    tilized = pd._div_up(S, pd.TILE) * (TILIZED_ROW_NS + TILIZED_WTILE_NS * pd._div_up(W, pd.TILE))
    return flat <= tilized


def create_program_descriptor(input_tensor, gamma, output_tensor, *, epsilon, compute_kernel_config):
    """Drop-in for rms_norm_program_descriptor.create_program_descriptor with the flat HEIGHT RM regime.

    Flat when expressible + fits L1 + (cost model prefers it). Otherwise the tilized path; if THAT does not fit L1
    (e.g. fp32 [32, 4096] shards: the tilized x row + staging + untilize buffers beside two resident shards), the
    flat path is used whenever it fits (its only per-stick L1 is accumulator tiles + a tail tile).
    """
    kw = dict(epsilon=epsilon, compute_kernel_config=compute_kernel_config)
    flat_ok = False
    if flat_expressible(input_tensor, gamma):
        plan = flat_plan(input_tensor, gamma, bool(compute_kernel_config.fp32_dest_acc_en))
        flat_ok = flat_fits(plan, pd.cb_l1_budget(input_tensor.device()))
        if flat_ok and (FLAT_FORCE or flat_preferred(plan)):
            return create_flat_program_descriptor(input_tensor, gamma, output_tensor, **kw)
    try:
        return pd.create_program_descriptor(input_tensor, gamma, output_tensor, **kw)
    except RuntimeError:
        if not flat_ok:
            raise
        return create_flat_program_descriptor(input_tensor, gamma, output_tensor, **kw)


def _backed_cb(index, tensor, dtype, page, core_ranges):
    """CB aliasing a sharded tensor's resident shard (zero-copy) with an explicit page size."""
    cb = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
    cb.core_ranges = core_ranges
    cb.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)]
    return cb


def create_flat_program_descriptor(input_tensor, gamma, output_tensor, *, epsilon, compute_kernel_config):
    device = input_tensor.device()
    shape = [int(d) for d in input_tensor.shape]
    W = shape[-1]
    rows = math.prod(shape[:-1])
    x_dtype = input_tensor.dtype
    elem = _datum_bytes(x_dtype)
    assert ttnn.tile_size(x_dtype) == 1024 * elem
    has_gamma = gamma is not None
    fp32_dest = bool(compute_kernel_config.fp32_dest_acc_en)

    mc = input_tensor.memory_config()
    ss = mc.shard_spec
    sh = int(ss.shape[0])
    row_major = ss.orientation == ttnn.ShardOrientation.ROW_MAJOR
    order = pd._ordered_shard_cores(ss.grid, row_major)
    n_shards = pd._div_up(rows, sh)
    assert len(order) >= n_shards
    cores = [ttnn.CoreCoord(x, y) for (x, y) in order[:n_shards]]
    sticks_of = [min(sh, rows - g * sh) for g in range(n_shards)]
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    plan = flat_plan(input_tensor, gamma, fp32_dest)
    pitch, nt, T = plan["pitch"], plan["nt"], plan["T"]
    assert output_tensor.buffer_aligned_page_size() == pitch
    staged, n_body, has_tail_cb, tail_bytes = plan["staged"], plan["n_body"], plan["has_tail_cb"], plan["tail_bytes"]
    stick_pages, tile_pages = plan["stick_pages"], plan["tile_pages"]
    stick_bytes = W * elem

    # ---- CBs (flat_plan) ----
    cbs = []
    for idx, (pages, page_bytes, dtype) in plan["cbs"].items():
        if pages is None:  # zero-copy: backed on the input / output shard
            cbs.append(
                _backed_cb(idx, input_tensor if idx == pd.CB_X else output_tensor, dtype, page_bytes, core_ranges)
            )
        else:
            cbs.append(pd._cb(idx, pages, page_bytes, dtype, core_ranges))
    g_read_bytes, g_page = 0, 0
    if has_gamma:
        g_page = gamma.buffer_aligned_page_size()
        g_read_bytes = g_page  # one read of the whole (aligned) page; <= nt flat gamma tiles
        assert g_read_bytes <= nt * ttnn.tile_size(gamma.dtype), (g_read_bytes, nt)

    defines = []
    if has_gamma:
        defines.append(("RMS_HAS_GAMMA", "1"))
    if staged:
        defines.append(("RMS_FLAT_STAGED", "1"))
    defines.extend((name, "1") for name in pd.PERF_ABLATE)
    defines.extend((name, "1") for name in FLAT_EXTRA_DEFINES)
    gamma_mcast = has_gamma and pd.GAMMA_MCAST_SHARDED_K1
    if gamma_mcast:
        defines.append(("RMS_GAMMA_MCAST", "1"))

    reader_ct = [n_body, 1 if has_tail_cb else 0, tail_bytes, pitch, T, stick_pages, stick_bytes]
    compute_ct = [n_body, 1 if has_tail_cb else 0, stick_pages, tile_pages]
    writer_ct = [
        n_body,
        1 if has_tail_cb else 0,
        tail_bytes,
        pitch,
        T,
        stick_pages,
        stick_bytes,
        g_read_bytes,
        nt,
        g_page,
    ]
    writer_ct.extend(
        (ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )

    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0
    inv_w_bits = pd._f32_bits(1.0 / W)
    eps_bits = pd._f32_bits(epsilon)
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, n in zip(cores, sticks_of):
        reader_rt[c.x][c.y] = [x_addr, n]
        compute_rt[c.x][c.y] = [n, inv_w_bits, eps_bits]
        writer_rt[c.x][c.y] = [out_addr, n, g_addr]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(FLAT_KERNEL_DIR / "rms_norm_flat_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        defines=defines,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(FLAT_KERNEL_DIR / "rms_norm_flat_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        defines=defines,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(FLAT_KERNEL_DIR / "rms_norm_flat_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        defines=defines,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=cbs)
    if gamma_mcast:
        # One group of every Tensix core; sender = core 0 (reads the gamma page once), writer NoC. One-shot.
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=False),
            core_ranges,
            len(cores),
            ttnn.McastFixedSenderConfig(sender_index=0),
            ttnn.McastCoreOrder.RowMajor,
        )
        mcast.attach(descriptor, "gamma_mcast", [writer], pd.SEM_GAMMA_MCAST_FIRST)
    descriptor.kernels = [reader, writer, compute]
    return descriptor
