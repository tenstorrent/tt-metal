# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off for lever B13 — stateful NoC transfers in reader AND writer.

NOT part of the rms_norm op.  This is a self-contained DRAM->CB->DRAM page copy
that reproduces exactly ONE thing from the op: the reader's / writer's
transaction ISSUE loop (N whole-page reads/writes of an interleaved DRAM tensor,
ONE barrier per chunk).  No compute, no gamma, no two-pass reduce.  Everything
else is held trivial so the measured delta is attributable to the issue loop.

WHY the isolation is faithful to the op's `read_tiles` / `write_tiles`:
  * same transaction shape  — one whole tile PAGE per transaction (lever B5/B6),
  * same barrier shape      — one `noc_async_read_barrier()` per chunk (lever B7),
  * same address source     — `TensorAccessor` over an interleaved DRAM tensor,
  * same per-core work unit — a contiguous run of page ids, chunked.

VARIANT MENU (reader `rvar` / writer `wvar`):
  0 baseline    `noc_async_read_page` / `noc_async_write_page`  == the op today
  1 one_packet  same address math, `..._one_packet` (drops the any-len dispatch)
  2 affine      + bank-table address math instead of TensorAccessor::get_noc_addr
  3 bank_state  + cmd-buffer state reuse, chunk walked BANK-MAJOR
  4 bank_trid   reader only: bank_state issued with a transaction id (lever B8's
                per-transaction cost, barriered on its own chunk)

Correctness is the pass/fail: the output tensor must equal the input BIT-EXACTLY
(it is a copy), so a wrong bank / wrong address is caught, not measured.

Run:
    scripts/run_safe_pytest.sh --profile \\
        ttnn/ttnn/operations/rms_norm/perf_experiments/stateful_noc/test_stateful_noc.py -s
    SN_SET=focus|safe|gate|final|sweep|domain|dtype|all

MEASURED (blackhole p150b, 1350 MHz, DEVICE KERNEL DURATION [ns], median of 3;
`rd_only` / `wr_only` = the other half's payload stubbed so the measured half is
alone on the core's NoC):

  focus (224 x 2048 B tile pages, 1 core, chunk 112)
    rd_only   baseline 8028/8090   one_packet 6981/6980   affine 7040/6994
              bank_state 7024   bank_trid 7116
    wr_only   baseline 8606/8609   one_packet 7553/7578   affine 7290/7293
              bank_state 7809
    copy      baseline 12462   one_packet 10817   affine 10655   bank_state 11601

  bf8b (224 x 1088 B pages, 1 core)   rd 7816 -> op 6781, aff 6747, bank 5002
                                      wr 8419 -> op 7370, aff 7085, bank 4709
  fp32 (224 x 4096 B pages, 1 core)   rd 12143 -> op 12068, aff 12165, bank 12550
                                      wr 12008 -> op 11947, aff 11956, bank 17639
  dram_bound (8192 pages / 64 cores)  rd 42420 -> op 42464, aff 42381, bank 49607
                                      wr 57215 -> op 56789, aff 57128, bank 62335
  wide (57344 pages / 64 cores)       rd 295887 -> op 298939, aff 300136, bank 405266
                                      wr 385012 -> op 383149, aff 383096, bank 459879
  rm (8192 x 2048 B sticks / 64)      rd 42424 -> op 42516, aff 42351, bank 50221
  rm_odd (2000 B partial pages / 32)  rd 6055 -> aff 6181, bank 6181
  tiny (1 page / 1 core)              rd 724 -> op 701, aff 790, bank 741
  tiny4 (4 pages / 1 core)            rd 835 -> op 770, aff 873, bank 807

MECHANISM (what the numbers say):
  * The issue loop is NOT the binding constraint for 2 KB+ pages.  `bank_state`
    halves ISSUE occupancy (30.7 -> 16.8 ns/txn on the reader) but the freed time
    reappears in the barrier: total read time is pinned at ~29 ns per 2 KB page
    per core (~70 GB/s), which is the per-core NoC/DRAM ceiling.  So the whole
    available win on the focus shape is ~13-15 %, and the ORDER-PRESERVING
    variants already collect all of it.
  * Bank-major order (the price of cmd-buffer state reuse) SERIALISES DRAM
    channels: a core issues 14-112 back-to-back requests to ONE bank instead of
    spreading them round-robin.  That is a measured regression wherever the
    transfer is bandwidth-bound (+17 % dram_bound, +37 % wide, +47 % fp32
    writes).  A per-core bank ROTATION (`bank_rot`) recovers only part of it
    (wide 405 -> 393 us), so the cause is intra-core bank concentration, not
    just cross-core phase alignment.
  * It INVERTS for small pages: at bfloat8_b's 1088 B the per-transaction rate
    is the limit, and bank-major wins 36 % (reads) / 44 % (writes).
  * Transaction ids (lever B8) cost ~+5.5 ns/txn over plain state reuse (the
    NIU_MST_REQS_OUTSTANDING_ID poll in
    ncrisc_noc_fast_read_with_transaction_id), which is more than the barrier
    they could hide on this shape (3.1 ns/txn in the op).
  * `one_packet` needs a COMPILE-TIME size guard: the API is only valid for
    size <= NOC_MAX_BURST_SIZE (16 KB on blackhole, 8 KB on wormhole).  Every
    TILE page qualifies (1088 / 2048 / 4096 B); a ROW_MAJOR stick chunk can
    exceed it at large W, so that path needs the guarded dual arm.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import ttnn

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
KERNEL_DIR = Path(__file__).parent / "kernels"
MANIFEST_PATH = Path("generated/stateful_noc_manifest.json")

N_WARMUP = 2
N_ITERS = 3

TILE_DIM = 32

RVAR_NAMES = {0: "baseline", 1: "one_packet", 2: "affine", 3: "bank_state", 4: "bank_trid", 5: "bank_rot"}
WVAR_NAMES = {0: "baseline", 1: "one_packet", 2: "affine", 3: "bank_state", 4: "bank_rot"}


# --- regimes ----------------------------------------------------------------
# (mode, shape, dtype, cores, chunk)  — mode 0 = TILE pages, 1 = ROW_MAJOR sticks.
# `tiles_per_core` is derived: total_pages / cores, and must be a multiple of chunk.
REGIMES = {
    # The focus case's per-core geometry: ONE core, Wt_core = 224 tile pages,
    # W-chunk 112 (rms_norm (1,1,32,7168) bf16, Regime B, BLOCK_HT = 1).
    "focus": dict(mode=0, shape=(1, 1, 32, 7168), dtype=ttnn.bfloat16, cores=1, chunk=112),
    # Regime A / full grid / DRAM-bandwidth-bound: 8192 tile pages over 64 cores.
    "dram_bound": dict(mode=0, shape=(1, 1, 8192, 1024), dtype=ttnn.bfloat16, cores=64, chunk=32),
    # The widest prefill: 57344 tile pages over 64 cores.
    "wide": dict(mode=0, shape=(1, 1, 8192, 7168), dtype=ttnn.bfloat16, cores=64, chunk=112),
    # B0 regime: ONE page on ONE core — where fixed set_state cost has nothing
    # to amortise over and the lever should REGRESS if it regresses anywhere.
    "tiny": dict(mode=0, shape=(1, 1, 32, 32), dtype=ttnn.bfloat16, cores=1, chunk=1),
    # Same as tiny but 4 pages, the (32,17)-ish end of the ladder.
    "tiny4": dict(mode=0, shape=(1, 1, 32, 128), dtype=ttnn.bfloat16, cores=1, chunk=4),
    # Byte-size sweep: the one-packet fast path is size-dependent
    # (NOC_MAX_BURST_SIZE), and so is the DRAM transfer itself.
    "fp32": dict(mode=0, shape=(1, 1, 32, 7168), dtype=ttnn.float32, cores=1, chunk=112),
    "bf8b": dict(mode=0, shape=(1, 1, 32, 7168), dtype=ttnn.bfloat8_b, cores=1, chunk=112),
    # ROW_MAJOR stick path: page = one row, transaction = row_bytes (< page for
    # the unaligned case).  Proves whether the mechanism is expressible when the
    # transfer is a PARTIAL page rather than a whole one.
    "rm": dict(mode=1, shape=(1, 1, 8192, 1024), dtype=ttnn.bfloat16, cores=64, chunk=32),
    "rm_odd": dict(mode=1, shape=(1, 1, 1024, 1000), dtype=ttnn.bfloat16, cores=32, chunk=8),
}


def _dtype_tile_bytes(dtype):
    return ttnn.tile_size(dtype)


def _geometry(reg):
    """Page count / transaction bytes / CB page bytes for one regime."""
    shape = reg["shape"]
    dtype = reg["dtype"]
    if reg["mode"] == 0:
        pages = (shape[-2] // TILE_DIM) * (shape[-1] // TILE_DIM)
        for d in shape[:-2]:
            pages *= d
        tb = _dtype_tile_bytes(dtype)
        return dict(pages=pages, read_bytes=tb, cb_page=tb, aps=tb, byte_off=0)
    # ROW_MAJOR: one page per row of the last dim.
    elem = _dtype_tile_bytes(dtype) // (TILE_DIM * TILE_DIM)
    row_bytes = shape[-1] * elem
    aps = (row_bytes + 63) // 64 * 64  # DRAM page alignment on this arch
    pages = 1
    for d in shape[:-1]:
        pages *= d
    # The CB page must be 16 B-aligned for the NoC destination; keep it at the
    # aligned page size so the writer's source addresses stay legal too.
    return dict(pages=pages, read_bytes=row_bytes, cb_page=aps, aps=aps, byte_off=0)


def make_tensors(device, reg):
    import torch

    torch.manual_seed(0)
    shape = reg["shape"]
    dtype = reg["dtype"]
    layout = ttnn.TILE_LAYOUT if reg["mode"] == 0 else ttnn.ROW_MAJOR_LAYOUT
    t = torch.randn(shape, dtype=torch.float32)
    x = ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)
    return x


def build_program(x, out, reg, rvar, wvar, rskip=0, wskip=0):
    device = x.device()
    geo = _geometry(reg)
    chunk = reg["chunk"]
    cores = reg["cores"]
    pages = geo["pages"]

    assert pages % cores == 0, f"{pages} pages over {cores} cores"
    per_core = pages // cores
    assert per_core % chunk == 0, f"{per_core} pages/core not a multiple of chunk {chunk}"
    num_chunks = per_core // chunk

    grid = device.compute_with_storage_grid_size()
    core_list = ttnn.grid_to_cores(cores, grid.x, grid.y, True)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_list])

    # CB: two chunks deep so the reader and the writer pipeline (the op's own
    # double-buffered shape).  A 2-chunk CB keeps the write pointer chunk-aligned,
    # so the multi-page contiguous access can never run off the end.
    cb = ttnn.CBDescriptor(
        total_size=2 * chunk * geo["cb_page"],
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=reg["dtype"], page_size=geo["cb_page"])
        ],
    )

    common = [reg["mode"], 0, geo["read_bytes"], geo["cb_page"], chunk, geo["byte_off"], 0]

    reader_ct = list(common)
    reader_ct[1] = rvar
    reader_ct[6] = rskip
    reader_ct.extend(ttnn.TensorAccessorArgs(x).get_compile_time_args())

    writer_ct = list(common)
    writer_ct[1] = wvar
    writer_ct[6] = wskip
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    in_addr = x.buffer_address()
    out_addr = out.buffer_address()
    for i, c in enumerate(core_list):
        reader_rt[c.x][c.y] = [in_addr, i * per_core, num_chunks, i]
        writer_rt[c.x][c.y] = [out_addr, i * per_core, num_chunks, i]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "sn_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "sn_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=[cb])


# --- Ledger-facing counterfactual arms --------------------------------------
# Written in the same `levers=dict(<knob>=<int>)` idiom the op's own bench uses,
# so `eval.verify_levers` can see that the B8 / B13 verdicts are RE-RUNNABLE
# rather than one-off kernel edits.  `sn_reader_variant` / `sn_writer_variant`
# are this bench's variant selectors (the `rvar` / `wvar` compile-time args);
# the integers index RVAR_NAMES / WVAR_NAMES, and the VARIANT MENU at the top of
# this file documents what each one does.
LEVER_ARMS = {
    # The baseline both verdicts are measured against: the op's pre-Perf-1 form
    # (noc_async_read_tile / noc_async_write_tile, i.e. the any-length dispatch).
    "baseline": dict(levers=dict(sn_reader_variant=0, sn_writer_variant=0)),
    # B13 - the ACTUAL set_state/with_state mechanism.  State reuse republishes
    # only TARG/RET_ADDR_LO, so it REQUIRES bank-major issue order, which
    # serialises DRAM channels: measured a regression outside small pages
    # (+17% on (1,1,8192,1024), +37% on the widest shape, +47% on fp32 writes),
    # and an inversion at bfloat8_b's 1088 B tiles (-36% / -44%).
    "B13": dict(levers=dict(sn_reader_variant=3, sn_writer_variant=3)),
    # B13's ORDER-PRESERVING half, which is what graduated into the op: pass a
    # compile-time size bound so the runtime any-length loop disappears.
    "B13_applied": dict(levers=dict(sn_reader_variant=1, sn_writer_variant=1)),
    # B8 - trid double-issue, on top of B13's state reuse.  Measured null: the
    # +5.5 ns/txn surcharge exceeds the barrier it could hide.
    "B8": dict(levers=dict(sn_reader_variant=4, sn_writer_variant=3)),
}


def dispatch_lever_arm(device, manifest, lever, reg_name, **kw):
    """Dispatch one LEVER_ARMS entry by name — the re-runnable form of a verdict."""
    arm = LEVER_ARMS[lever]["levers"]
    return dispatch_arm(
        device,
        manifest,
        f"{reg_name}/{lever}",
        reg_name,
        arm["sn_reader_variant"],
        arm["sn_writer_variant"],
        **kw,
    )


def run_copy(device, x, reg, rvar, wvar, rskip=0, wskip=0):
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)), x.dtype, x.layout, device, x.memory_config())
    pd = build_program(x, out, reg, rvar, wvar, rskip, wskip)
    return ttnn.generic_op([x, out], pd)


def check(x, y):
    """Bit-exact equality — the bench is a pure copy, so anything else is a bug."""
    import torch

    a = ttnn.to_torch(x).to(torch.float32)
    b = ttnn.to_torch(y).to(torch.float32)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    bad = int((a != b).sum())
    return bad == 0, f"{bad} of {a.numel()} elements differ"


def dispatch_arm(device, manifest, label, reg_name, rvar, wvar, iters=N_ITERS, rskip=0, wskip=0):
    """One arm = a correctness dispatch (when nothing is stubbed) + N timed ones.

    With `rskip` / `wskip` the corresponding half issues no NoC transfer, so the
    output is garbage BY CONSTRUCTION and correctness is not checkable here — it
    is established by the same variant's un-stubbed copy arm.  That is the whole
    point of the stub: it prices ONE half's issue loop without the other half's
    traffic competing for the same core's NoC bandwidth.
    """
    reg = REGIMES[reg_name]
    x = make_tensors(device, reg)
    geo = _geometry(reg)
    per_core = geo["pages"] // reg["cores"]

    stubbed = bool(rskip or wskip)
    n_extra = 0
    ok, detail = True, "not checked (payload stubbed)"
    if not stubbed:
        y = run_copy(device, x, reg, rvar, wvar)
        ttnn.synchronize_device(device)
        ok, detail = check(x, y)
        ttnn.deallocate(y)
        n_extra = 1

    n = 0
    for _ in range(N_WARMUP + iters):
        y = run_copy(device, x, reg, rvar, wvar, rskip, wskip)
        ttnn.deallocate(y)
        n += 1
    ttnn.synchronize_device(device)

    manifest.append(
        {
            "label": label,
            "regime": reg_name,
            "rvar": RVAR_NAMES[rvar],
            "wvar": WVAR_NAMES[wvar],
            "rskip": rskip,
            "wskip": wskip,
            "pages_per_core": per_core,
            "cores": reg["cores"],
            "correct": bool(ok),
            "detail": detail,
            # the correctness dispatch is profiled too — skip it plus the warm-ups
            "calls": n + n_extra,
            "profiled": iters,
        }
    )
    return ok, detail


def write_manifest(manifest, path=MANIFEST_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def report_from_csv(csv_path, manifest_path=MANIFEST_PATH):
    manifest = json.loads(Path(manifest_path).read_text())
    with open(csv_path) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("OP CODE") == "GenericOpDeviceOperation"]
    out, i = {}, 0
    for arm in manifest:
        i += arm["calls"] - arm["profiled"]
        window = rows[i : i + arm["profiled"]]
        i += arm["profiled"]
        vals = sorted(float(r[_DURATION_KEY]) for r in window if r.get(_DURATION_KEY))
        out[arm["label"]] = {
            "ns": vals[len(vals) // 2] if vals else None,
            "all": vals,
            "regime": arm["regime"],
            "rvar": arm["rvar"],
            "wvar": arm["wvar"],
            "correct": arm["correct"],
            "detail": arm["detail"],
        }
    return out


def print_report(csv_path, manifest_path=MANIFEST_PATH):
    rep = report_from_csv(csv_path, manifest_path)
    print(f"{'label':<44} {'ns':>10} {'ok':>5}  detail")
    for k, v in rep.items():
        ns = "-" if v["ns"] is None else f"{v['ns']:.0f}"
        print(f"{k:<44} {ns:>10} {'OK' if v['correct'] else 'FAIL':>5}  {v['detail']}")
    return rep


# --- per-zone attribution ----------------------------------------------------
# The wall of a reader+writer copy is a COMPOSITE (and on one core it saturates
# that core's NoC in both directions), so the number that prices the issue loop
# is the `rd_issue` / `wr_issue` zone, per transaction, per core.
DEVICE_LOG = "generated/profiler/.logs/profile_log_device.csv"


def zone_report(log_path=DEVICE_LOG, manifest_path=MANIFEST_PATH):
    """Fold ZONE markers into per-arm, per-zone ns (averaged over cores)."""
    import csv as _csv
    from collections import defaultdict

    manifest = json.loads(Path(manifest_path).read_text())
    with open(log_path) as fh:
        header = fh.readline()
        mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(_csv.DictReader(fh, skipinitialspace=True))
    runs = sorted({int(r["run host ID"]) for r in rows if r["run host ID"]})
    by_run = defaultdict(list)
    for r in rows:
        if r["run host ID"]:
            by_run[int(r["run host ID"])].append(r)

    def fold(run):
        stack, tot, cores = defaultdict(list), defaultdict(float), set()
        for r in by_run[run]:
            key = (r["core_x"], r["core_y"], r["RISC processor type"])
            cores.add((r["core_x"], r["core_y"]))
            cyc = int(r["time[cycles since reset]"])
            if r["type"] == "ZONE_START":
                stack[(key, r["zone name"])].append(cyc)
            else:
                st = stack[(key, r["zone name"])]
                if st:
                    tot[r["zone name"]] += (cyc - st.pop()) / mhz * 1000.0
        n = max(1, len(cores))
        return {k: v / n for k, v in tot.items() if not (k.endswith("-FW") or k.endswith("-KERNEL"))}

    out, i = {}, 0
    for arm in manifest:
        i += arm["calls"] - arm["profiled"]
        window = runs[i : i + arm["profiled"]]
        i += arm["profiled"]
        per = [fold(r) for r in window]
        med = {}
        for k in set().union(*[set(p) for p in per]) if per else []:
            vals = sorted(p.get(k, 0.0) for p in per)
            med[k] = vals[len(vals) // 2]
        out[arm["label"]] = {"zones": med, "pages": arm["pages_per_core"], "arm": arm}
    return out


def print_zone_report(log_path=DEVICE_LOG, manifest_path=MANIFEST_PATH):
    rep = zone_report(log_path, manifest_path)
    keys = ("rd_issue", "rd_barrier", "rd_reserve", "wr_issue", "wr_barrier", "wr_wait")
    print(f"{'label':<40} {'pages':>6} " + " ".join(f"{k:>11}" for k in keys) + "   ns/txn rd  wr")
    for label, v in rep.items():
        z, pg = v["zones"], v["pages"]
        cells = " ".join(f"{z.get(k, 0.0):>11.0f}" for k in keys)
        rd = z.get("rd_issue", 0.0) / pg
        wr = z.get("wr_issue", 0.0) / pg
        print(f"{label:<40} {pg:>6} {cells}   {rd:>8.2f} {wr:>6.2f}")
    return rep
