# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""dram_dynamic_balance: descriptor-side variants of groupnorm_sc_N_1_HW_C_b (DRAM interleaved only; sharded plans
pass through untouched).

install(spec) patches the op's program-descriptor module (plan_or_raise, create_program_descriptor) and returns a
restore callable. spec = '+'-joined parts:
  static:<src>   oracle static rebalance: rows per core proportional to the core's measured rate (rows / time)
                 from baseline_core_times.json[<case>] (<src> = p1 | p12: pass-1 time, or pass 1 + pass 2)
  dyn            dynamic chunk tickets (kernels/ built with GN_DYN): see create_program_descriptor below
  slack:<f>      dyn: resident-phase chunks N_B = floor(f * cpi * Kc) when the unit does not fit (default 1)
  kcap:<n>       dyn: cap the per-core residency at n chunks (default: the L1 maximum)
  kplus:<n>      dyn: cap the per-core residency at ceil(Nu / cpi) + n chunks (bounded stealing)
  split[:<f>]    pass-2 output written by both DM RISCs, each on its own NoC (kernels built with GN_SPLIT); f = the
                 fraction of every core's chunks on NOC1 (default 0.5); da:<n> / db:<n> output CB depths (chunks)
  split:d        dynamic split (GN_SPLITD): the packer hands each chunk to whichever consumer's CB has room
  tau:<c>        split:d: the reader takes a chunk only after the writer's current one has been out c cycles per tile
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import ttnn
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd

HERE = Path(__file__).parent
KDIR = HERE / "kernels"
CORE_TIMES = HERE / "baseline_core_times.json"

CFG = {"case": None}

# dyn CBs / semaphores (CB slots unused by the op)
CB_META = 25  # reader -> compute: one 16 B page per chunk (rows_valid | flags)
CB_IDS = 26  # reader -> writer: one 16 B page per pass-2 chunk (chunk id), END-terminated
META_PAGE = 16


def _parse(spec):
    cfg = {}
    for part in spec.split("+"):
        if not part:
            continue
        k, _, v = part.partition(":")
        cfg[k] = v if v else True
    return cfg


# ---------------------------------------------------------------------------------------------------------------
# Plan post-processing
# ---------------------------------------------------------------------------------------------------------------


def _residency(plan, B_max):
    """K_alloc / stream for a (new) per-core chunk maximum B_max, solved like the op does."""
    W, T, R, h = plan.W, plan.T, plan.R, plan.h
    kw = dict(W=W, T=T, R=R, is_rm=plan.is_rm, has_gamma=plan.has_gamma, has_beta=plan.has_beta)
    chunk = pd.BF16_TILE_BYTES * T
    f_fixed = pd.footprint_bytes(K_alloc=0, stream=True, **kw)
    f_res = pd.footprint_bytes(K_alloc=0, stream=False, **kw)
    K_stream = (plan.l1_budget - f_fixed) // chunk
    K_full = (plan.l1_budget - f_res) // chunk
    return K_stream, K_full, kw


def _static_rebalance(plan, src):
    times = json.loads(CORE_TIMES.read_text())[CFG["case"]]
    by_unit = {}
    for wk in plan.work:
        by_unit.setdefault(wk.unit_start, []).append(wk)
    for unit, cores in by_unit.items():
        cores.sort(key=lambda w: w.slot)
        rates = []
        for wk in cores:
            t = times.get(f"{wk.core.x},{wk.core.y}", {})
            tt = t.get("p1", 0) + (t.get("p2", 0) if src == "p12" else 0)
            rates.append(wk.rows_core / tt if tt > 0 else 1.0)
        tot = sum(rates)
        want = [plan.Ht * r / tot for r in rates]
        rows = [max(1, int(math.floor(w))) for w in want]
        rem = plan.Ht - sum(rows)
        order = sorted(range(len(cores)), key=lambda i: want[i] - rows[i], reverse=True)
        i = 0
        while rem != 0:
            j = order[i % len(order)]
            if rem > 0:
                rows[j] += 1
                rem -= 1
            elif rows[j] > 1:
                rows[j] -= 1
                rem += 1
            i += 1
        start = 0
        for wk, r in zip(cores, rows):
            wk.row_start, wk.rows_core = start, r
            start += r
    B_max = max(pd._ceil_div(wk.rows_core, plan.h) for wk in plan.work)
    K_stream, K_full, kw = _residency(plan, B_max)
    if K_full >= B_max:
        plan.K_alloc, plan.stream = B_max, False
    else:
        plan.K_alloc, plan.stream = min(B_max, K_stream), True
    plan.B_max = B_max
    plan.rows_core_max = max(wk.rows_core for wk in plan.work)
    plan.footprint = pd.footprint_bytes(K_alloc=plan.K_alloc, stream=plan.stream, **kw)


def _dyn_plan(plan, cfg):
    """Dynamic chunk tickets: the unit's Nu = ceil(Ht / h) chunks are handed out from counters on the unit root.
    Phase A (stream: dropped after pass 1, re-read in pass 2 from a second counter) then phase B (resident, at most
    Kc per core). Fits (Nu <= cpi * K_full): everything is phase B, the ring holds no streaming slots."""
    Nu = pd._ceil_div(plan.Ht, plan.h)
    K_stream, K_full, kw = _residency(plan, None)
    kcap = int(cfg["kcap"]) if "kcap" in cfg else None
    if "kplus" in cfg:  # bounded stealing: at most ceil(Nu / cpi) + n chunks per core
        kcap = pd._ceil_div(Nu, plan.cpi) + int(cfg["kplus"])
    if Nu <= plan.cpi * (K_full if kcap is None else min(K_full, kcap)):
        Kc = K_full if kcap is None else min(K_full, kcap)
        Kc = min(Kc, Nu)
        N_B, stream = Nu, False
    else:
        Kc = K_stream if kcap is None else min(K_stream, kcap)
        f = float(cfg.get("slack", 1.0))
        N_B = min(Nu, int(math.floor(f * plan.cpi * Kc)))
        stream = True
    plan.dyn = dict(Nu=Nu, N_A=Nu - N_B, N_B=N_B, Kc=Kc)
    plan.K_alloc, plan.stream = Kc, stream
    plan.footprint = pd.footprint_bytes(K_alloc=Kc, stream=stream, **kw) + 2 * (Kc + 8) * META_PAGE
    assert plan.footprint <= plan.l1_budget, (plan.footprint, plan.l1_budget)


# ---------------------------------------------------------------------------------------------------------------
# Descriptor (dyn): the op's create_program_descriptor + the dyn CBs, semaphores and args
# ---------------------------------------------------------------------------------------------------------------


CB_OUT_B = 27  # split: the reader's share of the pass-2 output (written on the reader's NoC)


def _split_mask(frac_b):
    """8-bit pattern with round(8 * frac_b) evenly spread bits: chunk q is the reader's iff bit q % 8 is set."""
    nb = max(0, min(8, int(round(8 * frac_b))))
    return sum(1 << i for i in range(8) if (i * nb) // 8 != ((i + 1) * nb) // 8)


def _split_m(plan, sw):
    """Per-core split mask: plan.split['f'] = the fraction of each core's pass-2 chunks written on NOC1. The reader is
    on NOC1 iff the core is swapped (odd DRAM column, not a root)."""
    if getattr(plan, "split", None) is None:
        return []
    f = plan.split["f"]
    return [_split_mask(f if sw else 1.0 - f)]


def _dyn_rt(plan):
    d = getattr(plan, "dyn", None)
    return [] if d is None else [d["N_A"], d["N_B"], d["Kc"], d["Nu"], plan.cpi, DYN_SEM0]


def _split_rt(plan, sw, output_tensor):
    if getattr(plan, "split", None) is None:
        return []
    return [output_tensor.buffer_address(), *_split_m(plan, sw)]


def _out_cbs(plan, T, cores):
    sp = getattr(plan, "split", None)
    if sp is None:
        return [pd._cb(pd.output_ring_bytes(T), cores, [(pd.CB_OUTPUT_TILES, ttnn.bfloat16, pd.BF16_TILE_BYTES)])]
    return [
        pd._cb(sp["da"] * T * pd.BF16_TILE_BYTES, cores, [(pd.CB_OUTPUT_TILES, ttnn.bfloat16, pd.BF16_TILE_BYTES)]),
        pd._cb(sp["db"] * T * pd.BF16_TILE_BYTES, cores, [(CB_OUT_B, ttnn.bfloat16, pd.BF16_TILE_BYTES)]),
    ]


CB_SEQ_A, CB_SEQ_B = 28, 29  # dynamic split: pass-2 indices of the chunks packed for the writer / the reader
SEQ_PAGES = 8


def _extra_cbs(plan, cores):
    out = []
    d = getattr(plan, "dyn", None)
    if d is not None:
        pg = (d["Kc"] + 8) * META_PAGE
        out += [
            pd._cb(pg, cores, [(CB_META, ttnn.uint32, META_PAGE)]),
            pd._cb(pg, cores, [(CB_IDS, ttnn.uint32, META_PAGE)]),
        ]
    sp = getattr(plan, "split", None)
    if sp is not None and sp["dynamic"]:
        out += [
            pd._cb(SEQ_PAGES * META_PAGE, cores, [(CB_SEQ_A, ttnn.uint32, META_PAGE)]),
            pd._cb(SEQ_PAGES * META_PAGE, cores, [(CB_SEQ_B, ttnn.uint32, META_PAGE)]),
        ]
    return out


def _extra_sems(plan, cores):
    if getattr(plan, "dyn", None) is None:
        return []
    return [ttnn.SemaphoreDescriptor(id=DYN_SEM0 + i, core_ranges=cores, initial_value=0) for i in range(4)]


def _defines(plan):
    out = []
    if getattr(plan, "dyn", None) is not None:
        out.append(("GN_DYN", "1"))
    if getattr(plan, "split", None) is not None:
        out.append(("GN_SPLIT", "1"))
        if plan.split["dynamic"]:
            out.append(("GN_SPLITD", "1"))
            if plan.split.get("tau"):
                out.append(("GN_SPLITD_TAU", str(plan.split["tau"])))
    return out


def _build_create():
    """The op's create_program_descriptor (its source, read from the op's file so the copy tracks the op), patched:
    kernels from KDIR; GN_DYN: meta / ids CBs, counter semaphores, 6 reader runtime args; GN_SPLIT: the output ring
    split into the writer's and the reader's share, the reader's output accessor (CT) + [out_addr, mask] (RT), the
    mask on compute / writer."""
    text = Path(pd.__file__).read_text()
    src = text[text.index("def create_program_descriptor(") :]
    reps = [
        ("def create_program_descriptor(", "def create_patched_program_descriptor("),
        (
            "            cbs.append(_cb(output_ring_bytes(T), cores, [(CB_OUTPUT_TILES, bf16, BF16_TILE_BYTES)]))\n",
            "            cbs.extend(X_OUT_CBS(plan, T, cores))\n",
        ),
        (
            "    cbs.append(_cb(stage, cores, [(CB_AFFINE_STAGING, bf16, stage)]))\n",
            "    cbs.append(_cb(stage, cores, [(CB_AFFINE_STAGING, bf16, stage)]))\n    cbs.extend(X_EXTRA_CBS(plan, cores))\n",
        ),
        (
            "    # Compute is built once per role too",
            "    if getattr(plan, 'split', None) is not None:\n"
            "        reader_ct.append(out_page_bytes)\n"
            "        reader_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())\n\n"
            "    # Compute is built once per role too",
        ),
        (
            "            root_virt.y,\n        ]\n        compute_rt[wk.is_root][x][y] = [wk.num_blocks, wk.rows_core, K_core, "
            "inv_n_bits, eps_bits]",
            "            root_virt.y,\n            *X_DYN_RT(plan),\n            *X_SPLIT_RT(plan, sw, output_tensor),\n"
            "        ]\n        compute_rt[wk.is_root][x][y] = [wk.num_blocks, wk.rows_core, K_core, inv_n_bits, eps_bits, "
            "*X_SPLIT_M(plan, sw)]",
        ),
        ("            plan.cpi,\n        ]", "            plan.cpi,\n            *X_SPLIT_M(plan, sw),\n        ]"),
        ("semaphores=[gather_sem]", "semaphores=[gather_sem, *X_EXTRA_SEMS(plan, cores)]"),
        (
            "    return descriptor",
            "    for _k in descriptor.kernels:\n        _k.defines = list(_k.defines) + X_DEFINES(plan)\n"
            "    return descriptor",
        ),
    ]
    for a, b in reps:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    ns = dict(vars(pd))
    ns.update(
        KERNEL_DIR=KDIR,
        X_OUT_CBS=_out_cbs,
        X_EXTRA_CBS=_extra_cbs,
        X_DYN_RT=_dyn_rt,
        X_SPLIT_RT=_split_rt,
        X_SPLIT_M=_split_m,
        X_EXTRA_SEMS=_extra_sems,
        X_DEFINES=_defines,
    )
    exec(compile(src, "<patched create_program_descriptor>", "exec"), ns)
    return ns["create_patched_program_descriptor"], src


# Counter semaphores: A, B, A2 on the unit root + the reader's private atomic-return slot (above the mcast channel's).
DYN_SEM0 = 12


def _split_plan(plan, cfg):
    """split[:f]: the pass-2 output split between the writer (its NoC) and the reader (the other NoC); f = the fraction
    of every core's chunks written on NOC1 (default 0.5). da / db: chunk depths of the two output CBs (default 1 + 1:
    the op's 2-chunk output ring, L1-neutral)."""
    dynamic = cfg["split"] == "d"
    f = 0.5 if (cfg["split"] is True or dynamic) else float(cfg["split"])
    chunk = plan.T * pd.BF16_TILE_BYTES
    if cfg.get("da") == "auto":
        # The writer keeps the op's double buffer when L1 has a spare chunk; the reader's slot is the extra one.
        # L1-bound plans (the ring took every spare chunk) split the op's 2-chunk output ring 1 + 1.
        spare = plan.l1_budget - plan.footprint - (2 * SEQ_PAGES * META_PAGE if dynamic else 0)
        da = 2 if spare >= chunk else 1
    else:
        da = int(cfg.get("da", 1))
    db = int(cfg.get("db", 1))
    plan.split = dict(f=f, da=da, db=db, dynamic=dynamic, tau=int(cfg.get("tau", 0)))
    plan.footprint += (da + db - pd.D_OUT) * plan.T * pd.BF16_TILE_BYTES + (2 * SEQ_PAGES * META_PAGE if dynamic else 0)
    assert plan.footprint <= plan.l1_budget, ("split", plan.footprint, plan.l1_budget)


# ---------------------------------------------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------------------------------------------


def install(spec):
    cfg = _parse(spec)
    orig_plan, orig_create = pd.plan_or_raise, pd.create_program_descriptor
    patched_create = _build_create()[0] if ("dyn" in cfg or "split" in cfg) else None

    def plan_or_raise(input_tensor, num_groups, **kw):
        plan = orig_plan(input_tensor, num_groups, **kw)
        if plan.sharded:
            return plan
        if "static" in cfg:
            _static_rebalance(plan, cfg["static"])
        if "dyn" in cfg:
            _dyn_plan(plan, cfg)
            print(
                f"DYN_PLAN case={CFG['case']} Ht={plan.Ht} h={plan.h} W={plan.W} cpi={plan.cpi} cores={len(plan.work)} "
                f"dyn={plan.dyn} stream={plan.stream} footprint={plan.footprint}/{plan.l1_budget}"
            )
        if "split" in cfg:
            _split_plan(plan, cfg)
            nb = max(wk.num_blocks for wk in plan.work)
            print(
                f"SPLIT_PLAN case={CFG['case']} split={plan.split} cpi={plan.cpi} blocks={nb} K={plan.K_alloc} "
                f"B={plan.B_max} footprint={plan.footprint}/{plan.l1_budget}"
            )
        return plan

    def create(plan, input_tensor, output_tensor, **kw):
        if getattr(plan, "dyn", None) is not None or getattr(plan, "split", None) is not None:
            return patched_create(plan, input_tensor, output_tensor, **kw)
        return orig_create(plan, input_tensor, output_tensor, **kw)

    pd.plan_or_raise, pd.create_program_descriptor = plan_or_raise, create

    def restore():
        pd.plan_or_raise, pd.create_program_descriptor = orig_plan, orig_create

    return restore
