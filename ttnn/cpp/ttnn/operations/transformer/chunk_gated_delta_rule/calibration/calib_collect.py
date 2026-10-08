# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One row per capture: the quantities the fused cost model is fitted to.

  python calib_collect.py <rows file> [<rows file> ...] [--json out.json] [--vt 4] [--out DIR]

A row is "label script args" (calib_rows.py / calib_batch.sh). The geometry comes from the args (--hv, --nv, --np,
--pool, --share, --nbuf, --rl, --phased); for a row with no pinned geometry (the op's own pick) the receiver and
producer cores are read from the capture's zones and the depth / share from the binding's pick for that geometry
(so run this on the tree that took the captures). The last run of the capture is analysed (the device profiler keeps
every run; the TRISC zones hold ~21 steps / 10 items per core, the BRISC and NCRISC zones the whole run).

Per capture:
  op     median device time of the ChunkGdn op over launches 2..n (ops_summary.txt; phased rows: prep + scan)
  item   TRISC_1 `prep_item` steady median (items after the first on each producer)        -> kWpUs, cross-check
  per    BRISC VALID-to-VALID period per producer (tx_valid ends), median                   -> kWpUs on production-bound rows
  fill   median over receivers of the first `scan_step` end                                 -> kFillA/B/C
  step   TRISC_1 `scan_step` steady median                                                  -> t_step(Vtl)
  pace   (median last `rx_wait_valid` end + step - fill) / (NC - 1)                         -> t_step + kChainSlopeUs on chain-bound rows
  last   median chunk-end over receivers; skew = max - median; tail = op - last             -> kSkewA/B, kTailUs
  swait  TRISC_0 `scan_wait_in` steady median per step (the compute's real input wait)
  txcr   BRISC `tx_wait_credit` per item; txcb BRISC `tx_wait_cb` per item                  -> the regime
  regime chain-bound (producers wait for credits) / production-bound (receivers wait for input) / balanced
  pools: per class (home / extra) items from the map, period, txcr, txcb, first VALID, finish
"""

import argparse
import glob
import re
import json
import os
import statistics as st
import sys

import pandas as pd

from ttnn._ttnn.operations import transformer as _t

PER_HEAD, POOL = 0, 1
med = lambda v: st.median(v) if v else float("nan")
r2 = lambda v: round(v, 2) if v == v else v


def parse_row_args(args):
    a = args.split()
    g = {
        "phased": "--phased" in a,
        "pool": "--pool" in a,
        "share": None,
        "nbuf": 0,
        "rl": -1,
        "nv": 0,
        "np": 0,
        "hv": 0,
        "T": 2048,
    }
    for k in ("hv", "nv", "np", "nbuf", "rl", "T"):
        if f"--{k}" in a:
            g[k] = int(a[a.index(f"--{k}") + 1])
    if "--share" in a:
        g["share"] = float(a[a.index("--share") + 1])
    g["BH"] = g["hv"]
    g["NC"] = g["T"] // 32
    return g


def op_median(d):
    try:
        txt = open(f"{d}/ops_summary.txt").read()
    except FileNotFoundError:
        return float("nan")
    vals = {}
    for line in txt.splitlines():
        f = line.split()
        if f and f[0].startswith("ChunkGdn"):
            vals[f[0]] = float(f[3])
    if "ChunkGdnDeviceOperation" in vals:
        return vals["ChunkGdnDeviceOperation"]
    if "ChunkGdnPrepOperation" in vals and "ChunkGdnScanOperation" in vals:
        return vals["ChunkGdnPrepOperation"] + vals["ChunkGdnScanOperation"]
    return float("nan")


def load_zones(d):
    """-> (zones {(x, y, risc, name): [(start_us, end_us), ...]}, xs, ys, run, runs) of the capture's last run."""
    csvs = glob.glob(f"{d}/reports/*/profile_log_device.csv")
    if not csvs:
        return None
    csv = max(csvs, key=os.path.getmtime)
    freq = float(open(csv).readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    df = pd.read_csv(csv, skiprows=1, usecols=[1, 2, 3, 5, 7, 10, 11])
    df.columns = ["x", "y", "risc", "t", "run", "zone", "type"]
    df["zone"] = df["zone"].astype(str).str.strip()
    df["risc"] = df["risc"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip()
    xs, ys = sorted(df.x.unique()), sorted(df.y.unique())
    runs = sorted(df.loc[df.zone == "prep_item", "run"].unique())
    if not runs:
        return None
    run = int(runs[-1])
    d0 = df[df.run == run].copy()
    d0["us"] = (d0.t - d0.t.min()) / freq
    d0 = d0.sort_values("t", kind="stable")
    Z = {}
    for key, g in d0.groupby(["x", "y", "risc", "zone"], sort=False):
        ty = g.type.values
        us = g.us.values
        s, e = us[ty == "ZONE_START"], us[ty == "ZONE_END"]
        n = min(len(s), len(e))
        Z[key] = list(zip(s[:n], e[:n]))
    return Z, xs, ys, run, [int(r) for r in runs]


def infer_geometry(Z, xs, ys, BH, NC, vt):
    """The geometry of a capture without a pinned one: receivers = cores with scan_step, producers = cores with
    prep_item; placement by matching the factory's core map; depth and share from the binding's pick."""
    Pz = {(x, y) for (x, y, r, z) in Z if z == "prep_item" and r == "TRISC_1"}
    Rz = {(x, y) for (x, y, r, z) in Z if z == "scan_step" and r == "TRISC_1"}
    gx, gy = len(xs), len(ys)
    phys = lambda xy: (xs[xy[0]], ys[xy[1]])
    if len(Rz) % BH:
        return None
    NV, P = len(Rz) // BH, len(Pz)
    tries = []
    if P % BH == 0:
        tries += [(1, P // BH), (0, P // BH)]
    tries.append((2, P))
    for pl, np_ in tries:
        try:
            recv, prod = _t.chunk_gdn_fused_placement(gx, gy, BH, NV, np_, pl)
        except Exception:
            continue
        if {phys(c) for c in recv} == Rz and {phys(c) for c in prod} == Pz:
            pick = _t.chunk_gdn_fused_geometry(gx, gy, BH, NC, vt, NV, np_, 0, POOL if pl == 2 else PER_HEAD)
            share = pick[7] / np_ if pl == 2 and pick[7] else (0.0 if pl == 2 else None)
            return {"nv": NV, "np": np_, "PL": pl, "nbuf": pick[3], "share": share, "geom_src": "zones+model"}
    return None


def metrics(Z, xs, ys, g, vt):
    BH, NV, NP, PL, NC = g["BH"], g["nv"], g["np"], g["PL"], g["NC"]
    gx, gy = len(xs), len(ys)
    recv, prod = _t.chunk_gdn_fused_placement(gx, gy, BH, NV, NP, PL)
    if PL == 2:
        nph = _t.chunk_gdn_fused_pool_home_producers(gx, gy, BH, NV, NP)
        nx = NP - BH * nph
        num = nx if g["share"] is None else int(round(g["share"] * NP))
        den = NP
    else:
        nph, nx, num, den = NP, 0, 0, 1
    items, owner = _t.chunk_gdn_fused_item_map(BH, NC, nph, nx, num, den)
    phys = lambda xy: (xs[xy[0]], ys[xy[1]])
    Pc = [phys(c) for c in prod]
    Rc = [phys(c) for c in recv]
    z = lambda core, r, name: Z.get((core[0], core[1], r, name), [])
    Pz = {(x, y) for (x, y, r, zn) in Z if zn == "prep_item" and r == "TRISC_1"}
    Rz = {(x, y) for (x, y, r, zn) in Z if zn == "scan_step" and r == "TRISC_1"}
    o = {
        "nph": nph,
        "nx": nx,
        "num": num,
        "den": den,
        "share": round(num / den, 4) if nx else (0.0 if PL == 2 else None),
        "roles_match": set(Pc) == Pz and set(Rc) == Rz,
    }
    if not o["roles_match"]:
        print(
            f"  WARNING: the capture's producer/receiver cores differ from the placement ({len(Pz)} / {len(Rz)} with zones, "
            f"{len(Pc)} / {len(Rc)} expected)",
            file=sys.stderr,
        )
    pi = {c: z(c, "TRISC_1", "prep_item") for c in Pc}
    steps = {c: z(c, "TRISC_1", "scan_step") for c in Rc}
    waits = {c: z(c, "NCRISC", "rx_wait_valid") for c in Rc}
    step = med([e - s for c in Rc for s, e in steps[c][1:]])
    fill = med([steps[c][0][1] for c in Rc if steps[c]])
    ends = [waits[c][-1][1] + step for c in Rc if waits[c]]
    last, last_max = med(ends), (max(ends) if ends else float("nan"))
    tv = {c: z(c, "BRISC", "tx_valid") for c in Pc}
    per = [(tv[c][-1][1] - tv[c][0][1]) / (len(tv[c]) - 1) for c in Pc if len(tv[c]) > 2]
    cnt = [len(l) for l in items]
    home = list(range(BH * nph))
    extra = list(range(BH * nph, NP)) if PL == 2 else []
    swait = med([e - s for c in Rc for s, e in z(c, "TRISC_0", "scan_wait_in")[1:]])
    txcr = med([e - s for c in Pc for s, e in z(c, "BRISC", "tx_wait_credit")[1:]])
    txcb = med([e - s for c in Pc for s, e in z(c, "BRISC", "tx_wait_cb")[1:]])
    o.update(
        {
            "n_home": max(cnt[p] for p in home),
            "n_extra": max(cnt[p] for p in extra) if extra else 0,
            "item": r2(med([e - s for c in Pc for s, e in pi[c][1:]])),
            "item1": r2(med([pi[c][0][1] - pi[c][0][0] for c in Pc if pi[c]])),
            "per": r2(med(per)),
            "fill": r2(fill),
            "step": round(step, 3) if step == step else step,
            "pace": round((last - fill) / (NC - 1), 3) if last == last else float("nan"),
            "last": r2(last),
            "skew": r2(last_max - last),
            "swait": r2(swait),
            "txcr": r2(txcr),
            "txcb": r2(txcb),
            "rxval": r2(med([e - s for c in Rc for s, e in waits[c][1:]])),
            "steps_seen": min((len(steps[c]) for c in Rc), default=0),
            "items_seen": min((len(pi[c]) for c in Pc), default=0),
        }
    )
    o["regime"] = (
        "chain-bound" if txcr >= 5 and txcb < 2 else "production-bound" if txcb >= 5 and txcr < 2 else "balanced"
    )
    if extra:
        for cname, idx in (("home", home), ("extra", extra)):
            tvc = {p: tv[Pc[p]] for p in idx}
            o[f"{cname}_per"] = r2(
                med([(tvc[p][-1][1] - tvc[p][0][1]) / (len(tvc[p]) - 1) for p in idx if len(tvc[p]) > 2])
            )
            o[f"{cname}_txcr"] = r2(med([e - s for p in idx for s, e in z(Pc[p], "BRISC", "tx_wait_credit")[1:]]))
            o[f"{cname}_txcb"] = r2(med([e - s for p in idx for s, e in z(Pc[p], "BRISC", "tx_wait_cb")[1:]]))
            o[f"{cname}_first_valid"] = r2(med([tvc[p][0][1] for p in idx if tvc[p]]))
            o[f"{cname}_finish"] = r2(med([tvc[p][-1][1] for p in idx if tvc[p]]))
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--vt", type=int, default=4)
    ap.add_argument(
        "--out", default=os.environ.get("CALIB_OUT", "generated/gdn_calib"), help="capture directory (CALIB_OUT)"
    )
    a = ap.parse_args()
    rows = []
    for rf in a.rows:
        for line in open(rf):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            label, script, args = line.split(None, 2)
            g = parse_row_args(args)
            d = f"{a.out}/{label}"
            op = op_median(d)
            r = {
                "label": label,
                "op": op,
                "BH": g["BH"],
                "NC": g["NC"],
                "phased": g["phased"],
                "nv": g["nv"],
                "np": g["np"],
                "PL": 2 if g["pool"] else (g["rl"] if g["rl"] >= 0 else 1),
                "nbuf": g["nbuf"],
                "share": g["share"],
                "geom_src": "args" if g["nv"] or g["np"] else "auto",
            }
            if op != op:
                r["note"] = "no capture"
                print(f"{label}: no capture", file=sys.stderr)
            elif not g["phased"]:
                print(f"{label}: parsing zones ...", file=sys.stderr, end=" ", flush=True)
                lz = load_zones(d)
                if lz is None:
                    r["note"] = "no device zones"
                else:
                    Z, xs, ys, run, runs = lz
                    if r["geom_src"] == "auto":
                        inf = infer_geometry(Z, xs, ys, g["BH"], g["NC"], a.vt)
                        if inf is None:
                            r["note"] = "geometry not recognised"
                            print("geometry not recognised", file=sys.stderr)
                            rows.append(r)
                            continue
                        r.update(inf)
                        g.update(inf)
                    elif (
                        r["nbuf"] == 0
                    ):  # pinned geometry, free depth: the label's _d<N> suffix, else the binding's pick
                        m = re.search(r"_d([1-8])(?:_|$)", label)
                        if m:
                            r["nbuf"] = g["nbuf"] = int(m.group(1))
                            r["geom_src"] = "args+label depth"
                        else:
                            pick = _t.chunk_gdn_fused_geometry(
                                len(xs),
                                len(ys),
                                g["BH"],
                                g["NC"],
                                a.vt,
                                g["nv"],
                                g["np"],
                                0,
                                POOL if g["pool"] else PER_HEAD,
                                -1.0 if g["share"] is None else g["share"],
                            )
                            r["nbuf"] = g["nbuf"] = pick[3]
                            r["geom_src"] = "args+model depth"
                            print(
                                f"(no --nbuf in the row and no _d<N> in the label: depth {pick[3]} taken from the tree's current model)",
                                file=sys.stderr,
                                end=" ",
                            )
                    g["PL"] = r["PL"]
                    r.update(metrics(Z, xs, ys, g, a.vt))
                    r["run"], r["runs"] = run, runs
                    print(
                        f"op {op:.1f} item {r['item']} per {r['per']} fill {r['fill']} step {r['step']} pace {r['pace']} {r['regime']}",
                        file=sys.stderr,
                    )
            rows.append(r)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    keys = [
        "label",
        "BH",
        "nv",
        "np",
        "PL",
        "nbuf",
        "share",
        "op",
        "n_home",
        "n_extra",
        "item",
        "per",
        "fill",
        "step",
        "pace",
        "skew",
        "swait",
        "txcr",
        "txcb",
        "regime",
    ]
    print(" ".join(f"{k:>{26 if k == 'label' else 16 if k == 'regime' else 6}s}" for k in keys))
    for r in rows:
        print(" ".join(f"{str(r.get(k, '')):>{26 if k == 'label' else 16 if k == 'regime' else 6}}" for k in keys))


if __name__ == "__main__":
    main()
