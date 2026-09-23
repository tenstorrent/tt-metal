"""Secondary lever: a row-cost-aware tile-row split (host side, RT args only).

Each scheme maps (cores, R) -> {(x, y) logical: tile-rows}. Weights are allocated by largest
remainder so the total is exactly R. Schemes:
  oracle     rows ~ 1 / T, T = the default program's BRISC-KERNEL end time per Tensix core on the
             focus shape (measured 2026-09-23, NoC0 coords). Shape- and session-specific: an upper
             bound on what any static rebalance can buy, not a rule.
  oracle_sq  rows ~ 1 / T^2 (more aggressive).
  geo_tail   geometry rule: the Tensix cores of the lower band (NoC0 y >= 7) east of the grid's
             x midpoint through the core just west of DRAM column 5 -- NoC0 x in {4, 6, 7, 8} --
             get weight 1 - d, every other core 1 (d = GEO_TAIL_DEFICIT).
  geo_tail2  as geo_tail, but the lower band's other four columns (NoC0 x in {1, 2, 3, 9}) get 1 + d
             and the upper band stays at 1.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_nr_spec = importlib.util.spec_from_file_location("noc_region_nr", Path(__file__).parent / "noc_region.py")

# default program, [1,1,16384,64], BRISC-KERNEL end cycles; rows = NoC0 y, cols = NoC0 x 1,2,3,4,6,7,8,9
_XS = (1, 2, 3, 4, 6, 7, 8, 9)
_T = {
    1: (20546, 20713, 20656, 20593, 21114, 20462, 21186, 20968),
    2: (20324, 20373, 21363, 21930, 21278, 20950, 19795, 21120),
    3: (19671, 19840, 19669, 21345, 21684, 20359, 20555, 19546),
    4: (18636, 18740, 18582, 21476, 20946, 19635, 18805, 18222),
    7: (19153, 18072, 17178, 21594, 22454, 21339, 20314, 17325),
    8: (18494, 16644, 18763, 19651, 22986, 22368, 21847, 17447),
    9: (17875, 18060, 20428, 22645, 24105, 23665, 22907, 17185),
    10: (19459, 18481, 19873, 22121, 24055, 23024, 22706, 16746),
}
GEO_TAIL_DEFICIT = 0.125


def _alloc(weights, R):
    tot = sum(weights.values())
    raw = {k: R * w / tot for k, w in weights.items()}
    out = {k: max(1, int(v)) for k, v in raw.items()}
    rem = R - sum(out.values())
    for k in sorted(raw, key=lambda k: raw[k] - int(raw[k]), reverse=True)[: max(0, rem)]:
        out[k] += 1
    return out


def make(scheme, phys):
    def fn(cores, R):
        w = {}
        for c in cores:
            px, py = phys(c)
            if scheme.startswith("oracle"):
                t = _T[py][_XS.index(px)]
                w[(c.x, c.y)] = 1.0 / (t * t if scheme == "oracle_sq" else t)
            elif scheme == "geo_tail":
                w[(c.x, c.y)] = 1.0 - GEO_TAIL_DEFICIT if (py >= 7 and px in (4, 6, 7, 8)) else 1.0
            elif scheme == "geo_tail2":
                # lower band: the 4 columns around DRAM column 5 lose d, the other 4 gain d
                lower = py >= 7
                w[(c.x, c.y)] = (
                    (1.0 - GEO_TAIL_DEFICIT if px in (4, 6, 7, 8) else 1.0 + GEO_TAIL_DEFICIT) if lower else 1.0
                )
            else:
                raise ValueError(scheme)
        return _alloc(w, R)

    return fn


def install(monkeypatch, device, scheme, rule_name=None):
    nr = importlib.util.module_from_spec(_nr_spec)
    _nr_spec.loader.exec_module(nr)
    nr._load_phys(device)
    nr.install(monkeypatch, device, rule_name, rows_fn=make(scheme, nr.phys))
