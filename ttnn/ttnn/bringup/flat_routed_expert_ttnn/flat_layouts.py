# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Flat routed expert layouts: the per-model layout / program choices (C++ FlatLayoutOptions, typed; part of the
plan cache key and the program hashes) instead of MIMO_FL_* environment defaults (ENV_CLEANUP.md).

    from ttnn.bringup.flat_routed_expert_ttnn.flat_layouts import GLM_FLASH, flat_layout
    FlatRoutedExpert(..., layout=GLM_FLASH)          # a preset (a dict of FlatLayoutOptions fields)
    FlatRoutedExpert(..., layout={**DRISC_ONLY, "max_down": 28})

Unset fields keep the plan's own choice (FlatLayoutOptions' defaults). Perf / debug probes are not layout: they stay
MIMO_FL_* environment variables read in one place (C++ FlatProbes)."""

import os

import ttnn

FIELDS = (
    "drisc",
    "drisc_dual",
    "drisc_in_overlap",
    "drisc_pure",
    "max_down",
    "x_helpers",
    "half_width",
    "hg_units",
    "row0",
    "row1",
    "x_path",
    "x_snakes",
    "x_pieces",
    "x_fwd_ncrisc",
    "x_fwd_noc",
    "x_fwd_depth",
    "x_relay_depth",
    "x_once",
    "dw_batch",
    "pin_ratio",
)

# DRAM-core (DRISC) gate/up weight readers only: no Tensix weight readers, no per-call Tensix fallback, inside the
# combine overlaps too (pure DRISC: every slot from the DRAM cores, tiles in switch mode)
DRISC_ONLY = dict(drisc_in_overlap=True, drisc_dual=False, drisc_pure=1)

# GLM-5.3-Flash on the K3 token-split block (models/demos/glm53_flash_d_p/tt/experts_ts.py): DRISC-only weight readers
# (pure by the plan's default: no reader down columns), 32 down cores (128 output tile columns, 4 a core), one x helper
# (a second one costs ~104 KB of every core's arena: the dispatch's manifest no longer fits beside it at S 640)
GLM_FLASH = dict(drisc_in_overlap=True, drisc_dual=False, max_down=32, x_helpers=1)

# GLM-5.3-Flash on the all-gather MoE path (experts_ag.py): 32 down cores, so both modes share one flat weight cache
GLM_FLASH_AG = dict(max_down=32)


def k3_gu64(acc_on_ports=True):
    """Kimi K3 token-split block (tt_k3_token_split_moe.py, K3_GU64): 64 gate/up cores (16-column units, 3 a core)
    with DRISC-only weight readers, 28 down cores (4 columns each); two x helpers per rectangle on the cells the
    accumulators-on-ports combine frees (one otherwise)."""
    return dict(
        drisc_in_overlap=True,
        drisc_dual=False,
        half_width=True,
        hg_units=3,
        max_down=28,
        x_helpers=2 if acc_on_ports else 1,
    )


def flat_layout(layout=None, **kw):
    """A FlatLayoutOptions from a preset dict / an existing FlatLayoutOptions / None, plus field overrides."""
    o = ttnn._ttnn.operations.bringup.FlatLayoutOptions()
    if isinstance(layout, ttnn._ttnn.operations.bringup.FlatLayoutOptions):
        layout = {f: getattr(layout, f) for f in FIELDS}
    for k, v in {**(layout or {}), **kw}.items():
        if k not in FIELDS:
            raise ValueError(f"flat layout: unknown field {k!r} (FlatLayoutOptions: {', '.join(FIELDS)})")
        setattr(o, k, v)
    return o


def _atoi(v):  # (C atoi: the leading integer, else 0 - what the old std::getenv reads parsed)
    v = v.strip()
    n = len(v) - len(v.lstrip("+-"))
    while n < len(v) and v[n].isdigit():
        n += 1
    try:
        return int(v[:n])
    except ValueError:
        return 0


# the old environment names of the layout fields: test harnesses / benches only (model code passes presets)
_ENV = {
    "MIMO_FL_DRISC": ("drisc", lambda v: _atoi(v) != 0),
    "MIMO_FL_DRISC_DUAL": ("drisc_dual", lambda v: _atoi(v) != 0),
    "MIMO_FL_CMB_DRISC": ("drisc_in_overlap", lambda v: _atoi(v) != 0),
    "MIMO_FL_DRISC_PURE": ("drisc_pure", lambda v: int(_atoi(v) != 0)),
    "MIMO_FL_ND": ("max_down", _atoi),
    "MIMO_FL_NH": ("x_helpers", _atoi),
    "MIMO_FL_HG": ("half_width", lambda v: _atoi(v) != 0),
    "MIMO_FL_HG_NU": ("hg_units", _atoi),
    "MIMO_FL_X_CHAIN": ("x_path", _atoi),
    "MIMO_FL_X_CHAIN_SNAKES": ("x_snakes", _atoi),
    "MIMO_FL_X_PIECES": ("x_pieces", _atoi),
    "MIMO_FL_X_FWD_NC": ("x_fwd_ncrisc", lambda v: int(_atoi(v) != 0)),
    "MIMO_FL_X_CHAIN_NOC": ("x_fwd_noc", lambda v: int(_atoi(v) != 0)),
    "MIMO_FL_X_FWD_DEPTH": ("x_fwd_depth", _atoi),
    "MIMO_FL_X_CHAIN_DEPTH": ("x_relay_depth", _atoi),
    "MIMO_FL_X1": ("x_once", lambda v: _atoi(v) != 0),
    "MIMO_FL_DW_BATCH": ("dw_batch", _atoi),
    "MIMO_FL_PIN_RATIO": ("pin_ratio", _atoi),
}


def layout_from_env(layout=None, env=None):
    """Test harnesses / benches only: `layout` (a preset) with the old MIMO_FL_* names in the environment applied on
    top (MIMO_FL_ROWS=y0,y1 -> row0 / row1), so existing sweep recipes keep working. Model code never calls this."""
    env = os.environ if env is None else env
    kw = dict(layout or {})
    for name, (field, conv) in _ENV.items():
        if name in env:
            kw[field] = conv(env[name])
    if "MIMO_FL_ROWS" in env:
        y0, y1 = (int(v) for v in env["MIMO_FL_ROWS"].split(","))
        kw["row0"], kw["row1"] = y0, y1
    return flat_layout(kw)
