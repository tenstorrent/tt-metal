# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Variant installer for the dram_stream_balance bake-off.

install(name) swaps the op's planner / descriptor entry points (plan_or_raise, create_program_descriptor) for
desc_variant.py's copies configured per VARIANTS[name], and returns a restore callable. "baseline" installs
nothing (the op's own descriptor and kernels). Names may be joined with "+" (e.g. "col+resid_fix").
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

KDIR = os.path.join(HERE, "kernels")

# Each entry is a partial CFG for desc_variant.
VARIANTS = {
    "copy": {},  # the descriptor copy with no knob turned: must equal baseline
    "row_rev": {"order": "row_rev"},
    "col": {"order": "col"},
    "noc_swap": {"noc_swap": True},
    "resid_fix": {"resid_fix": True},
    "kcopy": {"kernel_dir": KDIR},
    "tmux": {"tmux": True},
    "gx2": {"gx_mult": 2},
    "gx4": {"gx_mult": 4},
    "mix_checker": {"noc_mix": "checker"},
    "mix_cols": {"noc_mix": "cols"},
    "mix_rows": {"noc_mix": "rows"},
    "mix_hop": {"noc_mix": "hop"},
    "mix_hop_inv": {"noc_mix": "hop_inv"},
    "mix_unit": {"noc_mix": "unit"},
    "mix_unit_inv": {"noc_mix": "unit_inv"},
    "dyn_split": {"defines": [("DSB_DYN_SPLIT", "1")], "kernel_dir": KDIR},
}


def _cfg_for(name):
    cfg = {}
    for part in name.split("+"):
        if part.startswith("D:"):  # D:NAME -> reader + writer define NAME=1
            cfg.setdefault("defines", []).append((part[2:], "1"))
            cfg.setdefault("kernel_dir", KDIR)
        elif part.startswith("WD:"):
            cfg.setdefault("writer_defines", []).append((part[3:], "1"))
            cfg.setdefault("kernel_dir", KDIR)
        elif part.startswith("RD:"):
            cfg.setdefault("reader_defines", []).append((part[3:], "1"))
            cfg.setdefault("kernel_dir", KDIR)
        else:
            for k, v in VARIANTS[part].items():
                if isinstance(v, list):
                    cfg.setdefault(k, []).extend(v)
                else:
                    cfg[k] = v
    return cfg


def _install_graduate(real):
    """'grad': the op's own descriptor with graduate/graduate.patch applied (the exact graduation candidate)."""
    import importlib.util

    pkg = "ttnn.operations.groupnorm_sc_N_1_HW_C_b"
    path = os.path.join(HERE, "graduate", "groupnorm_sc_N_1_HW_C_b_program_descriptor.py")
    spec = importlib.util.spec_from_file_location(pkg + "._dsb_graduate_descriptor", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    mod.KERNEL_DIR = real.KERNEL_DIR  # the op's own kernels
    saved = (real.plan_or_raise, real.create_program_descriptor)
    real.plan_or_raise = mod.plan_or_raise
    real.create_program_descriptor = mod.create_program_descriptor

    def restore():
        real.plan_or_raise, real.create_program_descriptor = saved

    return restore


def install(name):
    if name in ("", "baseline"):
        return lambda: None
    if name == "grad":
        return _install_graduate(
            importlib.import_module(
                "ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor"
            )
        )
    import desc_variant

    real = importlib.import_module("ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor")
    cfg = dict(
        order="row",
        noc_swap=False,
        resid_fix=False,
        kernel_dir=None,
        defines=[],
        writer_defines=[],
        reader_defines=[],
        tmux=False,
        gx_mult=1,
        noc_mix=None,
    )
    cfg.update(_cfg_for(name))
    desc_variant.CFG.clear()
    desc_variant.CFG.update(cfg)
    saved = (real.plan_or_raise, real.create_program_descriptor)
    real.plan_or_raise = desc_variant.plan_or_raise
    real.create_program_descriptor = desc_variant.create_program_descriptor

    def restore():
        real.plan_or_raise, real.create_program_descriptor = saved

    return restore
