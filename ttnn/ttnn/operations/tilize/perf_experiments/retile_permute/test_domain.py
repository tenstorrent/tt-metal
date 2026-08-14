# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Domain sweep for the winning arms: the axes the focus case does not move.

  * DTYPE — the run length of the NoC-landed permutation is
    min(out_face_h, src_face_h) * 16 * elem_bytes, so the element width moves the
    transfer size directly. fp32 doubles it, uint8 halves it.
  * The 64 B KNEE — 1->32 (32 B runs) is the one geometry where the DRAM-direct
    arm is not the winner; 2->32 and 32->2 (64 B runs) bracket where that starts.
  * SHARDED OUTPUT — a resident output shard makes the op take W_REGION work
    assignment and (for the direct arms) aliases the CB the reader writes.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/retile_permute/test_domain.py
"""

import pytest
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.retile_permute import _harness as H  # noqa: E402

FOCUS = [1, 1, 1024, 1024]


def _table(rows, header):
    base = next(ns for v, _s, ns, _e in rows if v == 0)
    out = [f"=== {header}  (baseline {base:.0f} ns) ==="]
    for variant, slug, ns, exact in sorted(rows, key=lambda r: r[2]):
        out.append(f"  {variant} {slug:16s} {ns:10.0f} ns   x{base / ns:5.2f}   bit_exact={exact}")
    logger.info("\n".join(out))
    return out


@pytest.mark.parametrize("dtype_name,dtype", [("fp32", ttnn.float32), ("uint8", ttnn.uint8)], ids=["fp32", "uint8"])
@pytest.mark.parametrize("in_tile_h,tile_h", [(32, 8), (8, 32)], ids=["32to8", "8to32"])
def test_dtype(device, dtype_name, dtype, in_tile_h, tile_h):
    rows = []
    for variant in H.arms_for(in_tile_h):
        ns, exact = H.run(device, variant, FOCUS, in_tile_h, tile_h, dtype=dtype)
        rows.append((variant, H.VARIANTS[variant][0], ns, exact))
    _table(rows, f"{dtype_name} {in_tile_h}->{tile_h} {FOCUS}")
    assert all(e for *_r, e in rows), "an arm is not bit-exact"


@pytest.mark.parametrize(
    "in_tile_h,tile_h", [(2, 32), (32, 2), (4, 32), (32, 1)], ids=["2to32", "32to2", "4to32", "32to1"]
)
def test_knee(device, in_tile_h, tile_h):
    """Where does the DRAM-direct arm's transfer get too small to pay?"""
    rows = []
    for variant in H.arms_for(in_tile_h):
        ns, exact = H.run(device, variant, FOCUS, in_tile_h, tile_h)
        rows.append((variant, H.VARIANTS[variant][0], ns, exact))
    _table(rows, f"knee {in_tile_h}->{tile_h} {FOCUS}")
    assert all(e for *_r, e in rows), "an arm is not bit-exact"


@pytest.mark.parametrize("in_tile_h,tile_h", [(32, 8), (1, 32)], ids=["32to8", "1to32"])
def test_sharded_output(device, in_tile_h, tile_h):
    """A resident L1 output shard: W_REGION work assignment, and for the direct
    arms the CB the reader writes is the output tensor itself."""
    shape = [1, 1, 1024, 256]
    cfg = H.height_shard(shape, 8)
    rows = []
    for variant in H.arms_for(in_tile_h):
        ns, exact = H.run(device, variant, shape, in_tile_h, tile_h, out_mem_config=cfg)
        rows.append((variant, H.VARIANTS[variant][0], ns, exact))
    _table(rows, f"sharded-out {in_tile_h}->{tile_h} {shape}")
    assert all(e for *_r, e in rows), "an arm is not bit-exact"


# --- the two axes where the "direct" arms are structurally INEXPRESSIBLE ------
# They hand raw bytes to the writer, so anything the PACKER would have done on
# the way out (a dtype cast) or the pad stamp geometry has no owner. These two
# tests establish what the op does there at all, and whether arm 4 is wrong (a
# guard is then mandatory) or the case simply cannot be built.
def test_cast_retile(device):
    """A retile that also CASTS. Does the op even allow it, and is arm 4 wrong?"""
    import torch

    from ttnn.operations.tilize import tilize
    from ttnn.operations.tilize import tilize_program_descriptor as pd

    shape = [1, 1, 512, 256]
    orig = pd.KERNEL_DIR
    for variant in (0, 4, 5, 8):
        pd.KERNEL_DIR = H._shim_dir(variant)
        try:
            t = torch.randn(shape).to(torch.bfloat16)
            tt = ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                tile=ttnn.Tile([32, 32]),
            )
            out = tilize(tt, tile=ttnn.Tile([8, 32]), dtype=ttnn.float32)
            ttnn.synchronize_device(device)
            exact = torch.equal(ttnn.to_torch(out), t.to(torch.float32))
            logger.info(f"CAST-RETILE arm={variant} exact={exact}")
        except Exception as exc:
            logger.info(f"CAST-RETILE arm={variant} REFUSED/ERROR: {type(exc).__name__}: {str(exc)[:200]}")
        finally:
            pd.KERNEL_DIR = orig


def test_padded_retile(device):
    """A retile whose target is PADDED (H not a multiple of the output tile)."""
    import torch

    from ttnn.operations.tilize import tilize
    from ttnn.operations.tilize import tilize_program_descriptor as pd

    shape = [1, 1, 20, 256]  # in_tile_h=1 -> 20 rows; out tile 32 -> pads to 32
    orig = pd.KERNEL_DIR
    for variant in (0, 4, 2):
        pd.KERNEL_DIR = H._shim_dir(variant)
        try:
            t = torch.randn(shape).to(torch.bfloat16)
            tt = ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                tile=ttnn.Tile([1, 32]),
            )
            out = tilize(tt, tile=ttnn.Tile([32, 32]))
            ttnn.synchronize_device(device)
            exact = torch.equal(ttnn.to_torch(out), t)
            logger.info(f"PADDED-RETILE arm={variant} exact={exact}")
        except Exception as exc:
            logger.info(f"PADDED-RETILE arm={variant} REFUSED/ERROR: {type(exc).__name__}: {str(exc)[:200]}")
        finally:
            pd.KERNEL_DIR = orig


def test_blockfloat_retile(device):
    """bfloat8_b -> bfloat8_b retile. A block-float TILE page is NOT plain
    elements (it carries per-face exponent sections), so both the baseline's face
    arithmetic and arm 4's run arithmetic are written against the wrong page
    layout. Which of them is wrong here is the question."""
    import torch

    from ttnn.operations.tilize import tilize
    from ttnn.operations.tilize import tilize_program_descriptor as pd

    shape = [1, 1, 512, 256]
    orig = pd.KERNEL_DIR
    for variant in (0, 4, 5, 8):
        pd.KERNEL_DIR = H._shim_dir(variant)
        try:
            t = torch.randn(shape)
            tt = ttnn.from_torch(
                t,
                dtype=ttnn.bfloat8_b,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                tile=ttnn.Tile([32, 32]),
            )
            ref = ttnn.to_torch(tt)
            out = tilize(tt, tile=ttnn.Tile([8, 32]))
            ttnn.synchronize_device(device)
            exact = torch.equal(ttnn.to_torch(out), ref)
            logger.info(f"BFP8-RETILE arm={variant} exact={exact}")
        except Exception as exc:
            logger.info(f"BFP8-RETILE arm={variant} REFUSED/ERROR: {type(exc).__name__}: {str(exc)[:200]}")
        finally:
            pd.KERNEL_DIR = orig
