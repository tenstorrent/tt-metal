# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""bank_local_small: bank locality for small g6_clone plans (perf tournament round 1).

Installed into a bench process by bench_cases.py (this dir) when BLS_VARIANT is set; the op's files are
never modified, only its module globals are patched in that process.

Variants (env):
  BLS_VARIANT=""         the op as it is (baseline).
  BLS_VARIANT=ba         relaxed bank_affine_runs gate: a plan below the op's gate (some group core would
                         get 0 pages) still runs bank_affine_runs, each bank's pages going to
                         min(pages, group size, BLS_CPB) cores of that bank's anchor row (dram_rows
                         placement), each core a contiguous bank run moved as one transfer, both kernels
                         addressing the bank's NoC0-table endpoint (on the cores' row). Host-only: the
                         op's kernels run unchanged. Plans the op's gate already accepts are untouched.
      BLS_ORDER=anchor   which cores of a bank's row are used first: anchor = the group's own order
                         (anchor first, nearest the bank's DRAM column); early = the columns whose kernels
                         start first (nearest the middle DRAM column: west rows x descending, east rows x
                         ascending).
      BLS_CPB=0          most cores per bank (0 = as many as pages).
      BLS_CORE=x/y       one-page plans only: run the page on this logical core instead (endpoint control).
  BLS_VARIANT=rule       the integration candidate: ba with BLS_ORDER=early and BLS_CPB=2 (defaults, both
                         overridable), plus one-page logical_rows plans moved to the same core (bank 0's
                         first row core in that order) with both kernels addressing the NoC0 endpoint
                         (kernel copies). A forced-flat gate (BANK_AFFINE_MIN_CORE_PAGES > 1, used by the
                         unit tests) keeps flat_page_range.
  BLS_VARIANT=at         one-page plans only (any regime): keep the op's plan but move its core to
                         BLS_CORE (default: bank 0's NoC0 anchor). With BLS_EP=0|1 both kernels address
                         that NoC's endpoint table on the consecutive-page paths (kernel copies in
                         kernels/, define BLS_FLAT_ENDPOINT_NOC); without it each kernel addresses its own
                         NoC's endpoints (location-only control, the op's kernels).
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

HERE = Path(__file__).resolve().parent
BLS_KERNEL_DIR = HERE / "kernels"


def _env():
    return {
        "variant": os.environ.get("BLS_VARIANT", ""),
        "order": os.environ.get("BLS_ORDER", "anchor"),
        "cpb": int(os.environ.get("BLS_CPB", "0") or 0),
        "core": os.environ.get("BLS_CORE", ""),
        "ep": os.environ.get("BLS_EP", ""),
    }


def ordered_group(group, order: str):
    """A bank's row cores in the order they are used. `group` comes from bank_core_groups (row-major,
    i.e. x ascending; the anchor is its first core)."""
    if order == "anchor":
        return list(group)
    if order == "early":
        # Kernel start offset grows with the physical column's distance from the middle DRAM column
        # (phys x9): x11 < x6 < x12 < x13 < x5 < x4 < x14 < x15 < x3 < x1/x2. A west group
        # (anchor at logical x 0) is used from its east end, an east group (anchor x 6) from its anchor.
        anchor_x = group[0][0]
        return sorted(group, key=lambda c: -c[0]) if anchor_x == 0 else sorted(group, key=lambda c: c[0])
    raise ValueError(f"bls: unknown BLS_ORDER {order!r}")


def relaxed_bank_affine_roles(desc, orig_roles_fn, env):
    """The rule as code: _bank_affine_roles with the gate relaxed to 'some page per bank'."""

    def roles_fn(device, grid, tensor_pages, num_banks):
        roles = orig_roles_fn(device, grid, tensor_pages, num_banks)
        if env["variant"] == "rule" and desc.BANK_AFFINE_MIN_CORE_PAGES > 1:
            return roles
        # BLS_CAP_ABOVE=1 (exploratory): the core cap also applies to plans the op's gate accepts.
        if roles is not None and os.environ.get("BLS_CAP_ABOVE", "") != "1":
            return roles
        groups = desc.bank_core_groups(device, grid, num_banks, desc.BANK_AFFINE_PLACEMENT)
        bank_pages = [len(range(bank, tensor_pages, num_banks)) for bank in range(num_banks)]
        if any(pages and not group for group, pages in zip(groups, bank_pages)):
            return None
        roles = []
        for bank, group in enumerate(groups):
            pages = bank_pages[bank]
            if pages == 0:
                continue
            cores = ordered_group(group, env["order"])
            num_cores = min(pages, len(cores), env["cpb"] or len(cores))
            cores = cores[:num_cores]
            if tensor_pages == 1 and env["core"]:
                cores = [tuple(int(v) for v in env["core"].replace("/", ",").split(","))]
            run_start = 0
            for idx, (x, y) in enumerate(cores):
                num_pages = pages // num_cores + (1 if idx < pages % num_cores else 0)
                roles.append((ttnn.CoreCoord(x, y), "reader+writer", bank + num_banks * run_start, num_pages))
                run_start += num_pages
        return roles

    return roles_fn


def install():
    """Patch the op's modules in this process for BLS_VARIANT; returns the variant tag ('' = op)."""
    env = _env()
    if not env["variant"]:
        return ""
    if env["variant"] == "rule":
        env["order"] = os.environ.get("BLS_ORDER", "early")
        env["cpb"] = int(os.environ.get("BLS_CPB", "2") or 0)
    import importlib

    desc = importlib.import_module("ttnn.operations.g6_clone.g6_clone_program_descriptor")
    op = importlib.import_module("ttnn.operations.g6_clone.g6_clone")
    orig_compute_plan = op.compute_plan
    orig_create = op.create_program_descriptor

    if env["variant"] in ("ba", "rule"):
        desc._bank_affine_roles = relaxed_bank_affine_roles(desc, desc._bank_affine_roles, env)
    elif env["variant"] != "at":
        raise ValueError(f"bls: unknown BLS_VARIANT {env['variant']!r}")

    def compute_plan(device, tensor_pages, aligned_page_bytes, geometry=None, num_banks=1):
        plan = orig_compute_plan(device, tensor_pages, aligned_page_bytes, geometry, num_banks)
        if env["variant"] == "at" and tensor_pages == 1:
            if env["core"]:
                x, y = (int(v) for v in env["core"].replace("/", ",").split(","))
            else:
                anchor = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)[0]
                x, y = anchor.x, anchor.y
            ((_, role, start, count),) = plan["roles"]
            plan["roles"] = [(ttnn.CoreCoord(x, y), role, start, count)]
            plan["core_ranges"] = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))])
            if env["ep"] != "":
                plan["bls_flat_endpoint_noc"] = int(env["ep"])
        if env["variant"] == "rule" and tensor_pages == 1 and plan["regime"] == desc.REGIME_LOGICAL_ROWS:
            grid = device.compute_with_storage_grid_size()
            group = desc.bank_core_groups(device, grid, num_banks, desc.BANK_AFFINE_PLACEMENT)[0]
            x, y = ordered_group(group, env["order"])[0]
            ((_, role, start, count),) = plan["roles"]
            plan["roles"] = [(ttnn.CoreCoord(x, y), role, start, count)]
            plan["core_ranges"] = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))])
            plan["bls_flat_endpoint_noc"] = desc.BANK_AFFINE_ENDPOINT_NOC
        roles = [(c.x, c.y, s, n) for c, _, s, n in plan["roles"]]
        print(
            f"BLS_PLAN variant={env['variant']} order={env['order']} cpb={env['cpb']} core={env['core']} ep={env['ep']} "
            f"regime={plan['regime']} stride={plan['page_stride']} ep_noc={plan['bank_endpoint_noc']} "
            f"flat_ep={plan.get('bls_flat_endpoint_noc')} pages={tensor_pages} page_bytes={aligned_page_bytes} "
            f"cores={plan['num_cores']} block_pages={plan['block_pages']} rects={len(plan['core_ranges'].ranges())} "
            f"roles={roles}",
            flush=True,
        )
        return plan

    def create_program_descriptor(input_tensor, output_tensor, plan):
        descriptor = orig_create(input_tensor, output_tensor, plan)
        ep = plan.get("bls_flat_endpoint_noc")
        if ep is not None:
            for kernel, name in zip(descriptor.kernels, ("g6_clone_reader.cpp", "g6_clone_writer.cpp")):
                kernel.kernel_source = str(BLS_KERNEL_DIR / name)
                kernel.defines = list(kernel.defines) + [("BLS_FLAT_ENDPOINT_NOC", str(ep))]
        return descriptor

    op.compute_plan = compute_plan
    op.create_program_descriptor = create_program_descriptor
    return env["variant"]
