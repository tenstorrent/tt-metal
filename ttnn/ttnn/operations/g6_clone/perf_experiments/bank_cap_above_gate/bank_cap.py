"""Variant of g6_clone's _bank_affine_roles for the bank-cap experiment (perf tournament round 1).

BANK_CAP_BYTES=<B>: every bank-affine plan runs each bank's run on
    min(len(row), bank_pages, max(BANK_CAP_MIN or SMALL_PLAN_BANK_CORES, ceil(bank_bytes / B)))
of its row's cores (nearest the middle DRAM column first), instead of the whole row once every bank has
at least as many pages as its row has cores. Unset: the op's rule.
"""

import math
import os


def install(descriptor_module):
    # BANK_CORES=<n> (op at or after the 2-per-bank graduation): set BANK_AFFINE_CORES_PER_BANK; 6 restores
    # the whole-row split of the 5-6-core dram_rows groups. BANK_CAP_BYTES below targets the op before it.
    cores = os.environ.get("BANK_CORES", "")
    if cores:
        descriptor_module.BANK_AFFINE_CORES_PER_BANK = int(cores)
        return
    cap = int(os.environ.get("BANK_CAP_BYTES", "0") or 0)
    if not cap:
        return
    import ttnn

    m = descriptor_module
    original_compute_plan = m.compute_plan

    def compute_plan(device, tensor_pages, aligned_page_bytes, geometry=None, num_banks=1):
        def roles(device, grid, tensor_pages, num_banks):
            groups = m.bank_core_groups(device, grid, num_banks, m.BANK_AFFINE_PLACEMENT)
            bank_pages = [len(range(bank, tensor_pages, num_banks)) for bank in range(num_banks)]
            if any(pages and not group for group, pages in zip(groups, bank_pages)):
                return None
            bank_cores = []
            for group, pages in zip(groups, bank_pages):
                if not pages:
                    bank_cores.append([])
                    continue
                floor = int(os.environ.get("BANK_CAP_MIN", "0") or 0) or m.SMALL_PLAN_BANK_CORES
                want = max(floor, math.ceil(pages * aligned_page_bytes / cap))
                bank_cores.append(m.bank_row_start_order(group)[: min(len(group), pages, want)])
            out = []
            for bank, cores in enumerate(bank_cores):
                run_start = 0
                for idx, (x, y) in enumerate(cores):
                    n = bank_pages[bank] // len(cores) + (1 if idx < bank_pages[bank] % len(cores) else 0)
                    out.append((ttnn.CoreCoord(x, y), "reader+writer", bank + num_banks * run_start, n))
                    run_start += n
            return out

        saved = m._bank_affine_roles
        m._bank_affine_roles = roles
        try:
            return original_compute_plan(device, tensor_pages, aligned_page_bytes, geometry, num_banks)
        finally:
            m._bank_affine_roles = saved

    m.compute_plan = compute_plan
    # g6_clone.py imported compute_plan by name.
    import importlib

    importlib.import_module("ttnn.operations.g6_clone.g6_clone").compute_plan = compute_plan
