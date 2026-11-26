# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: run the op's golden suite with a dram_stream_balance variant installed.

  PYTHONPATH=<this dir>:$PYTHONPATH DSB_VARIANT=mix_cols+resid_fix [DSB_GOLDEN_FILTER=memory_layout=INTERLEAVED] \
      scripts/run_safe_pytest.sh --run-all eval/golden_tests/groupnorm_sc_N_1_HW_C_b/ -p dsb_golden_plugin

DSB_GOLDEN_FILTER: comma-separated terms, each a '|'-separated list of substrings; a test is kept iff every
term has a substring its node id contains.
"""

import os

import variant_descriptor

VARIANT = os.environ.get("DSB_VARIANT", "baseline")
variant_descriptor.install(VARIANT)
print(f"DSB_GOLDEN_PLUGIN: variant {VARIANT}")


def pytest_collection_modifyitems(config, items):
    """DSB_GOLDEN_FILTER applies to the parametrized test_op cells; DSB_GOLDEN_DROP (comma-separated substrings)
    removes any item whose node id contains one of them."""
    subs = [s for s in os.environ.get("DSB_GOLDEN_FILTER", "").split(",") if s]
    drops = [s for s in os.environ.get("DSB_GOLDEN_DROP", "").split(",") if s]
    if not subs and not drops:
        return

    def ok(nid):
        if any(d in nid for d in drops):
            return False
        if "::test_op[" in nid:
            return all(any(a in nid for a in t.split("|")) for t in subs)
        return True

    keep = [it for it in items if ok(it.nodeid)]
    kept = {id(it) for it in keep}
    drop = [it for it in items if id(it) not in kept]
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep
    print(f"DSB_GOLDEN_PLUGIN: kept {len(keep)} items (filter {subs}, drop {drops})")
