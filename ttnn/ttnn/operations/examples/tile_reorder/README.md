# tile_reorder

**Trick:** on a DRAM-bound move, coalesce into whole-page transfers and batch barriers — relocate whole tiles instead of writing sub-tile faces. Read on NoC0, write on NoC1.

**Op:** `tile_reorder(t, method="relocate" | "scatter")` — reverses column-tile order. `relocate` = one whole-page write/tile; `scatter` = 4 faces + barrier each (baseline).

**Run:** `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_tile_reorder.py`
