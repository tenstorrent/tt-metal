# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier), for Quasar on ttsim: one `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec`, FlashAttention form, thread plan (4, 4, 2).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], memory_config=[DRAM], sharding=[interleaved], is_causal=[False, True], kv_heads_mode=[equal, grouped], batch=[single], alignment=[tile_aligned] (= PHASE0)
- **Accuracy achieved**: PCC=0.999998, max_abs_err≤0.011, rel_rms_err=0.0016–0.0020 (≈ the bfloat16 output-rounding floor), ULP p99 ≤ 8 (measured on 5 shapes × causal/full via test_scaled_dot_product_attention_precision_baseline.py)
- **Golden suite at Phase 0**: 18 / 18 supported cells passing (14 cells + 4 numeric regression cases), 6 / 6 rejections, 210 xfail_expected, 0 supported_fail / xpass_drift / xfail_wrong_mode (per `golden_results/verifier_report.json`)
- **Issues encountered** (verifier fixes, no behaviour change):
  - The fit predicate restated every ring capacity in separate closed forms. It now evaluates the one ring table (`ring_entries()`), which the `DataflowBufferSpec`s and the L1 total also use; checked identical on 322,560 cases.
  - The constants ring's entry indices are single-sourced in `kernels/sdpa_geometry.hpp`; dead duplicates were removed from Python and the compute kernel, and the reader asserts its push order against them.
  - `l1_ledger.md`'s worked footprint table was regenerated from the code (it predated the Float32 constants ring and counted one K/V ring where there are two).
- **Tests added**: test_scaled_dot_product_attention.py (harness acceptance), test_scaled_dot_product_attention_plan.py, test_scaled_dot_product_attention_debug.py, test_scaled_dot_product_attention_precision_baseline.py (verifier)

### Implementer notes (Phase 0)

Built as op_design.md specifies: one `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec`; FlashAttention
form (resident and streamed key regimes, same kernels); NEO units of one key/value head dealt C per cluster
round with K/V shared through ALL rings (read once per cluster per round); whole rounds dealt to clusters;
thread plan `THREAD_PLAN = (4, 4, 2)` from one host constant; every block knob (`q_chunk_tiles`,
`k_chunk_tiles`, `d_block_tiles`, `Q_BUFFER_DEPTH`, `KV_BLOCKS_PER_READER`, `OUT_BUFFER_DEPTH`) live and
single-sourced from the fit predicate.

SUPPORTED = PHASE0. Golden suite: every run cell passes (14 PHASE0 cells PCC >= 0.999998, rel-RMS ~0.002;
4 numeric regression cases; 6 rejections); every other cell is refused by validate().

Tests beyond the acceptance test (tests/ttnn/unit_tests/operations/scaled_dot_product_attention/):
- `test_scaled_dot_product_attention_plan.py`: regime-pinned (resident, streamed kc=4, ragged kc=3, qc=2,
  qc=2 streamed; causal and not), five other thread plans bit-identical to (4,4,2), per-NEO tagging
  (`tag_threads`), one device session (tilize program, this op, ttnn.to_layout untilize), the traced calls'
  argument forms (SDPAProgramConfig, WormholeComputeKernelConfig) at LoFi/HiFi2/HiFi3/HiFi4.
- `test_scaled_dot_product_attention_debug.py`: two-chunk deterministic cases (the streamed-regime bug).

Deviations from the design (advisory parts) and why:
- `cb_constants` follows the accumulator format (Float32) and holds 2 tiles when causal; the full-mask tile
  is replaced by `fill_tile_bitcast(-inf)` in DEST (matmul skipped for those tiles). See l1_ledger.md.
- The compute kernel clears the math thread's ALU-format latch before each FPU op init
  (`invalidate_alu_formats()`).
- On Quasar every unpack op bakes its operand's buffer descriptor at the op init.
  The kernel re-inits per ring.
- Ring names are `query_block` / `key_block` / `value_block`: `v_block` is an SFPI macro.

Structural counts (ttsim has no clock; no timing is claimed): K and V cross DRAM once per cluster round
(the design's data-movement budget; e.g. (1,2,512,64) non-causal: 8 rounds x 16 K tiles); Q and O once per
unit; one NoC barrier per Q / K / V block; output written with implicit sync, one tagged write per tile.
