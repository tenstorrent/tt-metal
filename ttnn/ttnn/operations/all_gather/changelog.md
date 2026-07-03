# Changelog: all_gather

## Phase 0 — Core Implementation
- **Date**: 2026-07-03
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Self-contained Python CCL op:
  `generic_op` + `MeshProgramDescriptor` running a bidirectional store-and-forward
  ring on newly-authored dataflow kernels (`all_gather_reader.cpp`,
  `all_gather_writer.cpp`). Two worker cores per device (`core_fwd`, `core_bwd`),
  three single-owner GlobalSemaphores (barrier + fwd/bwd counting), fabric egress via
  the `ccl_helpers_dataflow.hpp` safety-by-construction helper.
- **SUPPORTED at Phase 0**:
  - dtype = [bfloat16, float32]
  - layout = [TILE]
  - topology = [Linear]
  - gather_dim = [-4]  (negative convention; -4 ≡ dim 0 for rank-4 shards)
  - alignment (INPUT_TAGGERS) = [tile_aligned, non_tile_aligned]
  - EXCLUSIONS = []
- **Accuracy achieved**: PCC = 1.0, max_abs_err = 0.0, mean_abs_err = 0.0,
  rel_rms_err = 0.0 — bit-exact identity gather across 4 shapes × {bf16, f32}
  (measured via `test_all_gather_precision_baseline.py`). All 8 devices agree
  bit-for-bit (replicated output).
- **Golden suite at Phase 0**: **16 / 384 cells passing** (`supported_pass = 16`,
  `xfail_expected = 304`, `invalid_skipped = 64`; loud categories `supported_fail`,
  `xpass_drift`, `xfail_wrong_mode` all = 0) — per `verifier_report.json`.
  Run via `scripts/run_multidevice_sim_pytest.py --op all_gather` in 5 dtype/layout
  `-k` chunks (a full single-process golden run exceeds the wall-clock backstop
  because the CCL golden `mesh_device` fixture re-inits the 8-device fabric per cell).
- **Issues encountered / fixed during verification**:
  - Simplified a dead-code branch in `validate()`: `page % 16 != 0 and page != 16`
    → `page % 16 != 0` (the `and page != 16` conjunct was unreachable). Behaviour
    unchanged.
  - Unblocked shared test infra: `tests/ttnn/utils_for_testing.py` referenced
    `ttnn.fp8_e4m3` (from FP8-enablement commit `079872566e`) which the built binary
    predates, breaking collection of every test importing `assert_with_pcc`. Guarded
    the entry with `hasattr(ttnn, "fp8_e4m3")`. Not an all_gather defect.
  - No SUPPORTED drift (`xpass_drift = 0`) — no auto-promotions needed.
- **Tests added**:
  - `test_all_gather.py` (acceptance; pre-existing, 9/9 PASS)
  - `test_all_gather_precision_baseline.py` (pre-existing, 8/8 PASS, bit-exact)
  - `test_all_gather_extended.py` (**new**; preallocated-output path +
    validate() rejection behaviour; 2/2 PASS)
- **Refinement queue set up** (`op_requirements.md`): 3 refinements covering the
  TARGET − SUPPORTED gap —
  1. Format axes: bfloat8_b + ROW_MAJOR (`/memory-layouts`, `/numeric-formats-metal`)
  2. Non-contiguous concat addressing: gather_dim −3/−2/−1 (verifier-authored)
  3. Ring topology (verifier-authored; **verification infra-blocked** — no ring
     topology in the multidevice sim matrix yet)

## Refinement 1 — Format axes: bfloat8_b dtype + ROW_MAJOR layout
- **Date**: 2026-07-03
- **What was done**: Promoted `ttnn.bfloat8_b` into `SUPPORTED["dtype"]` and
  `ttnn.ROW_MAJOR_LAYOUT` into `SUPPORTED["layout"]`. **No kernel or program-descriptor
  change was needed** — all_gather is pure byte movement (it never (un)tilizes), and the
  reader/writer already move whole pages by page-index via `TensorAccessor`, with the
  relay-CB `data_format` derived from the input dtype and the page size from
  `buffer_page_size()` / `buffer_num_pages()`. Both new axes are therefore native
  in-kernel format flexibility, NOT a `to_layout`/`tilize` wrapper:
  - **ROW_MAJOR**: the relay CB page is the logical row (`buffer_page_size()`, L1-aligned
    for the CB slot). At `gather_dim=-4` an RM shard is still a contiguous page range, so
    the existing contiguous-slice walk is unchanged. Confirmed (via the `point_to_point`
    sibling RM CCL op + the TensorAccessor implementation) that for **interleaved DRAM**
    tensors `get_noc_addr(page_id)` re-aligns the DRAM page stride internally through
    `InterleavedAddrGen`, so passing the logical page size is correct (not a stride bug).
  - **bfloat8_b**: the whole block-float tile page (1088 B, 16-B aligned) is relayed
    intact; because tiles arrive already-packed and are NEVER re-tilized, the shared-face
    exponents survive even for non-tile-aligned shards (memory-layouts §5) — so
    `bfloat8_b × non_tile_aligned` needs **no EXCLUSIONS** (the verifier's flagged
    possible-exclusion did not materialize; EXCLUSIONS stays `[]`).
  - `bfloat8_b × ROW_MAJOR` is structurally impossible (INVALID in feature_spec, skipped
    by the harness); the op file is agnostic to it (no INVALID block, per the model).
  - `validate()` needed no structural change — the registry axis gate enforces the
    extended SUPPORTED automatically.
- **Accuracy achieved**: identity gather is bit-exact byte movement, so the only error is
  the pre-op `from_torch` dtype quantization. Measured on the golden `(0.999, 0.02)`
  (PCC, relative-RMS) bar across all 8 golden shapes (incl. non_tile_aligned 1×1×48×64):
  - RM bfloat16: PCC ≈ 0.999+ (bf16 round-trip), well within tolerance.
  - RM float32: PCC ≈ 1.0.
  - bfloat8_b (TILE): PCC = 0.999971, relative-RMS = 0.0076 (host round-trip probe),
    matched by the on-device gather — comfortably clears 0.999.
- **Golden test progress**: **40 / 384** cells passing (was 16 / 384 at Phase 0) — the
  24 previously-xfail `gather_dim=-4, topology=Linear` cells for the three new format
  combinations (RM×bf16 = 8, RM×f32 = 8, bf8b×TILE = 8) all flipped to `supported_pass`.
  Verified on the multidevice WH sim (`run_multidevice_sim_pytest.py --op all_gather`)
  in `-k` chunks. Non-regression + no-drift confirmed on a 1×1×32×32 all-Linear sweep:
  TILE bf16/f32 gd=-4 still pass, and every gd∈{-3,-2,-1} cell still xfails with the
  correct reason (no xpass-drift, no xfail_wrong_mode). bf8b×RM cells correctly skipped
  (INVALID). Loud verifier categories (supported_fail / xpass_drift / xfail_wrong_mode)
  all 0.
- **Issues encountered**: None. The verifier-flagged risk (`bf8b × non_tile_aligned`
  block-float sub-tile edge) did not occur — non-aligned bf8b gives identical PCC to
  aligned bf8b because the op never re-tilizes.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/all_gather/test_all_gather_formats.py` (**new**):
    format/layout matrix — RM×{bf16,f32}, bf8b×TILE, plus TILE regression, across
    tile-aligned and non-tile-aligned shard shapes, asserting output dtype+layout are
    preserved. bf8b×RM omitted (INVALID).
  - `test_all_gather_extended.py` (**updated**): the `validate()` rejection test now gates
    on `topology=Ring` (still unsupported) since ROW_MAJOR is now SUPPORTED.
