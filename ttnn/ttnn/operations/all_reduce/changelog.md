# Changelog: all_reduce

## Phase 0 — Core Implementation
- **Date**: 2026-07-25
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Self-contained Python CCL **+ compute** op on
  `ttnn.generic_op` + `ttnn.MeshProgramDescriptor`, with newly-authored reader
  (NCRISC) / compute (TRISC) / writer (BRISC) kernels. Algorithm: **broadcast-all
  then local N-way sum** — every device duplex-line-MULTICASTs its shard to all
  peers from one worker core (last packet a fused write+atomic-inc with
  `flush=true`, so a peer's arrival costs no extra packet), then folds its own shard
  plus the N-1 received slots with pairwise `add_tiles` in a single DEST register
  (`ceil(N/2)` FPU ops per output tile, odd-N handled via a `copy_tile` seed).
  Wraps no existing `all_reduce` / `reduce_scatter` / `all_gather`.
- **SUPPORTED at Phase 0**: `dtype=[bfloat16, float32]`, `layout=[TILE]`,
  `topology=[Linear]`, `alignment=[tile_aligned]`; `EXCLUSIONS=[]`. Interleaved DRAM
  or L1; 1-D line mesh `(1, N)`, `N ≥ 2`; single worker core per device.
  **`TARGET − SUPPORTED` is empty on every axis** — Phase 0 covers the full declared
  universe.
- **Accuracy achieved**: bfloat16 PCC=0.999994, max_abs=0.0625, mean_abs=0.0062,
  rel_rms=0.0037; float32 PCC=0.99999996, max_abs=0.0060, mean_abs=0.00098,
  rel_rms=0.00044. Measured on 4 shapes × 2 dtypes (8 cells) via
  `test_all_reduce_precision_baseline.py` against an fp32-accumulated torch oracle,
  N=8 addends. Error is shape-independent (a streaming per-tile fold: depth is N,
  not P). The fp32 rel_rms of 2⁻¹¹ is the Wormhole FPU's 19-bit SrcA/SrcB operand
  format, not an accumulation defect (DEST *is* fp32 via `fp32_dest_acc_en`).
- **Golden suite at Phase 0**: **6 / 6 registry cells passing** —
  `supported_pass=6`, `xfail_expected=0`, `invalid_skipped=0`
  (`INVALID = []`), and all five loud categories **0**
  (`eval/results/all_reduce/verifier_report.json`). Plus 5/5 `test_translated.py`
  cells passing and the `topology=Ring` translated cell correctly xfailing with
  `UnsupportedAxisValue`. Unit-test directory: **21/21**. All runs on the
  deterministic craq-sim WH multi-device runner
  (`--op all_reduce` → `wh_t3k_allmmio_all_reduce`, mesh `(1,8)`, `FABRIC_1D`),
  aggregate exit 0.
- **Issues encountered** (all fixed by the verifier, all re-verified green):
  1. **Latent cross-device semaphore reuse.** The op-internal `GlobalSemaphore` was
     cached in a module-level `{id(mesh_device): sem}` dict that outlives the
     device; `MeshDevice` has no weakref support and CPython reuses freed
     addresses, so a fresh mesh could be handed a closed device's semaphore. Now
     bound to the device object itself
     (`mesh_device._ttnn_all_reduce_recv_semaphore`), which cannot be aliased
     across devices. Still one creation + one `synchronize_device` per mesh, still
     parked on the descriptor, still no per-call barrier.
  2. **`validate()` ordering.** The per-axis `SUPPORTED` gate ran *after* the
     dtype-dependent framing gates (`page_size % l1_alignment`,
     `ccl_packet_dims(...).page_segments`), so a future out-of-SUPPORTED dtype could
     have been refused with `ValueError` instead of `UnsupportedAxisValue`
     (→ `xfail_wrong_mode`). The axis + `EXCLUSIONS` gate now runs before them.
  3. **Helper under-use in the writer.** Plain fabric payload pages hand-rolled
     `addrgen_detail::get_noc_address(...)`; they now use the duplex tier's
     `DuplexWriteChannel::write_page(l1, page_idx, accessor)` convenience (the
     `all_gather` precedent). The one remaining manual resolution is on the fused
     page, where the channel has no page overload, and is commented as such.
  4. **Dead code / clarity**: removed unused `_num_line_devices()`; typed
     `output_tensor: ttnn.Tensor | None`; replaced the bare `program.kernels[2]`
     with a named `_WRITER_KERNEL_IDX`.
  No correctness bug was found in the algorithm, kernels, CB balance, routing or
  compute fold; SUPPORTED was **not** widened (`xpass_drift = 0`) and no
  `EXCLUSIONS` entry was needed.
- **Coverage holes closed** (new `test_all_reduce_extended.py`, 3/3 green):
  odd-N compute fold via a `(1,3)` submesh (the `seeded` branch was dead code on an
  even-N mesh, and it is exactly the branch the shipped C++ reference gets wrong);
  L1-interleaved memory (accepted by `validate()` but previously untested);
  back-to-back calls with no intervening host sync (design Risk 5 — probed, does
  **not** reproduce on the deterministic sim; recorded as residual risk).
- **Tests added**: `test_all_reduce.py` (acceptance, 10 — implementer),
  `test_all_reduce_precision_baseline.py` (8 — implementer, run + tabulated by the
  verifier), `test_all_reduce_extended.py` (3 — verifier).
- **Refinement queue**: 2 entries (`op_requirements.md`) — R1 `Ring` topology (the
  only named failing cell), R2 `non_tile_aligned` alignment. Short by construction:
  `TARGET − SUPPORTED` is empty, so the queue is anchored on the translated Ring
  refinement cell and the tagger-only `alignment` axis rather than on a TARGET gap.
