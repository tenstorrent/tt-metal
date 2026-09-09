# Claude review round 1 — 2026-09-08

Reviewer: `claude-opus-5`, high effort, fresh session
`2e7c9a07-1c84-49dc-a22b-832690a45b8c`. Verdict: **CHANGES REQUIRED**.
The full original review and CLI transcript are preserved under
`generated/reduce_migration_reviews/round_01_20260908/`.

| Finding | Resolution / validation status |
| --- | --- |
| R1: three new device headers missing from runtime install | Added the Moreh layer-norm forward/backward and attention-softmax header globs. Native build passed. Isolated `ttnn-runtime` installs of both operation packages contained all three headers under `/tmp/reduce-review1-install-20260908-5kpf6zlc/libexec/tt-metalium`. |
| R2: full-sync H negate reader mismatch | Reader and negate compute now share the host chunk size, including the width-sharded one-column FIFO. The planned H hardware configuration now follows the compute full-sync setting. T178 exercises both sync modes and both FP32 destination settings. |
| R3: implicit changes to format reconfiguration | Restored NONE for dot and native groupnorm calls; INPUT for RMS all-gather, sampling, small attention MAX, and same-format generic/Quasar calls. Groupnorm Add retains INPUT_AND_OUTPUT because it changes the operand pair from input/scaler to input/input. Mixed-format generic row-major calls retain output reconfiguration after tilize. These exceptions are documented at construction. |
| R4: scalar-bit recipe discriminator | Replaced it with explicit worker-role flags: `is_second_stage_reader` for RMS all-gather and `skip_global_scale` for sharded layernorm. Removed the layernorm context's unused packed cross-core scalars. |
| R5: BGE test untracked | Already committed in `7d4229a0b76`, after the review started; verified tracked at the review checkpoint. |
| S1: lost H column grouping | Restored each per-core rectangle rather than treating every column as a batch. The existing two-tile FIFO uses native per-tile consumption while independent column accumulators remain in DEST; it cannot satisfy a grouped multi-row Add wait. The reader uses the plan's output group width. Quasar receives the same correction. |
| S2: packer-mask setup in output loop | Moved additive reduction mask setup outside the output loops, with one matching clear per call. All 111 helper cases, including exact-zero padding checks, passed. |
| S3: shape contracts only checked in kernels | Added host fatal checks for RMS worker/post-stat counts, sharded layernorm logical/full/tail widths and runtime dispatch, and groupnorm nonempty/full/tail/empty block geometry. |
| S4: rowvalid alias points to removed CB | Aliases to the allocated input-mask CB c_7 when row masking is disabled. |
| S5: missing other-architecture JIT evidence | Blackhole mock JIT checks compiled the helper matrix, KDA, indexer, sparse attention, attn-res and DiT programs without compiler errors. Quasar mock attempts failed before kernel compilation, as detailed below. Numerical validation still requires the corresponding accelerator and mesh. The user has been asked for available runner configurations. |
| S6: Falcon dropped BF8 attention-mask coverage | Restored `bfloat8_b` in the local test configuration. |
| S7: source-composition key no longer describes kernel | Renamed the key to `fused_distributed_rmsnorm_post`, retaining the existing `rms_ln` test case ID. |

The review accepted the new BF16 gradient relative-L2 check based on the
recorded legacy comparison and unchanged upstream tests. Its optional
single-sample maximum-error assertion was not adopted: the parameterized
suite's aggregate bound and unchanged upstream elementwise tests remain the
regression contract.

The review also confirmed all inventoried consumers were migrated, all five
retired kernels were unreferenced, and the inspected argument offsets, CB
bindings and auxiliary depths agreed. The branch fixes logical-width RMS
normalization/padding, explicit empty groupnorm output, narrow SFPU H planning,
and two previously uncompilable BGE writer branches, as recorded in the journal.

Follow-up on R3: the blanket native-NONE policy also affected the newly introduced
sharded groupnorm mean, which does not inherit the old manual accumulation's
format setup. Additional checks on 2026-09-09 exposed this error. Its native mode
now explicitly reconfigures inputs after masking; see
`review_followup_groupnorm_2026-09-09.md` for reproduction and validation.

Validation:

- The complete available N300 sanity selection passed **61/61, no skips**, after
  the review fixes (`reduce-migration-mvkrxglb`).
- `cmake --build build --target ttnn unit_tests_ttnn --parallel 8` passed
  (`/tmp/reduce-review1-build-20260908.log`).
- T178: **42/42 passed**, `reduce-migration-vp33d1m3`.
  The test uses exact BF16 addends for SUM to isolate ordering from cancellation,
  random extrema for MAX/MIN, and partial heights for interleaved layouts.
  Repeated width-sharded batches use a tile-aligned height because the existing
  fill-pad operation rejects non-tile-aligned multi-batch shards before reduction.
- T152: BF8-mask Falcon case passed; T175: **111/111 helper cases passed**,
  `reduce-migration-5q2tigkm`; T176 source composition also passed (113 total).
- Blackhole mock compilation: recurrence compiled **15 programs, zero errors**;
  indexer, sigmoid-gated RMSNorm, sparse SDPA/MSA and the helper matrix compiled
  **121 programs, zero errors**. Logs:
  `/tmp/reduce-review1-blackhole-precompile-20260908.log` and
  `/tmp/reduce-review1-blackhole-matrix-precompile-v2-20260908.log`.
  These are JIT compilation checks only; NO_DISPATCH numerical assertions are
  deliberately not evaluated as correctness evidence.
- Attn-res Blackhole 2x4 mock compilation built **12 programs, zero errors**.
  The precompile plugin marked its result incomplete because its single-device
  count check compared 12 compiled mesh artifacts to 5 unique operations. The
  wrapper still exited 0; neither its status nor the swallowed numerical checks
  are presented as a passing device test.
  Log: `/tmp/reduce-review1-attnres-precompile-20260908.log`.
- Quasar mock setup initially lacked dispatch-engine cores. Enabling the shipped
  interim Tensix dispatch configuration then failed its 28-bank L1 allocator
  validation and crashed before any test body. Slow dispatch opens the mock device,
  but ordinary input preparation uses unsupported legacy data-movement kernels.
  A temporary single-chip shape-only input adapter advanced to factory creation:
  prefill failed with `std::get: wrong index for variant`, and decode rejected the
  existing `out_o` multi-binding, which is only supported on WH/BH. **No Quasar
  kernels compiled.** These failures preceded operation capture; the plugin
  swallowed them, so its pytest status is not a passing-test result.
  Logs: `/tmp/reduce-review1-quasar-precompile-v3-20260908.log` and
  `/tmp/reduce-review1-quasar-precompile-v5-20260908.log`. The decode multi-binding
  code is unchanged by this migration, and both selected model tests explicitly
  construct a Wormhole compute configuration. These attempts do not substitute
  for running the Quasar lane on a supported Quasar setup.
- T026/T031 boundary rerun: **60 passed, 10 upstream skips**,
  `reduce-migration-xo3l2yrq`. Corrected the earlier journal's mistaken counting
  of those skips as passes. T072 N300 RMS all-gather: **4/4 passed**,
  `reduce-migration-j62gkd5k`.
- Blackhole mock DiT TP1 compiled in the first 5-program/zero-error run; TP2
  needed the documented `WAN_GALAXY_LINKS=2` setting. Its rerun compiled
  **4 programs, zero errors**, including `dit_rmsnorm_fused_compute`; the mesh
  count again exceeds the 3 unique operations and the plugin labels it incomplete.
  Logs: `/tmp/reduce-review1-dit-precompile-20260908.log` and
  `/tmp/reduce-review1-dit-precompile-v2-20260908.log`.
