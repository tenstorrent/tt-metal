# Metal 2.0 Port Report — `experimental/transformer/nlp_create_qkv_heads_vit`

## Outcome

**PORTED.** `NlpCreateHeadsVitDeviceOperation` now runs through a nested `NlpCreateQkvHeadsVitProgramFactory::create_program_artifacts` on `ProgramSpecFactoryConcept`. Both op kernels are converted. The live path passes every test that exists (see [Verification](#verification)). The compile-time-dead `transpose_k_heads` path is carried over but untested, because nothing can reach it.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(from the `Port_Recipe` checkout; the Metal_Ports checkout carries no recipe docs)*
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## Verification

- **Build:** `./build_metal.sh --build-tests` succeeded with no errors (main checkout, `build_Release`).
- **Legality checks:** all 9 `skip_validation` sites in `tt_metal/impl/metal2_host_api/` were forced to `false`. These are uncommitted working-tree scaffolding and are not in this diff. Both markers appear in the test log, 18× each: `METAL2_CHECKS_FORCED (program_spec.cpp:3670)` and `(program_run_args.cpp:673)`.
- **Watcher:** `TT_METAL_WATCHER=10`. The watcher server initialized with no disabled features, and nothing tripped.
- **Tests:** `pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_vit.py -v`: **17 passed** in 14.2 s on the 8×8 Wormhole.
  - 16 `test_nlp_create_qkv_heads_vit_test` cases: seq 224 / 4k × BFLOAT8_B / BFLOAT16 × in0 DRAM / L1 × out DRAM / L1.
  - `test_nlp_create_qkv_heads_vit_with_program_cache`: it asserts 2 cache entries, so cache-hit tensor-binding refresh and unchanged cache keying are both exercised.
- **Test set:** found by a sweep of `tests/` and `models/` for the op name. This file is the op's only coverage; there are no gtests, sweeps or model callers. The set was not separately confirmed with the invoker; the brief names the same file.
- **Not verified:** the dead `transpose_k_heads` path (compute specs, `k_in` / `k_out` DFBs, `TRANSPOSE_K_HEADS` kernel arms). It is unreachable: `const bool transpose_k_heads = false;`.
- **Self-audit** (10 code files scanned): zero hits for buffer-address / `Buffer*` / `emplace_runtime_args`, `TensorAccessorArgs`, `cb`-shaped names / `CircularBuffer` / `CBDescriptor`, positional `get_compile_time_arg_val` / `get_arg_val`, varargs, `.id`, `allow_instance_multi_binding`, and `.md` citations from code.
  - TT_FATAL/ASSERT census unchanged: device_operation.cpp 10, program_factory.cpp 6.
  - The diff touches no `tt_metal/` file.
  - `opt_level`: both dead-path compute specs are built by one lambda that sets O3 (`program_factory.cpp:109`). The DM specs keep the O2 default, matching legacy.
  - `hw_config`: reader and writer use the reader / writer default helpers, the same values as `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`. Compute uses `ComputeHardwareConfig{}`, the same values as `ComputeConfigDescriptor{}`.

## TTNN ProgramFactory

- **Concept realized:** `ProgramSpecFactoryConcept`, as the audit chose. No `override_runtime_arguments`; the four tensor bindings (`input`, `q`, `k`, `v`) are the whole cache-hit refresh, the same as the legacy descriptor's address patching.
- **Device-op-class edit forced (exception 3, direct-descriptor op):** the device-op-level `create_descriptor` (`device/nlp_create_qkv_heads_vit_device_operation.hpp:21-27` pre-port) moved into a nested `struct NlpCreateQkvHeadsVitProgramFactory`. The header gained `using program_factory_t = std::variant<NlpCreateQkvHeadsVitProgramFactory>;`, and its cache-hit comment moved onto the struct. The factory takes its pre-#57409 name, as the brief asked. The rest of the device-op class (`validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`, `nlp_create_qkv_heads_vit`) is byte-identical.
- **Custom hash:** none, so nothing to leave intact.
- **Pybind entry points removed:** none. `nlp_create_qkv_heads_vit_nanobind.cpp` binds only the user-facing function.
- **TTNN gate waiver (inherited):** the readiness-sheet row is stale. It still reads `legacy device-op`, has a phantom `NlpCreateQkvHeadsVitProgramFactory` row and says `yes (with PD step)`. The user waived the gate on 2026-10-07 ("If it's just the stale sheet, waive it"). The sheet refresh is still owed to the readiness-sheet owner, and should now read `ProgramSpecFactoryConcept` / ported.

## Handoff points

- **Device-op-class edit (exception 3).** Recorded above. Not a ticket, just the prominent record the recipe asks for. No user-visible API change.
- **Readiness sheet owner:** refresh this op's row (see the waiver above).

No capitulation, no boundary-rule violation, no kernel-lib gap.

## Successes

- **Recipe §Legacy inventory → "where the factory methods live".** Recording the direct-descriptor shape up front made the exception-3 restructure a planned step rather than a concept error at first build (recipe §Legacy inventory, first bullet).
- **Recipe §Compiler options.** The dead-path compute specs carry an explicit `opt_level = O3` (`program_factory.cpp:109`). The legacy `ComputeConfigDescriptor{}` never mentioned it, which is exactly the absent-line case the section warns about.
- **Recipe §Hardware configuration, Style B.** The legacy compute config was `ComputeConfigDescriptor{}` with all defaults. The port builds `ComputeHardwareConfig{}` directly rather than going through `to_compute_hardware_config`, whose high-performance defaults would have flipped fields. The sibling `nlp_create_qkv_heads` sets `enable_32_bit_dest` for Float32 here. That is *its* legacy behavior and was deliberately not copied: this op's legacy never set it.
- **The brief's "two wrappers, one DFB" watch-for** caught the only non-obvious kernel point. Both kernels construct `dfb_qv` and `dfb_k` from the one `dfb::qv` token on the live path.

## Friction

- **Gap — compile-time-dead branches.** The audit already raised this (its Recipe notes). The recipe has no rule for a branch behind a hard-coded `const bool … = false`. I carried it faithfully: the user answered the audit's Question 1 on 2026-10-07 with "Keep the behaviour unchanged, stick with the default". The cost: about 70 lines of host spec plus two kernel `#ifdef` arms that no test can reach (`program_factory.cpp:98-137,224-256,284-286`). A one-line rule ("carry / drop / ask") would save the next porter the question.
- **Confusion — the brief vs rule 5 on the dummy `in1_tensor_addr`.** The brief says "carry them as plain named args with value `0`; they are not a tensor binding". That is right for `in1_tensor_tile_id`, a tile index, which is carried. But `in1_tensor_addr` is a buffer base address, and its only consumer is `TensorAccessor(in1_args, in1_tensor_addr)` under the never-defined `READ_FROM_INPUT_TENSOR_KV`. Carrying it would put an address-shaped named RTA in the spec (rule 5, and the "No buffer address survived in the run-args" self-audit). I dropped it as Dropped Plumbing instead. The `#ifdef` arm now reads `TensorAccessor(tensor::input_kv)` (reader `:29-31`), the same way the already-ported `nlp_create_qkv_heads` reader treats this lineage. Zero functional change: no compiled configuration reads the value. Suggest the audit's "dummy RTA" wording separate address slots from scalar slots.
- **Confusion — dead `#ifdef` arms naming non-existent bindings.** Under `READ_FROM_INPUT_TENSOR_KV` the reader references `tensor::input_kv`, which this op never declares, so that arm would fail to compile if someone defined the macro. Legacy had the same latent shape: `in1_args` read a `TensorAccessorArgs()` placeholder that was never filled. I kept the arm rather than delete it (rule 8, comment and structure preservation). The recipe could say whether a never-defined macro arm should be kept, translated, or deleted.
- **Tooling — worktree git guard.** In this job's isolated worktree, compound shell commands containing `git` (e.g. the recipe's own TT_FATAL census one-liner `diff <(git grep …) <(git grep …)`) were refused as "too complex to verify". I had to split them into separate commands. Not a recipe issue, but the census snippet can't be pasted as written in this environment.
- **Process — duplicate instances of one job.** At least three live instances of the same background job (same job id and transcript) worked in the same worktree at once. They overwrote each other's plan, kernel and factory files within seconds of each other. Two stood down and one committed the WIP. After the user confirmed that only one instance remained, the surviving instance re-read every file, checked the factory ↔ kernel binding names (`tensor::input` / `q` / `k` / `v`, `dfb::qv` / `dfb::k`, named args), and then built, tested and committed. All the competing drafts were faithful ports and differed only in names. Worth knowing for whoever schedules these jobs.

## Open items for downstream

- **Shared kernel touches.**
  - (a) `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`.
  - (b) Rung: **reused the existing `_metal2` fork** `ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp`. No new file, no change to either copy. This op binds the fork only on its dead `transpose_k_heads` path.
  - (c) Remaining legacy consumers of `transpose_wh.cpp`: `experimental/transformer/nlp_create_qkv_heads_boltz`, `experimental/transformer/split_query_key_value_and_split_heads` (per the audit's team-only section).
- **Dead `transpose_k_heads` branch** (`program_factory.cpp:98-137,224-256,284-286`; reader/writer `#ifdef TRANSPOSE_K_HEADS` arms). It is untestable as written. The ops team should either delete it or plumb it through as a real attribute the way `nlp_create_qkv_heads` does. The port carries it so that decision stays theirs.
- **Dead `READ_FROM_INPUT_TENSOR_KV` arms and the `in1_tensor_tile_id` RTA** (reader `:19,29-31,58-66,75-83`). This is dead plumbing from the `nlp_create_qkv_heads` lineage. A cleanup PR could remove the arms and the always-`0` RTA together.
- **Hard-coded geometry** (`program_factory.cpp:37-51`): `q_num_tiles_per_tensor = 24`, 12 heads, `q_out_w_tiles = 2`. It is consistent with validation (`[B,1,S,2304]` only) but would silently mis-split a relaxed shape (audit misc anomaly, carried forward).
- **`compute_output_specs` sharded path** (`device_operation.cpp:53-55`) returns `{}` after `TT_ASSERT(false)`. Release builds would hand an empty spec vector to `create_output_tensors`. Unreachable today because validation rejects non-interleaved output (audit misc anomaly; device-op class off-limits, so not touched).
- **Quasar uplift — two `DataflowBuffer` objects on one DFB.** On the live path both kernels build `dfb_qv` and `dfb_k` from the same `dfb::qv` token (reader `:33,37`, writer `:35,39`), mirroring the legacy pair of `CircularBuffer`s on index 1. On WH/BH the two objects share one `LocalCBInterface`, so this behaves exactly like legacy. On Quasar, though, each `DataflowBuffer` object carries its own implicit-sync counters (`dataflow_buffer.h`, `ptxn_*` / `ctxn_*` members), so two objects interleaving FIFO calls on one DFB may not be equivalent there. The sibling `nlp_create_qkv_heads` port uses a `DataflowBuffer& dfb_k = dfb_qv;` alias instead. The Quasar pass should switch to that alias or confirm the two forms are equivalent.
- **Test coverage.** The only test is `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_vit.py`. No gtest, sweep, or model caller in-tree exercises this op. The dead transpose path has no coverage at all.
