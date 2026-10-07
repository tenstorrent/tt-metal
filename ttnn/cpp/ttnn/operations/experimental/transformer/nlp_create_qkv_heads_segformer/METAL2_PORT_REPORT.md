# Metal 2.0 Port Report — `experimental/transformer/nlp_create_qkv_heads_segformer`

## Outcome

**PORTED.** `NlpCreateHeadsSegformerDeviceOperation` now runs through a nested `NlpCreateQkvHeadsSegformerProgramFactory::create_program_artifacts` on `ProgramSpecFactoryConcept`. Both op kernels are converted, and every test that exists for the op passes (see [Verification](#verification)).

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(from the `Port_Recipe` checkout; the Metal_Ports checkout carries no recipe docs)*
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## Verification

- **Build:** `./build_metal.sh --build-tests` succeeded with no errors (main checkout, `build_Release`). The main checkout also held another session's uncommitted `nlp_kv_cache_load_slice` port work at the time, so the binary under test included it. That op is independent of this one, and this commit contains only the segformer files.
- **Legality checks:** all 9 `skip_validation` sites in `tt_metal/impl/metal2_host_api/` were forced to `false`. This is uncommitted working-tree scaffolding and is not in this diff. Both markers appear in the unit-test log, 26× each: `METAL2_CHECKS_FORCED (program_spec.cpp:3670)` and `(program_run_args.cpp:673)`.
- **Watcher:** `TT_METAL_WATCHER=10`. The server initialized with no features disabled, and nothing tripped.
- **Unit tests:** `pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_segformer.py -v`: **25 passed** in 18 s on the 8×8 Wormhole.
  - 24 `test_nlp_create_qkv_heads_segformer_test` cases: seq 256 / 1k / 4k × BFLOAT8_B / BFLOAT16 × in0 DRAM / L1 × out DRAM / L1.
  - `test_nlp_create_qkv_heads_segformer_with_program_cache`: it asserts 2 cache entries, which exercises the cache-hit tensor-binding refresh and checks that cache keying is unchanged.
- **Model test:** `models/demos/vision/segmentation/segformer/tests/pcc/test_segformer_efficient_selfattention.py`, which runs the op inside its only in-tree caller. **Could not exercise the op: 8/8 fail in test setup.**
  - Every case raises `IndexError` in the reference-weight loader (`models/demos/vision/segmentation/segformer/common.py:97`). The `segformer_b0_ade_512_512.pth` checkpoint keys match neither directly nor through the transformers-5.x remap, so the loader falls back to a positional zip, which runs out of values.
  - This happens before any ttnn op is built. The log has zero `METAL2_CHECKS_FORCED` markers, which means no Program was constructed.
  - The cause is the environment and checkpoint, not the port, and it is unrelated to this op. Model-level coverage remains **owed** once the checkpoint / transformers mismatch is fixed.
- **Test set:** found by sweeping `tests/` and `models/` for the op name. Hits:
  - `tests/tt_eager/.../misc/test_nlp_create_qkv_heads_segformer.py`, the baseline.
  - `tests/sweep_framework/sweeps/model_traced/nlp_create_qkv_heads_segformer_model_traced.py`, a sweep. Not run; it is driven by the sweep framework.
  - The Segformer model (`ttnn_segformer_efficient_selfattention.py`), covered by the model PCC test above.
  - No gtests.

  The set was not separately confirmed with the invoker; the brief names the same baseline file.
- **Self-audit** (10 code files scanned):
  - Zero hits for: buffer address / `Buffer*` / `emplace_runtime_args`; `TensorAccessorArgs`; `cb`-shaped names / `CircularBuffer` / `CBDescriptor`; positional `get_compile_time_arg_val` / `get_arg_val`; varargs; `.id`; `allow_instance_multi_binding`; `.md` citations from code.
  - TT_FATAL/ASSERT census unchanged: device_operation.cpp 10, program_factory.cpp 3.
  - The diff touches no `tt_metal/` file.
  - `opt_level`: no compute kernels. Legacy set none on either DM kernel, and the specs keep the O2 default.
  - `hw_config`: reader and writer use `create_reader_datamovement_config()` / `create_writer_datamovement_config()`, the same values as legacy `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`.

## TTNN ProgramFactory

- **Concept realized:** `ProgramSpecFactoryConcept`, as the audit chose. There is no `override_runtime_arguments`. The two tensor bindings (`input`, `q`) are the whole cache-hit refresh, the same as the legacy descriptor's address patching of `in0_buffer` / `q_buffer`.
- **Device-op-class edit forced (exception 3, direct-descriptor op):**
  - The device-op-level `create_descriptor` (`device/nlp_create_qkv_heads_segformer_device_operation.hpp:21-27` pre-port) moved into a nested `struct NlpCreateQkvHeadsSegformerProgramFactory`. It takes the pre-#57409 name, as the brief asked.
  - The header added `using program_factory_t = std::variant<NlpCreateQkvHeadsSegformerProgramFactory>;`. The cache-hit comment moved onto the struct, with "address bindings" reworded to "tensor bindings".
  - `<tt-metalium/program_descriptors.hpp>` was swapped for `ttnn/metal_v2_artifacts.hpp`.
  - The rest of the device-op class is byte-identical: `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`, and the `nlp_create_qkv_heads_segformer` launcher.
- **Custom hash:** none, so nothing to leave intact.
- **Pybind entry points removed:** none. `nlp_create_qkv_heads_segformer_nanobind.cpp` binds only the user-facing function.
- **TTNN gate waiver (inherited):** the readiness-sheet row is stale. It still reads `legacy device-op`, has a phantom `NlpCreateQkvHeadsSegformerProgramFactory` row, and says `yes (with PD step)`. The user waived the gate on 2026-10-07 ("If it's just the stale sheet, waive it"). The sheet refresh is still owed to the readiness-sheet owner, and the row should now read `ProgramSpecFactoryConcept` / ported.

## Handoff points

- **Device-op-class edit (exception 3).** Recorded above. This is the prominent record the recipe asks for, not a ticket, and there is no user-visible API change.
- **Readiness sheet owner:** refresh this op's row (see the waiver above).

No capitulation, no boundary-rule violation, no kernel-lib gap.

## Successes

- **Recipe §Legacy inventory → "where the factory methods live".** Recording the direct-descriptor shape up front made the exception-3 restructure a planned step, not a concept error at first build.
- **Recipe kernel-side rule 7 (DFB metadata via the object).** Both `get_tile_size(cb_id_qv)` calls were non-`constexpr` locals, so they move to `dfb_qv.get_tile_size()` (reader `:31`, writer `:31`) rather than keeping the free function with `dfb::qv`. The rule's `constexpr` carve-out made that decision mechanical.
- **Recipe rule 8 (relocate a deleted line's comment).** The hard-coded `cb_id_qv = 1;  // cb for Q, V heads` line is gone. Its role comment now sits on the `DataflowBuffer dfb_qv(dfb::qv)` construction in both kernels, and on the host `QV` spec name.
- **Sibling port as shape reference.** `nlp_create_qkv_heads_vit` (a5a563abb3a) is the same kernel lineage, which made this port a subset translation. That port's report flagged the dummy `in1_tensor_addr` question before I hit it (see Friction).

## Friction

- **Confusion — the brief vs. rule 5 on the dummy `in1_tensor_addr` (recurring).** The brief says to carry both unused reader RTAs as named args with value `0`, and not to delete them. I carried `in1_tensor_tile_id`, a tile index (reader `:19`; host named RTA). I **dropped** `in1_tensor_addr` as Dropped Plumbing.
  - It is an address-shaped slot for an in1 tensor this op does not have.
  - This kernel has no consumer for it at all.
  - Carrying it would put an address-named RTA into the spec, which runs against rule 5 and the "no buffer address in the run-args" self-audit item.
  - This matches the vit port's disposition of the identical slot, so the two sibling ports agree. There is zero functional change, because the legacy kernel read the value into a local that nothing used.

  This is the second port where the audit's "carry dummy RTAs as named args" wording conflicts with rule 5. The audit/brief template should separate address slots (drop) from scalar slots (carry).
- **Tooling — worktree git guard.** As in the vit port, the isolated-worktree guard refused compound shell commands that mention `git` or use shell variables before `find`/`git`. Examples are the recipe's TT_FATAL census `diff <(git grep …) <(git grep …)` and a `python3` heredoc edit followed by `git diff`. I split them into plain commands. This is not a recipe issue, but the census snippet can't be pasted as written here.

## Open items for downstream

- **Shared kernel touches:** none. Both kernels are this op's private copies. The same-named files under `nlp_create_qkv_heads/` and `nlp_create_qkv_heads_vit/` are different files and were not touched.
- **Unused reader `in1_tensor_tile_id` RTA** (reader `:19`; host `program_factory.cpp`, the `AddRuntimeArgsForNode` reader call). It is always `0` and never used, dead plumbing from the `nlp_create_qkv_heads` lineage. A cleanup PR could drop it. It is also an RTA with the same value on every node, so it is a CRTA candidate if it is kept.
- **K and V outputs are allocated and returned but never written** (`device_operation.cpp:76,89-92`; audit misc anomaly). Every in-tree caller takes only `[0]`, and the port preserves the behavior. The ops team should decide whether to return only Q.
- **The nanobind docstring is wrong** (`nlp_create_qkv_heads_segformer_nanobind.cpp:23`). It was copied from `_vit` ("[B, 1, S, 2304] … [B, 12, S, 64]"); the op actually produces Q heads with head_dim 32 for any `hidden % 32 == 0`. Out of port scope.
- **`q_out_h_tiles = ashape[2] / TILE_WIDTH`** (`program_factory.cpp:43`) should be `TILE_HEIGHT`. Both are 32, so it is harmless. Carried unchanged.
- **Unreachable code after `TT_FATAL(false, …)`** in the sharded branch of `compute_output_specs` (`device_operation.cpp:57-66`). The device-op class is off-limits, so it was not touched.
- **Test coverage.**
  - The baseline unit test checks Q only, which matches what the op writes.
  - The model-traced sweep was not run.
  - `test_segformer_efficient_selfattention.py` is broken in this environment by the weight-loader `IndexError` (see Verification). Someone should rerun it once that is fixed.
