# Metal 2.0 Port Report — `experimental/transformer/rotate_half`

## Outcome

**PORTED.** The op's single program (`create_descriptor` on the device op →
`RotateHalfProgramFactory::create_program_artifacts`) is now on `ProgramSpecFactoryConcept`. The confirmed test set passes
7/7 with Watcher on (`TT_METAL_WATCHER=10`) and the Metal 2.0 legality checks forced:
- `tests/tt_eager/python_api_testing/unit_testing/misc/test_rotate_half.py`: 5/5, including `test_rotate_half_program_cache`,
  which moves the input address across cache hits and asserts exactly one cache entry
- `tests/ttnn/integration_tests/falcon7b/test_falcon_rotary_embeddings.py::test_functional_rotary_embeddings`: 2/2

Both `METAL2_CHECKS_FORCED` markers (`program_spec.cpp`, `program_run_args.cpp`) appeared in the run log, 8 times each.
The legacy baseline on the same card, Release build, was also 7/7. The tests compare with `torch.equal`, so they confirm
bit-exactness for the compute code-path change described below.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
  (from the `/localdev/edwinlee/Port_Recipe` checkout; this workspace has no copy of the docs).
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **TTNN gate waiver (inherited from the audit):** the readiness-sheet row is stale. It predates #57409 and shows
  `Concept = legacy device-op` plus a phantom `RotateHalfProgramFactory`. The user waived the gate on 2026-10-07
  ("waive"), and the sheet refresh is still owed by its owner. The port now *creates* a nested struct named
  `RotateHalfProgramFactory` (exception 3 naming), so the row's factory name becomes right again. Its `Concept` and
  definition path (`…_program_factory.hpp`, deleted) are still wrong.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. There was no `override_runtime_arguments`. The framework's tensor-binding
refresh (`input` → reader `tensor::src`, `output` → writer `tensor::dst`) is the whole cache-hit refresh, which matches
the legacy `Buffer*` RTA patching. All other RTAs derive from the hashed padded shape.

### Device-op-class edits
- **Exception 3 (direct-descriptor op):** added the nested `RotateHalfProgramFactory` with `create_program_artifacts`
  and `using program_factory_t = std::variant<RotateHalfProgramFactory>` (`device/rotate_half_device_operation.hpp`).
  The cache-hit comment moved with the method. Its wording changed from "runtime-arg bindings" to "tensor bindings".
  Includes: dropped `program_descriptors.hpp`, added `<variant>` and `ttnn/metal_v2_artifacts.hpp`. Nothing else in
  the class changed, and the TT_FATAL census is identical per file.
- Pybind entry points removed: none (`rotate_half_nanobind.cpp` binds only the user function).
- Custom `compute_program_hash`: none.

### Open items
- Relaxation candidates: none.

## Handoff points

- **Direct-descriptor → program factory (exception 3)**: recorded above. The change is structural but mechanical.
- No capitulations, boundary-rule violations, kernel-lib gaps or framework gaps.

## Successes

- **Brief's define translation for the stale fork.** The audit worked out that `bcast_hw_metal2.cpp` reads
  `BCAST_OP` / `BCAST_LLKOP` / `BCAST_DIM`, not the `CHAIN_*` pair that rotate_half passed to the legacy kernel. It
  pointed at `bcast_op_utils::get_defines(HW, MUL)` + `BCAST_SCALAR`, mirroring the bcast HW factory. The compute spec
  compiled and ran bit-exact on the first build. Binding the fork with the legacy defines would not compile (not tried):
  the fork references `BCAST_LLKOP` / `BCAST_DIM` / `BCAST_OP` unconditionally.
- **Compiler options rule 2.** `grep -n opt_level` on the legacy factory printed nothing. The rule ("absent ≠ no
  setting; compute resolves to O3") made the explicit `O3` on the compute spec mechanical (`rotate_half_program_factory.cpp`, compute `KernelSpec`).
- **Whitelist rule 7 / §A keyed on the declaration.** Both DM kernels declared `const uint32_t tile_bytes =
  get_tile_size(cb_id)` (non-`constexpr`), so these became the member getters `dfb_in_no_mul.get_tile_size()` /
  `dfb_out_no_mul.get_tile_size()`. The `DataflowBuffer` constructions moved up a few lines so the object exists first.
- **Self-audit's `emplace_runtime_args` / `Buffer*` search.** The legacy addresses arrived only as `Buffer*` through
  `emplace_runtime_args`. A search for `->address()` alone would have found nothing even before the port.

## Friction

- **Gap (repeat of the nlp_kv_cache_load_slice report):** the recipe's DM snippet `create_reader_datamovement_config(device->arch())`
  doesn't match the helper, which takes no arch argument. I used `create_reader_datamovement_config()` /
  `create_writer_datamovement_config()`.
- **Confusion (minor): one DFB, two host names.** Legacy CB 2 was named `src_no_mul_cb_index` on the reader side and
  `output_no_mul_cb_index` (an alias variable) on the writer side. The brief said to declare one spec. I named it
  `no_mul`, with per-kernel accessor names `in_no_mul` / `out_no_mul` matching each kernel's legacy local. Nothing in
  the recipe covers "two host variables, one CB index". The patterns catalog could say that the spec name is a free
  choice and accessor names follow each kernel's vocabulary.
- **Workflow (not a doc issue).** Another port session (`nlp_create_qkv_heads_falcon7b`) left uncommitted edits in the
  shared main checkout during this port, and the `build` symlink had been flipped to `build_Debug` beforehand. Both
  builds here therefore compiled their in-progress tree too, and each was a full 963-step rebuild. The builds were
  clean. This port's commit contains only rotate_half paths.

## Open items for downstream

- **Shared kernel touches.**
  - (a) `ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_hw.cpp`.
  - (b) **Reused the existing `_metal2` fork** `bcast_hw_metal2.cpp` (rung 1). No new file was created, and neither
    the fork nor the original was edited.
  - (c) Remaining legacy binders of the original: **none**. rotate_half was the last; the bcast HW factory already
    binds the fork. **The legacy `bcast_hw.cpp` can be sunset.** Note for the bcast owners: the fork never got the
    `kernel_lib` `eltwise_chain` migration (#55584, `8cf12808550`) that the original received after the fork was cut.
    Once the original is deleted, the fork's hand-rolled `bcast_init` / `mul_tiles_bcast` loop is the only copy. The
    original's comment ("this file serves the consumers still on the legacy API") is now stale. I left it, because
    rung 1 forbids touching the original.
  - Consequence for this op: rotate_half's compute body moved from `eltwise_chain` to the fork's loop. It is the same
    math (scalar-broadcast multiply by −1; the scalar tile is waited on once and never popped). The test compares with
    `torch.equal`, and the output is bit-exact for bf16 on both test shapes.
  - Reader and writer are op-owned and only rotate_half binds them, so they were converted in place.
- **Pre-existing anomalies, preserved unchanged (from the audit, confirmed while porting):**
  - The scalar DFB (`src_scalar`) is declared with the input's data format but a `Float16_b`-sized entry, and the reader
    always writes bf16 bits. Non-bf16 inputs (fp32, bfp8_b) are untested and likely wrong. `validate_on_program_cache_miss`
    does not restrict dtype. The Metal 2.0 validator accepted the bf16 case; non-bf16 was not exercised.
  - Reader comment "Fill tile with zeros" (it writes one `-1.0` scalar), and the writer's "read a ublock of tiles from
    src to DFB" comment (it writes to dst). I only changed `CB` → `DFB` in that wording.
- **Tidy-up for a later pass (not port work):** reader and writer `start_id` are always `0` (single core), and the
  compute `B` / `Ht` are constant `1`. On a single node none of these is meaningfully a CRTA candidate. Left as RTAs.
- **Test coverage.** No gtests or sweeps call `ttnn.experimental.rotate_half`. Coverage is the two pytest files above,
  bf16 only, both single-core shapes. The user confirmed this set on 2026-10-08.
