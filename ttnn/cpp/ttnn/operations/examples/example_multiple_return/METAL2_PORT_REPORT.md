# Metal 2.0 Port Report — `examples/example_multiple_return`

## Outcome

**PORTED** — the op's single program (`ExampleMultipleReturnProgramFactory`, previously the direct `create_descriptor`) now targets `ProgramSpecFactoryConcept`. Verified on the 8×8 Wormhole b0 card with `TT_METAL_WATCHER=10` and forced Metal 2.0 legality checks. Both `METAL2_CHECKS_FORCED` markers fired (`program_spec.cpp`, `program_run_args.cpp`; 6 each, one per spec-built program). `tests/ttnn/unit_tests/operations/debug/test_examples.py`: **7/7 passed**. That is 3 presence combinations, 3 program-cache runs (one cache entry each, cache hits at moved addresses), and the sibling `test_composite_example`, which still binds the legacy `eltwise_sfpu.cpp` that received the pointer comment. The test set was not confirmed with the invoker: the broad `find tests -iname '*example*'` sweep plus a grep for `composite_example_multiple_return` turned up only this file, and the op has no gtests.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit waiver (inherited):** the TTNN factory concept gate was RED only because the readiness-sheet row predates PD batch #57409 (`f5093e705ae`). The row still lists `Concept` = `legacy device-op` and a phantom `SingleCore` factory. The user waived the gate on 2026-10-07 ("Okay waive it if it's just the stale row"). The sheet row still needs refreshing by its owner.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. There is no `override_runtime_arguments`. Per-dispatch state is just the tensor bindings (`src`, plus `dst1` / `dst2` when present), which the framework refreshes on cache hits. The tile-count RTAs come from the input spec, which is part of the program hash.

### Device-op-class edits
- **Exception 3 (direct-descriptor shape):** `device/example_multiple_return_device_operation.hpp`.
  - Replaced the static `create_descriptor` with `struct ExampleMultipleReturnProgramFactory { static ProgramArtifacts create_program_artifacts(...); }` and `using program_factory_t = std::variant<ExampleMultipleReturnProgramFactory>;`.
  - Swapped the `<tt-metalium/program_descriptors.hpp>` include for `"ttnn/metal_v2_artifacts.hpp"` plus `<variant>`.
  - Rewrote the comment above the method, which described the descriptor / `emplace_runtime_args` mechanism, for the spec shape.
  - Everything else in the class is untouched.
- **Pybind entry points removed:** none. `example_multiple_return_nanobind.cpp` binds only the composite.
- **Custom `compute_program_hash`:** none (default reflection hash), untouched.

### Open items
- Relaxation candidates: none.
- No capability gaps for this concept.

## Handoff points

- **Device-op-class edit (exception 3), TTNN owners FYI.** The op arrived in the direct-descriptor shape (`create_descriptor` on `ExampleMultipleReturnDeviceOperation`, no `program_factory_t`). It now has the conventional nested factory plus a one-alternative variant. This is the same shape as the earlier #57409-batch ports (`test_hang_device`, `concatenate_heads`, `create_qkv_heads_from_separate_tensors`).
- **Readiness sheet, owner.** The `examples/example_multiple_return` / `SingleCore` row is stale (see Provenance). After this port it should read `ProgramSpecFactoryConcept`, with factory `ExampleMultipleReturnProgramFactory`.

## Successes

- **Conditional / optional resource bindings** (`port_patterns.md`). The audit and the brief flagged the writer's `if (dst_addrN != 0)` raw-address presence test up front. The pattern's "optional tensors make the `#ifdef` mandatory" line led straight to the conditional `TensorBinding` + `RETURN_OUTPUT1` / `RETURN_OUTPUT2` define + `#ifdef` shape (`single_core_program_factory.cpp` writer block; `writer_multiple.cpp`). There was no temptation to keep an address-valued RTA as a flag.
- **Shared-kernel Caution, rung 1 and rung 2.**
  - The reader reused `reader_unary_interleaved_start_id_metal2.cpp` as-is. Its `dfb::in` / `tensor::src` / `num_pages` / `start_id` vocabulary fit without a rename on the kernel side.
  - The compute kernel got a new `eltwise_sfpu_metal2.cpp`. The "concurrent ports collide by design" note mattered: an unmerged branch (`origin/anasuya/metal2_port_eltwise_unary`, `229aab0b082`) already creates the same fork. This port adopts that file and its pointer comment byte-for-byte, so the merge is an identical add/add rather than a conflict.
- **Compiler options.** The `grep -n opt_level` check on the legacy factory printed nothing, which made the `O3` rule for the compute spec impossible to miss (`single_core_program_factory.cpp`, compute `KernelSpec`).

## Friction

### Gaps
- **Recipe's compute-config shape is stale for this tree.** `metal2_port.md` → *Hardware configuration / Compute kernels* talks about a `ComputeGen1Config` reached via `std::get<ComputeGen1Config>(compute_hw)`. In this checkout, `ComputeHardwareConfig` (`tt_metal/api/tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp`) is a flat struct with common fields plus an optional `config_1xx` (`bfp_pack_precision_mode`). There is no Gen1/Gen2 variant. The value mapping still held, and Style B was a direct `ComputeHardwareConfig{...}`, but the recipe's code snippets don't compile as written.
- **New legacy field not in the recipe's mapping table.** `ComputeConfigDescriptor` now carries `enable_trisc2_rvv` (`tt_metal/api/tt-metalium/program_descriptors.hpp:110`), which has no Metal 2.0 counterpart in the table. Here it is default `false`, so nothing to carry. A port whose op sets it would have no guidance.

### Confusion
- **Duplicate port sessions.** Two background sessions were started on this same port. They landed in the same worktree name (`.claude/worktrees/metal2-example-multiple-return`). The other session wrote `METAL2_PORT_PLAN.md` before noticing and stopping. This port kept that plan (its inventory was accurate) and edited the spec-name sections to match the implementation. No code was clobbered. This is a workflow issue, not a recipe one.
- **First commit aborted by the clang-format hook,** as the recipe warns. The only change was splitting the long reader-fork path literal across two string literals. It was a pure reformat, so no re-verification was needed.

## Open items for downstream

- **Shared kernel touches:**
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`: **rung 1, reused the existing fork** `reader_unary_interleaved_start_id_metal2.cpp`. No new file. Remaining legacy binders: `examples/example` (single- and multi-core factories), `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `reduction/topk` (`topk_route_prep_program_factory.cpp`).
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`: **rung 2, created the fork** `eltwise_sfpu_metal2.cpp` beside it. The pointer comment landed at the top of the original. Remaining legacy binders: `eltwise/unary` (`unary_op_utils.cpp:1212` → `unary_program_factory.cpp`), `examples/example` (single- and multi-core), `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:284`. The same fork also exists on the unmerged `origin/anasuya/metal2_port_eltwise_unary`. Content is identical, so whichever lands second resolves trivially.
  - `device/kernels/writer_multiple.cpp`: op-owned, not lent. Converted in place.
- **Presence-test semantics (behaviour-preserving, worth a reviewer's eye).** The legacy writer decided per output at runtime with `dst_addrN != 0`. The host passed a null `Buffer*` (emitting 0) for an absent output. The port moves that decision to compile time (`RETURN_OUTPUTN`), keyed on whether the output tensor exists. The two agree unless a present output were allocated at address 0, which the allocator never does. `return_output1` / `return_output2` are already in the default hash, so each presence combination was already its own cached program.
- **Pre-existing anomalies, left untouched** (from the audit, re-observed):
  - `writer_multiple.cpp` still includes `api/debug/dprint.h` without using it.
  - `operation_attributes_t::attribute` / `some_other_attribute` are hard-wired placeholders (`device/example_multiple_return_device_operation.cpp:63`) that feed the hash but are never read.
  - `return_output1` / `return_output2` are declared `uint32_t` but initialised from `bool` (`device/example_multiple_return_device_operation.hpp:22-23`).
  - The legacy factory computes `output_dtype` from whichever output exists (`single_core_program_factory.cpp`), assuming both share a dtype. They do, because both come from the same `TensorSpec`.
- **RTA → CRTA candidates (not converted):** none worth noting. The op runs on one core.
- **Test coverage:**
  - The op has no C++ gtest. Coverage is `tests/ttnn/unit_tests/operations/debug/test_examples.py`: `test_composite_example_multiple_return` (all three presence combinations, one 64×128 shape) and `..._program_cache` (three presence combinations; cache hits at moved addresses with fresh data).
  - Both outputs are checked with `assert_equal` (exact).
  - Nothing exercises a non-bfloat16 dtype or a row-major layout. The DFB `data_format_metadata` / `entry_size` derivations are therefore only tested for bf16 tiles.
