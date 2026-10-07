# Metal 2.0 Audit Findings — `examples/example_multiple_return`

- **`ExampleMultipleReturnDeviceOperation`** (`device/example_multiple_return_device_operation.hpp`)
  - *(no named factory)*: **direct-descriptor shape**. `create_descriptor` is a static member of the device-op itself (`device/example_multiple_return_device_operation.hpp:72`), with no `program_factory_t`. The body lives in `device/single_core_program_factory.cpp:12`. The framework runs it through the `DirectDescriptorFactory` shim. Before PD batch #57409 (`f5093e705ae`, 2026-09-25) this was the legacy `SingleCore` factory struct.
- Kernels bound:
  - reader: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` (borrowed, eltwise/unary)
  - writer: `device/kernels/writer_multiple.cpp` (own)
  - compute: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (borrowed, eltwise/unary)

**Scope:** TTNN op, Gen1 (WH/BH) target. Within scope of `audit/metal2_audit.md`. First audit of this op (no prior `METAL2_PREPORT_AUDIT.md`).

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/examples/example_multiple_return` |
| **Overall** | **GREEN (user waiver)**. The original finding was RED: the only blocker was a stale readiness-sheet row from PD batch #57409, waived by the user on 2026-10-07. Every other gate is GREEN. |
| **DOps / Factories** | `ExampleMultipleReturnDeviceOperation` → direct `create_descriptor` (sheet still lists `SingleCore`) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok (no function-call escapes; two borrowed kernel files, see Heads-ups) |
| *Feature Support* — overall | GREEN (all Appendix A rows N/A) |
| *Feature Support* — Variadic-CTA | Ok (none) |
| *TTNN Readiness* — `Is able to port?` (the gate) | Sheet cell `yes (with PD step)`, **but the sheet is broken for this op**: `Concept` conflicts with the code, and the `SingleCore` row is a phantom. → **GATE**, routed to the readiness-sheet owner. |
| *TTNN Readiness* — Concept (current) | Code: `descriptor` (direct-descriptor shape). Sheet: `legacy device-op` (stale). |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (needs a new `ExampleMultipleReturnProgramFactory` struct, per `ttnn_factory.md` exception 3) |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | `src` Case 1 · `dst1` Case 1 (optional) · `dst2` Case 1 (optional) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none: no accessor passes a 3rd arg |
| *Port work* — CB endpoints | legal (c_0 and c_2 are both plain 1P+1C FIFOs) |

## Result

**GREEN (user waiver) → brief issued.** On 2026-10-07 the user waived the stale-sheet gate below ("Okay waive it if it's just the stale row"). This was confirmed as the only RED. The original finding is kept unchanged below, and the sheet row still needs refreshing by its owner.

*Original finding:* **RED → blocked on the readiness sheet (spreadsheet-broken), routed to the readiness-sheet owner.** The sheet's row for this op predates PD-migration batch #57409. It still shows `Concept` = `legacy device-op` with a factory named `SingleCore`. That PR deleted the `SingleCore` struct and moved the op to a direct `create_descriptor` on the device-op. This is the only RED. Device 2.0, feature compatibility, offset base pointers and the TensorAccessor 3rd argument all clear. Once the sheet catches up, or the user waives this stale-row finding, the op is GREEN. This is a whole-op RED: there is only one factory, so no subset applies.

The blocker clears on the **sheet side, not in the op's code**. So per the recipe's Red-outcome exception, the informational subjects were run anyway: re-audit would read the same code.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED, spreadsheet-broken.** The sheet row (`Op` = `examples/example_multiple_return`, `Device operation` = `ExampleMultipleReturnDeviceOperation`, `Factory (variant)` = `SingleCore`) reads:
  - `Concept`: `legacy device-op`
  - `Op Classification`: `Legacy Op`
  - `Is able to port?`: `yes (with PD step)`
  - `Porting Target`: `ProgramSpecFactoryConcept`
  - `Custom hash`: `no` · `Backdoor custom hash`: `no` · `Runtime-args update (get_dynamic_runtime_args)`: `no` · `Override runtime args method?`: `n/a` · `Pybind descriptor`: `no` · `Smuggled pointer`: `no`
  - `Known op issues`: *(empty)* · `TensorParameter relaxation`: `none` · `Op-owned tensors?`: `no`

  Cross-check against the code:
  - **`Concept`: conflict.** The code is `descriptor`. `create_descriptor` returns a `ProgramDescriptor` and is declared on the device-op (`device/example_multiple_return_device_operation.hpp:72`), defined at `device/single_core_program_factory.cpp:12`. There is no `create()` and no `override_runtime_arguments`.
  - **Factory-set match: phantom row.** No `SingleCore` exists in the code. `f5093e705ae` (#57409) deleted `struct SingleCore` and `using program_factory_t = std::variant<SingleCore>`. The code's single program comes from the direct-descriptor shim, which has no row.
  - `Custom hash` `no`: matches (no `compute_program_hash`, no `attribute_values` / `to_hash`).
  - `get_dynamic_runtime_args` `no`: matches.
  - `Override runtime args method?` `n/a`: consistent (no `override_runtime_arguments`).
  - `Pybind descriptor` `no`: matches. `example_multiple_return_nanobind.cpp:13` binds only the composite function.
  - `Smuggled pointer` `no`: matches. Every address goes in as a `Buffer*` binding (`single_core_program_factory.cpp:117-118`).
  - Cross-column invariants: none violated.

  The `Is able to port?` cell is `yes (with PD step)`, and the PD step has since landed, so the sheet's own verdict likely flips to plain `yes` once the row is refreshed. Under the recipe, a primary-column conflict plus a phantom row is still "spreadsheet is broken", and that is a GATE. **Route:** readiness-sheet owner (Diego), to refresh the row for the post-#57409 shape. This is the same stale-row pattern already seen for the other #57409 ops (`test/hang_device`, `transformer/concatenate_heads`, `transformer/create_qkv_heads_from_separate_tensors`, `transformer/nlp_concat_heads_boltz`, `topk_router_gpt`).

- **Device 2.0 (every kernel used):** **GREEN.**
  - `writer_multiple.cpp` (own): `Noc` (`:24`), `CircularBuffer cb_out` (`:25`), `TensorAccessor` (`:27-28`), `noc.async_write` / `async_write_barrier`, and `cb_out.wait_front` / `pop_front`. No CB-index free-function holdovers and no addr-gen.
  - `reader_unary_interleaved_start_id.cpp` (eltwise/unary): `Noc`, `DataflowBuffer`, `TensorAccessor`. Its one CB-index free function is `get_local_cb_interface(cb_id_in0).fifo_page_size` (`:25`), which is **sanctioned**.
  - `eltwise_sfpu.cpp` (eltwise/unary, compute): `DataflowBuffer` wrappers for FIFO sync. Compute LLK calls taking CB indices (`copy_tile`, `pack_tile`, `compute_kernel_hw_startup`) are compute APIs, not Device 2.0 data movement.

- **Feature compatibility:** GREEN, no gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer`, `global_circular_buffer` field, or `remote_index` |
  | CBDescriptor `address_offset` (non-zero) | N/A | neither `CBDescriptor` (`single_core_program_factory.cpp:47`, `:59`) sets `address_offset` |
  | GlobalSemaphore | N/A | no semaphores at all |

- **CB endpoints (GATE-free):** all legal; details under Port-work.
- **Offset base pointers:** **GREEN.** There are three address args, all `Buffer*` bindings with no arithmetic: reader `src_buffer` (`:117`), and writer `dst_buffer1` / `dst_buffer2` (`:118`). The op is not in the 2026-07-19 triage tables. No `narrow` and no `address_offset`.
- **TensorAccessor 3rd argument:** **N/A.** No accessor passes a 3rd argument: `TensorAccessor(dst1_args, dst_addr1)` / `(dst2_args, dst_addr2)` (`writer_multiple.cpp:27-28`) and `TensorAccessor(src_args, src_addr)` (reader `:30`). `s1.get_aligned_page_size()` at `writer_multiple.cpp:35,40` is a transfer-size argument to `noc.async_write`, not an accessor page-size override.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding). All three use the `Buffer*`-binding form, which is correct on cache hit today and is routine port work:
  - `src` (input): **Case 1**. Reader feeds it to `TensorAccessor` (reader `:30`). Bind via the existing fork's `tensor::src`.
  - `dst1` (output 0, **optional**): **Case 1**. `TensorAccessor(dst1_args, dst_addr1)` (`writer_multiple.cpp:27`).
  - `dst2` (output 1, **optional**): **Case 1**. `TensorAccessor(dst2_args, dst_addr2)` (`writer_multiple.cpp:28`).
  - Optional outputs: the kernel also reads the raw address as a **presence flag**: `if (dst_addr1 != 0)` (`:34`) and `if (dst_addr2 != 0)` (`:39`). The host passes a null `Buffer*` (emits 0) and builds `TensorAccessorArgs(nullptr)` (`:82-83`) for an absent output. In Metal 2.0 the address leaves the RTA list, so this sentinel has to become the **conditional / optional resource binding** pattern (`port_patterns.md`): bind `dst1` / `dst2` only when present, emit a matching define, and `#ifdef`-gate the kernel's accessor and write block. Presence is driven by `return_output1` / `return_output2` in `operation_attributes_t`, which the default reflection hash covers, so each presence combination is already its own program. A compile-time gate therefore keeps behaviour the same.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (single config; one core):
  - `c_0` (input, `:47`): reader FIFO-produces (`reserve_back` / `push_back`), compute FIFO-consumes. Plain 1P+1C.
  - `c_2` (output, `:59`): compute FIFO-produces, writer FIFO-consumes (`writer_multiple.cpp:32,44`). Plain 1P+1C.
  - No dead CBs, self-loops or multi-binding.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. I looked for a hidden second writer: no kernel raw-writes either CB.
- **Cross-op / shared kernels:**
  - Reader `eltwise/unary/.../dataflow/reader_unary_interleaved_start_id.cpp`: a **`_metal2` fork already exists** beside it (`reader_unary_interleaved_start_id_metal2.cpp`). It fits this op as-is: `dfb::in`, `tensor::src`, named `args::num_pages` / `args::start_id`, page size from `dfb.get_entry_size()` (equal to the legacy `fifo_page_size`), and `#ifdef BACKWARDS`, which this op does not set. Bind it; don't re-fork. **Sunset list** (legacy-copy binders still on the original, *not authorization to convert in place*): `examples/example` (single- and multi-core factories), `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `reduction/topk` (`topk_route_prep_program_factory.cpp`).
  - Compute `eltwise/unary/.../compute/eltwise_sfpu.cpp`: **no `_metal2` fork yet**, so this port creates `eltwise_sfpu_metal2.cpp` beside the original (rung 2). It is broadly shared: `eltwise/unary`'s own `UnaryProgramFactory` uses it as the default compute kernel (`unary/common/unary_op_utils.cpp:1212`, consumed at `unary/device/unary_program_factory.cpp:414`). **Sunset list:** `eltwise/unary` (`unary_program_factory.cpp`), `examples/example` (single- and multi-core), and `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:284` (a test, not a factory). The kernel has `#ifdef SFPU_OP_CHAIN_0` (`:36`), which this op does not define. Name the fork's bindings from the kernel's vocabulary (e.g. `dfb::in` / `dfb::out`), not this op's locals.
- **RTA varargs:** none. Writer (`:12-15`), reader (`:16-18`) and compute (`:18`) all read fixed, distinct indices.
- **Direct-descriptor shape:** the op has no factory struct (`device/example_multiple_return_device_operation.hpp:72`). Per `ttnn_factory.md` exception 3, the port must add `struct ExampleMultipleReturnProgramFactory` with `create_program_artifacts` and `using program_factory_t = std::variant<ExampleMultipleReturnProgramFactory>;`. Record this under Handoff points.
- **Compute `opt_level`:** the compute descriptor sets no `opt_level` (`single_core_program_factory.cpp:94-101`), so legacy resolves it to `O3`. The spec must carry that explicitly, per `metal2_port.md`, Compiler options.
- **Test:** `tests/ttnn/unit_tests/operations/debug/test_examples.py` (`test_composite_example_multiple_return`, `test_composite_example_multiple_return_program_cache`). It runs on one core and fits the 8×8 Wormhole here.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.** No kernel `#include`s anything outside `tt_metal/*` (LLK / HAL / dataflow / compute APIs), so there is no function-call escape. File-path coupling: 2 borrowed kernels.

  | Op kernel | Donor | Class | Status |
  |---|---|---|---|
  | reader | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` | cross-family borrowed file | ✓ (fork exists) |
  | compute | `eltwise/unary/.../compute/eltwise_sfpu.cpp` | cross-family borrowed file | ✓ (no fork yet; this port creates it) |

- **Relaxation candidates:** none (no custom hash).
- **TTNN factory analysis:** descriptor concept via the direct-descriptor shim. No op-owned tensors (single `ProgramDescriptor`, no `WorkloadDescriptor`). No custom hash, no `get_dynamic_runtime_args`, no `override_runtime_arguments`, no pybound `create_descriptor`. Target: `ProgramSpecFactoryConcept`.

## Misc anomalies

- `writer_multiple.cpp:9` includes `api/debug/dprint.h`, and nothing in the kernel uses it.
- `operation_attributes_t::attribute` and `some_other_attribute` (`device/example_multiple_return_device_operation.hpp:20-21`) are hard-wired to `true` / `42` (`device/example_multiple_return_device_operation.cpp:63`), never read, and still fed to the default reflection hash. They are harmless placeholders in an example op.
- `return_output1` / `return_output2` are declared `uint32_t` but initialised from `bool` (`device/example_multiple_return_device_operation.hpp:22-23`).

## Questions for the user

1. **Waive the stale-sheet gate?** The only RED is the #57409 stale row (`SingleCore` / `legacy device-op`). **Answered 2026-10-07: waived** ("Okay waive it if it's just the stale row").

## Recipe notes

- **Unrecognized `Is able to port?` value.** The cell reads `yes (with PD step)`. The recipe's Routing only covers `yes` and `no`. Here it doesn't change the verdict, because the spreadsheet-broken triggers fire independently. But an auditor seeing this value on a clean row would have to guess whether it clears the gate. The recipe and `ttnn_op_porting_readiness.md` should say how to read `yes (…)` qualifiers.
- **Direct-descriptor ops and "Factory-set match".** The cross-check assumes each sheet row maps to a named factory. A direct-descriptor op has no factory name in the code, so "missing row" and "phantom row" need a rule for the shim. Here the outcome is the same either way: `SingleCore` is a phantom.
