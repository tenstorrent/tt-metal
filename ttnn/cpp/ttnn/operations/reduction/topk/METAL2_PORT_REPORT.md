# Port Report — topk

Metal 2.0 port of `reduction/topk`. **Single-core factory ported; multi-core factory deferred on legacy.**
Tests not yet run — pending orchestrator build/test (`tests/ttnn/unit_tests/operations/reduce/test_topk.py`).

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept` (`create_program_spec`) for `TopKSingleCoreProgramFactory`. `TopKMultiCoreProgramFactory` stays on the legacy `ProgramDescriptorFactoryConcept` (`create_descriptor`). The `program_factory_t` variant runs **mixed concepts**; `select_program_factory`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors` are unchanged.

### Device-op-class edits
- `device/topk_device_operation.hpp`: `TopKSingleCoreProgramFactory::create_descriptor` → `static ttnn::device_operation::ProgramArtifacts create_program_spec(...)`. Added `#include "ttnn/device_operation.hpp"` and `#include "ttnn/metal2_artifacts.hpp"`; **kept** `<tt-metalium/program_descriptors.hpp>` because the multi-core factory still returns `ProgramDescriptor`.
- Custom `compute_program_hash` deleted: none (default reflection hash).
- Pybind entry points removed: none.

### Open items
- Strict tensor matching kept (default).

## Successes

- **Runtime-dynamic CB selection in `compute/topk.cpp`** (`metal2_port_patterns.md` — runtime-dynamic CB selection). The kernel reassigns `uint32_t cb0..cb3` to different staging CBs per insertion-sort case, then uses them both at LLK call sites (`copy_tile`, `pack_results`) and to build `DataflowBuffer` objects. Ported with `constexpr uint32_t name = (uint32_t)dfb::name;` aliases for the eight CB ids, `DataflowBuffer obj(cbX)` (uint16_t ctor) for the dynamic locals, and direct `dfb::name`/`uint32_t` pass-through to every LLK. The per-case `cb0/cb1/cb2/cb3` branching logic is byte-for-byte preserved.
- **Real staging self-loops** (`c_2..c_5`). The compute kernel both fills and drains `transposed_*` and `result_prep_*` (`reserve_back`/`push_back` then `wait_front`/`pop_front` on the same kernel). Bound as genuine PRODUCER+CONSUMER self-loop DFBs on the compute kernel (the legitimate accumulator-style self-loop, not the fake-CB hack) — each carries real FIFO semantics.
- **Clean Case-1 TensorAccessors.** `input` (reader), `value`/`index` (writer) all read/write page-by-page (`{.page_id = ...}`); ported as `TensorParameter` + `ta::name` with no base-address bridge.
- **Conditional optional-indices binding.** `GENERATE_INDICES` is hardcoded `"1"` (GH #36329), so the precomputed-indices read path is dead. Kept `GENERATE_INDICES` as a `compiler_options.defines` flag and `#ifdef`-gated the `ta::indices` accessor; on the live path no `indices` TensorParameter is bound — exactly the documented conditional-binding shape.

## Friction

- **Stale build header for `DFBAccessor::operator uint32_t`.** `build_Release/libexec/.../dataflow_buffer.h` lacks the `operator uint32_t()` that the source `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h` (line 53) has. The `(uint32_t)dfb::name` aliases and direct LLK pass-through rely on it; a full install build (regenerating the libexec copy) is required, consistent with PR #44646 / `akertesz/dfb-accessor-implicit-conv`.
- **Shared kernel-source disambiguation.** `reader_create_index_tensor.cpp` / `writer_binary_interleaved.cpp` / `compute/topk.cpp` are single-core-only; the multi-core path uses entirely distinct kernels (`*_local_topk`, `*_final_topk`, `topk_local/final`, plus `topk_common_funcs.hpp`). Disjoint sets — porting single-core left every multi-core kernel untouched.

## Grounded stop — multi-core factory

Left on legacy `create_descriptor`. It does not fit the documented Metal 2.0 patterns:

1. **Cross-core remote-CB write.** `writer_local_topk` computes the *final* core's CB write pointer locally (`final_values_cb.get_write_ptr()` on an `all_cores`-allocated CB) and NoC-writes into it at `{.noc_x = noc_final, ...}`, while `reader_final_topk` independently `reserve_back`/`push_back`-produces the same `c_4`/`c_5` on the final core. One DFB, multiple producers across disjoint node sets, targeted by raw remote address — no `metal2_port_patterns.md` entry expresses a remote-core CB-address write, and the DFB endpoint invariant rejects it.
2. **Custom semaphore-multicast handshake** (`receiver_sem.set_multicast` / `sender_sem.wait`/`up`) coupled to (1).
3. **Allocation-order-pinned L1 layout** across two core sets, relied on so `get_write_ptr()` yields a remotely-targetable address; Metal 2.0 DFB placement is derived, not order-pinned.

Per the orchestrator brief, a finished single-core factory with multi-core cleanly deferred is a valid deliverable.

## Open items for downstream

- **Multi-core factory remains to be ported** once a sanctioned pattern exists for remote-core CB-address writes (or the algorithm is restructured to a gather-into-local-DFB shape). Sites: `device/topk_multi_core_program_factory.cpp` + its six kernels.
- **Cross-op kernel touches:** none.
- **Test coverage:** `tests/ttnn/unit_tests/operations/reduce/test_topk.py` exercises both single-core (small / large-dim-with-UInt32 / K>64 paths) and multi-core (large power-of-two dim, K≤64). Not yet run — pending orchestrator build/test.
