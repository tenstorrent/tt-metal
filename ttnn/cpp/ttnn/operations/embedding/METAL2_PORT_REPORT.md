# Metal 2.0 Port Report - `ttnn/cpp/ttnn/operations/embedding`

*Opened at the start of the port; friction captured as it occurred, polished at the end.*

## Outcome

`PORTED` - all three factories (`EmbeddingsRMProgramFactory`, `EmbeddingsFusedProgramFactory`, `EmbeddingsTilizedIndicesProgramFactory`) of `EmbeddingsDeviceOperation` converted to `MetalV2FactoryConcept` in one pass. Build green (`./build_metal.sh -e --enable-fake-kernels-target --build-tests`, exit 0). Confirmed test baseline green: **318 passed, 0 failed, 2 skipped** across `tests/ttnn/unit_tests/operations/data_movement/test_embedding.py`, `tests/tt_eager/python_api_testing/unit_testing/misc/test_embedding.py`, and `tests/ttnn/docs_examples/test_embedding_examples.py`. The 2 skips are pre-existing hardware gates (`test_tg_llama_sharded_embedding` / `test_tg_llama_sharded_rm_embedding` need a 7×10 core grid; this card is 7×9) and skip identically before the port. Anti-pattern self-audit clean on every checklist item.

## Provenance

- **Recipe docs (this port):** `d28425ca5cf 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `d28425ca5cf 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for all three factories. Each factory's static `create_descriptor(...)` (returning `tt::tt_metal::ProgramDescriptor`) is replaced by `create_program_artifacts(...)` (returning `ttnn::device_operation::ProgramArtifacts`). No disagreement with the audit's decision.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** - the op never had one.
- Pybind entry points removed: **none** - `embedding_nanobind.cpp` binds only the user op `&ttnn::embedding`; no factory entry point was pybound.
- `select_program_factory`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors` on the device-op class are unchanged.

### Open items
*(see Open items for downstream)*

## Handoff points

- **No capitulation.** Metal 2.0 expressed every construct in this op.
- **No boundary-rule assumption violations.** No out-of-op call site required a `sem::`/`tensor::` handle (the op uses no semaphores; tensor accessors are consumed inside the op's own kernels and the forked donor kernels).
- **No removed pybind surface.**

## Successes

- **Brief's "Fused weights accessor base offset" heads-up prevented a silent mis-address.** The brief (Watch for → the load-bearing one) flagged that a naive `TensorAccessor(tensor::weights)` would drop `weight_offset` and mis-address block/width-sharded fused output. Following it, the offset was relocated to the accessor read's `offset_bytes` at `embeddings_tilize.cpp` (per-token reads) and `embeddings_common.hpp` (`read_token_async` / `prepare_local_cache`). This exact path (GENERIC + WIDTH/BLOCK-sharded) is exercised by `test_embedding_tiled_sharded_output`, so the relocation is verified, not just asserted.
- **Borrowed-memory + self-loop guidance ([Sync-free / single-ended → self-loop](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)) fit the sharded output cleanly.** The recipe's statement that a `borrowed_from` reference satisfies the TensorParameter referential-integrity rule (`program_spec.cpp:545`) was exactly right - the sharded-output `TensorParameter` needs no kernel `TensorBinding`, only the `OUT.borrowed_from = OUTPUT` reference. No validator complaint.
- **The conditional-binding pattern reused the op's own type define.** The weight-cache DFB is bound only for PADDED/BINARY, and the kernel already gated its `dfb::weight_cache` reference on the `PADDED`/`BINARY` type define (`embeddings_common.hpp`). No separate `KernelSpec::compiler_options.defines` flag was needed - the existing type define does double duty, matching the [Conditional / optional DFB bindings](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings) "promote a CTA gate to a define" spirit (here the gate was already a define).
- **The demoting-CTA anti-pattern warning fired correctly.** The fused compute's per-group `per_core_block_cnt` was kept as a compile-time arg across two `KernelSpec`s (`compute_g1` / `compute_g2`) over disjoint core groups (`embeddings_fused_program_factory.cpp` `make_compute`), not demoted to a runtime arg - following [Demoting per-group CTA to RTA](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta).
- **DFB-handle-to-LLK conversion made the compute forks trivial.** `compute_kernel_hw_startup(dfb::in, dfb::out)` and `compute_kernel_lib::tilize<…, dfb::in, dfb::out, …>` (and `is_fp32_input_format<dfb::in>()`) worked via the implicit `DFBAccessor → uint32_t` conversion with no `.id` extraction - matching [Pass DFB handles directly to LLKs](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers).

## Friction

- **`weight_offset` relocation reaches further than the brief stated. (Gap)** The brief said to fold the fused reader's `weight_offset` into "the `weight_chunk_offset` that `read_token_async` already passes as `offset_bytes`." That covers the main token reads, but the same `weights` accessor is *also* read inside `prepare_local_cache` (the PADDED/BINARY cache fill), which the legacy base-shift also offset. To stay faithful for the PADDED/BINARY + block/width-sharded fused case, the relocation had to be applied to `prepare_local_cache`'s accessor reads too (a `weight_base_offset` parameter added to both helpers). A future porter hitting a "separate-offset arg added to an accessor base" shape should relocate the offset at **every** site that reads through that accessor, not only the one the brief points at. (The audit's own Recipe-notes section already asks for a named "separate-offset arg added to an accessor base" Case-1 sub-shape; this port confirms it and adds the multi-site nuance.)
- **Migration guide's "identical WorkUnitSpec membership" reads stricter than what is validated. (Confusion)** The [migration guide troubleshooting](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/migration_guide.md#troubleshooting) states "A local DFB's producer and consumer kernels must share identical WorkUnitSpec membership." The fused factory's shape (reader/writer in **both** `wu_g1` and `wu_g2`, `compute_g1`/`compute_g2` each in **one**) does *not* satisfy that literal reading, yet it is exactly the [Demoting per-group CTA to RTA](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) correct-port shape and the `dataflow_buffer_spec.hpp` invariant (per-node 1P+1C, with same-role multi-bindings allowed across non-overlapping node sets) permits it. I relied on the anti-pattern's worked example over the troubleshooting line. Suggest the guide reword the one-liner to the per-node rule so a porter doesn't read it as forbidding the split-compute shape.
- **`std::cref` on `TensorArgument` - recipe vs. reference disagree. (Confusion)** The recipe/migration guide say to pass the `MeshTensor` directly (`{INPUT, input}`) and *not* wrap in `std::cref`; the `transpose_wh_sharded` reference port wraps in `TensorArgument{std::cref(...)}`. The bare form the recipe prescribes compiles and runs (confirmed by the smoke test, which exercises `tensor_args`). This is a case where the wider codebase disagrees with the recipe and the recipe is right - noting it because a porter glancing at the reference would copy the `std::cref` noise.

## Open items for downstream

*(populated during the port)*

- **Cross-op kernel forks (`_metal2`).** Three shared donor kernels were forked (CB→DFB + named-token rewrite) because their other consumers are not yet on Metal 2.0. Legacy copies retained unchanged. Each fork is a faithful translation of the *whole* legacy kernel (all `#ifdef` variants preserved) so it is a drop-in for any consumer. Sunset each fork - and delete it - when its last legacy consumer ports. Any bug fix to a legacy copy during the window should be mirrored onto its fork.

  | fork (new file) | forked from | remaining unmigrated consumers |
  |---|---|---|
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp` | `writer_unary_stick_layout_interleaved_start_id.cpp` (same dir) | `data_movement/concat`, `data_movement/slice` |
  | `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_metal2.cpp` | `tilize.cpp` (same dir) | `tilize`, `tilize_with_val_padding`, `untilize`, `untilize_with_unpadding`, `moreh/moreh_getitem`, `pool/upsample`, `sliding_window/halo`, `deepseek_prefill/combine`, `quasar/tilize_with_val_padding` |
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` | `writer_unary_interleaved_start_id.cpp` (same dir) | broadly shared (~22 op families) |

  Notes on the forks:
  - The stick-layout writer fork drops the legacy dead CTA at index 1 (`output_page_size`, never read); its output page size rides on the `stick_size` runtime arg.
  - The interleaved writer fork reads the page size off the `DataflowBuffer` object (`get_entry_size()`) instead of `get_local_cb_interface(cb_id).fifo_page_size`, and preserves the `OUT_SHARDED` / `BACKWARDS` build variants.
  - The `tilize` compute fork sources the input/output CB indices from `dfb::in` / `dfb::out` and passes them to `compute_kernel_hw_startup` / `compute_kernel_lib::tilize` via the implicit `DFBAccessor → uint32_t` conversion.

- **Untested paths (compile-coverage only).** No test exercises `EmbeddingsType::PADDED` or `BINARY` (all embedding tests use `GENERIC`). The weight-cache conditional DFB binding, the pad-token named RTA, and the `prepare_local_cache` `weight_offset` relocation are therefore compile-covered but not numerically validated.

- **Latent bug corrected as a side effect (routed to ops team).** The TilizedIndices reader passed `pad_token_arg_idx=6`, but its factory places the pad token at RTA 7 (arg 6 is `starting_index`), so on the PADDED + TILE-layout-index path the legacy kernel read `starting_index` as the pad token (audit Misc anomaly). Metal 2.0 reads args by name, so the faithful named-arg conversion reads the real pad token and the positional-aliasing bug cannot be reproduced. The invoker approved landing this correction (path is untested, so no test regresses). Ops team should confirm the corrected behavior is intended.

- **Dead code left in place (not the port's to fix).** `#define RISC_CORES_PER_TENSIX 2` (`embedding_device_operation.cpp:13`, never used); vestigial `#include "api/debug/dprint.h"` in `embedding_ind_tilized.cpp` (no `DPRINT` usage). Both left as-is per scope discipline.

- **Dead compile-time args dropped by the named-arg model.** Fused reader `weight_page_size` (legacy CTA 4), TilizedIndices reader `input_page_size` (CTA 3) and `input_block_size_bytes` (CTA 6), and the stick-layout writer's `output_page_size` (CTA 1) are all emitted-but-unread in the legacy kernels; the named-arg conversion emits only the args the kernels read, so these fall away with zero functional change.
