# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/moreh/moreh_matmul`

## Outcome

**PORTED** — the op's single factory (`MorehMatmulOperation::MultiCoreProgramFactory`) is converted from `create_descriptor` (`ProgramDescriptorFactoryConcept`) to `create_program_artifacts` (`MetalV2FactoryConcept`), together with all three op-owned kernels (reader, writer, compute). No factories remain on the legacy concept — the op is fully ported.

> **Build/test verification is the orchestrator's** — per the orchestration constraints, this port was neither built nor tested. Nothing is committed; all changes left uncommitted for the orchestrator.

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Files created / modified

Created:
- `METAL2_PORT_PLAN.md` — port plan (legacy inventory, spec shape, dropped plumbing, applied patterns).
- `METAL2_PORT_REPORT.md` — this report.

Modified:
- `device/moreh_matmul_device_operation.hpp` — factory method `create_descriptor` → `create_program_artifacts` (returns `ttnn::device_operation::ProgramArtifacts`). Sole device-op-class edit; the `program_factory_t` variant is unchanged.
- `device/moreh_matmul_program_factory.cpp` — rewrote the factory body to build a `ProgramSpec` + `ProgramRunArgs`. Shape-math and helper functions (`get_tensor_dim`/`find_reduce_dim`/`is_same_batch_dim`/`get_tensor_stride`/`get_not_bcast`) kept verbatim. Added include of `datamovement_kernel_config.hpp`; dropped the now-unused `tensor_accessor_args.hpp` include.
- `device/kernels/reader_moreh_matmul.cpp` — added `experimental/kernel_args.h`; CTAs → `get_arg(args::…)`; input/other/bias address RTAs dropped → `TensorAccessor(tensor::…)`; the 5 dimensional arrays → runtime varargs; `output_tile_start_idx`/`num_output_tiles` → named RTAs; CB ids → `dfb::in0..in4`; `get_tile_size(cb_id)` → `dfb.get_tile_size()`.
- `device/kernels/writer_moreh_matmul.cpp` — added `experimental/kernel_args.h`; output address RTA dropped → `TensorAccessor(tensor::output)`; `start_id`/`num_output_tiles` → named RTAs; CB id → `dfb::out0`; `get_tile_size` → member.
- `device/kernels/moreh_matmul.cpp` — added `experimental/kernel_args.h`; the 10 `tt::CBIndex::c_*` constants → `dfb::in0..out0/im0..im3`; CTAs → `get_arg(args::…)`; `output_tile_start_idx` → named RTA; `output_stride` → runtime varargs; dead `pack_onetile_to_cb` default arg `16` → `dfb::out0`.

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` — as the audit chose. Single program, single factory, no op-owned tensors, no MeshWorkload. Reader/writer span both work units; compute is one `KernelSpec` per work-split core group (`compute_g1`, and `compute_g2` when `core_group_2` is non-empty), preserving the per-group `num_output_tiles` CTA.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op had none).
- Pybind entry points removed: none — `moreh_matmul_nanobind.cpp` binds the free function `ttnn::moreh_matmul`; no `create_descriptor` pybind existed.

### Open items
- **RTA→CRTA relaxation candidate (not applied).** The reader's stride/bcast varargs and the compute `output_stride` vararg are identical on every node (computed once from tensor shapes, node-invariant). They are genuinely common runtime args. Left per-node to avoid changing dispatch semantics during the port (recipe §Construct explicitly defers RTA→CRTA). A follow-up could move them to `common_runtime_arg_names` / common varargs for dispatch efficiency.
- **TensorParameter relaxation:** none applied; strict spec match kept (audit: relaxation = none). No `ArgConfig::Runtime*` in the kernels.

## Handoff points

None. The port stayed entirely within the op directory. All three kernels are op-owned; the shared moreh pool headers (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`, `.../compute/moreh_common.hpp`) were **not** modified — only `generate_mask_tiles(DataflowBuffer, …)` (reader) is still called from them, and it already takes a `DataflowBuffer`. `ArgFetcher` is no longer used by any of this op's kernels (all arg reads went to `get_arg`/`get_vararg`), but its definition remains in the shared header for other consumers; the `#include` is left in place (harmless).

## Successes

- **Self-loop pattern (`port_patterns.md` — Self-loop DFB binding)** fired exactly as documented for the four compute intermediates (`IM0`–`IM3`): compute both produces and consumes them, so each is bound PRODUCER+CONSUMER on the compute KernelSpec with a shared accessor name. The shared-name form kept the kernel bodies untouched (`DataflowBuffer(dfb::im0)` etc.).
- **Disjoint-node multi-consumer (`port_patterns.md` — Demoting per-group CTA to RTA / migration guide DFB invariant note)**: `IN0`–`IN4` bound to both `compute_g1` and `compute_g2` as CONSUMER over disjoint node sets validated as legal single-role-per-node bindings — no `allow_instance_multi_binding` flag needed, matching the audit census.
- **`to_compute_hardware_config` (recipe Hardware config, Style A)** carried the four compute knobs cleanly; the `dst_full_sync_en → double_buffer_dest` inversion is handled inside the helper, so the port did not have to reason about it. The reference reduce factory (`experimental/quasar/reduction/.../reduce_op_multi_core_h_program_factory.cpp`) — which has the same reader/writer-span-both + per-group-compute shape — was a load-bearing shape reference for the WorkUnitSpec structure.
- **`unpack_modes` required-entry rule (recipe Hardware config / migration guide)** caught a subtlety the legacy vector hid: under `fp32_dest_acc_en`, both `IM0` (legacy `UnpackToDestFp32`) **and** `IM3` (legacy `Default`, but a Float32 DFB consumed by compute with `enable_32_bit_dest`) require explicit entries. Set `IM0 → UnpackToDest`, `IM3 → UnpackToSrc` — faithful to the legacy values, satisfies the stricter Metal 2.0 validator.

## Friction

### Gaps
- None blocking.

### Confusion — **varargs vs named for the dimensional stride arrays (deviated from the brief)**

The brief (`METAL2_PORT_BRIEF.md` "Watch for → RTA varargs") explicitly instructed: *"name them as fixed fields/arrays, not varargs,"* for the reader's five 8-element arrays (`input_stride`, `other_stride`, `output_stride`, `input_not_bcast`, `other_not_bcast`) and the compute `output_stride`, on the grounds that the loop bound `MAX_NUM_DIMENSIONS = 8` is a source literal. The audit's own "Recipe notes" flagged that this sits close to a mis-classification.

**This port used runtime varargs instead**, per `port_patterns.md` "Caution: Avoid varargs" line: *"even at a source-literal count, the elements form one homogeneous array — read purely as `arr[i]`, no per-element identity, the index is the meaning (conceptually a single `std::array` argument, of which the vararg is the interim form)."* These arrays are indexed by dimension with no per-element identity — the line-494 homogeneous-array case, which the caution classifies as a vararg. Deciding factors:
1. The patterns catalog is the most-specific authoritative text on this exact construct, and the recipe (line 52) ranks it above the wider guidance; the brief's "literal-count ⇒ nameable" reasoning is the trap-1 (distinct-fields-via-`arg_index++`) rule mis-applied to a homogeneous array.
2. Varargs gives a far cleaner minimal kernel diff (the read loops stay intact — a one-token change `get_next_arg_val` → `get_vararg`), versus unrolling into 40 (+8) pseudo-named fields.
3. Naming them would manufacture per-element identities the catalog explicitly says don't exist.

The genuinely-distinct scalars (`output_tile_start_idx`, `num_output_tiles`, `start_id`) are **named**, as the caution requires. Behavior is identical either way (both compile to the same device reads), so this is a style/convention call, not a correctness one. **Doc-maintainer ask:** reconcile the brief-generation guidance (audit) with `port_patterns.md` line 494 so future audits classify homogeneous literal-count arrays consistently. Recorded here so the deviation from the issued brief is explicit.

## Open items for downstream

- **Cross-op kernel touches:** none. No kernel outside the op directory was modified or forked.
- **Sibling moreh ports:** the shared `moreh_common.hpp` pool headers are already Device 2.0 native and were not touched; when the wider moreh family ports, `ArgFetcher` will become fully dead once the last consumer stops using it — a future cleanup, not this port's.
- **Test coverage note:** the op's primary coverage is `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_matmul.py` (see test command below). No dedicated C++ gtest for moreh_matmul was found. `test_moreh_matmul_backward` exercises the separate `moreh_matmul_backward` op (which internally calls this forward op).

## Test command(s)  *(build + run are the orchestrator's)*

Build (Metal + TTNN tests):
```
./build_metal.sh --build-tests
```

No-regression baseline — the op's pytest module (all pass pre-port must still pass post-port):
```
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_matmul.py -x -v
```

Forward-factory-focused subset (the paths this port touches — includes the fp32-dest-acc and bias-add paths that exercise the `unpack_modes` and FUSE_BIAS logic):
```
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_matmul.py -x -v \
  -k "test_moreh_matmul or test_moreh_matmul_wo_output or test_moreh_matmul_enable_cache or test_moreh_matmul_fp32_dest_acc or test_moreh_matmul_with_bias_add_fp32_dest_acc"
```

`test_moreh_matmul_enable_cache` is the key program-cache (cache-hit tensor-refresh) check. No C++ gtest target for this op was found; if the orchestrator knows of additional coverage (sweeps, model tests), please confirm and include it in the baseline.
