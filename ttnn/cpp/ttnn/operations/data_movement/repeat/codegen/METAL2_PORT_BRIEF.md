# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/repeat/codegen`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `8c5559389da 2026-09-22 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

**Scope of the port:** `RepeatCodegenDeviceOperation` → its single factory `RepeatCodegenProgramFactory` (`repeat_codegen_program_factory.cpp`), all three internal branches (TILE / RM last-dim / RM higher-dim). The native `RepeatDeviceOperation` under `../device/` is already Metal 2.0 and is **not** part of this port. Do not read `ttnn/cpp/ttnn/operations/experimental/quasar/**` for precedent.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `static ProgramDescriptor create_descriptor(...)` at `repeat_codegen_program_factory.hpp:39-42`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer (cell is `none`) · `get_dynamic_runtime_args` (absent). Also absent on this op (none of these gate, recorded for completeness): custom `compute_program_hash`, `override_runtime_arguments`, pybound `create_descriptor`.
- **Device-op shape to keep:** `program_factory_t = std::variant<RepeatCodegenProgramFactory>` (`repeat_codegen_device_operation.hpp:20`); `select_program_factory` returns the one alternative. `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`, `create_op_performance_model` and the `repeat_codegen(...)` launcher (`repeat_codegen_device_operation.cpp`) are host-only and untouched by the port apart from the factory-concept wiring.
- **Attribute struct:** `RepeatCodegenParams` (`repeat_codegen_program_factory.hpp:22-31`) already carries every value the kernels need as CTAs/RTAs (`rep_dim`, `num_repeats`, `lower_pages`, `rep_dim_pages`, `total_out_pages`, `stick_size`, `output_mem_config`) — the default attribute hash covers them; no hash work.

## Construct — to do

**Tensor bindings** (per binding; identical roles in all three branches):

- `src` (input, `tensor_args.input`) — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::src)`. Legacy delivery is the `Buffer*`-binding form (`src_buffer` in `emplace_runtime_args`, `repeat_codegen_program_factory.cpp:152, 212, 266`) plus positional `TensorAccessorArgs(*src_buffer).append_to(reader_ct_args)` (`:115, 184, 237`) — both disappear.
- `dst` (output, `tensor_return_value`) — **Case 1** → `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::dst)`. Legacy: `dst_buffer` at `:158, 213, 267`, `TensorAccessorArgs(*dst_buffer)` at `:136, 198, 253`.
- Kernel-side sites that take the base pointer today: `reader_repeat_last_dim_rm.cpp:35,47` · `reader_repeat_higherdim_rm.cpp:20,33` · `writer_repeat_rm.cpp:26,36` · (fork of) `reader_tile_interleaved_unified.cpp:160-169` (`base->src_addr`) · (fork of) `writer_interleaved.cpp:15,28`.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** drop the redundant page-size argument (Class 2, no `dynamic_tensor_shape`) at — in the **`_metal2` forks**, never in the legacy originals:
- `common/kernels/codegen/reader_tile_interleaved_unified.cpp:169` — `TensorAccessor(src_args, base->src_addr, source_page_size)` → 2-arg form on `tensor::src`. (`source_page_size` resolves to `src_args.get_aligned_page_size()` because every current consumer passes `src_page_pitch = 0`.)
- `common/kernels/codegen/reader_tile_interleaved_unified.cpp:292` — dead-branch twin in `SEQ_CONCAT` (`a->src_addr_1`); drop identically so the fork compiles with one accessor idiom.
- `common/kernels/codegen/writer_interleaved.cpp:28` — `TensorAccessor(dst_args, dst_addr, destination_page_size)` → 2-arg form on `tensor::dst`.
- The op's own RM kernels already use the 2-arg form — nothing to drop there.

**CB endpoints:** all legal. One DFB per branch (legacy `buffer_index 0`, `kRepeatCbDepth = 8` entries, page size = the branch's aligned page): bind the reader **PRODUCER** and the writer **CONSUMER** on `(CB0, TILE)`, `(CB0, RM last-dim)`, `(CB0, RM higher-dim)`. No self-loop, no 1P+1C reassignment, no multi-binding flag, no dead-CB drop, no conditional DFB.

**Branch structure to preserve** (`repeat_codegen_program_factory.cpp:91-92, 99, 167, 222`): three mutually exclusive `KernelSpec` pairs + one `DataflowBufferSpec` each, selected by `is_row_major` / `is_last_dim_rm`. Only the TILE branch binds the shared-pool forks; the two RM branches bind the op's own kernels. Per-core work split (`split_work`, `:46-66`, column-major enumeration — keep `row_wise=false`) feeds per-node RTAs `n` / `start`.

**CTA inventory to name** (all positional today except the TILE reader's four named ones):
- TILE reader (`repeat_codegen_program_factory.cpp:114-132`): `seq_id` (=1, keep named), `cb_id` → `dfb::` binding, `batch`, `src_page_pitch` (see Watch for), `TensorAccessorArgs` → `tensor::src`.
- TILE writer (`:135-137`): `cb_out` → `dfb::`, `REQUESTED_WRITE_SIZE` (= dst aligned page), `TensorAccessorArgs` → `tensor::dst`, `BATCH`.
- RM last-dim reader (`:183-187` ↔ `reader_repeat_last_dim_rm.cpp:39-45`): `stick_size`, `in_read_size`, `out_l1_stride`, `TensorAccessorArgs` → `tensor::src`, `cb_id` → `dfb::`, `NUM_REPEATS`, `BATCH`.
- RM higher-dim reader (`:236-242` ↔ `reader_repeat_higherdim_rm.cpp:24-31`): `xfer_size`, `l1_stride`, `TensorAccessorArgs` → `tensor::src`, `cb_id` → `dfb::`, `NUM_REPEATS`, `LOWER_PAGES`, `REP_DIM_PAGES`, `BATCH`.
- RM writer (`:197-199`, `:252-254` ↔ `writer_repeat_rm.cpp:30-34`): `cb_out` → `dfb::`, `xfer_size`, `l1_stride`, `TensorAccessorArgs` → `tensor::dst`, `BATCH`.
- Several of these drive `if constexpr` (`reader_repeat_last_dim_rm.cpp:73-74, 100-132`; `reader_repeat_higherdim_rm.cpp:52-58`; both writers' `BATCH > 1`) — the named CTA reads must remain `constexpr`-evaluable.

**RTA inventory to name** (per node): TILE reader `num_pages, start_id, num_repeats, lower_pages, rep_dim_pages` (address goes to `tensor::src`); all other kernels `num_pages, start_page`/`start_id` (address goes to the tensor binding). No CRTAs today.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** the TILE branch binds two kernels this op does **not** own, in `ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/`:
  - `reader_tile_interleaved_unified.cpp` → caution per `port_patterns.md` → *Caution: Porting a shared kernel* (**borrowed** shape). **No fork yet** — this port creates `reader_tile_interleaved_unified_metal2.cpp` beside the original (rung 2) and adds the pointer comment to the original; nothing else in that directory is yours. Its `#include "sequencers.h"` stays as-is (scalar-only signatures, no rewrite).
  - `writer_interleaved.cpp` → same caution, same rung; fork `writer_interleaved_metal2.cpp` beside it.
  - Other binding ops (both files): `repeat_interleave/codegen/repeat_interleave_codegen_program_factory.cpp:38-40, 113, 124, 188` — **sunset list, not authorization to convert the kernel in place.** Name the fork's bindings for the kernel's role vocabulary (`tensor::src`, `tensor::dst`, `dfb::` for the staging buffer, `seq_id`/`batch` as-is), not for this op's locals, because `repeat_interleave` will bind the same fork with `seq_id = SEQ_REPEAT_INTERLEAVE`.
  - `untilize/codegen/kernels/reader_tile_interleaved_unified.cpp` is a *same-named private copy* at a different path — not a consumer; leave it alone.
- **RTA varargs:** none — prefer named RTAs everywhere. But note the borrowed reader's read pattern: it overlays a struct on the RTA block via `reinterpret_cast<const ArgsRepeat*>(get_arg_addr(0))` (`reader_tile_interleaved_unified.cpp:160, 179`; struct at `:44-54`). Under `SEQ_REPEAT` that is six fixed fields → six named args; replace the overlay with per-field named reads in the fork. The `SEQ_SLICE`/`SEQ_PERMUTE` branches of the same file (`:197-199, 210-211`) read genuine variable-length tails — they are dead for every current consumer; if you leave them in the fork they must still compile, so decide (and record in the port report) whether the fork keeps all ten sequencer branches or only the two with binders.
- **`src_page_pitch` named CTA** (`repeat_codegen_program_factory.cpp:131`; read unconditionally at `reader_tile_interleaved_unified.cpp:161-163`): its only effect is the 3rd-arg value you are dropping. After the drop it is a dead read under `SEQ_REPEAT`; keep supplying it if the fork still reads it (the co-consumer also passes `0`), and record whatever you decide.
- **Sanctioned free functions with a DFB equivalent:** `get_local_cb_interface(cb_id).fifo_page_size << cb_addr_shift` (`reader_tile_interleaved_unified.cpp:164`, `writer_interleaved.cpp:39`) → the DFB entry-size getter (`cb_dfb_api_whitelist.md`, section B). Confirm units (bytes) against the DFB header before swapping; don't swap blind.
- **Firmware coordinate globals** `my_x[noc.get_noc_id()]` / `my_y[…]` (`reader_repeat_last_dim_rm.cpp:79-80`; dead PAD branch of the borrowed reader `:240-241`) feed a `UnicastEndpoint` self-address for the L1→L1 stick replication. Not a Device 2.0 idiom to touch — leave them.
- **`cb_in.get_write_ptr()` peek + `noc.async_read(self_ep, cb_in, …)`** in the last-dim reader (`reader_repeat_last_dim_rm.cpp:60, 84-89`) and the `CoreLocalMem` RISC copies (`:100-140`) address the CB slot by absolute L1 address. Under a DFB the write-pointer getter must return the same absolute base for this to keep working — confirm rather than assume when swapping `CircularBuffer` → `DataflowBuffer`.
- **Baseline first, kernels are JIT'd from the working tree:** run `tests/ttnn/unit_tests/operations/data_movement/test_repeat.py` and `tests/ttnn/nightly/unit_tests/operations/data_movement/test_repeat_codegen_routing.py` (pins the codegen leg through `ttnn._ttnn.operations.data_movement.repeat_force_codegen`) before the first kernel edit, and purge the kernel cache between baseline and post-port runs. All three branches need coverage: TILE (any tile-aligned repeat), RM last-dim (`rep_dim == 3`, e.g. `[1,1,1,32] x (1,1,1,4)`), RM higher-dim (e.g. the gpt_oss `[1,1,1,32] x (1,1,4,1)` call).
