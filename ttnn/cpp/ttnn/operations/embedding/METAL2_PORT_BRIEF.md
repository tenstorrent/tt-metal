# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/embedding`

> **Scoped brief — two of three factories.** The audit cleared every gate for `EmbeddingsRMProgramFactory` and `EmbeddingsTilizedIndicesProgramFactory`. **`EmbeddingsFusedProgramFactory` is blocked and out of scope for this port** — do not touch it. This is your actionable input for the two cleared factories; the full record is in `METAL2_PREPORT_AUDIT.md`.

**In scope:**

- `EmbeddingsRMProgramFactory` (`device/embeddings_rm_program_factory.cpp`) — row-major output
- `EmbeddingsTilizedIndicesProgramFactory` (`device/embeddings_tilized_indices_program_factory.cpp`) — TILE-layout indices

**Out of scope — blocked:** `EmbeddingsFusedProgramFactory` (`device/embeddings_fused_program_factory.cpp`) and its kernels `device/kernels/dataflow/embeddings_tilize.cpp` and `device/kernels/compute/tilize_chunked.cpp`. It carries a Type-2 offset-base-pointer wall at `embeddings_tilize.cpp:36` that the ops team must resolve first. Leave the factory, its two kernels, and their host-side wiring exactly as they are.

**Gates cleared (for the two in-scope factories):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `0efcf9f88ae 2026-08-17 docs(metal_2.0): CTA varargs are in, and five columns read present-tense` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); both in-scope factories port to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — each factory is a `static ProgramDescriptor create_descriptor(const EmbeddingParams&, const EmbeddingInputs&, Tensor&)`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`, for both.
- **One device operation, three factories in the variant.** `EmbeddingsDeviceOperation::program_factory_t` (`device/embedding_device_operation.hpp:24-28`) holds all three, and `select_program_factory` (`device/embedding_device_operation.cpp:17-26`) dispatches on index layout and the `tilized` attribute. The blocked factory stays in the variant on the legacy path while the two in-scope ones move — expect the device-op to carry a mixed set of factory concepts for the duration.
- **No custom hash, no `override_runtime_arguments`, no pybound `create_descriptor`** — nothing to preserve, translate, or delete on any of those three fronts.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a non-`none` `TensorParameter relaxation` · `get_dynamic_runtime_args` (deprecated hook).

## Construct — to do

**Tensor bindings** (per binding, per factory):

`EmbeddingsRMProgramFactory`:

- `input_tensor_arg` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::<name>)`. The legacy `Buffer*` push at `embeddings_rm_program_factory.cpp:264` and the `TensorAccessorArgs(*a.buffer()).append_to(...)` at `:188` both disappear, as does `embeddings.cpp:15,37,39`.
- `weight_arg` — **Case 1** → same treatment. Legacy sites: `Buffer*` at `:265`, args at `:189`, kernel at `embeddings.cpp:16,38,40`.
- `output` — **Case 1 in the interleaved configs**, where a writer kernel exists. Legacy sites: `Buffer*` at `:280` (chunked) / `:283` (non-chunked), args at `:219` / `:232`, kernel at `embeddings_rm_writer_chunked.cpp:15,24,26` / `writer_unary_stick_layout_interleaved_start_id.cpp:12,18,20`.
- `output` — **clean in the height-sharded config.** No writer kernel is created (`:211`); the tensor is reached through the borrowed-memory CB set up at `:133-135` (`out_cb_desc.buffer = out_buffer`). Port it as `DataflowBufferSpec::borrowed_from` the output `TensorParameter` — not as a Case-1 accessor.

`EmbeddingsTilizedIndicesProgramFactory`:

- `input_tensor_arg` — **Case 1**. Legacy sites: `Buffer*` at `embeddings_tilized_indices_program_factory.cpp:209`, args at `:144`, kernel at `embedding_ind_tilized.cpp:16,33,35`.
- `weight_arg` — **Case 1**. Legacy sites: `Buffer*` at `:210`, args at `:145`, kernel at `embedding_ind_tilized.cpp:17,34,36`.
- `output` — **Case 1**. Legacy sites: `Buffer*` at `:224`, args at `:170`, kernel at `writer_unary_stick_layout_interleaved_start_id.cpp:12,18,20`. This factory has no sharded-output branch, so there is one config here, not two.

Every pointer above arrives via the **`Buffer*`-binding form** — the factories push the `Buffer*` object into `KernelDescriptor::RTArgList`, never `->address()`. That means none of them is the stale-address correctness hazard; they are routine typed-binding conversions.

**TensorParameter relaxation:** none — the only value that reaches a brief.

**TensorAccessor 3rd arg:** drop the redundant page-size argument at `embeddings_rm_writer_chunked.cpp:26` (`TensorAccessor(dst0_args, dst_addr, output_page_size)` → the two-argument form, then the token form). It is **Class 2** — an interleaved accessor receiving the true logical page size, so the value is inert twice over. No `dynamic_tensor_shape` needed. No other accessor in either in-scope factory passes a 3rd argument.

**CB endpoints:**

`EmbeddingsRMProgramFactory`:

- `c_0` (output staging), interleaved configs (chunked and non-chunked) — **legal 1:1**, no action. Reader is the locked producer (`embeddings.cpp:71,77`), the writer the locked consumer. The chunked writer's `get_read_ptr()` at `embeddings_rm_writer_chunked.cpp:34` is a peek on its own consumer binding, not a third endpoint — do not read it as one.
- `c_0` (output), height-sharded config — **self-loop** (one toucher: the reader; no writer kernel exists) **and `borrowed_from`** the output `TensorParameter`. Bind the reader both PRODUCER and CONSUMER.
- `c_1` (index scratch), all configs — **self-loop**. Reader only: `reserve_back(1)` at `embeddings.cpp:47`, `get_write_ptr()` at `:48`, `push_back(1)` at `:96`.
- `c_2` (local weight cache), `PADDED` / `BINARY` — **self-loop**. Reader only, via `prepare_local_cache` (`embeddings_common.hpp:38-41`, `:44-51`).
- `c_2`, `GENERIC` — **conditional DFB.** The legacy factory already allocates it only under `PADDED`/`BINARY` (`embeddings_rm_program_factory.cpp:151-173`), so make the `DataflowBufferSpec` conditional on the same predicate. **Keep the CTA carrying its index conditional too** — all three readers currently read that index unconditionally (`embeddings.cpp:25`) even though nothing uses it under `GENERIC`, and an unconditional binding token pointing at a buffer that was never specced is a different proposition from an unused `uint32_t`.

`EmbeddingsTilizedIndicesProgramFactory`:

- `c_0`, all configs — **legal 1:1**, no action. Note `output_cb_index = src0_cb_index` at `embeddings_tilized_indices_program_factory.cpp:132`: one CB is both the reader's weight-staging buffer and the writer's output CB. That is what makes it a genuine producer/consumer pair, not two CBs to merge.
- `c_1` (index scratch), all configs — **self-loop** (reader only: `embedding_ind_tilized.cpp:47,48,128`).
- `c_2`, `PADDED` / `BINARY` — **self-loop**; `GENERIC` — **conditional DFB**, exactly as in the RM factory (`embeddings_tilized_indices_program_factory.cpp:108-130`).

Nothing needs the multi-binding advanced option, and there is no dead CB to drop.

## Watch for

- **CB endpoints (multi-binding):** none. The op declares **no semaphores at all**, so the hidden-second-writer face cannot occur here — you can skip that hunt with confidence rather than by assumption.

- **The pad-token runtime-arg slot is wrong in `EmbeddingsTilizedIndicesProgramFactory`, and naming the args would silently fix it.** The reader takes its pad token from slot 6 (`embedding_ind_tilized.cpp:42` passes `pad_token_arg_idx = 6`, consumed at `embeddings_common.hpp:37`), but this factory puts `col_offset % FACE_HEIGHT` in slot 6 (`embeddings_tilized_indices_program_factory.cpp:215`) and the real pad token in slot **7** (`:217`). Slot 7 is never read. The other two readers have the pad token at slot 6 and are correct.

  **Preserve the current behavior.** A named-argument port naturally binds `pad_token` to the pad token, which *changes* what the kernel reads under `EmbeddingsType::PADDED` — a functional change, which this port does not make. Reproduce the existing mapping (the value the kernel reads today is the face-column index), note it prominently in the port report, and leave the fix to the ops team. If reproducing it cleanly turns out to be awkward under named args, stop and raise it rather than picking a behavior.

- **Cross-op / shared kernels:** both in-scope factories instantiate `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` from the shared pool. **No `_metal2` fork exists yet — this port creates the first one** beside the original; do not convert the original in place. Other factories binding the same file, as a **sunset list, not authorization to convert the kernel in place**: `data_movement/concat` (`concat_program_factory.cpp:234`, row-major path) and `data_movement/copy` (`copy_same_memory_config_program_factory.cpp:39`, row-major interleaved path). The legacy copy retires when the last of the three migrates.

  Watch the name: `data_movement/slice` has its **own** near-identically-named file (`slice/device/kernels/dataflow/slice_writer_unary_stick_layout_interleaved_start_id.cpp`) and is *not* a co-borrower. Don't fork the wrong one, and don't count slice as a consumer.

- **`embeddings_common.hpp` is shared with the blocked factory.** All three readers include it — including `embeddings_tilize.cpp`, which is out of scope. Anything you change in that header lands on the blocked factory too. Its shared surface is four file-scope globals (`pad_token`, `pad_local_addr`, `zero_local_addr`, `one_local_addr` — `:24-27`) and two function templates (`prepare_local_cache` at `:30`, `read_token_async` at `:59`). If the CB→DFB conversion forces a signature or type change there, you cannot make it in place without touching the blocked kernel — treat that as an assumption-violation stop and raise it.

- **RTA varargs:** none. Every runtime arg in both factories is read at a constant index as a distinct field, and every compile-time arg at a constexpr index — name them all. The argument that looks variable is not: `prepare_local_cache`'s `pad_token_arg_idx` (`embeddings_common.hpp:35`) is a compile-time-constant default supplied at each call site, so the read is at a fixed slot. No CTA varargs either.

- **Includes need converting, not just swapping.** Both readers, the chunked writer, and `embeddings_common.hpp` include `api/dataflow/circular_buffer.h` and use the `CircularBuffer` wrapper. Moving to `DataflowBuffer` / `api/dataflow/dataflow_buffer.h` is part of the binding change. For the target shape, `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` is a good in-tree reference: it is already on `DataflowBuffer` and has a checked-in `_metal2` fork beside it.

- **`prepare_local_cache` reserves without committing, on purpose.** `embeddings_common.hpp:38-41` and `:44-51` call `reserve_back` + `get_write_ptr` and never `push_back` — the `c_2` CB is a local scratch cache nothing drains. Don't "balance" it. The same is nearly true of `c_1`, which each reader reserves once at the top and commits at the very end (`embeddings.cpp:47` / `:96`, `embedding_ind_tilized.cpp:47` / `:128`) purely so the CB is left balanced; the accompanying comment says so. Preserve both as-is — they are why the self-loop bindings are the right expression.

- **Confirm rather than swap blind: `get_local_cb_interface`.** The in-scope kernels don't use it, but the shared writer's sibling (`writer_unary_interleaved_start_id.cpp:24`) does, and if your fork of the stick writer needs page-size metadata, the DFB exposes its own accessor set — reach for the object's method, not the free function.

- **One warning about a directory you will trip over.** `ttnn/cpp/ttnn/operations/experimental/quasar/` holds whole-op pre-port copies, including `_metal2` kernels that look like solved versions of problems in front of you. They are deliberately hacky and carry idioms this port forbids. Nothing there is a precedent, a naming source, or a fork to reuse.
