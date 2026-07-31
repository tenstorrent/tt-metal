# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

## Shape of the op

One device operation, one factory, one instantiation shape — interleaved + TILE only, enforced by `validate_on_program_cache_miss` (sharded and non-TILE inputs are rejected outright). Three CBs, three kernels, no semaphores, no multicast, no sharding branches.

- `GeluBackwardDeviceOperation` → `GeluBackwardProgramFactory::create_descriptor` (`device/gelu_backward_program_factory.cpp:17`)
- Kernels (all `SourceType::FILE_PATH`, all over `all_cores`):
  - reader — `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` (**borrowed**)
  - writer — `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (**borrowed**)
  - compute — `device/kernels/compute/eltwise_bw_gelu_approx_tanh.cpp` when `approximate == "tanh"`, else `device/kernels/compute/eltwise_bw_gelu_poly.cpp` (**op-owned**, sole consumer — port in place)

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `create_descriptor()` returning `tt::tt_metal::ProgramDescriptor`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (the readiness sheet's own `Porting Target` column independently agrees).
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported replacement) · pybind `create_descriptor` — all gate conjuncts — plus other migration-risky pybind, which surfaces as a `safe` warning that also fails the gate. All `no` on this op.

## Construct — to do

**Tensor bindings** (three, all Case 1 — express each as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)` and both the legacy `Buffer*` RTA slot *and* its `TensorAccessorArgs(...).append_to(...)` CTA plumbing disappear):

- `grad_output` — **Case 1**. Legacy: `Buffer*` in reader RTA slot 0 (`gelu_backward_program_factory.cpp:152`), CTAs appended @ `:85`; kernel `TensorAccessor(src0_args, src0_addr)` @ `reader_binary_interleaved_start_id.cpp:46`.
- `input` — **Case 1**. Legacy: `Buffer*` in reader RTA slot 1 (`:152`), CTAs @ `:86`; kernel `TensorAccessor(src1_args, src1_addr)` @ `reader…:53`.
- `output` — **Case 1**. Legacy: `Buffer*` in writer RTA slot 0 (`:156`), CTAs @ `:99`; kernel `TensorAccessor(dst_args, dst_addr)` @ `writer_unary_interleaved_start_id.cpp:31`.

No Case 2 anywhere — no kernel does raw pointer arithmetic on a tensor base, so you will not need the `get_bank_base_address` bridge.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — all three accessor constructions are two-argument. Nothing to drop.

**CB endpoints:** all legal — bind each of the three DFBs with one PRODUCER and one CONSUMER, exactly mirroring the legacy topology. No self-loop, no 1P+1C assignment call to make, no multi-binding flag, no dead-CB drop.

| DFB (legacy CB) | PRODUCER | CONSUMER |
|---|---|---|
| `c_0` — grad_output tiles (2 × `tile_size(grad_output.dtype())`) | reader | compute |
| `c_1` — input tiles (2 × `tile_size(input.dtype())`) | reader | compute |
| `c_2` — grad_in / output tiles (2 × `tile_size(output.dtype())`) | compute | writer |

**Runtime args — all nameable, none vararg.** Every kernel reads its args as distinct fields at fixed constant indices; name each one:

- reader (`:17-23`): `src0_addr`, `src1_addr`, `num_tiles`, `start_id`, `block_height`, `block_width`, `num_cores_y` — the first two become `tensor::` bindings; the remaining five stay named RTAs. **Keep all five**, including `block_height`/`block_width`/`num_cores_y`, even though this op passes `0u, 0u` for two of them and the branch that consumes all three is compile-time-dead here (CTA 0 is hardcoded `0` @ `:84`). The kernel is shared and reads them unconditionally at the top of `kernel_main`; dropping them is out of scope.
- writer (`:11-13`): `dst_addr` (→ `tensor::`), `num_pages`, `start_id`.
- compute (tanh `:24` / poly `:22`): `num_tiles`.

## Watch for

- **CB endpoints (multi-binding):** none. The hidden-second-writer hunt and the multi-reader hunt both came back empty — there is no `get_write_ptr()` / `fifo_wr_ptr` raw write anywhere in the four kernels, and no semaphore exists to coordinate one. Each compute-side `copy_tile(cb, …)` / `pack_tile(0, cb)` is a peek on a binding that kernel already holds, not a second toucher.

- **Cross-op / shared kernels — this is the substantive part of the port.** Both dataflow kernels are borrowed and **no `_metal2` fork exists** for either. A tree-wide check confirms **zero** `_metal2` files anywhere outside `experimental/quasar/**`, so rung 1 (reuse) is unavailable and you are at **rung 2: create the fork beside the original**, leave the original untouched apart from the pointer comment.

  - `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` → create `reader_binary_interleaved_start_id_metal2.cpp` in that same directory. Other binders — **sunset list, not authorization to convert in place**: `eltwise/unary_backward/gelu_bw`, `eltwise/unary_backward/tanh_bw`, and `tests/ttnn/unit_tests/gtests/test_generic_op.cpp`. *(Heads-up: this file's own header comment @ `:5-7` calls it a temporary copy expected to be deleted or refactored — worth a glance before you invest in the fork.)*
  - `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` → create `writer_unary_interleaved_start_id_metal2.cpp` in that same directory. This one is **very** broadly shared — roughly 34 non-quasar factories bind this exact path (tilize, tilize_with_val_padding, reduction/generic, reduction/prod, transpose, slice, concat, copy, permute, reshape_on_device, bcast, typecast, embedding, examples, attn_matmul, nlp_concat_heads, kv_cache, `gelu_bw`, `tanh_bw`, …). Full list in `METAL2_PREPORT_AUDIT.md` → Heads-ups. Again: **sunset list, not authorization.**
  - **Name the fork bindings for the kernel, not for gelu-backward.** These names become the interface every later consumer inherits, and they will not be able to rename them. Take them from the kernel's own vocabulary — `tensor::src0` / `tensor::src1` / `tensor::dst`, `dfb::in0` / `dfb::in1` / `dfb::out` — not from this factory's locals (`src0_buffer`, `cb_grad_out`).
  - **Same-named private copies are not consumers.** `matmul/`, `data_movement/slice/`, `point_to_point/`, and the layernorm families each carry their *own* `writer_unary_interleaved_start_id*.cpp`. Different files; don't touch them and don't count them.
  - **Negative pointer — stay out of `ttnn/cpp/ttnn/operations/experimental/quasar/`.** It holds a copy of the reader (`quasar/binary/.../reader_binary_interleaved_start_id.cpp`) and several `*_metal2.cpp` writer variants (`quasar/reduction/generic/…`, `quasar/tilize_with_val_padding/…`). Those are not forks to reuse and not a naming source — at least one carries idioms the whitelist forbids. If a search lands you there, close the file.
  - The two **compute** kernels are op-owned with no other consumer (`gelu_bw` binds its own same-named private copies under `eltwise/unary_backward/gelu_bw/device/kernels/compute/`). Convert them in place — no fork.

- **RTA varargs:** none — prefer named RTAs throughout (see the arg list above).

- **CB-metadata free functions become DFB methods.** Two sanctioned-at-Device-2.0 lookups move onto the DFB object under kernel-side whitelist rule 7: `get_tile_size(cb_id_in0/in1)` @ `reader…:45,52` and `get_local_cb_interface(cb_id_out).fifo_page_size` @ `writer…:19`. Both are inside the kernels you are forking, so they are yours to convert.

- **Compute kernels still include `api/dataflow/circular_buffer.h`** (tanh `:18`, poly `:19`) and construct `CircularBuffer` objects. These are op-owned files converting in place — the DFB swap applies to them too; don't leave the stale include behind.
