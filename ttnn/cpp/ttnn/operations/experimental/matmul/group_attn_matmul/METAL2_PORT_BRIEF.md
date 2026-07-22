# Metal 2.0 Port Brief — `experimental/matmul/group_attn_matmul`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `b6dadc46ee0 2026-07-21 docs: fix metal_2.0 doc links (READ_ME_FIRST rename + two stragglers)` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — single DeviceOperation (`GroupAttnMatmulDeviceOperation`), single ProgramFactory (`GroupAttnMatmulProgramFactory::create_descriptor`).
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no`. (Note: the factory comments *mention* `compute_program_hash`, but no such override exists — the op uses the default hash. See the audit's Misc anomalies.)

## Construct — to do

**Tensor bindings** (per binding — each has a per-config split; the sharded path is a borrowed-memory DFB, the interleaved path a `TensorAccessor`):

- `input_tensor_a` (in0) — **Case 1** (via `TensorAccessor`) in the **interleaved** config → express as `TensorParameter` / `TensorBinding`; the writer builds `TensorAccessor(tensor::name)` (replacing `TensorAccessor(in0_args, src0_addr)` at `writer_transformer_group_attn_matmul.cpp:65`). In the **IN0_SHARDED** config it is a borrowed-memory DFB read from `c_0` → `DataflowBufferSpec::borrowed_from(input_tensor_a)`; kernel access unchanged.
- `input_tensor_b` (in1) — **Case 1** (via `TensorAccessor`) in the **interleaved** config → `TensorParameter`; the reader builds `TensorAccessor(tensor::name)` (replacing `TensorAccessor(in1_args, src1_addr)` at `reader_mcast_transformer_group_attn_matmul.cpp:83`). In the **IN1_SHARDED** config it is a borrowed-memory DFB read from `c_2` → `borrowed_from(input_tensor_b)`.
- `output` — **Case 1** (via `TensorAccessor`) in the **interleaved** config → `TensorParameter`; the writer builds `TensorAccessor(tensor::name)` (replacing `TensorAccessor(out_args, dst_addr)` at `writer_transformer_group_attn_matmul.cpp:71`). In the **OUT_SHARDED** config it is borrowed-memory (`c_5`) → `borrowed_from(output)`.

All three are delivered today as framework `Buffer*` runtime args (`src0_buffer`, `src1_buffer`, `dst_buffer`); replacing each with a typed `TensorParameter` retires the `Buffer*` arg and its `TensorAccessorArgs` plumbing. No Case 2 — no raw-pointer arithmetic to preserve.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — all three accessor sites are already 2-arg.

**CB endpoints:**
- `c_2` (in1 sharded borrowed CB) — **self-loop** (single toucher: the reader's raw `get_read_ptr` peek), IN1_SHARDED config only. Bind the reader as both PRODUCER and CONSUMER.
- All other CBs (`c_0`, `c_1`, `c_3`, `c_4`, `c_5`) — legal **1:1** (one locked producer + one locked consumer); bind roles to match their FIFO ops.
- Sharded configs back `c_0` / `c_2` / `c_5` with `borrowed_from` (see the tensor bindings above); no allocation of your own for those.

## Watch for

- **CB endpoints (multi-binding):** none — no multi-binding and no hidden second writer to hunt. The mcast fill of `c_1` is the single reader source writing to remote `c_1` instances against its own producer binding; it does **not** create a second local endpoint. Bind `c_1` plainly 1P (reader) + 1C (compute).
- **Cross-op / shared kernels:** none — all three kernels are op-owned and in-directory. No shared-kernel port-together coupling.
- **RTA varargs:** the reader consumes a variable-length RTA tail — `in1_mcast_sender_noc_x[num_x]` then `in1_mcast_sender_noc_y[num_y]`, read by pointer via `get_arg_addr(i)` with the cursor advanced by a runtime count (`reader_mcast_transformer_group_attn_matmul.cpp:61-64`). Port this block as **RTA varargs** (kernel-side vararg mechanism per kernel-side whitelist rule 4) — do **not** try to name each element. All the other RTAs read at distinct constant indices → name those normally.
