# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/fill_pad`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `c1349c0d941 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

Two factories under one `FillPadDeviceOperation`, both ported together — they share `fill_pad_compute.cpp` and `fill_pad_dataflow_common.hpp`:

- `FillPadProgramFactory` — DRAM interleaved + DRAM-sharded
- `FillPadL1ShardedProgramFactory` — all L1-sharded (HEIGHT/WIDTH/BLOCK)

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); both factories port to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (both factories)
- **Op-owned tensors:** none
- **Target concept:** `MetalV2FactoryConcept`
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — plus op-owned tensors and genuine multi-program. All `no` / absent on this op.

## Construct — to do

**Tensor bindings** (single binding `input`, classification differs by factory):

- `input` @ **`FillPadProgramFactory`** — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernels build `TensorAccessor(tensor::name)` instead of `TensorAccessor(src_args, buf_addr, tile_bytes)`. The RTA-slot-0 address (`tens_buffer`) and the `TensorAccessorArgs<10>()` plumbing both disappear. Sites: `fill_pad_reader.cpp:86-87`, `fill_pad_writer.cpp:80-81`; host `fill_pad_program_factory.cpp:173,192,293,295`.
- `input` @ **`FillPadL1ShardedProgramFactory`** — **Case 2** (raw pointer) → bind the tensor, pull the base via `get_bank_base_address`, leave the raw `UnicastEndpoint` address arithmetic unchanged. Do **not** rewrite it into `TensorAccessor` iteration. Sites: `fill_pad_sharded_reader.cpp:46,70,90`, `fill_pad_sharded_writer.cpp:54,92`; host `fill_pad_program_factory.cpp:619-622`.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** drop the redundant page-size arg (Class 2) @ `fill_pad_reader.cpp:87` and `fill_pad_writer.cpp:81`. These are folded into the Case-1 rewrite above (`TensorAccessor(tensor::name)` supplies `aligned_page_size` implicitly). No `dynamic_tensor_shape`.

**CB endpoints:** all legal — every CB is a plain 1P+1C FIFO. No self-loop, no multi-binding, no dead-CB drop. CBs: `cb_data_in` (c_0, reader→compute), `cb_right_mask` (c_1, writer→compute, under `has_right_pad`), `cb_bot_mask` (c_2, writer→compute, under `has_bottom_pad`), `cb_data_out` (c_16, compute→writer).

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** no borrowed kernels. In-op, `fill_pad_compute.cpp` and `fill_pad_dataflow_common.hpp` are shared by both factories — do the CB→DFB / named-token rewrite once and keep both factories consistent in the same change.
- **RTA varargs:** none — all RTAs are fixed-count, distinct fields; name each. (Reader/writer slots 0–6, compute 0–2, sharded 0–4.)
