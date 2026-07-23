# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** provenance not pinnable (metal_2.0 doc tree absent from checkout; recipe supplied standalone). Repo HEAD at audit: `033960ede6d 2026-07-23`. *(Carry this into the port report's Provenance section.)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. One `DeviceOperation` (`TilizeWithValPaddingDeviceOperation`), four `descriptor` factories — port them together.

- **Current concept:** `descriptor` (all 4 factories: SingleCore, MultiCoreDefault, MultiCoreBlockInterleaved, MultiCoreSharded)
- **Op-owned tensors:** none
- **Target concept:** `MetalV2FactoryConcept`
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · migration-risky pybind. All `no` / safe.

## Construct — to do

**Tensor bindings** (per binding, per factory):

- **SingleCore / MultiCoreDefault / MultiCoreBlockInterleaved:**
  - `input` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`. Drop the `Buffer* src0_buffer` RTA slot and the `TensorAccessorArgs(*src0_buffer).append_to(...)` CTA plumbing.
  - `output` — **Case 1** → same, on the eltwise/unary writer; drop the `Buffer* dst_buffer` RTA slot and `TensorAccessorArgs(*dst_buffer)` CTA.
- **MultiCoreSharded:**
  - `input` — **clean** (borrowed-memory DFB `c_1`) → bind via `DataflowBufferSpec::borrowed_from` the input buffer (replaces `cb_src0.buffer = a.buffer()`, `…sharded_program_factory.cpp:75`).
  - `output` — **clean** (borrowed-memory DFB `c_16`) → bind via `borrowed_from` the output buffer (replaces `cb_output.buffer = dst_buffer`, `:111`).

No Case 2 (no raw-pointer base arithmetic anywhere).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor is 2-arg; nothing to drop.

**CB endpoints:**

- **Self-loop** (single-toucher scratch/staging — bind the one kernel PRODUCER *and* CONSUMER):
  - MultiCoreBlockInterleaved `c_1` (temp alignment-staging buffer, `reader_unary_pad_multicore_both_dims.cpp:96-99`).
  - MultiCoreSharded `c_1` (borrowed input; self-loop **plus** `borrowed_from` input buffer).
  - MultiCoreSharded `c_2` (pad scratch, `reader_unary_pad_height_width_sharded.cpp:33,37`).
- **Legal 1:1** (bind PRODUCER + CONSUMER as-is, no action): all `c_0` input/staging CBs, all `c_16` output CBs.
- No dead CBs, no multi-binding.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader. (The block-interleaved reader's raw writes into `c_1` and the sharded reader's raw peeks are all single-toucher self-loops, not co-fills.)
- **Cross-op / shared kernels:** the op instantiates shared kernels it does not own — each is a single rewrite the whole port-together set must adopt at once:
  - `eltwise/unary/…/writer_unary_interleaved_start_id.cpp` and `…_wh.cpp` (broadly shared writers).
  - `data_movement/sharded/…/writer_unary_sharded.cpp` (shared sharded writer).
  - `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` (shared compute pool).
  - `data_movement/tilize/…/compute/tilize_wh.cpp` (in-family sibling compute).
  - `data_movement/common/kernels/common.hpp` `tt_memmove` (function-call escape; Device-2.0 native, pass `dfb`/`tensor` tokens through the `Noc`-first overload).
- **RTA varargs:** `reader_unary_pad_dims_split_rows_multicore.cpp:143-169` (MultiCoreDefault) reads a variable-count block-representation stream via a running `rt_arg_idx` bounded by runtime `n_block_reps` — port via the kernel-side vararg mechanism, don't name each element. The other three readers use nameable fixed-field RTAs.
