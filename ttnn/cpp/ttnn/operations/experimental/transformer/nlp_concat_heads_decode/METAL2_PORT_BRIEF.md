# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** *not pinnable* — the `metal_2.0/` docs tree is untracked in this checkout (`git log` over it prints nothing); carry this note into the port report's Provenance section.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — both factories (`NLPConcatHeadsDecodeProgramFactory`, `NLPConcatHeadsDecodeSubcoregridsProgramFactory`), one shared DeviceOperation (`NLPConcatHeadsDecodeDeviceOperation`), selected by `operation_attributes_t::on_subcoregrids`. Port both; `create_program_artifacts` must live in a `program_factory_t` variant for each.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (both factories).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a non-`none` `TensorParameter relaxation` · `get_dynamic_runtime_args` (deprecated hook). Also absent here, for the record (none of these gate): custom `compute_program_hash`, `override_runtime_arguments`, pybound `create_descriptor` — this op has **none of them**, so there is no hash to preserve, no cache-hit method to translate, and no pybind to delete.

## Construct — to do

**Tensor bindings** (per binding; identical in both factories):

- `input` — **Case 2** (raw pointer) → bind as `TensorParameter` / `TensorBinding`. Today the factory pushes `Buffer* in_buffer` into the `RTArgList` (`device/nlp_concat_heads_decode_program_factory.cpp:130`, `device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp:137`) and the kernel consumes the base **raw** as `q_start_addr` (arg 1), assembling remote NoC addresses itself (`{.noc_x, .noc_y, .addr = qkv_read_addr}`). Port: the kernel pulls the base via the sanctioned `TensorAccessor::get_bank_base_address` bridge and keeps the existing raw walk **unchanged** — do not rewrite the walk into `TensorAccessor` iteration.
- `output` — **clean** (borrowed-memory DFB) → `DataflowBufferSpec::borrowed_from` the output `TensorParameter`, replacing `CBDescriptor{.buffer = output.buffer()}` (`device/nlp_concat_heads_decode_program_factory.cpp:46-55`, `..._subcoregrids_program_factory.cpp:56-65`). Kernel keeps its `get_write_ptr()`-based raw writes (on the DFB object after the CB→DFB swap).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — the op constructs no `TensorAccessor` anywhere today.

**CB endpoints:** assign **1P+1C** on `(c_16 q_output, both factories, all configs)` — the only CB. Two touchers per node: the Reader-config and Writer-config instances of the same kernel source, both sync-free raw writers (no FIFO ops exist in either kernel). Bind one instance PRODUCER, the other CONSUMER — cosmetic on Gen1, satisfies the validator. No self-loops, no multi-binding flag, no dead CBs.

## Watch for

- **CB endpoints (dual-instance work-split, not multi-binding):** each factory instantiates the *same* kernel source twice (`ReaderConfigDescriptor` + `WriterConfigDescriptor`) over the full `q_cores` range, split by CTA index 6 (phase 1 = left half-tile lines, phase 2 = right). Only two touchers, nothing consumes the CB → **1P+1C**, do not reach for the multi-binding advanced option.
- **Cross-op / shared kernels:** none — both kernel files are op-owned and single-consumer; no `_metal2` fork exists (this port creates none, since nothing is shared). Do not look at `experimental/quasar/**` for precedent — no quasar copy of this op exists anyway.
- **RTA varargs (both kernels):** the NoC-coordinate blocks are genuine varargs — args `2 .. 2+num_x+num_y` (default kernel: `reader_tm_tile_layout_nlp_concat_heads_decode.cpp:31-32`) / `2 .. 2+2*in_num_cores` (subcoregrid kernel: lines 31-32), counts CTA-driven and varying per instantiation, read today via a raw `(tt_l1_ptr uint32_t*)(get_arg_addr(2))` pointer walk. Port them with the vararg mechanism and replace the pointer walk with vararg indexing (note: `get_vararg` is read-only; these kernels only read, so no local copy needed). **Name the two leading scalars** — arg 0 `in_tile_offset_by_head` stays a named per-core RTA; arg 1 `q_start_addr` disappears into the input tensor binding. The coord blocks are identical across cores (host builds them once), so CRTA varargs are an option; arg 0 is genuinely per-core.
- **CTA plumbing that dissolves:** CTA 2 carries the CB index (`q_output_cb_index`) — replaced by the `dfb::` token; drop it from the CTA list rather than carrying a dead arg. CTA 6 (phase selector) is set positionally on a copied vector (`writer_compile_time_args[6] = 2`, `device/nlp_concat_heads_decode_program_factory.cpp:103` / `..._subcoregrids_program_factory.cpp:110`) — give it a named CTA per instance.
- **Byte-identical RTAs on both instances:** the factories build one `rt_args` per core and emplace it into both the reader and writer descriptors — keep that symmetry in the spec; only the phase CTA differs between the two KernelSpecs.
- **Kernel-side `CircularBuffer cb_q_out(cb_id_q_out)` → DFB token:** the kernels are already Device 2.0, so this port is a binding-layer change (CB wrapper → `DataflowBuffer` from the `dfb::` token; `api/dataflow/circular_buffer.h` include → the DFB header), not an idiom rewrite — the `Noc`/`UnicastEndpoint`/`CoreLocalMem` read machinery stays untouched.
- **Compute opt level:** no compute kernels exist in this op, so the Metal 2.0 KernelSpec O2-default trap for compute kernels does not apply here.
