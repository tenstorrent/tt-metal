# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** none
- **MeshWorkload:** not needed
- **Pybind `create_descriptor`:** none
- **Other risky pybind:** none
- **Custom `override_runtime_arguments`:** none

## Construct — to do

**Tensor bindings** (per binding):

- `input` (`in_buffer`) — **Case 1** (assumed) → re-express via `TensorParameter` / `TensorBinding` (kernel builds `TensorAccessor(ta::input)`). The factory pushes `in_buffer` as a `Buffer*` RTA at `nlp_concat_heads_decode_program_factory.cpp:130` and `nlp_concat_heads_decode_subcoregrids_program_factory.cpp:137`. The kernel reads it as `q_start_addr = get_arg_val<uint32_t>(1)` and uses it as a raw base address for sub-tile face-row NoC reads. **Note:** there is an open question (see `METAL2_PREPORT_AUDIT.md` → Questions for the user) about whether the sub-tile face-row access granularity is expressible via `TensorAccessor` (Case 1) or requires the `get_bank_base_address` bridge (Case 2). Confirm with the user before starting construction on this binding.

- `output` (fake CB) — resolved via the **fake-CB workaround** rather than a standard `TensorParameter` binding. The output CB (`c_16`) is declared with `.buffer = output.buffer()` but has no LLK FIFO producer or consumer — it is used purely as an L1 address anchor via `cb_q_out.get_write_ptr()`. The port replaces this with the sanctioned fake-CB workaround (see the porting recipe). The write-pointer call (`get_write_ptr()` at `reader_tm_tile_layout_nlp_concat_heads_decode.cpp:49` and `reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp:48`) becomes the address-extraction point in the workaround.

**Custom hash:** none

## Watch for

- **Fake CBs:** CB `c_16` in both factories — `.buffer = output.buffer()` at `nlp_concat_heads_decode_program_factory.cpp:46–55` and `nlp_concat_heads_decode_subcoregrids_program_factory.cpp:56–65`. No FIFO semantics; the port applies the fake-CB workaround from the recipe. The kernel's `cb_q_out.get_write_ptr()` call becomes a direct base-address extraction in the Metal 2.0 form.

- **Dual-instantiation kernel pattern:** Both factories instantiate the same kernel file twice (reader + writer) with the same RTAs but different CTA `[6]` (`PHASES_TO_READ = 1` vs. `2`). This is intentional (RISC0/RISC1 phase split). When translating to two `KernelSpec`s, preserve the `ReaderConfigDescriptor` / `WriterConfigDescriptor` distinction and the per-instantiation CTA difference.

- **Cross-op / shared kernels:** none — both kernel files are owned by this op.

- **RTA varargs:** none — the `get_arg_addr(2)` pointer-cast pattern reads noc-coord arrays whose sizes are compile-time-known CTAs; not true RTA varargs.
