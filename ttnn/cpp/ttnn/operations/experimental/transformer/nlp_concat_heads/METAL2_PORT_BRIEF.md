# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** version cannot be pinned — the `metal_2.0` docs tree is untracked in this checkout (`git log` prints nothing for it). *(Carry this line into the port report's Provenance section.)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `NLPConcatHeadsProgramFactory::create_descriptor` (`device/nlp_concat_heads_program_factory.cpp:19`), single factory in `program_factory_t`, two config branches (interleaved / sharded).
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a non-`none` `TensorParameter relaxation` · `get_dynamic_runtime_args`. Also absent (none of these gate, but this op happens to have none): custom hash, `override_runtime_arguments`, pybound `create_descriptor`.
- The op is a **pure data-movement op** — no compute kernel in either config.

## Construct — to do

**Tensor bindings** (per binding, per config branch):

- `input` — interleaved: **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; reader kernel builds `TensorAccessor(tensor::<name>)`, and the legacy `Buffer*` RTA (factory line 201) + `TensorAccessorArgs` CTA plumbing (factory line 117, kernel `TensorAccessorArgs<4>()` line 27) disappear.
- `input` — sharded: **clean** — borrowed-memory CB → `DataflowBufferSpec::borrowed_from` the input `TensorParameter` (legacy `CBDescriptor .buffer = in0_buffer`, factory line 150).
- `output` — interleaved: **Case 1** → bind as `TensorParameter`; the `_metal2` writer fork already consumes it as `tensor::dst` (legacy `Buffer*` RTA at factory line 210, `TensorAccessorArgs` at line 119).
- `output` — sharded: **clean** — borrowed-memory CB → `borrowed_from` (legacy `.buffer = out_buffer`, factory lines 153–165).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor in the op passes one.

**CB endpoints:**

- `cb0`, interleaved: legal 1:1 — reader PRODUCER (FIFO `reserve_back`/`push_back`; its own `get_write_ptr` peek at reader kernel line 41 rides the PRODUCER binding), writer-fork CONSUMER (`dfb::out`).
- `cb0` + `cb16`, sharded: two touchers each — the two same-source instances of the sharded kernel (dual-instance work-split). **Intended disposition: 1P+1C** (both touchers are role-free raw peeks: `get_read_ptr`/`get_write_ptr`, sharded kernel lines 43–44) — **but** the vestigial `reserve_back` calls (lines 35–36) strictly lock both instances as producers. Resolve the open question first (below, and `METAL2_PREPORT_AUDIT.md` → Questions): if the dead sync is stripped (ops-team approval), bind 1P+1C; if it must stay, a CONSUMER-bound instance keeping its `reserve_back` is the blocker — fall back to the multi-binding advanced option and note the 2-producer/0-consumer shape in the port report.
- `cb16` exists **only** in the sharded config — the legacy factory already allocates it conditionally (`if (out_sharded)`, factory line 153); keep the DFB spec conditional the same way. Do **not** drop it.

## Watch for

- **CB endpoints (sharded):** no hidden second writer and no third toucher — the census is exactly the two visible kernel instances; the only wrinkle is the dead `reserve_back` pair above. Do not silently delete those lines yourself — that removal is off the kernel-side whitelist; it needs the explicit go-ahead recorded in the audit's open question.
- **Cross-op / shared kernels:** `writer_unary_interleaved_start_id.cpp` (eltwise/unary) → **`_metal2` fork already exists** at `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` — **bind it, don't re-fork**, and adopt its vocabulary: `dfb::out` (bind `cb0` under this name for the writer's CONSUMER side), `tensor::dst` (the output binding), named args `args::num_pages`, `args::start_id`. Fit confirmed: this op's writer RTAs map 1:1 (`num_blocks_per_core * per_tensor_tiles` → `num_pages`, `num_blocks_written * per_tensor_tiles` → `start_id`); no `OUT_SHARDED`/`BACKWARDS` defines needed (this op uses the fork only on the interleaved path). The fork is **read-only** to you (it has consumers). Ignore the duplicate fork in `copy/typecast/` (`tensor::output`) — bind the eltwise/unary one. Other legacy-copy binders (concat, reshape_on_device, slice, tilize, transpose, gelu_bw/tanh_bw, embedding, examples, attn_matmul, nlp_concat_heads_boltz, matmul multicore, reduction/generic; issue #52228): **sunset list, not authorization to convert the kernel in place**.
- **RTA varargs:** none — every arg in all three kernels is a fixed nameable scalar; prefer named RTAs throughout. Note the sharded reader instance's two zeros (`{nheads_first_risc, 0, 0}`, factory lines 174–180) are real named fields (`start_read_offset_bytes`, `start_write_offset_bytes`) that happen to be 0 for that instance.
- **Own kernels are unshared** — both `reader_tm_tile_layout_nlp_concat_heads*.cpp` files are bound only by this op (verified); convert them in place, no fork.
- **Dual instantiation of one source (sharded):** the legacy factory pushes the same `kernel_source` into two `KernelDescriptor`s differing only by Reader/Writer config and RTAs, sharing one CTA vector (factory lines 86–109; reader copies, writer moves). Mirror as two `KernelSpec`s off one source with per-instance runtime args.
- **Sharded kernel local-copy idiom:** L1→L1 self-copy via `UnicastEndpoint` + `my_x[noc_id]`/`my_y[noc_id]` (sharded kernel lines 38–44) — ports as-is; not a Device 2.0 holdover, no swap needed.
- **Legacy writer is already part-modernized:** the legacy copy itself uses kernel-side `DataflowBuffer` — the port for the writer is purely a re-point at the existing fork plus factory-side binding/args wiring; no kernel edit on that file at all.
