# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `bcf38615192 2026-08-03 docs(metal_2.0): add the op-porting recipe set` *(carry this line into the port report's Provenance section)*

## Shape of the op (read this first)

One device operation, **one** program factory — but `NLPConcatHeadsProgramFactory::create_descriptor`
branches on `in_sharded` into two configs that share **no** kernel, no CB layout, and no binding
classification. Almost every item below is answered per *config*, not per factory:

| | **INTERLEAVED** (`in_sharded == false`) | **SHARDED** (`in_sharded == true`) |
|---|---|---|
| Reader | `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` (op-private) | `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` (op-private), instantiated **twice** |
| Writer | `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` (**borrowed**) | same source as the reader, `WriterConfigDescriptor` |
| CBs | `cb_src0` (idx 0), plain double-buffered | `cb_src0` (idx 0) borrowed from input, `cb_out0` (idx 16) borrowed from output |
| Tensor access | `TensorAccessor` both ends | borrowed-memory DFB both ends; no accessor at all |

No compute kernel. No semaphores. Output CB index 16 exists only in the SHARDED config.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `create_descriptor(const NlpConcatHeadsParams&, const Tensor& input, Tensor& output)` returning a `ProgramDescriptor` (`device/nlp_concat_heads_program_factory.hpp:15-16`).
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`, plain.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash ·
  `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` · other
  migration-risky pybind. All verified against the code, not just the sheet.

## Construct — to do

**Tensor bindings** (per binding; classification splits by config):

- `input` — **INTERLEAVED: Case 1** (via `TensorAccessor`). Legacy delivers it as a bare `Buffer*` in
  reader RTA[0] (`device/nlp_concat_heads_program_factory.cpp:201`); the kernel reads
  `get_arg_val<uint32_t>(0)` (`...nlp_concat_heads.cpp:17`) and feeds it to
  `TensorAccessor(in0_args, in0_tensor_addr)` (`:31`). → Express as a `TensorParameter`; the kernel
  builds `TensorAccessor(tensor::input)`, and both the address RTA and the `TensorAccessorArgs<4>` CTA
  block disappear.
- `output` — **INTERLEAVED: Case 1**. Same shape: `Buffer*` in writer RTA[0] (`:210`) →
  `TensorAccessor(dst_args, dst_addr)` (`writer_unary_interleaved_start_id.cpp:13,31`).
- `input` — **SHARDED: clean** (borrowed-memory DFB). `cb_src0` is backed by the input buffer
  (`:150`), read via `cb_in0.get_read_ptr()` (`...sharded.cpp:43`). → `DataflowBufferSpec::borrowed_from`
  the input `TensorParameter`. No accessor, no address arg.
- `output` — **SHARDED: clean** (borrowed-memory DFB). `cb_out0` backed by the output buffer (`:163`),
  written via `cb_out0.get_write_ptr()` (`...sharded.cpp:44`). → `borrowed_from` the output
  `TensorParameter`.

Neither `Buffer*` RTA is a correctness hazard today (the framework patches `BufferBinding`s on cache
hits) — they are enumerated because the port replaces both with typed bindings.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor in this op passes an explicit page size. Nothing to drop.

**CB endpoints:**

- `(cb_src0, INTERLEAVED)` — **legal 1:1, no action.** Reader is the locked producer
  (`reserve_back`/`push_back`/`get_write_ptr` @ `...nlp_concat_heads.cpp:47,59,41`); the borrowed writer
  is the locked consumer (`wait_front`/`pop_front`/`async_write` @ `writer_unary_interleaved_start_id.cpp:40,43,41`,
  bound to index 0 via its `cb_id_out` CTA at `nlp_concat_heads_program_factory.cpp:118`).
- `(cb_src0, SHARDED)` — **set the multi-binding advanced option.**
- `(cb_out0, SHARDED)` — **set the multi-binding advanced option.**

  Why, since this looks like the dual-instance work-split that normally resolves to 1P+1C: it *is* that
  shape structurally (one kernel source pushed into two `KernelDescriptor`s differing only by
  `ReaderConfigDescriptor` / `WriterConfigDescriptor` and the head-split RTAs, `:95-109`, `:169-187`),
  **but the two co-touchers are not sync-free.** Each instance issues a real FIFO
  `cb_in0.reserve_back(block_size)` *and* `cb_out0.reserve_back(block_size)`
  (`...sharded.cpp:35-36`), which locks it to PRODUCER on both CBs. Two locked producers per node, on
  each CB, with no consumer anywhere (`push_back` is commented out at `:62`; nothing `wait_front`s).
  That census cannot be relabelled into 1P+1C, so the flag is forced. Do **not** "fix" this by deleting
  the vestigial `reserve_back` calls — that is a functional kernel change, off the whitelist.

- No dead CB; no CB to drop.

## Watch for

- **The `in_sharded && !out_sharded` hole — resolve this before you build the SHARDED spec.**
  `validate_on_program_cache_miss` (`device/nlp_concat_heads_device_operation.cpp:48-51`) only forbids a
  `HEIGHT_SHARDED` output when the input is sharded, so an **INTERLEAVED** output on a sharded input is
  reachable and produces a well-formed output spec. But the factory allocates `cb_out0` **only when
  `out_sharded`** (`nlp_concat_heads_program_factory.cpp:153`), while the SHARDED kernel
  unconditionally constructs and uses it (`...sharded.cpp:33,36,44`). Legacy tolerates this silently
  (UB, untested — no test parametrizes the mixed combination). Metal 2.0 will not: the kernel names
  `dfb::out`, so the spec must supply that binding, and a conditional binding is exactly the trap in
  `metal2_audit.md`'s conditional-named-arg territory — the CTA/name lookup is not gated by an
  `if constexpr` on the host side. **This is an ops-team question, not a porter decision.** Do not
  invent a semantic (do not add a `TT_FATAL`, do not silently always-allocate the CB) — surface it and
  get a ruling on whether the mixed config should be rejected outright.
- **CB endpoints (multi-binding):** `(cb_src0, SHARDED)` and `(cb_out0, SHARDED)`. The hidden-2nd-writer
  hunt is already done and came back **negative** — the op has zero semaphores, and both co-touchers are
  the two visible instances of one kernel source. You do not need to re-run it; the flag here comes from
  the FIFO-role rule, not from a concealed endpoint.
- **Cross-op / shared kernels:** the INTERLEAVED writer is a **borrowed** kernel —
  `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`,
  owned by `eltwise/unary`. **Rung 1 fails locationally: no `_metal2` sibling exists in that directory**
  (verified by `ls` at audit time — re-verify at port time, several metal2 branches are in flight).
  **Take rung 2**: create `writer_unary_interleaved_start_id_metal2.cpp` beside the original in
  `eltwise/unary/device/kernels/dataflow/`, convert the copy, point `KernelSpec::source` at it, and add
  the pointer comment to the legacy original. Name its bindings for the *kernel's* role
  (`dfb::out`, `tensor::output`, `args::num_pages`, `args::start_id`), not for this op — you are
  authoring the pool's canonical Metal 2.0 interface.

  ⚠ **A near-identical fork exists in the wrong place — read it, don't bind it.**
  `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`
  (landed in `cbde3d44ff3`, *[Cleanup] Port Typecast to Metal 2.0 #51397*) is a converted fork of this
  exact donor, but it sits in typecast's own tree rather than beside its donor — the same misplacement
  commit `5ecda11bb71` corrected for `transpose_wh`. It is **not** a rung-1 reuse target (the test is
  locational) and it is **not** yours to edit or relocate. Treat it as a reference for what the
  converted body should look like; put *your* fork beside the donor.

  The op's other two kernels (`reader_tm_tile_layout_nlp_concat_heads.cpp`,
  `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`) are **private to this op** — a repo-wide grep
  finds no other binder — so convert them in place, no fork needed.

  **Sunset list** (co-borrowers of the legacy `writer_unary_interleaved_start_id.cpp`, so the fork's
  retirement is trackable — **this is a coordination list, not authorization to convert the file in
  place**): ~35 non-quasar factories, across `data_movement/{concat,copy,tilize,tilize_with_val_padding,transpose,slice,permute,reshape_on_device,bcast}`,
  `reduction/{generic,prod}`, `matmul`, `embedding`, `kv_cache`, `eltwise/unary_backward/{gelu_bw,tanh_bw}`,
  `experimental/matmul/attn_matmul`, `experimental/unary_backward/gelu_backward`,
  `experimental/transformer/nlp_concat_heads_boltz`, `examples/example`, plus two
  `tt_metal/programming_examples` and two generic-op tests.

- **RTA varargs:** none. Every RTA in all three kernels is read at a literal index as a distinct field
  (interleaved reader 0–3, borrowed writer 0–2, sharded kernel 0–2) — no counted loop, no `arg_index++`,
  no data-selected index. **Name all of them.**

- **`experimental/quasar/**` is out of bounds.** Grepping for any of this op's kernel names or for
  `writer_unary_interleaved_start_id` will surface quasar copies and their `_metal2` files. They are not
  precedents, not reuse targets, and not a naming source.
