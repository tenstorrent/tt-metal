# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding`

> **ADDENDUM (post-port, 2026-09-09) — the brief's scope no longer matches the tree.**
> This brief scoped **all five** factories, and the port delivered all five. Owners then asked for a
> **partial port**: `MultiCoreInterleaved` and `MultiCoreBlockInterleaved` were **reverted to
> `create_descriptor`** because they are the only two reachable from untilize codegen's live-L1
> native fallback. `SingleCore`, `MultiCoreSharded` and `MultiCoreNDSharded` remain on
> `ProgramSpecFactoryConcept`. The brief's findings below are unchanged and still accurate as audit
> *findings* — but its "five factories, all in scope" framing, and the shared-kernel table's fork
> rows for the two reverted factories, describe a state the tree no longer has. Current state and the
> reachability derivation: [`METAL2_PORT_REPORT.md`](METAL2_PORT_REPORT.md).
>
> Also corrected by the port: this brief lists
> `device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp` among "8 op-owned writers …
> none is dead code" and does not flag it as shared. It **is** shared — `data_movement/untilize`'s
> block factory binds it by full path.

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `9c1a0466220 2026-09-07 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

## Scope at a glance

One device-operation, five factories, all in scope — no subset, nothing deferred:

| Factory | File (`device/factories/`) |
|---|---|
| `UntilizeWithUnpaddingSingleCoreProgramFactory` | `…_single_core_program_factory.cpp` |
| `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory` | `…_multi_core_interleaved_program_factory.cpp` |
| `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` | `…_multi_core_sharded_program_factory.cpp` |
| `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory` | `…_multi_core_block_interleaved_program_factory.cpp` |
| `UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory` | `…_multi_core_nd_sharded_program_factory.cpp` |

17 kernel files are in play: 8 op-owned writers under `device/kernels/dataflow/`, and 9 borrowed (all
the readers, all the compute). The op owns no reader and no compute kernel. All 8 own writers are
live — none is dead code.

The Sharded factory selects one of five writers by configuration
(`…_multi_core_sharded_program_factory.cpp:204-240`); several findings below are scoped to one of those
branches, so read that dispatch first.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — one `static ProgramDescriptor create_descriptor(...)` per factory,
  declared at `device/factories/*_program_factory.hpp:14`.
- **Op-owned tensors:** none — the `descriptor` concept cannot carry them.
- **Target concept:** `ProgramSpecFactoryConcept` (plain — no op-owned tensors, no custom variant).
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a `TensorParameter relaxation`
  that is neither `none` nor an analysis pointer (it is `none` on all five rows) · `get_dynamic_runtime_args`.
  A custom hash, an `override_runtime_arguments`, and a pybound `create_descriptor` are **not** in that
  list — none of them gate — but on this op all three happen to be absent too. So: no hash to preserve,
  no override method to translate, and no pybind binding to delete. **This port carries no user-visible
  API change.**

## Construct — to do

### Tensor bindings

Fifteen bindings across the five factories. **Every one is Case 1 or clean — there is no Case 2**, so
you will not need the `get_bank_base_address` bridge anywhere in this op.

**Case 1 — express as `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`,
and the legacy base arg plus its `TensorAccessorArgs` plumbing both disappear:**

- SingleCore — `input` (`:187`) and `output` (`:190`)
- MultiCoreInterleaved — `input` (`:237`) and `output` (`:202`)
- BlockInterleaved — `input` (`:295`) and `output` (`:300`)
- Sharded, **cross-shard-type config only** — `output` (`:310`)
- Sharded, **HEIGHT_SHARDED → interleaved config only** — `output` (`:356`)
- Sharded, **W/B_SHARDED → interleaved config only** — `output` (`:415`)
- NDSharded — `output` and `input` on the writer (`:275`, two bindings on one kernel), `input` on the
  reader (`:272`)

**clean — a borrowed-memory DFB read; the DFB *is* the tensor access, so there is no binding work item
beyond the `borrowed_from` below:**

- Sharded — `input` via CB `c_0` (`.buffer = a.buffer()`, `:152`). The reader
  (`reader_unary_sharded.cpp`) only `push_back`s; it never touches a tensor address.
- Sharded, **same-shard-type sharded-output configs only** — `output` via CB `c_17`
  (`.buffer = output.buffer()`, `:181`).

**Delivery note — read this before you start converting.** Every base in this op already arrives as a
`Buffer*` entry pushed into `emplace_runtime_args` / an `RTArgList`, **never** as `->address()`. There
is not a single `->address()` expression in the op. So the framework already registers these as
`BufferBinding`s and patches them on cache hits: you are doing a routine conversion to the typed
binding, **not** repairing a stale-pointer hazard. Don't let the Case-1 label suggest urgency it
doesn't have — and don't go hunting for an address RTA that isn't there.

### TensorParameter relaxation

`none`. No relaxation is applied, and no analysis doc is referenced.

### TensorAccessor 3rd arg

**Drop the redundant page-size argument at two sites, and drop the host CTA feeding each.** Both are
Class 2 (redundant/inert); neither is Class 1, so **no `dynamic_tensor_shape`** is set anywhere in this
port.

| Kernel site | Host CTA to drop |
|---|---|
| `device/kernels/dataflow/writer_unary_stick_layout_split_rows_multicore.cpp:36` — `TensorAccessor(dst_args, dst_addr, writer_page_size)` | `…_multi_core_interleaved_program_factory.cpp:124` (`writer_page_size`, CTA index 2) — and the `writer_page_size` computation at `:110-119` becomes dead |
| `device/kernels/dataflow/writer_unary_unpad_cross_sharded.cpp:35` — `TensorAccessor(dst_args, dst_addr, writer_page_size)` | `…_multi_core_sharded_program_factory.cpp:206` (`cross_writer_page_size`, CTA index 0) — and its computation at `:205` |

Removing each CTA shifts the kernel's `TensorAccessorArgs<N>` offset: `TensorAccessorArgs<3>` →
`<2>` in the first kernel (`:32`), `TensorAccessorArgs<1>` → `<0>` in the second (`:32`). Adjust both.
(The Metal 2.0 binding supplies `aligned_page_size` implicitly, which is exactly the value both sites
compute by hand today — see the audit's classification table for why each config's value is provably
equal.)

### CB endpoints

Almost everything is an ordinary 1-producer / 1-consumer FIFO needing **no** action. Three items do:

- **Self-loop `c_17`** — Sharded factory, same-shard-type sharded output (both the general writer
  `writer_unary_unpad_batch_rows_sharded.cpp` and the W=16 writer
  `writer_unary_unpad_width_16_sharded.cpp`). Its only toucher is the writer, which `reserve_back`s,
  fills by `get_write_ptr()`, and `push_back`s — **nothing drains it**, because the CB *is* the output
  buffer. Bind that one kernel **PRODUCER and CONSUMER**. Legal on Gen1 for a DM kernel; kernel code is
  untouched.
- **Borrowed-memory → `DataflowBufferSpec::borrowed_from`** — two CBs, both in the Sharded factory:
  `c_0` ← `a.buffer()` (`:152`, all configs) and `c_17` ← `output.buffer()` (`:181`). Note both carry
  `address_offset` at its **default zero** — this is the borrowed-memory pattern, a mechanical
  translation, *not* the gated `address_offset` feature.
- **Conditional DFB for `c_17`** — it is allocated only under `out_sharded && !cross_shard_type`
  (`…_multi_core_sharded_program_factory.cpp:169`) and does not exist in the other three Sharded
  configurations. Gate its `DataflowBufferSpec` the same way the factory gates its allocation. This is
  the easy kind of conditional: the host-side branch already exists, so you are translating it, not
  inventing it. **Do not drop `c_17`** — it is live where it is allocated.

**No multi-binding advanced option anywhere, and no dead CB anywhere.** The census found no CB with ≥3
touchers and none with two kernels locked to the same FIFO role, in any factory or configuration.

## Watch for

- **CB endpoints (multi-binding):** none — see above. In particular, the audit actively hunted the
  hidden-second-writer face and found none; the op declares **no semaphores at all**, so the
  semaphore-gated raw co-fill shape cannot exist here.

- **Two same-source kernel instances that are *not* a dual-instance work-split.** The BlockInterleaved
  factory pushes the same reader source and the same writer source into **two** `KernelDescriptor`s each
  (`…_multi_core_block_interleaved_program_factory.cpp:168-171`), and the compute source **four** times
  (`:221-232`). This looks like the co-touching shape, and it isn't: every instance covers a **disjoint**
  core set, and the two buffer sets bind **different CB indices** — `c_0`/`c_16` for the full set,
  `c_2`/`c_17` for the cliffrow set (assigned in `make_block_plan`,
  `ttnn/cpp/ttnn/operations/data_movement/common/common.cpp:906-923`; disjointness asserted by
  `buffer_set_for_core`, `:928`). Each node sees exactly one reader, one writer and one compute
  instance → ordinary 1:1. Don't reach for a 1P+1C assignment or the multi-binding flag here. The same
  goes for the MultiCoreInterleaved factory's full/cliff compute split (`:155-184`).

  Note also that this factory's CB indices come from the **shared** `BlockBufferSet` in
  `data_movement/common`, not from the factory itself — the reader/writer/compute kernels take the index
  as a CTA rather than hardcoding `c_0`/`c_16`. Keep that indirection intact.

- **Cross-op / shared kernels — 9 borrowed files, 4 with a fork you bind and 5 you fork first.** The op
  owns none of its readers and none of its compute kernels, so this is the bulk of the port's cross-op
  cost.

  | Borrowed kernel | `_metal2` fork status |
  |---|---|
  | `eltwise/unary/…/reader_unary_interleaved_start_id.cpp` | **exists** — `…_metal2.cpp`, bind it, don't re-fork |
  | `eltwise/unary/…/reader_unary_sharded.cpp` | **exists** — bind it |
  | `data_movement/sharded/…/reader_unary_nd_sharded_blocks.cpp` | **exists** — bind it |
  | `data_movement/untilize/…/compute/untilize.cpp` | **exists** — bind it |
  | `data_movement/untilize/…/compute/untilize_variable_num_blocks.cpp` | **exists** — bind it |
  | `ttnn/kernel/compute/eltwise_copy.cpp` | **exists** — bind it |
  | `eltwise/unary/…/reader_unary_interleaved_wh_multicore.cpp` | **no fork** — this port creates the first, beside the original |
  | `data_movement/untilize/…/compute/untilize_wh.cpp` | **no fork** — this port creates the first |
  | `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | **no fork** — this port creates the first |

  Other ops binding these files — a **sunset list**, telling you when each legacy copy can go; it is
  **not** authorization to convert any kernel in place:
  `reader_unary_interleaved_start_id.cpp` → `examples/example`, `examples/example_multiple_return`,
  `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `reduction/topk`.
  `reader_unary_sharded.cpp` → `data_movement/tilize`, `data_movement/untilize`,
  `data_movement/sharded_partial/sharded_to_interleaved_partial`, `experimental/slice_write`.
  `reader_unary_interleaved_wh_multicore.cpp`, `untilize_wh.cpp`, `untilize_variable_num_blocks.cpp` →
  `data_movement/untilize`. `untilize.cpp` → `data_movement/fold`.
  `eltwise_copy.cpp` → `data_movement/copy`, `data_movement/sharded/interleaved_to_sharded`,
  `data_movement/sharded_partial/interleaved_to_sharded_partial`,
  `data_movement/sharded_partial/sharded_to_interleaved_partial`.
  `reader_unary_nd_sharded_blocks.cpp` and `writer_unary_stick_layout_interleaved_blocks.cpp` → no other
  consumer; this op is the only one.

- **Stay out of `ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/`.** A copy of this
  op lives there carrying a deliberately hacky shortcut port, and quasar copies show up as apparent
  co-borrowers of six of the nine kernels above. It is not a production port and not a precedent — its
  `_metal2` files are whole-op pre-port copies that do **not** count as forks to reuse, and it ships
  idioms this recipe forbids (a stale `api/dataflow/circular_buffer.h` include, `cb_*` handle naming)
  sitting inline with code that reads perfectly well. The audit excluded it from every table; you should
  not read it either.

- **RTA varargs — two genuine blocks, port them with the vararg mechanism rather than naming each:**

  1. `device/kernels/dataflow/writer_unary_stick_layout_split_rows_multicore.cpp:75-88` — **RTA**
     vararg. The loop runs `n_block_reps` times (a runtime value, RTA 3) pulling five args per group
     through `rt_arg_idx`, advanced **inside** the loop at `:84`. Host side:
     `…_multi_core_interleaved_program_factory.cpp:218-232`, emitting five values per `BlockRep` run — a
     count that varies per core and per shape.

     **The four leading args are *not* part of the block — name them:** `dst_addr` (0, which becomes the
     tensor binding), `padded_X_size` (1), `start_stick_id` (2), `n_block_reps` (3). All are read at
     constant indices at the top of the kernel (`:19-22`), before the loop.

  2. `device/kernels/dataflow/writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:100-105` —
     **CRTA** vararg. Two loops read the output shape then the input shape via `get_common_arg_val`,
     bounded by the CTA `tensor_rank` (`:39`). A CTA-bounded count still varies across instantiations, so
     this is a vararg, not an unrolled name set. Host side:
     `…_multi_core_nd_sharded_program_factory.cpp:175-183`.

  **Do not convert these** — they are fixed fields read repeatedly, and they get names:
  `writer_unary_stick_layout_wh_multicore.cpp:66-71` re-reads RTA indices 2-7 inside a `third_dim` loop,
  but at **constant** indices. Every other kernel in the op reads each RTA a fixed number of times at a
  constant index. **CTA varargs: none** — no kernel here calls `get_compile_time_arg_val` at a varying
  index.

- **Two kernels are still on the `CircularBuffer` wrapper while their siblings use `DataflowBuffer`.**
  `device/kernels/dataflow/writer_unary_unpad_sharded_to_interleaved.cpp:44` and the borrowed
  `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp:77`. The latter also passes
  `CircularBuffer&` into its own file-local helper (`:15`), so converting it means changing that
  signature too — it is file-local, so that is yours to change inside the fork, not a donor-coupling
  problem. Both are fully Device 2.0-legal today; this is ordinary CB→DFB port work, just unevenly
  distributed across the op's kernels, so don't read the inconsistency as a signal about either kernel.

- **`get_tile_size(cb_id)` appears three times and is sanctioned — the DFB swap is exact, not a guess.**
  Sites: `writer_unary_unpad_width_16_sharded.cpp:22`, `reader_unary_interleaved_wh_multicore.cpp:27`,
  and (as `get_local_cb_interface`) `reader_unary_interleaved_start_id.cpp:25`. Moving these onto the
  object (`dfb.get_tile_size()`) is kernel-side whitelist rule 7, and it is a faithful swap rather than
  something to verify empirically: the free `get_tile_size(operand)`
  (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:280`) and `DataflowBuffer::get_tile_size()`
  (`.../dataflow_buffer.h:248`) read the **same** JIT `unpack_tile_size[]` descriptor array.

- **One in-family donor call, and it needs nothing from you.** Both
  `writer_unary_stick_layout_split_rows_multicore.cpp:12` and `writer_unary_unpad_cross_sharded.cpp:10`
  include `ttnn/operations/data_movement/common/kernels/common.hpp` and call
  `tt::data_movement::common::noc_async_write_sharded(Noc, uint32_t l1_addr, AddrGenType tensor, ...)`.
  The signature takes `Noc` by value (Device 2.0 native) and the accessor **by value** — Shape 1,
  ✓ excellent. Construct `TensorAccessor(tensor::name)` and pass it. No donor-side change, no fork of
  the header. The remaining `uint32_t l1_addr` parameter is a CB read pointer, not a resource handle.

- **A `TensorAccessor` is constructed on the *input* inside the ND-sharded writer, purely for shard
  geometry.** `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:42-43` builds
  `accessor_src` from a second `TensorAccessorArgs` block (offset via
  `dst_args.next_compile_time_args_offset()`), and uses it only for `accessor_src.shard_pages(shard_id)`
  (`:110`) — it never reads input data. That is still a real tensor binding on the writer: bind the
  input on this kernel as well as the output, and keep the two `TensorAccessorArgs` offsets chained
  correctly when the binding replaces them.
