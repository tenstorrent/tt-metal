# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/slice`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `c07a9a48c61 2026-09-12 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

**Shape of the op:** one device operation, `ttnn::prim::SliceDeviceOperation`, with five program factories — `SliceRmProgramFactory`, `SliceRmShardedProgramFactory`, `SliceRmStrideProgramFactory`, `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory`. All five clear, so the port covers the whole op with no scoped subset.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `CustomProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — each factory defines a `create_descriptor` returning a `tt::tt_metal::ProgramDescriptor`.
- **Op-owned tensors:** none.
- **Target concept:** **`CustomProgramSpecFactoryConcept`** for all five factories, because every factory declares an `override_runtime_arguments`. You translate that method into one returning a `ProgramRunArgs` rather than deleting it — see *Translating `override_runtime_arguments`* below, which is the largest single piece of work in this port.
- **Custom hash:** `SliceDeviceOperation::compute_program_hash` at `device/slice_device_operation.cpp:348`. **Leave it exactly as it is.** It does not enter the concept choice and the port does not touch it.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args` (the deprecated hook). A custom hash, an `override_runtime_arguments`, and a pybound `create_descriptor` are **not** in this list: none of them gate, and this op has all three.

### Translating `override_runtime_arguments` — read before you start

All five `override_runtime_arguments` methods are one-line delegations to a single shared function, `ttnn::prim::patch_slice_program_addresses`, defined at `device/slice_program_factory_rm_sharded.cpp:357-416` and declared at `device/slice_device_operation.hpp:71-78`. That one function is the entire cache-hit refresh for the op, and it branches into three distinct shapes you translate separately:

1. **`SliceRmShardedProgramFactory`** (`:365-371`) — patches only the two borrowed-CB addresses, via a CB-address-only `ProgramDescriptor` handed to `apply_descriptor_runtime_args`. Everything else in that factory's arguments is cache-keyed. Under Metal 2.0 this becomes the DFB `borrowed_from` refresh and should mostly disappear.
2. **`SliceRmProgramFactory`, `SliceRmStrideProgramFactory`** (`:374-392`) — patches reader slot 0 and writer slot 0 with the input and output base addresses, through a `patch_slot0` helper that **skips any slot holding zero**. The zero check exists because `SliceTileProgramFactory` writes a literal `0u` into writer slot 0 on no-op cores; see the *Watch for* note on that inconsistency.
3. **`SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory`** (`:396-412`) — patches the addresses **and** re-emits the per-core scalars through `slice_tile_dynamic_args` (`device/slice_program_factory_tile.cpp:198`). The scalar re-emission is not optional: those scalars are excluded from the hash, and a divergent-partition cache hit otherwise leaves the writer at `num_pages = 0` and the output all zeros (issue #52651, recorded at `device/slice_program_factory_tile.hpp:29`). `slice_tile_dynamic_args` reproduces `create_descriptor`'s work split exactly, and its own comment at `:205` warns that any divergence leaves stale scalars.

`patch_slice_program_addresses` is **shared with another op** — see the MeshPartition entry under *Watch for*. Whatever you do to its signature or semantics has to work for that caller too.

## Construct — to do

### Tensor bindings (per binding, per factory)

Twelve bindings across the five factories. Ten are **Case 1**; two are **clean** borrowed-memory DFB reads. There is no Case 2 in this op — no kernel does hand-rolled bank arithmetic on a tensor base, so you never need the `get_bank_base_address` bridge.

| Factory | Binding | Case | Action |
|---|---|---|---|
| `SliceRmProgramFactory` | `input` | **Case 1** | express as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::input)` |
| `SliceRmProgramFactory` | `output` | **Case 1** | same, for the writer |
| `SliceRmShardedProgramFactory` | `input` | **clean** | borrowed-memory DFB — `DataflowBufferSpec::borrowed_from` the input `TensorParameter` (`device/slice_program_factory_rm_sharded.cpp:299`) |
| `SliceRmShardedProgramFactory` | `output` | **clean** | same, from the output `TensorParameter` (`device/slice_program_factory_rm_sharded.cpp:311`) |
| `SliceRmStrideProgramFactory` | `input` | **Case 1** | `TensorAccessor(tensor::input)` |
| `SliceRmStrideProgramFactory` | `output` | **Case 1** | `TensorAccessor(tensor::output)` |
| `SliceTileProgramFactory` | `input` | **Case 1** | delivered today as a **common** runtime arg (`device/slice_program_factory_tile.cpp:143`) |
| `SliceTileProgramFactory` | `output` | **Case 1** | per-core runtime arg (`device/slice_program_factory_tile.cpp:180`) |
| `SliceTileTensorArgsProgramFactory` | `input` | **Case 1** | common runtime arg (`device/slice_program_factory_tile_tensor_args.cpp:182`) |
| `SliceTileTensorArgsProgramFactory` | `start_tensor` | **Case 1** | common runtime arg (`:183`) |
| `SliceTileTensorArgsProgramFactory` | `end_tensor` | **Case 1** | common runtime arg (`:184`) |
| `SliceTileTensorArgsProgramFactory` | `output` | **Case 1** | per-core runtime arg (`:151`, `:168`) |

**All ten Case-1 bindings already use the `Buffer*`-binding form** — the factory pushes the `Buffer*` object itself into the argument list, not a `->address()` expression. That form is correct on cache hits today, so none of these is a latent-correctness fix; they are routine binding work. What disappears with each is the kernel-side `TensorAccessorArgs<N>()` declaration, the raw `uint32_t` base variable, and the corresponding `TensorAccessorArgs(*buffer).append_to(...)` call on the host.

**TensorParameter relaxation:** `none`. The readiness sheet reads `none` on all five factory rows, so apply no relaxation and set no `dynamic_tensor_shape`.

**TensorAccessor 3rd arg:** none — no accessor in this op passes one. Nothing to drop.

### CB endpoints

Seven CBs. Four are already a plain one-producer / one-consumer FIFO and need nothing. Three are single-ended and take a **self-loop** — bind the one kernel as both PRODUCER and CONSUMER:

- **`SliceRmShardedProgramFactory` / `c_0`** (borrowed from the input buffer) — the reader is the only toucher, and only through a raw peek at `device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp:41`. Role-free → self-loop.
- **`SliceRmShardedProgramFactory` / `c_16`** (borrowed from the output buffer) — the reader is the only toucher (`reserve_back` / `push_back` plus a raw peek). That factory builds exactly one kernel, so a second toucher is structurally impossible.
- **`SliceTileTensorArgsProgramFactory` / `c_1`** (`tensor_cb_index`, the single-tile staging buffer for the start and end index tensors) — the reader runs the full `reserve_back` → `push_back` → `wait_front` → `pop_front` handshake against itself, twice, at `device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:52-83`.

No dead CB to drop, no multi-binding advanced option to set, no conditional DFB spec.

## Watch for

- **CB endpoints (multi-binding):** none. No CB in this op has three or more touchers on a node, and no two kernels lock the same FIFO role. There are no semaphores anywhere in the op, so the hidden-second-writer shape cannot arise.

- **Cross-op / shared kernels.** `SliceTileTensorArgsProgramFactory` instantiates a kernel it does not own: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (`device/slice_program_factory_tile_tensor_args.cpp:133`). **A `_metal2` fork already exists beside it** — `writer_unary_interleaved_start_id_metal2.cpp`, same directory — so **bind that fork; do not create another and do not convert the legacy file in place.** Thirteen other factories still instantiate the legacy copy (`data_movement/concat`, `data_movement/reshape_on_device`, four `data_movement/tilize` factories, `embedding`, `copy/typecast`, `eltwise/unary_backward/tanh_bw`, `experimental/matmul/attn_matmul`, `experimental/transformer/nlp_concat_heads_boltz`, and the two `examples/example` factories). That list is a **sunset list, not authorization to convert the kernel in place** — it records when the legacy copy can finally go, which issue #52228 tracks.

- **A same-basename trap between the two tile factories.** Slice owns its *own* `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`, which differs from the borrowed one only in reading its DFB index through `get_named_compile_time_arg_val("dfb_id_out")` so the fusion infrastructure can remap it. `SliceTileProgramFactory` instantiates the **slice-owned** copy (`device/slice_program_factory_tile.cpp:157`); `SliceTileTensorArgsProgramFactory` instantiates the **borrowed** one. Key on the full path, never the basename — the two factories sit a hundred lines apart and read almost identically.

- **MeshPartition calls slice's factories directly and will not compile after this port.** `ttnn/cpp/ttnn/operations/ccl/mesh_partition/device/mesh_partition_program_factory.cpp` bypasses `ttnn::prim::slice` and drives the factories itself: `SliceOp::validate_on_program_cache_miss` and `SliceOp::select_program_factory` at `:126-127`, `Factory::create_descriptor` at `:131`, and `ttnn::prim::patch_slice_program_addresses` at `:155`. It also stores a `SliceDeviceOperation::program_factory_t` in its own `shared_variables_t` (`mesh_partition_device_operation.hpp:46-50`). Every one of those entry points changes in this port. **Confirm with the user whether MeshPartition moves in the same change before you start** — the answer changes the size of the diff substantially. Related note: `device/slice_program_factory_rm.cpp:290-291` records that MeshPartition also bypasses `ttnn::slice`'s resharding guard, which is why the `check_accessor_page_size` assertion at `:292-308` exists.

- **The pybound `create_descriptor` is a user-visible API removal.** `slice_nanobind.cpp:167-179` exposes `SliceTileProgramFactory.create_descriptor` to Python. The port **deletes** this binding; give it its own entry in the port report as the user-visible change it is. The readiness sheet's `Pybind descriptor` cell reads `PR`, so a removal may already be in flight — check before you write the deletion. The neighbouring bindings at `slice_nanobind.cpp:138-166` (`SliceParams`, `SliceInputs`, and `SliceDeviceOperation::create_output_tensors` / `compute_output_specs`) are **not** factory entry points and stay.

- **RTA and CRTA varargs — six kernels.** These read genuinely variable-count blocks and need the vararg mechanism; do not try to name the elements. In every case the scalars *before* the block are ordinary named arguments, so name those and let only the counted block ride as varargs.

  | Kernel | Site | What is variable-count | Bounded by |
  |---|---|---|---|
  | `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | `:31-33` | three `num_dims`-long RTA blocks — `num_unpadded_sticks`, `num_padded_sticks`, `id_per_dim`; indexed by the loop variable at `:76-84` and `:106-114` | RTA 3, `num_dims` |
  | `slice_reader_unary_unpad_dims_rm_sharded.cpp` | `:26-30` | `read_noc_x` / `read_noc_y` (two per source core), `num_stick_chunks`, then `chunk_start_id` / `chunk_num_sticks` (two per chunk) | RTA 0, `num_cores_read`, plus a per-core chunk count that is itself runtime data |
  | `reader_unary_unpad_dims_interleaved_start_id.cpp` | `:17-18` (CRTA), `:23` (RTA) | CRTA `num_unpadded_tiles` + `num_padded_tiles`, each `num_dims` long; RTA `id_per_dim` | CTA 0, `num_dims` |
  | `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | `:25-26` and `:91-92` (CRTA), `:31` (RTA) | CRTA `num_unpadded_tiles`, `num_padded_tiles`, `input_shape_args` — three `num_dims`-long blocks; RTA `id_per_dim` | CTA 2, `num_dims` |
  | `reader_multicore_slice_nd.cpp` | `:73-87` | five `tensor_rank`-long RTA blocks — `input_dims`, `output_dims`, `slice_starts`, `slice_ends`, `slice_steps` | RTA 1, `tensor_rank` |
  | `writer_multicore_slice_nd.cpp` | `:73` | one `tensor_rank`-long RTA block — `output_dims` | RTA 1, `tensor_rank` |

  A CTA-bounded count is still a vararg: the block length varies across instantiations, so there is no stable name to attach.

  The other four kernels get **named** arguments, including the two 4D stride kernels. `reader_multicore_slice_4d.cpp` and `writer_multicore_slice_4d.cpp` use a running `rt_args_idx++` at the top of the kernel, which looks loop-like but is a fixed run of distinct fields — name each one. Same for `slice_writer_unary_stick_layout_interleaved_start_id.cpp` (ten constant indices) and both copies of `writer_unary_interleaved_start_id.cpp` (three constant indices).

- **No CTA varargs.** Every `get_compile_time_arg_val` in the op uses a literal constant index. You do not need `KernelAdvancedOptions::compile_time_varargs`.

- **Two kernel files in the op's directory are dead — do not port them.** `device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp` and `device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp` are instantiated by no factory anywhere in the repository. `SliceRmStrideProgramFactory` uses the `*_multicore_slice_4d` / `*_multicore_slice_nd` pair instead (`device/slice_program_factory_rm_stride.cpp:44-54`). Both dead files also address DFB index 24, which no slice factory allocates, so they would not run against any current program. Leave them alone; their cleanup is an ops-team matter recorded in the audit.

- **Do not read `ttnn/cpp/ttnn/operations/experimental/quasar/slice`.** A quasar copy of this op exists there. It is a deliberately hacky shortcut port done to unblock downstream work, it carries idioms the port recipe forbids, and it is not a precedent, a naming source, or evidence that anything is portable. It will also surface if you grep the dated triage analyses for "slice", since one of them carries a `quasar/slice` row two lines above the mainline row.

- **The one place slice's tile factories differ in argument layout, which the binding change resolves.** `SliceTileProgramFactory` emits a literal `0u` in writer slot 0 for no-op cores (`device/slice_program_factory_tile.cpp:176`) but binds `dst_buffer` on active cores (`:180`), while `SliceTileTensorArgsProgramFactory` binds `dst_buffer` on **every** core including no-op ones, and says why at `device/slice_program_factory_tile_tensor_args.cpp:148-151`. The zero-slot skip inside `patch_slot0` (`device/slice_program_factory_rm_sharded.cpp:373-383`) exists only to tolerate the first shape. A typed binding is uniform per kernel, so the distinction goes away — just be aware the two factories do not currently agree, so you cannot copy one's argument handling onto the other without thinking.

- **A Device 2.0 breadcrumb in the borrowed kernel — confirm, do not swap blind.** The donor `eltwise/unary/.../writer_unary_interleaved_start_id.cpp:27` reads its page size through `get_local_cb_interface(cb_id_out).fifo_page_size`. That free function is **sanctioned** under Device 2.0, which is why it did not gate the audit. Slice's own copy of the same kernel already reaches the same value through the object, as `dfb_out.get_entry_size()` (`device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:26`). When you bind the `_metal2` fork, check what that fork already does rather than assuming either form.
