# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/slice`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `1167faf7b42 2026-09-04 docs(metal_2.0): binary_ng relaxation analysis; invariant checks over commit stamps` *(carry this line into the port report's Provenance section)*

**Shape of the job:** one DeviceOperation, **five** factories, **eleven** referenced kernels (ten owned, one borrowed). Every kernel is already fully Device 2.0 — `Noc`, `DataflowBuffer`, `CoreLocalMem`, `TensorAccessor` throughout — so this is a **binding-layer port**, not an idiom rewrite. No semaphores anywhere in the op. Two kernel files in `device/kernels/dataflow/` are referenced by nothing and are out of scope: `strided_slice_reader_rm_interleaved_nd.cpp`, `strided_slice_writer_rm_interleaved.cpp`.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `CustomProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — all five factories define `create_descriptor(const SliceParams&, const SliceInputs&, Tensor&) -> ProgramDescriptor`.
- **Op-owned tensors:** none.
- **Target concept:** **`CustomProgramSpecFactoryConcept`** (all five). Selected by `Override runtime args method? == yes`: each factory defines `override_runtime_arguments` —
  `slice_program_factory_rm.cpp:424` · `slice_program_factory_rm_sharded.cpp:415` · `slice_program_factory_rm_stride.cpp:178` · `slice_program_factory_tile.cpp:189` · `slice_program_factory_tile_tensor_args.cpp:195`.
- **One body, five call sites.** All five `override_runtime_arguments` are one-line delegations to a shared free function, `ttnn::prim::patch_slice_program_addresses` (declared `slice_device_operation.hpp:73`, defined `slice_program_factory_rm_sharded.cpp:354-413`), which `std::visit`s the factory variant and branches internally. **The `ProgramRunArgs` translation is one function, not five** — and it is shared with another op (see *Watch for*).
- **Custom hash present, leave it alone:** `SliceDeviceOperation::compute_program_hash` at `slice_device_operation.cpp:348-432`. Not a gate; the port does not touch it.
- **Pybound `create_descriptor` to delete:** `slice_nanobind.cpp:168-179` binds `SliceTileProgramFactory::create_descriptor` as `nb::class_<SliceTileProgramFactory>(...).def_static("create_descriptor", ...)`. The port deletes it; that is a user-visible API change and gets its own entry in the port report. (The neighbouring `nb::class_` bindings of `SliceParams`, `SliceInputs` and `SliceDeviceOperation`'s `create_output_tensors` / `compute_output_specs` at `slice_nanobind.cpp:138-166` are not `create_descriptor` bindings — read the surrounding code before removing anything beyond the one `def_static`.)
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args` (deprecated hook). A custom hash, an `override_runtime_arguments`, and a pybound `create_descriptor` are **not** in this list: none of them gate, and all three are present here.

## Construct — to do

### Tensor bindings (per binding)

Twelve bindings across five factories. **Eight Case 1, two clean, zero Case 2** — nothing needs the `get_bank_base_address` bridge.

Every address in this op is already delivered as a **`Buffer*` pushed into an `RTArgList`** (never a bare `->address()` in an arg list), so the framework patches it on cache hits today. That means: routine port work, no correctness urgency, and the kernel-side change is the mechanical Case-1 rewrite in every instance.

| Factory | Binding | Today | Port to |
|---|---|---|---|
| `SliceRmProgramFactory` | `input` | `Buffer*` RTA[0] (`rm.cpp:405`) → `TensorAccessor(src_args, src_addr)` (`slice_reader_..._rm_interleaved_start_id.cpp:40`) | **Case 1** — `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::input)`; the RTA slot and its `TensorAccessorArgs` plumbing (`rm.cpp:364`) both disappear |
| `SliceRmProgramFactory` | `output` | `Buffer*` RTA[0] (`rm.cpp:413`) → `TensorAccessor(dst_args, dst_addr)` (`slice_writer_..._interleaved_start_id.cpp:29`) | **Case 1** |
| `SliceRmShardedProgramFactory` | `input` | **borrowed-memory DFB** `c_0`, `.buffer = input.buffer()` (`rm_sharded.cpp:290`); kernel reads via `dfb_in.get_write_ptr()` (`slice_reader_..._rm_sharded.cpp:41`) | **clean** — `DataflowBufferSpec::borrowed_from` the `TensorParameter`. No accessor, no address arg. |
| `SliceRmShardedProgramFactory` | `output` | **borrowed-memory DFB** `c_16`, `.buffer = output.buffer()` (`rm_sharded.cpp:302`) | **clean** — `borrowed_from` |
| `SliceRmStrideProgramFactory` | `input` | `Buffer*` RTA[0] (`rm_stride.cpp:128` 4D / `:147` ND) → `TensorAccessor(src_args, src_addr)` | **Case 1** |
| `SliceRmStrideProgramFactory` | `output` | `Buffer*` RTA[0] (`rm_stride.cpp:136` 4D / `:160` ND) → `TensorAccessor(dst_args, dst_addr)` | **Case 1** |
| `SliceTileProgramFactory` | `input` | `Buffer*` **CRTA**[0] (`tile.cpp:143`) → `TensorAccessor(src_args, src_addr)` | **Case 1** — note it rides a *common* runtime arg, not a per-core one |
| `SliceTileProgramFactory` | `output` | `Buffer*` RTA[0] (`tile.cpp:180`) → `TensorAccessor(dst_args, dst_addr)` | **Case 1** |
| `SliceTileTensorArgsProgramFactory` | `input` | `Buffer*` **CRTA**[0] (`tile_tensor_args.cpp:182`) | **Case 1** |
| `SliceTileTensorArgsProgramFactory` | `start_tensor` | `Buffer*` **CRTA**[1] (`tile_tensor_args.cpp:183`) → `TensorAccessor(start_args, start_addr)` (`reader_..._tensor_args.cpp:44`) | **Case 1** |
| `SliceTileTensorArgsProgramFactory` | `end_tensor` | `Buffer*` **CRTA**[2] (`tile_tensor_args.cpp:184`) → `TensorAccessor(end_args, end_addr)` (`reader_..._tensor_args.cpp:45`) | **Case 1** |
| `SliceTileTensorArgsProgramFactory` | `output` | `Buffer*` RTA[0] (`tile_tensor_args.cpp:151,168`) — consumed by the **borrowed** eltwise writer | **Case 1** |

Two things worth holding onto while you work the table:

- **The sharded factory is the odd one out.** `SliceRmShardedProgramFactory` alone has no writer kernel, no `TensorAccessor`, and no address argument of any kind — both tensors reach the single reader through borrowed-memory DFBs. The *same* `input` / `output` tensors are Case 1 in every other factory. Don't carry a conclusion from one factory to the next.
- **The tile factories bind through CRTAs.** `SliceTileProgramFactory` and `SliceTileTensorArgsProgramFactory` put their read-side `Buffer*`s in `emplace_common_runtime_args` (`tile.cpp:141-145`, `tile_tensor_args.cpp:180-186`), with per-dim scalar blocks appended after. When the bindings lift out, the CRTA list shrinks to just those scalar blocks.

### TensorParameter relaxation

`none` — the port applies no relaxation. (All five sheet rows read `none`; there is no analysis doc and none is needed.)

### TensorAccessor 3rd arg

**None.** No accessor in this op passes a page-size argument — all 12 construction sites across the 11 kernels use the two-arg form. Nothing to drop, no `dynamic_tensor_shape` to set.

Two notes so you don't go looking:

- The dated 3rd-arg triage (`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md:75,139`) *does* list `slice` as Class 1 + Special. **It is stale.** The change is documented in the factory itself at `slice_program_factory_rm.cpp:283-290`.
- `slice_program_factory_rm.cpp` carries a host-side guard, `check_accessor_page_size()` (`:291-307`, called at `:336-339`), that asserts the per-shard page size equals the buffer's `aligned_page_size()`. It exists precisely because the kernels now take the accessor's compile-time page size. **Leave it in place** — it is not 3rd-arg residue, it is the invariant that made dropping the 3rd arg safe.

### CB endpoints

Seven CBs across the five factories. Four are already legal 1:1; three need a **self-loop**. No multi-binding flag, no dead-CB drop, no conditional DFB. The census does not flip with config within any factory (the RM factory's chunked-vs-unchunked branch changes CB *sizing* only).

| Factory | CB | Disposition |
|---|---|---|
| `SliceRmProgramFactory` | `c_0` | legal 1:1 — reader PRODUCER, writer CONSUMER |
| `SliceRmShardedProgramFactory` | `c_0` (borrowed) | **self-loop** — the reader is the only toucher, and only raw-peeks (`get_write_ptr()`, `slice_reader_..._rm_sharded.cpp:41`); bind it PRODUCER **and** CONSUMER |
| `SliceRmShardedProgramFactory` | `c_16` (borrowed) | **self-loop** — the reader `reserve_back`/`push_back`s it (`:40,89`) and nothing drains it |
| `SliceRmStrideProgramFactory` | `c_0` | legal 1:1 (both the 4D and ND kernel pairs) |
| `SliceTileProgramFactory` | `c_0` | legal 1:1 |
| `SliceTileTensorArgsProgramFactory` | `c_0` | legal 1:1 |
| `SliceTileTensorArgsProgramFactory` | `c_1` | **self-loop** — the reader alone drives both FIFO roles on this single-tile staging buffer (`reader_..._tensor_args.cpp:52,58,59,66` and `69,75,76,83`); the kernel comment calls it "the producer/consumer handshake" |

Both `SliceRmShardedProgramFactory` self-loops sit on a **DM** kernel (`ReaderConfigDescriptor`). Legal on Gen1; the spec validator rejects a DM self-loop only on Gen2, so record it as Quasar-uplift debt for that later audit and move on.

### One kernel-side breadcrumb (whitelist rule 7)

The **borrowed** eltwise writer reads its page size through a sanctioned free function:

```
ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:27
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;
```

`get_local_cb_interface(cb_id)` is on the Device 2.0 sanctioned list, so this is **not** a Device 2.0 holdover and did not gate — but the Metal 2.0 port moves the lookup onto the DFB object per kernel-side whitelist rule 7. You have a worked reference for the exact replacement: slice's own near-identical copy already does it — `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:26` uses `dfb_out.get_entry_size()`.

## Watch for

- **CB endpoints (multi-binding):** none. No hidden second writer exists — the op declares no semaphores at all, and every raw `get_write_ptr()` / `get_read_ptr()` in the op is a peek by the DFB's own bound endpoint. You can skip that hunt.

- **Cross-op / shared kernels:**
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — borrowed by `SliceTileTensorArgsProgramFactory` (`tile_tensor_args.cpp:133`). **A `_metal2` fork already exists beside it** (`writer_unary_interleaved_start_id_metal2.cpp`, same directory, not a quasar copy) — **bind the existing fork; do not re-fork, and do not convert the legacy file in place.** The legacy file's header comment and issue **#52228** carry the rationale and sunset plan.
  - Other ops binding that same legacy kernel — **sunset list, not authorization to convert the kernel in place** (14 found): `experimental/transformer/nlp_concat_heads_boltz`, `experimental/transformer/nlp_concat_heads`, `experimental/matmul/attn_matmul`, `embedding` (fused), `examples/example` (multi-core and single-core), `data_movement/reshape_on_device`, `data_movement/tilize` (four factories), `data_movement/concat`, `eltwise/unary_backward/tanh_bw`.
  - **Same-basename trap.** Slice *also owns* a file named `writer_unary_interleaved_start_id.cpp`, in its own `device/kernels/dataflow/`, used by `SliceTileProgramFactory` (`tile.cpp:157`). They are different files with different arg conventions: slice's copy takes its DFB index from a **named** CTA (`get_named_compile_time_arg_val("dfb_id_out")`, `:19`) and uses `dfb_out.get_entry_size()`; the eltwise copy takes a **positional** CTA 0 (`:23`) and uses `get_local_cb_interface(...)`. Check which path you are in before editing either.

- **`ccl/mesh_partition` drives these factories directly — the biggest break risk in this port.** It does not borrow slice's kernels; it reuses slice's *host* entry points:
  - `mesh_partition_program_factory.cpp:126-134` calls `SliceOp::validate_on_program_cache_miss`, then `SliceOp::select_program_factory`, then `Factory::create_descriptor(slice_attrs, slice_tensor_args, tensor_return_value)` under a `std::visit`, and wraps the result as `Program{descriptor}`.
  - It stores the chosen `prim::SliceDeviceOperation::program_factory_t` in its own `shared_variables_t` (`mesh_partition_device_operation.hpp:47-50`) specifically so the cache-hit path can patch the slot layout that factory baked.
  - On a cache hit it calls `ttnn::prim::patch_slice_program_addresses(...)` (`mesh_partition_program_factory.cpp:155`) — the same shared function that *is* slice's `override_runtime_arguments`.

  Both entry points change under this port. Decide early whether MeshPartition moves with slice or gets a compatibility shim, and say which in the port report — this is a cross-op decision, not a mechanical edit.

- **RTA varargs — six kernels have genuine variable-count blocks; reach for the vararg mechanism, don't try to name them:**

  | Kernel | Site | Kind | Why |
  |---|---|---|---|
  | `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | `:31-33` | RTA | three blocks of `num_dims` at `get_arg_addr(13)`; `num_dims` is a **runtime** RTA (arg 3). Indexed by loop var at `:76-84`, `:106-114`. |
  | `slice_reader_unary_unpad_dims_rm_sharded.cpp` | `:26-30` | RTA | arg addresses are **computed at runtime** — `get_arg_addr(1 + num_cores_read * 2)`, `(1 + num_cores_read * 3)`, with `num_cores_read` from arg 0. Blocks walked at `:47-86`. |
  | `reader_unary_unpad_dims_interleaved_start_id.cpp` | `:17-18` (CRTA), `:23` (RTA) | **both** | `get_common_arg_addr(1)` holds `2 × num_dims` entries; `get_arg_addr(2)` holds `num_dims`. `num_dims` is a CTA here — still a vararg (it varies across instantiations), per whitelist rule 4. |
  | `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | `:25-26`, `:91-92` (CRTA), `:31` (RTA) | **both** | CRTA blocks at `get_common_arg_addr(3)` (2 × `num_dims`) and `get_common_arg_addr(3 + 2 * num_dims)` (`num_dims`); RTA block at `get_arg_addr(2)`. |
  | `reader_multicore_slice_nd.cpp` | `:73-87` | RTA | five consecutive blocks, `rt_args_idx` advanced by `tensor_rank` (a **runtime** RTA) between each. |
  | `writer_multicore_slice_nd.cpp` | `:73` | RTA | one `output_dims` block of length `tensor_rank`. |

  **Everything else gets a name.** In particular the 4D strided kernels are *not* varargs despite appearances: `reader_multicore_slice_4d.cpp:53-77` (25 args) and `writer_multicore_slice_4d.cpp:53-61` (9 args) are fixed runs of `rt_args_idx++` over a constant set — legacy positional plumbing that dissolves into named args. Same for `slice_writer_unary_stick_layout_interleaved_start_id.cpp:13-24` (fixed indices 0-9) and both `writer_unary_interleaved_start_id.cpp` copies (fixed 0, 1, 2).

  Two checks already done for you: **no kernel has nameable scalars trailing a vararg block** (every kernel reads its named scalars first, blocks last), so nothing needs rescuing off the end of a run; and **there are no CTA varargs** — every `get_compile_time_arg_val` / `TensorAccessorArgs<N>` read in the op is at a constexpr index, including the chained `TensorAccessorArgs<...next_compile_time_args_offset()>` in `reader_..._tensor_args.cpp:17-19`.

- **`patch_slice_program_addresses` also hand-patches slots the `Buffer*` bindings already cover.** For `SliceRmProgramFactory` / `SliceRmStrideProgramFactory` it re-writes RTA slot 0 directly via `GetRuntimeArgs` (`slice_program_factory_rm_sharded.cpp:372-389`), even though those slots are `Buffer*` bindings the framework patches on cache hits (see the factories' own comments at `slice_program_factory_rm.cpp:401-402,409-410`). The likely reason is MeshPartition's hand-built `Program{descriptor}`, which may not get binding injection. When you translate `override_runtime_arguments` into a `ProgramRunArgs`, decide which mechanism survives rather than porting both — and note that the manual patch deliberately **skips slots holding 0** (no-op cores that `create_descriptor` left zero-filled, `:374-377`), a subtlety worth preserving or consciously dropping.

- **The RM reader hardcodes its DFB index.** `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:42` has `constexpr uint32_t dfb_id_in0 = 0;` inline in the kernel rather than reading a CTA — the host never passes it (`rm.cpp:363-364` sends only `TensorAccessorArgs`). It becomes a `dfb::` token like any other, but there is no CTA to delete on the host side for this one. Its *writer* counterpart does take the index as CTA 0 (`rm.cpp:360`), so the two sides of the same CB differ.

- **`#ifdef OUT_SHARDED` is dead on every slice path.** Both `writer_unary_interleaved_start_id.cpp` copies branch on it, and no slice factory sets kernel `defines` — so only the `#else` path is ever compiled for slice. Relevant if you are reading the borrowed kernel and wondering which half matters.

- **The stale-triage flag, in case you read the analyses.** `analyses/2026-07-19_offset_base_pointers.md:63` still lists `slice_program_factory_rm.cpp` as the *canonical* Type-2 accessor-fed offset. That fold is gone: the base arrives clean as a `Buffer*` and the W-begin shift rides each read as a separate scalar (`rm.cpp:99` → kernel arg 12 `src_offset_bytes`, applied at `slice_reader_..._rm_interleaved_start_id.cpp:98` and `:64-67`). The kernel comment at `:38-39` explains the design. **There is no offset gate on this op** — don't re-derive one from the doc.

- **`ttnn/cpp/ttnn/operations/experimental/quasar/slice/` exists and is out of bounds.** It holds a whole-op shortcut copy of this op, including same-named kernels and `_metal2` files. Nothing in it is a precedent, a naming source, or evidence that a construct ports. Don't read it, and don't let a grep hit from it into the port.
