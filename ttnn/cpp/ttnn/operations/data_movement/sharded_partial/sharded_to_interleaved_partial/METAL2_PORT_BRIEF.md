# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

**Code baseline:** every file named below matched `origin/main` @ `2b7bf3396eb` (2026-08-05) at audit time. That baseline includes `6abdf94214d` (PR #51747), which cleared an offset-base fold in the shared row-major writer, and `0fb47949a27` (PR #51179), which migrated the shared compute kernel to Device 2.0. **If your checkout predates either, rebase before starting** — on an older base you would be porting code the audit did not clear.

## Read this first — sequencing

This op binds **the same four kernels** as `data_movement/sharded/sharded_to_interleaved`, which is also GREEN and also queued. Before you start, confirm with your invoker **which of the two ports runs first**:

- **If the sibling ports first** (recommended, and the audit's recommendation): all four `_metal2` forks already exist beside their originals. You are at **rung 1 on every kernel** — bind the existing forks, adopt their binding names, create nothing. Check each fork's named-arg set fits before committing, and remember a fork with a consumer is **read-only** to you.
- **If you go first**: you are at **rung 2 on all four** — you create every fork, and *your* binding names become the interface the sibling port inherits. Name them for the **kernel's role**, not for this op.

Either way, **do not run the two ports concurrently** — they would race on four shared files.

## Scope — what you are porting

One `DeviceOperation`, one factory, four borrowed kernels (the op owns none):

- **`ShardedToInterleavedPartialDeviceOperation`** → `ShardedToInterleavedPartialProgramFactory` (`device/sharded_to_interleaved_partial_program_factory.cpp`)

| Role | Kernel | Selected when |
|---|---|---|
| Reader | `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | always |
| Writer (tiled) | `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | `input.layout() == TILE` — **always, in practice** |
| Writer (row-major) | `data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `input.layout() == ROW_MAJOR` — **unreachable; see Watch for** |
| Compute (copy) | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | `convert_df` (cache-tensor dtype ≠ input dtype) |

Configs — kernel choice is a runtime branch inside the one factory (`program_factory.cpp:178-186`, `:191-197`), not separate factories:

- **C1** — TILE, no conversion: reader + tiled writer. *(Reachable.)*
- **C2** — TILE, conversion: reader + tiled writer + compute. *(Reachable.)*
- **C3** — ROW_MAJOR: reader + RM writer. **Unreachable** — `validate_on_program_cache_miss` rejects non-TILE input (`device_operation.cpp:24`). Port it, don't test it, don't delete it.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`device/sharded_to_interleaved_partial_program_factory.hpp:15-18`).
- **Op-owned tensors:** none. *(The op writes into a caller-supplied `cache_tensor`, but that is a preallocated output, not an op-owned buffer — see the wiring note below.)*
- **Target concept:** `ProgramSpecFactoryConcept`. *(Derived from `Concept` + `Op-owned tensors?`, and confirmed by the readiness sheet's own `Porting Target` cell.)*
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported replacement) · pybind `create_descriptor` — all gate conjuncts — plus other migration-risky pybind, which would have surfaced as a `safe` warning. All `no` on the sheet and confirmed against the code.

## Construct — to do

**Tensor bindings** (per binding):

- **`input_tensor`** — **clean (borrowed-memory DFB)**, not Case 1 or Case 2. `c_0` is Buffer-backed today (`cb.buffer = src_buffer`, `program_factory.cpp:41`, from the `push_s2i_partial_cb_pair` call at `:140-147`); the reader only `push_back`s pages that are already resident and builds no `TensorAccessor`. Port via `DataflowBufferSpec::borrowed_from` on the input `TensorParameter`. Do not give it an accessor.
- **`cache_tensor`** (the output) — **Case 1** (via `TensorAccessor`). The base reaches the writer as a `Buffer*` binding today (`writer_rt.push_back(dst_buffer)`, `program_factory.cpp:243` tiled / `:294` RM → arg 0). Express it as a `TensorParameter` / `TensorBinding`; each writer builds `TensorAccessor(tensor::<out>)`. Two things then disappear together: the **arg-0 base RTA** and the **`TensorAccessorArgs` CTA plumbing** — host-side `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` (`program_factory.cpp:177`) and kernel-side `constexpr auto dst_args = TensorAccessorArgs<1>()` (tiled `:23`, RM `:20`). Mechanical, low-risk.

  **Wiring note — the output is also an input.** `create_output_tensors` returns `tensor_args.cache_tensor` **itself** (`device_operation.cpp:56-60`), and `compute_output_specs` returns that tensor's spec (`:50-54`). So the tensor you bind as the writer's destination is the same object the op received as a tensor arg. That is one binding, not two — declare it from the output tensor as usual; just don't be surprised that it aliases a `tensor_args_t` member, and don't add a second `TensorParameter` for `cache_tensor`.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — both accessor constructions are already 2-argument (tiled `:28`, RM `:25`). Nothing to drop.

**CB endpoints:** all legal 1P+1C on every node in every config — nothing to self-loop, assign, flag, or drop. For reference when you write the specs:

| DFB | Configs | PRODUCER | CONSUMER |
|---|---|---|---|
| `c_0` (`src0_cb_index`, `borrowed_from` the input) | C1, C2, C3 | reader (`dfb.push_back`) | C1: tiled writer · C2: compute · C3: RM writer |
| `c_16` (`out_cb_index`) | C2 only | compute (`cb_out.reserve_back`/`push_back`) | tiled writer |

Preserve the aliasing: when `!convert_df`, `out_cb_index == src0_cb_index == c_0` (`program_factory.cpp:129`) — the writer drains the *same* borrowed DFB the reader fills. `c_16` is allocated **only** under `convert_df` (`:149-160`) and, unlike `c_0`, is **not** buffer-backed (`bound_buffer = nullptr`).

**RTA naming.** Both writers read their args at fixed constant indices, so every arg is nameable (no varargs anywhere). Two things to get right:

- **Tiled writer** reads 0–8 (`:11-19`). Arg 0 is the base (disappears with the binding); 1–8 become named args. `start_id = start_id_base + start_id_offset` (`:20`) is a kernel-local sum of args 8 and 7, not a ninth arg.
  **Arg 8 (`start_id_base`, host `starting_idx_h`) is load-bearing in *this* op** — unlike in the sibling, where it is always `0`. Here `num_slices` / `slice_index` are real user inputs, so `calculate_starting_idx_h` (`sharded_common.cpp:16-28`) returns a genuine per-slice tile offset. It is a **page index**, not an address; name it and pass it through unchanged.
- **RM writer** reads 0, **2**, 3, 4, 5, 6 (`:12-17`) — the factory pushes **7** args and index **1** (`num_units_per_row`, `:295`) is **never read**. Name only the six the kernel actually reads; do not invent a name for the dead slot. Dropping the host-side push is a behavior-neutral cleanup owned by the ops team, **not** this port — leave `program_factory.cpp:295` alone and note the mismatch in your port report.

## Watch for

- **The row-major branch is unreachable — carry it, don't touch it.** `validate_on_program_cache_miss:24` is `TT_FATAL(input_tensor.layout() == Layout::TILE, "Currently, only tile layout is supported for partial S->I")`, and layout is part of the program hash, so a row-major input always misses the cache and always hits that assert. The factory's entire `else` branch — RM unit sizing (`:94-107`), RM writer selection (`:182-186`), the RM per-core RTA block (`:259-308`) — is dead code that nonetheless compiles and binds a kernel. **Port it faithfully and identically to the tiled path's treatment; do not delete it, do not "simplify" it, and do not treat its untestability as licence to change it.** Deleting live-but-unreachable code is a behaviour change outside port scope, and the restriction reads as intended-temporary. Say plainly in your port report that this branch could not be exercised. *(If the sibling op ports first, its RM writer fork already exists and is the one you bind — which is the cleanest outcome, since the sibling **can** test that path.)*

- **CB endpoints (multi-binding):** none. The audit hunted all three faces with positive evidence — no `get_write_ptr` / `get_read_ptr` / `get_local_cb_interface` / `fifo_*_ptr` / `evil_set_*` in any of the four kernels, **no semaphores of any kind** in the op, and no dual-instance work-split (each kernel source goes into exactly one `KernelDescriptor`, `:311-315`). You should not find a hidden co-filler; if you do, the audit was wrong — stop and report it.

- **Cross-op / shared kernels — all four are borrowed.** Rung depends on sequencing (see *Read this first*). Fork targets and the sunset list, for whichever rung applies:

  | Kernel | Fork lives / lands in | Other ops binding it — **sunset list, not authorization to convert in place** |
  |---|---|---|
  | `reader_unary_sharded.cpp` | `eltwise/unary/device/kernels/dataflow/` | broadly shared: `sharded_to_interleaved`, `tilize` (×2), `transpose_wh_sharded`, `untilize` (×3), `untilize_with_unpadding`, `slice_write` (×2) |
  | `writer_unary_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded/device/kernels/dataflow/` | `sharded_to_interleaved` |
  | `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded/device/kernels/dataflow/` | `sharded_to_interleaved` |
  | `eltwise_copy.cpp` | `ttnn/cpp/ttnn/kernel/compute/` | `copy` (×2), `interleaved_to_sharded`, `sharded_to_interleaved`, `interleaved_to_sharded_partial`, `untilize_with_unpadding` |

  Note all three of the non-reader kernels live **outside this op's directory** (two under `data_movement/sharded/`, one in the shared compute pool), so every fork you create at rung 2 exercises the sanctioned peer-directory carve-out: add `<stem>_metal2.cpp` beside the original, add the pointer comment to the original, change nothing else there.

  **Open decision on the reader — confirm with the invoker before you fork it.** A real, non-quasar Metal 2.0 fork of `reader_unary_sharded.cpp` already exists on `main` at `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_unary_sharded_metal2.cpp` (PR #51397) — but in **typecast's** tree, not beside the original, so the rung-1 locational check misses it. Its shape fits this op's reader closely (`DataflowBuffer dfb(dfb::in); dfb.push_back(get_arg(args::num_tiles_per_core));`). Binding it versus creating a fork beside the original is a convention call the audit raised for the user (`METAL2_PREPORT_AUDIT.md` → Questions #2), and it should be answered **once for both this op and the sibling**. Do not decide it yourself.

- **`experimental/quasar/sharded_to_interleaved/` is a pre-port copy of the sibling op and binds these same kernels — stay out of it.** It will read as a finished, working answer to every question this port raises. It is not one: those copies are deliberate shortcut ports carrying idioms the current whitelist forbids (a stale `api/dataflow/circular_buffer.h` include, `cb_*` handle naming). Do not read it, template from it, lift its binding names, or count its `_metal2` files as forks to reuse.

- **RTA varargs:** none — every arg in every kernel is nameable (see the RTA-naming note above). If you find yourself reaching for the vararg mechanism here, re-read the arg list; the audit found no counted loop, no running `arg_index++`, and no data-selected index in any of the four kernels.

- **Recently-fixed site — don't "restore" it.** The RM writer's accessor takes a **clean base**, with the per-core column shift riding each write as a destination `offset_bytes`:

  ```cpp
  const auto s0 = TensorAccessor(dst_args, dst_addr);          // :25  — base only, deliberately
  noc.async_write(dfb_out, s0, block_width_bytes,
                  {.offset_bytes = cb_read_offset},
                  {.page_id = stick_id, .offset_bytes = input_width_offset_bytes});   // :34-39
  ```

  The previous form folded the offset into the accessor base, which Metal 2.0 cannot express — it was the sibling op's blocking gate until PR #51747 split it out (the in-file comment at `:22-24` says so). Keep the base clean and keep `input_width_offset_bytes` on the per-write destination args; it becomes an ordinary named RTA and nothing else about the site changes.

- **Known pre-existing issues you should *not* fix in this port.** The audit filed these with the ops team; they are noise you will trip over while reading, not work items: missing `cache_tensor` layout/dtype/device validation (`device_operation.cpp:12-48`), `output_mem_config` being a fully unused but hash-keyed attribute, `output_dtype`'s only check being vacuous, and `is_l1_aligned` hardcoded `true` (`program_factory.cpp:55`) making a guard unconditional. None affects the port; none belongs in the port diff.
