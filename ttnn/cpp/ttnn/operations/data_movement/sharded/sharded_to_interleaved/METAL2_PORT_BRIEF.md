# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

**Code baseline:** every file named below matched `origin/main` @ `2b7bf3396eb` (2026-08-05) at audit time. Two of this op's gates were cleared by recent merges — `0fb47949a27` (Device 2.0 on `eltwise_copy.cpp`) and `6abdf94214d` / PR #51747 (offset base pointer on the RM writer). **If your checkout predates either, rebase before starting**; on an older base you will be porting code the audit did not clear.

## Scope — what you are porting

One `DeviceOperation`, one factory, **four borrowed kernels** (the op owns none of them):

- **`ShardedToInterleavedDeviceOperation`** → `ShardedToInterleavedProgramFactory` (`device/sharded_to_interleaved_program_factory.cpp`)

| Role | Kernel | Selected when |
|---|---|---|
| Reader | `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | always |
| Writer (tiled) | `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | `input.layout() == TILE` |
| Writer (row-major) | `data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `input.layout() == ROW_MAJOR` |
| Compute (copy) | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | `convert_df` (input dtype ≠ output dtype; TILE only) |

Kernel choice is a **runtime branch inside the one factory** (`program_factory.cpp:177-185`, `:190-196`), not separate factories — so all four kernels and all three configs are in scope for a single change. The three configs:

- **C1** — TILE, no conversion: reader + tiled writer.
- **C2** — TILE, conversion: reader + tiled writer + compute.
- **C3** — ROW_MAJOR (never converts; a dtype mismatch requires TILE per `validate_inputs:67-71`): reader + RM writer.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`device/sharded_to_interleaved_program_factory.hpp:15`).
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`. *(Both derived from `Concept` + `Op-owned tensors?` and confirmed by the readiness sheet's own `Porting Target` cell.)*
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported replacement) · pybind `create_descriptor` — all gate conjuncts — plus other migration-risky pybind, which would have surfaced as a `safe` warning. All `no` on the sheet and confirmed against the code.

## Construct — to do

**Tensor bindings** (per binding):

- **`input_tensor`** — **clean (borrowed-memory DFB)**, not Case 1 or Case 2. `c_0` is Buffer-backed today (`cb.buffer = src_buffer`, `program_factory.cpp:41`, from the `push_s2i_cb_pair` call at `:140-147`); the reader only `push_back`s pages that are already resident and builds no `TensorAccessor`. Port via `DataflowBufferSpec::borrowed_from` on the input `TensorParameter`. Do not give it an accessor.
- **`output_tensor`** — **Case 1** (via `TensorAccessor`), in **all three configs**. The base reaches both writers as a `Buffer*` binding today (`writer_rt.push_back(dst_buffer)`, `program_factory.cpp:242` tiled / `:293` RM → arg 0). Express it as a `TensorParameter` / `TensorBinding`; each writer builds `TensorAccessor(tensor::<out>)`. Two things then disappear together: the **arg-0 base RTA** and the **`TensorAccessorArgs` CTA plumbing** — host-side `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` (`program_factory.cpp:176`) and kernel-side `constexpr auto dst_args = TensorAccessorArgs<1>()` (tiled `:23`, RM `:20`). Mechanical, low-risk.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — both accessor constructions are already 2-argument (tiled `:28`, RM `:25`). Nothing to drop.

**CB endpoints:** all legal 1P+1C on every node in every config — nothing to self-loop, assign, flag, or drop. For reference when you write the specs:

| DFB | Configs | PRODUCER | CONSUMER |
|---|---|---|---|
| `c_0` (`src0_cb_index`, `borrowed_from` the input) | C1, C2, C3 | reader (`dfb.push_back`) | C1: tiled writer · C2: compute · C3: RM writer |
| `c_16` (`out_cb_index`) | C2 only | compute (`cb_out.reserve_back`/`push_back`) | tiled writer |

Note the aliasing you must preserve: when `!convert_df`, `out_cb_index == src0_cb_index == c_0` (`program_factory.cpp:129`) — the writer drains the *same* borrowed DFB the reader fills. `c_16` is allocated **only** under `convert_df` (`:149-160`), and unlike `c_0` it is **not** buffer-backed (`bound_buffer = nullptr`).

**RTA naming — one gotcha per writer.** Both writers read their args at fixed constant indices, so every arg is nameable (no varargs anywhere; see Watch for). But the legacy index sets are not contiguous, and one arg is dead:

- **Tiled writer** reads 0–8 (`:11-19`). Arg 0 is the base (disappears with the binding); 1–8 become named args. Note `start_id = start_id_base + start_id_offset` (`:20`) is a kernel-local sum of args 8 and 7, not a ninth arg.
- **RM writer** reads 0, **2**, 3, 4, 5, 6 (`:12-17`) — the factory pushes **7** args and index **1** (`num_units_per_row`, `:294`) is **never read**. Name only the six the kernel actually reads; do not invent a name for the dead slot. Dropping the host-side push is a behavior-neutral cleanup that belongs to the ops team, **not** to this port — leave `program_factory.cpp:294` alone unless you are told otherwise, and note the mismatch in your port report.

## Watch for

- **CB endpoints (multi-binding):** none. The audit hunted all three faces with positive evidence — no `get_write_ptr` / `get_read_ptr` / `get_local_cb_interface` / `fifo_*_ptr` / `evil_set_*` in any of the four kernels, **no semaphores of any kind** in the op, and no dual-instance work-split (each kernel source goes into exactly one `KernelDescriptor`, `:310-314`). You should not find a hidden co-filler; if you do, the audit was wrong — stop and report it.

- **Cross-op / shared kernels — all four are borrowed, and no `_metal2` sibling exists beside any of them.** Rung 2 for each: fork **into the original's own directory** (`<stem>_metal2.cpp`), convert the copy, point your `KernelSpec::source` at it, and add the pointer comment to the legacy original. Name the bindings for **the kernel's role**, not for this op — three of these four are shared, and your names become every later consumer's interface.

  | Kernel | Fork lands in | Other ops binding it — **sunset list, not authorization to convert in place** |
  |---|---|---|
  | `reader_unary_sharded.cpp` | `eltwise/unary/device/kernels/dataflow/` | broadly shared: `sharded_to_interleaved_partial`, `tilize` (×2), `transpose_wh_sharded`, `untilize` (×3), `untilize_with_unpadding`, `slice_write` (×2) |
  | `writer_unary_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded/device/kernels/dataflow/` | `sharded_to_interleaved_partial` |
  | `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded/device/kernels/dataflow/` | `sharded_to_interleaved_partial` |
  | `eltwise_copy.cpp` | `ttnn/cpp/ttnn/kernel/compute/` | `copy` (×2), `interleaved_to_sharded`, `sharded_to_interleaved_partial`, `interleaved_to_sharded_partial`, `untilize_with_unpadding` |

  **Open decision on the reader — confirm with the invoker before you fork it.** A real, non-quasar Metal 2.0 fork of `reader_unary_sharded.cpp` already exists on `main`, at `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_unary_sharded_metal2.cpp` (PR #51397) — but in **typecast's** tree, not beside the original, so the rung-1 locational check misses it. Its shape is a close fit for this op's reader (`DataflowBuffer dfb(dfb::in); dfb.push_back(get_arg(args::num_tiles_per_core));`). Binding it, versus creating a second fork beside the original, is a convention call the audit raised as a question for the user (`METAL2_PREPORT_AUDIT.md` → Questions #1). **Do not decide it yourself** — get the answer, then apply rung 1 or rung 2 accordingly. The other three kernels are unambiguous rung 2.

- **`experimental/quasar/sharded_to_interleaved/` is a pre-port copy of this exact op — stay out of it.** It will read as a finished, working answer to every question this port raises. It is not one: those copies are deliberate shortcut ports that carry idioms the current whitelist forbids (a stale `api/dataflow/circular_buffer.h` include, `cb_*` handle naming). Do not read it, template from it, or lift its binding names — and do not count its `_metal2` files as forks to reuse. The same applies to the other quasar copies that bind these four kernels.

- **RTA varargs:** none — every arg in every kernel is nameable (see the RTA-naming note above). If you find yourself reaching for the vararg mechanism here, re-read the arg list; the audit found no counted loop, no running `arg_index++`, and no data-selected index in any of the four kernels.

- **Recently-fixed site — don't "restore" it.** The RM writer's accessor takes a **clean base**, with the per-core column shift riding each write as a destination `offset_bytes`:

  ```cpp
  const auto s0 = TensorAccessor(dst_args, dst_addr);          // :25  — base only, deliberately
  noc.async_write(dfb_out, s0, block_width_bytes,
                  {.offset_bytes = cb_read_offset},
                  {.page_id = stick_id, .offset_bytes = input_width_offset_bytes});   // :34-39
  ```

  That shape is exactly what makes this op portable — it is the ops team's Metal-2.0 unblock (PR #51747, and the in-file comment at `:22-24` says so). The previous form folded the offset into the accessor base, which Metal 2.0 cannot express. Keep the base clean and keep `input_width_offset_bytes` on the per-write destination args; it becomes an ordinary named RTA and nothing else about the site changes.
