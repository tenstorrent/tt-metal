# Metal 2.0 Port Brief — `data_movement/sharded/interleaved_to_sharded`

Porter-facing brief. Mirrors the porter-relevant items of `METAL2_PREPORT_AUDIT.md`; consult that
document for the full gate detail, the census evidence, and the team-only sections.

**Audit result:** GREEN. Audited 2026-09-11 at `e3ca937f19c` against the readiness sheet of that
date, which produced a RED on one cell (`Custom hash` = `no` vs. the code's `compute_program_hash`).
The gate was re-run 2026-09-14 against the live sheet, which now reads `yes`; the conflict is gone
and every other gate was clear throughout.

**Audit docs provenance (inherited):** recipe read from `origin/akertesz/op-porting-recipe` @
`4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's
current state`.

**Scope:** one device-op, one factory, no bundling —
`InterleavedToShardedDeviceOperation` → `InterleavedToShardedProgramFactory`
(`device/interleaved_to_sharded_program_factory.cpp:58`).

## TTNN factory analysis

- **Target concept: `ProgramSpecFactoryConcept`** (base). The ported-from factory has no
  `override_runtime_arguments`, so there is nothing to translate and no custom concept.
- **Custom `compute_program_hash`: present** — declared `device/interleaved_to_sharded_op.hpp:35`,
  defined `device/interleaved_to_sharded_op.cpp:144-162`. **Leave it exactly as it is.** It keys on
  the whole input `TensorSpec` (plus the output's when pre-allocated), so it is at least as strict as
  the strict `TensorParameter` match the port introduces — no `TensorSpec` legality failure is
  expected on cache hits.
- **Op-owned tensors:** none.
- **Pybind:** no `nb::class_` of the device op and no pybound `create_descriptor` — nothing to delete
  in `interleaved_to_sharded_nanobind.cpp`.
- **`TensorParameter` relaxation: `none`.** Declare strict; no `analyses/relaxations/` doc applies.
  (The sheet's informational `Provisional relaxation finding (Edwin)` cell reads
  `fix merged, then match_padded_shape` — a *later* roadmap item, not this port's.)

## Config vocabulary

Three axes; several items below are per-config.

| axis | values | decided by |
|---|---|---|
| layout | `TILE` / `RM` | `input.layout()` — factory:98 / :120 |
| output buffer | `dst-L1` / `dst-DRAM` | `dst_buffer->buffer_type()` — factory:94 |
| format conversion | `convert_df` / plain | input dtype != output dtype — factory:90; **TILE-only**, rejected on RM at `device/interleaved_to_sharded_op.cpp:92-96` |

Six reachable combinations: `TILE·{plain,convert_df}·{dst-L1,dst-DRAM}`, `RM·{dst-L1,dst-DRAM}`.

## Tensor bindings

| binding | config | today | case |
|---|---|---|---|
| `input` | all six | `Buffer*` pushed into the reader's `RTArgList` (factory:291 `TILE`, :388 `RM`) → framework `BufferBinding`; fed to `TensorAccessor(src_args, src_addr)` | **Case 1** — `TensorParameter` + `TensorBinding`; the address RTA and the `TensorAccessorArgs` CTAs both drop |
| `output` | `dst-DRAM` | `Buffer*` in the writer's `RTArgList` (factory:306, :406) → `TensorAccessor(dst_args, dst_addr)` | **Case 1** — same treatment |
| `output` | `dst-L1` | borrowed-memory CB: `cb.buffer = dst_buffer` (factory:46, set :174); the writer only `wait_front`/`pop_front`s | **clean** — no Case 1/2 work; `DataflowBufferSpec::borrowed_from = OUTPUT` |

Neither Case 1 is the silent-wrong hazard: the op never smuggled a raw `->address()`, so both are
routine. **No Case 2 anywhere**, and no compute-kernel tensor binding.

## Offset base pointers

**None.** No `->address()` / `.address()` expression exists anywhere in the factory — both bases ride
the `Buffer*` push form, which carries the base only and has no host expression a fold could hide in.
Every per-core displacement already travels as its own scalar and is applied kernel-side
(`aligned_width_offset` / `aligned_offset` on `RM`, `curr_idx_h + curr_idx_w` / `starting_idx_h` on
`TILE`, `start_id` / `start_id_base` on the `dst-DRAM` writers). #51747 did this split-out; nothing is
left for the porter.

## TensorAccessor 3rd argument

**None — no sites.** All four accessor constructions use the 2-arg form
(`reader_unary_sharded_blocks_interleaved_start_id.cpp:40`,
`reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:36`,
`writer_unary_sharded_blocks_start_id.cpp:29`, `writer_unary_sharded_stick_layout_start_id.cpp:24`).

## CB endpoints

Verify each against your own kernel-touch census; these are the audit's, and the census is the
authority.

| CB | config | touchers on a node | disposition |
|---|---|---|---|
| `c_0` as **input** CB (`convert_df` only, factory:151-163) | `TILE·convert_df` | reader locked producer · compute locked consumer | plain 1:1 — bind 1P + 1C |
| `c_0` as **output** CB (`out_cb_index == input_cb_index` when `!convert_df`, factory:145) | `TILE·plain`, `RM` | reader locked producer · writer locked consumer | plain 1:1; **borrowed** from the output tensor under `dst-L1` (`cb.buffer = dst_buffer`, factory:174) |
| `c_16` (output CB when `convert_df`, factory:152 + :167-174) | `TILE·convert_df` | compute locked producer · writer locked consumer | plain 1:1; borrowed under `dst-L1` |
| `c_1` (alignment scratchpad, factory:176-192) | `RM` | **1** — the `RM` reader only (`reserve_back` :62, `push_back` :137, raw peek `get_write_ptr()` :79) | **self-loop** — bind the `RM` reader PRODUCER *and* CONSUMER |
| `c_1` | `TILE` | **0** — no kernel receives the index | **conditional DFB** — build **no** spec under `TILE`. **Do not drop it outright**: it is live under `RM`. |

The `(0, 0)` result under `TILE` was established by tracing the index, not by assuming: the
allocation condition at factory:179 is layout-independent (its last disjunct is the hardcoded
`keep_l1_aligned = true` at factory:65), so the CB is allocated in every config, but its index reaches
a kernel only through the `RM` reader's CTA list (factory:207). No helper takes a CB index, no index
is computed or aliased, and there are no `#ifdef`-gated kernel variants.

**Nothing in this op needs the multi-binding advanced option.** The hidden-second-writer hunt came
back negative in every config: the only raw-pointer accesses are the `RM` reader's own
`dfb_in1.get_write_ptr()` (:79) and the local L1 loopback read addressed from it (:113-121) — one
kernel peeking at the buffer it binds. No `fifo_*_ptr` writes, no `evil_set_*`, no semaphores.

## Watch for

- **Shared kernels — the op owns none; all six come from the in-family pool
  `data_movement/sharded/device/kernels/`.** Rungs, per the shared-kernel Caution:

  | kernel | rung |
  |---|---|
  | `dataflow/writer_unary_sharded.cpp` | **1 — reuse `writer_unary_sharded_metal2.cpp`** beside it. Interface: `dfb::out` CONSUMER, `args::num_units`. Read-only to you. |
  | `compute/eltwise_copy.cpp` | **1 — reuse the *sibling* `compute/eltwise_copy_metal2.cpp`**, and mind which fork. Two forks of this stem exist; the sibling reads `per_core_tile_cnt` as a **runtime** named arg, the one under `ttnn/cpp/ttnn/kernel/compute/` as a **`constexpr`**. i2s emits the count **per core** (factory:425), so only the sibling fits. |
  | `dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp` | **2 — create the fork** beside the original. |
  | `dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | **2** |
  | `dataflow/writer_unary_sharded_blocks_start_id.cpp` | **2**. Near-miss: `writer_unary_sharded_blocks_interleaved_start_id_metal2.cpp` is a *different* kernel (s2i's writer), not a fork of this one. |
  | `dataflow/writer_unary_sharded_stick_layout_start_id.cpp` | **2**. Same near-miss shape (`writer_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp` is s2i's). |

  Remaining consumers of the four rung-2 originals, for the report's sunset list:
  `interleaved_to_sharded_partial` (blocked on its own gate — `Is able to port? = no`) and the tt-metal
  DM microbenchmark `tests/tt_metal/tt_metal/data_movement/interleaved_to_sharded_hardcoded/`. That is
  a **sunset list, not authorization** to convert anything in place.

- **Nearest in-tree precedent — the mirror op `sharded_to_interleaved`**, already on Metal 2.0 on
  `main`, same family, same kernel pool. Use it as an interface reference, not a template. Two details
  are directly reusable knowledge:
  - it binds **one DFB from two kernels under different accessor names** (`.dfb_spec_name = convert_df ? OUT_DFB : IN_DFB, .accessor_name = "out"`, :156-162) — exactly the shape i2s needs for `c_0` when `!convert_df`, where the reader calls it `in` and the writer calls it `out`;
  - its compute `KernelSpec` states `compiler_options = {.opt_level = KernelBuildOptLevel::O3}` because a legacy all-default `ComputeConfigDescriptor` resolves to **O3** while Metal 2.0's `CompilerOptions` defaults to **O2**. i2s's compute descriptor is also all-default (factory:245), so it needs the same explicit `O3`.

- **Two sanctioned free-function CB-index lookups become port work** (whitelist rule 7 — move onto the
  DFB object; do not swap blind): `get_tile_size(cb_id_in0)`
  (`reader_unary_sharded_blocks_interleaved_start_id.cpp:36` — declared `constexpr` and used as a
  **template argument** at :42, so it keeps the free-function form with the binding token) and
  `get_local_cb_interface(cb_id_in1).fifo_page_size`
  (`reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:63` — a plain local, so the
  member getter). Also `get_tile_size(cb_id_out)` at `writer_unary_sharded_blocks_start_id.cpp:27`,
  declared `const`, so the member getter.

- **The `RM` reader is the delicate file.** It carries TRID-tagged multi-slot pipelining (:85-133), a
  local L1 loopback read whose address is built from `dfb_in1.get_write_ptr()` (:79, :119), and a
  sticky-register reset at the end (:141-147). The port is a binding-layer change only — nothing in
  that control flow moves.

- **RTA varargs: none.** Every kernel reads its args at fixed constant indices; all are nameable.

- **Dead RTA in the `RM` reader.** The factory pushes ten reader args (factory:387-398) but the kernel
  reads indices 0 and 2-9: arg **1** (`num_units_per_row`, factory:389) is never read. Under named args
  there is no slot to carry it, so it has no schema entry — the same treatment `sharded_to_interleaved`
  gave its own dead arg 1. Record it.

- **`keep_l1_aligned` is inert.** A documented Python kwarg plumbed into the attributes, but the
  factory hardcodes `true` and never reads the attribute (factory:64-65). Preserve that exactly —
  it is behavior, bug included.
