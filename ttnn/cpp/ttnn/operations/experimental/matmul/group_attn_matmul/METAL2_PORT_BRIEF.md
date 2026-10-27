# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `akertesz/op-porting-recipe` @ `4bd4bf42bfe` *(carry this line into the port report's Provenance section; the audit read the recipe branch's `docs/…/metal_2.0/` tree directly from `origin/akertesz/op-porting-recipe`, which this branch does not carry)*

**Code audited:** `origin/main` @ `0df0dde3316` (2026-10-27); all `file:line` references below are against it.
One factory, three owned kernels (no borrowed files, no forks), six CBs (one conditional), two semaphores, three
tensor bindings whose case flips with sharding.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (PR #45074). The factory lives in the `program_factory_t` variant
  (`device/group_attn_matmul_device_operation.hpp:173`) — no direct-descriptor conversion needed; the port is a
  method swap inside `GroupAttnMatmulProgramFactory` (`device/group_attn_matmul_program_factory.hpp:216-227`).
- **Op-owned tensors:** none.
- **Target concept:** **`ProgramSpecFactoryConcept`** — no `override_runtime_arguments` exists; the framework
  refreshes the tensor bindings on a cache hit.
- **Custom `compute_program_hash`:** none (default reflection hash; the former custom hash was removed by
  PR #47232 — see the stale-comment note under *Watch for*).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that is
  neither `none` nor an analysis pointer (cell reads `none`) · `get_dynamic_runtime_args`. A pybound
  `create_descriptor` is also absent (`group_attn_matmul_nanobind.cpp:105` binds only the public wrapper; nothing
  to delete).
- **One public entry point.** `ttnn::experimental::group_attn_matmul` (`group_attn_matmul.cpp:39`) always passes
  `num_tokens = nullopt`, `transpose_hw = nullopt` (`:79-80`); the KV-cache attributes on the device op are not
  reachable from Python or any C++ caller today. Port the factory as written (the `transpose_hw` CTA and
  arithmetic stay — the port changes no semantics), but you will not be able to exercise that branch.

## Construct — to do

**Tensor bindings** (per binding; all three arrive in the `Buffer*`-binding form of PR #49134). Each binding is
**conditional on the interleaved config**; in the sharded config the same `TensorParameter` is reached through
`borrowed_from` on the matching DFB and the kernel's accessor is already `#ifndef`-gated:

- `input_tensor_b` — interleaved: **Case 1** (via `TensorAccessor`,
  `device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul.cpp:83`) → `TensorParameter` /
  `TensorBinding` on the reader `KernelSpec`(s); kernel uses `TensorAccessor(tensor::<in1>)` inside the existing
  `#ifndef IN1_SHARDED` (`reader:81-84`). Drops: reader RTA slot 2 (`device/group_attn_matmul_program_factory.cpp:361`)
  and the `TensorAccessorArgs` CTA tail (`:253`, kernel `TensorAccessorArgs<3>()` `reader:82`). Sharded
  (`in1_is_sharded`, `:160`): **clean** — `in2` borrows it (`borrowed_from`, legacy `:203`); no reader binding.
- `input_tensor_a` — interleaved: **Case 1** (`device/kernels/dataflow/writer_transformer_group_attn_matmul.cpp:65`)
  → `TensorParameter` / `TensorBinding` on the writer; accessor inside `#ifndef IN0_SHARDED` (`writer:64-66`).
  Drops: writer RTA slot 1 (`:406`) and the first writer CTA tail (`:260`, `writer:62`). Sharded (`in0_is_sharded`,
  `:159`): **clean** — `in0` borrows it (`:174`).
- `output` — interleaved: **Case 1** (`writer:71`) → `TensorParameter` / `TensorBinding` on the writer; accessor
  inside `#ifndef OUT_SHARDED` (`writer:70-72`). Drops: writer RTA slot 2 (`:407`) and the second CTA tail (`:261`,
  `writer:69` — note it is chained off the first tail's `next_compile_time_args_offset()`; both go). Sharded
  (`output_is_sharded`, `:161`): **clean** — `out` borrows it (`:243`).

The `Buffer*` RTAs are present in every config today even where the kernel ignores them; they simply vanish.
Declare all three `TensorParameter`s unconditionally (the validator counts a `borrowed_from` as a use,
`tt_metal/impl/metal2_host_api/program_spec.cpp:649-650`); make only the *bindings* conditional.

**TensorParameter relaxation:** `none`. Keep strict.

**TensorAccessor 3rd arg:** none — no accessor in the op passes one (`reader:83`, `writer:65`, `:71`).

**Semaphores:** two `SemaphoreDescriptor`s (`:142-153`; ids 0 / 1, `all_device_cores`, `initial_value = INVALID`
= 0 = the Metal 2.0 default) → two `SemaphoreSpec`s with `target_nodes` = the full grid, bound on every reader
`KernelSpec` (see the `in1` item below for why there are two). The kernel builds `Semaphore<> sender_sem(sem::<sender>)`
/ `receiver_sem(sem::<receiver>)` in place of the two id RTAs (`reader:54-55`, host `:389-390`). The remote
operations — `sender_sem.up(noc, x, y, 1)` (`reader:315-316`) and `receiver_sem.set_multicast(noc, …)`
(`:295-301`) — rely on every node holding the semaphore at the same address, which full-grid placement gives you.

**CB endpoints** (six CBs; all on the full device grid, `:60-62`; census per node class — **Q** = q-work nodes,
**M** = mcast-only nodes (`has_work_for_q_heads = 0`, `has_work_for_mcast_kv_heads = 1`, i.e. cores `q_heads … 31`,
present only when `q_heads < 32`), **I** = idle nodes):

- `in0` (CB 0, `:163-176`, capacity `Kt` tiles interleaved / the whole shard sharded) — legal 1:1: writer PRODUCER
  (`reserve_back` `writer:98`, `push_back` `:113`), compute CONSUMER (`wait_front` `compute:74`, `pop_front` `:174`).
  `borrowed_from = input_tensor_a` when `in0_is_sharded` (`:174`).
- `in1` (CB 1, `:178-190`, `2 · in1_block_num_tiles`) — **Q nodes: legal 1:1** (reader PRODUCER `reader:168` /
  `:323`, compute CONSUMER `compute:79-80` / `:142`). **M nodes: reader self-loop** — the reader both pushes and pops
  there (`reader:327-328`, lockstep with the mcast grid) and compute never runs. Express it with **two reader
  `KernelSpec`s** (same source, CTAs, defines, run-arg schema): `reader` bound PRODUCER of `in1` in the Q + I
  `WorkUnitSpec` alongside compute and writer; `reader_mcast_only` bound PRODUCER **and** CONSUMER of `in1` in a
  second `WorkUnitSpec` whose `target_nodes` are exactly the M nodes, **with no compute / writer kernel** (they
  return at their first statement there today, `compute:20-21`, `writer:15-18`). Emit the second work unit only when
  the M set is non-empty. Do **not** set `allow_instance_multi_binding` — the per-node invariant is satisfied by the
  disjoint node sets (`dataflow_buffer_spec.hpp:41-50`; `WorkUnitSpec` rules at `program_spec.cpp:2329-2399`).
- `in2` (CB 2, `:192-205`, the whole `in1` shard, **only when `in1_is_sharded`**) — **self-loop on the reader** (its
  only toucher is the `get_read_ptr()` peek `reader:123` feeding raw NoC reads `:189-196` / `:239`); bind it
  PRODUCER + CONSUMER on *both* reader `KernelSpec`s. **Conditional DFB:** declare the `DataflowBufferSpec` and the
  bindings only under `in1_is_sharded`, `borrowed_from = input_tensor_b` (`:203`); gate the kernel-side references
  — see *Watch for*.
- `intermed0` (CB 3, `:207-219`) — legal 1:1: compute PRODUCER (`compute:145`, `:151`), writer CONSUMER
  (`writer:131`, `:141`).
- `intermed1` (CB 4, `:221-230`, `MtNt` tiles) — legal 1:1: writer PRODUCER (`writer:115`, raw loopback writes
  through the `get_write_ptr()` peek `:116` / `:133-139`, `push_back` `:153`), compute CONSUMER (`compute:162`, `:171`).
- `out` (CB 5, `:232-245`) — legal 1:1: compute PRODUCER (`compute:164`, `:169`), writer CONSUMER (interleaved
  `writer:156` / `:166`; `OUT_SHARDED` `wait_front` only `:176`). `borrowed_from = output` when `output_is_sharded`
  (`:243`).

No multi-binding flag; no dead CB.

## Watch for

- **CB endpoints (multi-binding):** none. If your own census of `in1` tempts you toward the flag, recount per
  node set: on no node are there two *touching* consumers — the second consumer only appears if one reader
  `KernelSpec` is bound PRODUCER + CONSUMER over the *whole* grid. Split the node sets instead (above).
- **`in2` references outside `#ifdef IN1_SHARDED`.** The legacy reader declares `cb_id_in2` (`reader:71`),
  constructs the object (`:75`) and peeks it (`:123`) unconditionally, and consumes the peeked address at `:239`
  inside `if (mcast_in1_to_local_cb)` — a runtime bool only ever set under `IN1_SHARDED` (`:129-132`) but whose
  branch compiles in both builds. Post-port `dfb::<in2>` does not exist in the interleaved build, so gate
  `:71` / `:75` / `:123` under `#ifdef IN1_SHARDED` and either gate `:238-254` too or give `in1_sharded_cb_addr` an
  `#else` fallback (*Pattern: Conditional / optional resource bindings*). Preprocessor-level only; no logic change.
- **Dead `get_tile_size(cb_id_in1)` in the writer (`writer:59`).** Unused, and the writer binds no `in1` (no token,
  no object). Drop the line and record the drop in the port report — pending the invoker's answer to the audit's
  Question 2. `onetile` (`writer:57`, `compute:60`) is an unused constant; leave it.
- **Cross-op / shared kernels:** none — all three kernels are owned, unshared, unforked. This port creates no
  `_metal2` fork and the originals are rewritten in place.
- **RTA varargs: one genuine block, in the reader.** The mcast-sender coordinate tables are read through pointers
  into the arg buffer (`get_arg_addr`, `reader:61-64`) and indexed in loops (`:96-97`, `:107-108`); their counts
  `in1_mcast_sender_num_x` / `_y` (`:59-60`, host `:393-394`) are the x / y extents of the sender grid (`:123-137`),
  appended as two vectors (`:396-399`). → `advanced_options.num_runtime_varargs = num_x + num_y` per node, values
  `[noc_x…, noc_y…]`, read with `get_vararg(i)` (x entries at `0 … num_x-1`, y at `num_x … num_x+num_y-1`); the two
  counts stay **named**. Name everything else — reader (`reader:17-60`): `has_work_for_mcast_kv_heads`,
  `has_work_for_q_heads`, `Mt`, `Nt`, `num_kv_heads`, `in1_CKtNt`, `in1_CKtNt_mul_32`, `blocks`, `in1_start_id`,
  `in0_block_w`, `out_block_w`, `in1_num_subblocks`, `in1_num_blocks`, `in1_block_num_tiles`, `Nt_bytes`,
  `in1_block_w_tile_bytes`, `out_last_subblock_w`, `in1_last_block_w_tile_read_bytes`, `in1_last_block_addr_skip`,
  the four `in1_mcast_dest_noc_*`, `in1_mcast_num_dests`, `in1_mcast_num_cores`, `in1_mcast_grid_size`,
  `in1_mcast_sender_size_bytes`, `in1_mcast_sender_id`, `in1_mcast_sender_num_x`, `in1_mcast_sender_num_y` (slot 2
  → tensor binding, the two semaphore ids → `sem::` bindings); writer (`writer:15-39`): `has_work_for_q_heads`,
  `Mt`, `Kt`, `Nt`, `MtKt`, `blocks`, `in0_start_id`, `out_start_id`, `in0_block_w`, `in1_num_subblocks`,
  `in1_num_blocks`, `out_num_tiles`, `bfloat16_row_bytes`, `bfloat16_Nt_bytes`, `bfloat16_last_row_bytes_read`
  (slots 1-2 → tensor bindings); compute (`compute:20-39`): `has_work_for_q_heads`, `batch`, `Mt`,
  `num_kv_heads_skip`, `num_kv_heads_remaining`, `in0_block_w`, `out_subblock_h`, `in1_num_subblocks`,
  `in1_num_blocks`, `in0_block_num_tiles`, `in1_block_num_tiles`, `out_num_tiles`, `in0_subblock_num_tiles`,
  `in1_per_core_w`. CTAs are all named: reader `transpose_hw`, `row_major`, `out_subblock_w` (`reader:66-68`);
  writer `out_subblock_w`, `intermediate_num_tiles` (`writer:42-43`; CTA 0 is the output CB index → `dfb::<out>`,
  not an arg); compute `transpose_hw`, `out_subblock_w`, `out_subblock_num_tiles`, `intermediate_num_tiles`
  (`compute:42-45`). The accessor tails (`:253`, `:260-261`) go away. Several of these are constants in practice
  (`in1_start_id = 0`, `in1_num_subblocks = 1`, `out_subblock_h = 1`, `blocks = 1` on active cores) — keep them
  as named args with the same values; the port changes no values.
- **Full-grid kernels with flag-steered idle cores (`:331-447`).** All kernels, CBs and semaphores are on the
  whole device compute grid; the run-arg loop walks every device core in `grid_to_cores_with_noop` order
  (`:332-337`) and sets `has_work_for_q_heads` / `has_work_for_mcast_kv_heads` per core (`:352-353`), with
  `num_output_blocks_per_core = 0` beyond `num_cores` (`:347-349`). Keep the full-grid placement (less the M
  nodes, which move to the second work unit) and the per-core named values exactly.
- **Loopback L1→L1 copies — keep as is.** Reader `:189-196` (sharded `in1` staging copy from `in2`'s address into
  `in1`, under `IN1_SHARDED`) and writer `:133-139` (untilized row gather from `intermed0` into `intermed1`) build a
  `UnicastEndpoint` from `my_x[noc.get_noc_id()]` / `my_y[…]` (`reader:117-119`, `writer:87-89`) plus a peeked CB
  address. This is the Device 2.0 migration guide's *"Avoid (Device 2.0 API, but hand-built loopback endpoint)"*
  form (guide `:526-551`); the preferred local-handle form is a transfer rewrite, out of port scope. Only the peeks
  change spelling (`CircularBuffer` → `DataflowBuffer`).
- **The DFB is the multicast destination.** `noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(mcast_src, cb_in1_obj, …)`
  (`reader:243-253`) and the two plain multicasts (`:257-267`, `:272-282`) target `in1` at `.offset_bytes = 0` from
  a `CoreLocalMem` built on a peeked address (`:239` the `in2` read pointer, `:271` the `in1` write pointer). Pass
  the `DataflowBuffer` where the `CircularBuffer` was; nothing else changes. The Blackhole-only
  `noc.async_writes_flushed()` (`:288-292`) stays.
- **Cursor peeks.** `reader:169` (`in1` write pointer after `reserve_back`), `reader:123` (`in2` read pointer once
  at entry — static, the borrowed shard never moves), `writer:116` (`intermed1` write pointer after
  `reserve_back`), `writer:132` (`intermed0` read pointer after `wait_front`). Public peeks
  (`cb_dfb_api_whitelist.md` §C), same method names on `DataflowBuffer`. No cursor surgery.
- **Defines pair with the conditional resources.** `IN0_SHARDED` → writer (`:265-267`), `IN1_SHARDED` → reader
  (`:268-270`, both reader `KernelSpec`s), `OUT_SHARDED` → writer (`:271-273`) → `compiler_options.defines`; each
  shares its condition with one `borrowed_from` DFB and the absence of one `TensorBinding`.
- **Compute `hw_config` — Style A with dropped fields.** The op resolves the TTNN config
  (`get_compute_kernel_config_args`, `:34-35`) but copies only `math_fidelity` and `fp32_dest_acc_en` onto the
  `ComputeConfigDescriptor` (`:317-320`); `math_approx_mode` and `dst_full_sync_en` are resolved and dropped
  (descriptor defaults `false` / `false`), `packer_l1_acc` has no counterpart. So: translate with
  `to_compute_hardware_config(device->arch(), config)`, then on the returned `ComputeGen1Config` force
  `sfpu_precision_mode = Precision::Precise` and `double_buffer_dest = true` regardless of the caller's values
  (legacy behaviour; a silently-ignored knob is preserved, not repaired). `bfp_pack_precision_mode` stays default.
  **`unpack_modes`:** the legacy descriptor sets no `unpack_to_dest_mode` (`Default` → `UnpackMode::UnpackToSrc`);
  the validator demands an explicit entry for every Float32 DFB the compute kernel consumes while
  `enable_32_bit_dest` is on. The compute kernel consumes `in0`, `in1`, `intermed1`. `in0` / `in1` are Float32 when
  the input dtypes are (`:37-38`; validation then forces `fp32_dest_acc_en`, `:48-51`); `intermed1` carries
  `interm_data_format`, Float32 exactly when `fp32_dest_acc_en && in0 == Float32` (`:39-41`). Add a
  `{<dfb>, UnpackMode::UnpackToSrc}` entry for each of those three, gated on that DFB's format being `Float32`.
  `test_group_attn_matmul_fp32` (`test_attn_matmul.py:462`, `in_dtype = float32`) exercises the branch.
- **`opt_level`:** none set anywhere (`grep -n opt_level` on the factory returns nothing) → the compute spec
  resolves to **O3** and must say `KernelBuildOptLevel::O3` explicitly; both DM specs resolve to O2 (defaults).
- **DM `hw_config`:** reader = `RISCV_1` + `preferred_noc_for_dram_read(arch)` (`:275`, `:287-290`), writer =
  `RISCV_0` + the other NOC (`:276-277`, `:300-303`). The helper returns `NOC_0` for every arch today
  (`tt_metal/api/tt-metalium/kernel_types.hpp:140-145`), so the resolved triples equal the reader / writer defaults
  (`DM_DEDICATED_NOC`) → `create_reader_datamovement_config(device->arch())` /
  `create_writer_datamovement_config(device->arch())`. Both reader `KernelSpec`s take the reader helper. Record in
  the port report that the match is by value as of `0df0dde3316`.
- **`constexpr` metadata:** none from CB metadata. `reader:78` `in1_tile_bytes`, `writer:61` `in0_tile_bytes`,
  `writer:68` `out_tile_bytes` are runtime `const` → `dfb.get_tile_size()` getters.
- **LLK calls take `dfb::name` directly** (*Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers*):
  `compute_kernel_hw_startup<SrcOrder::Reverse>(c_0, c_1, c_3)` (`compute:64`), `matmul_init` (`:67`),
  `pack_untilize_dest_init<N>(c_3)` / `pack_untilize_dest<N>(c_3)` / `pack_untilize_uninit(c_3)` (`:83`,
  `:147-148`), `matmul_tiles` (`:116`), `reconfig_data_format_srca(c_1, c_4)` / `pack_reconfig_data_format(c_3, c_5)`
  (`:160-161`), `tilize_init_short_with_dt(c_1, c_4, …, c_5)` / `tilize_block(c_4, …, c_5)` / `tilize_uninit(c_4, c_5)`
  (`:167-168`, `:172`) — all `uint32_t` cb ids; no `.id` extraction, no temporary wrappers. The compute kernel's
  `#ifdef ARCH_GRAYSKULL` branch (`:90-106`) is byte-identical to the `#else` branch — port both as they are.
- **Stale hash comments — do not carry them.** `group_attn_matmul_program_factory.hpp:217-222` and
  `group_attn_matmul_program_factory.cpp:155-158` claim `padded_shape` is folded into `compute_program_hash()`;
  that custom hash was removed by PR #47232 and the op is on the default reflection hash, which keys on the full
  `TensorSpec`s — so the shape-keyed capacities (`:165`, `:180`, `:209`, `:223`, `:234`) stay safe. `:58-59`
  ("CB-size re-application on cache hit") is contradicted by `:155-158`; drop both.
- **Tests (no-regression baseline, confirm with the invoker):**
  `tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py` — `test_group_attn_matmul` (`:228`; bf8,
  `q_heads` 32 / 32 / 16, all sharding combinations), `test_group_attn_matmul_with_program_cache` (`:327`),
  `test_group_attn_matmul_fp32` (`:462`; float32 + bf16, all-sharded, `q_heads` 32 / 32 / 2),
  `test_group_attn_matmul_with_program_cache_exhaustive` (`:630`; `q_heads` 16-64 × sharding × orientation ×
  buffer type × optional output). The `q_heads < 32` cases (`16`, `24`, `2`) are the ones that instantiate the
  M-node work unit. The `test_attn_matmul*` cases in the same file are a different op. Production callers:
  `models/demos/falcon7b_common/tt/falcon_attention.py:698`, `:805`;
  `models/demos/ttnn_falcon7b/tt/falcon_attention.py:167`, `:230`;
  `models/demos/t3000/falcon40b/tt/falcon_attention.py:520`, `:575` (16 q-heads per chip — the M-node shape in
  production).
