# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/matmul`

**Audit scope: ONE ProgramFactory — `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`.**
The op directory holds two DeviceOperations and eight ProgramFactories; this audit covers a single
factory by request. The other seven are named below for disambiguation only and were **not**
audited — no statement in this report is a verdict on any of them.

- **`MatmulDeviceOperation`** (`device/matmul_device_operation.hpp`)
  - **`MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`** ← **audited**
    (`device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.{hpp,cpp}`)
  - `MatmulMultiCoreProgramFactory` — not audited
  - `MatmulMultiCoreReuseOptimizedProgramFactory` — not audited
  - `MatmulMultiCoreReuseMcast1DProgramFactory` — not audited
  - `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` — not audited
  - `MatmulMultiCoreReuseMcast2DProgramFactory` — not audited
  - `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` — not audited
- **`SparseMatmulDeviceOperation`** (`device/sparse/sparse_matmul_device_operation.hpp`)
  - `SparseMatmulMultiCoreReuseMcast1DProgramFactory` — not audited

> **Structure — unusually simple, and worth stating because several sibling factories are not.**
> The `.cpp` (1,092 lines) contains exactly **one** program builder,
> `create_program_dram_sharded_descriptor` (44-933), reached from
> `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory::create_descriptor` (937-1090). There is no
> second legacy builder in the file, no sibling factory sharing it, and no
> `override_runtime_arguments` anywhere. Every finding below is unambiguously in scope.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`

**Readiness sheet:** fetched live this session via the Google Drive connector (486 rows, 28 columns);
exactly one row matches this factory.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/matmul` |
| **Overall** | **GREEN** (for `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` only) |
| **DOps / Factories** | `MatmulDeviceOperation` → `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` (1 of 8 factories audited) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 3 bound kernels are Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | **Ok** — all three kernels are matmul-owned; no donor function-call escapes |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — GlobalCircularBuffer / `address_offset` / GlobalSemaphore | N/A · N/A · N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD | N/A (not a `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | **No** (framework-visible) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** |
| *TTNN Readiness* — `override_runtime_arguments` | **No** — absent from the factory entirely |
| *TTNN Readiness* — Pybind `create_descriptor` | **Yes** — `matmul_nanobind.cpp:1309-1323` |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (base) |
| *Port work* — Offset base pointer | **none** — no host-folded offset anywhere |
| *Port work* — Tensor bindings (per binding) | 4: **2 Case 2** (in1, bias) + **2 clean** (in0, output — borrowed DFB) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **N/A** — no kernel constructs a `TensorAccessor` at all |
| *Port work* — CB endpoints | **2 self-loops** (sync-free borrowed), 1 compute self-loop, aliased pair, rest plain 1:1 |

---

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, alongside this file).

Every gate cleared, and this is the most straightforward port of the matmul factories audited so far:
the base concept (no override to translate), one builder, three kernels of which only one is shared,
and no dropped compute-config field. The substance is in the CB dispositions — two sync-free borrowed
CBs that need self-loops, an aliased pair, and a compute self-loop — plus the usual absent
`opt_level`.

There is no blocked code path within this factory and no subset to carve.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.** The sheet's row for
  (`matmul`, `MatmulDeviceOperation`, `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`) reads
  **`yes`**, with `Diego validation = yes`, `Op Classification = PD Op (pointer-patching)`,
  `Execution Model = SPMD`, `Porting Target = ProgramSpecFactoryConcept`. Note that
  `Pointer patching perf issue?` and `Formerly custom hashed?` are both **blank** on this row.

  Lightweight cross-check clean on every primary column:

  | Column | Sheet | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor` returning `ProgramDescriptor` is the struct's **only** member (hpp:14-18) | ✓ |
  | `Custom hash` | `no` | No `compute_program_hash` override on the device-op — see below | ✓ |
  | `Backdoor custom hash` | `no` | No `attribute_values` / `to_hash` anywhere in the op | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | Zero hits across the op's `.cpp` / `.hpp` | ✓ |
  | `Override runtime args method?` | `no` | Zero hits for `override_runtime_arguments` in the factory | ✓ |
  | `Pybind descriptor` | `PR` | Present at `matmul_nanobind.cpp:1309-1323` | ✓ (in-flight PR; still on this checkout) |
  | `Smuggled pointer` | `no` | **Independently verified** — see below | ✓ |
  | Factory-set match | 8 rows | 8 factory structs in code, 1:1 — no phantom or missing row | ✓ |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no` on a `descriptor` row, and
  `Op-owned tensors?` is empty (only ever `yes` on `WorkloadDescriptor`).

  **`Smuggled pointer = no` verified against the code.** Only **two** address expressions exist in
  the whole file, both in the in1 sender/writer args (`:796` in1, `:797` bias). Both are placeholders
  in a `std::vector<uint32_t>` that is converted to a variant vector and rebound before dispatch:

  ```
  :910   // Build variant args: positions [1] and [2] are buffer addresses
  :917   in1_writer_args[1] = in1_tensor;
  :919   if (bias.has_value()) { in1_writer_args[2] = *bias; }
  :921   in1_sender_writer_kernel_desc.emplace_runtime_args(core, in1_writer_args);
  ```

  The bias rebind is guarded by exactly the same `bias.has_value()` condition as the push at `:797`,
  so when bias is absent slot [2] holds `0u` rather than an address — complete and correct. The other
  runtime-arg emplaces carry no address at all: `:772` is the **non-worker** core path (its vector
  holds only `is_worker_core = false`), and `:704` / `:722` / `:734` are the in0 sender / receiver /
  idle paths, whose vectors hold only `worker_core_type`, `sender_id` and a `last_ktile_w` value.
  **in0 and the output never appear as an address at all** — they reach the kernels through borrowed
  CBs (below).

  **On the custom hash.** The device-op declares `compute_descriptor_program_hash`
  (`device/matmul_device_operation.hpp:50`) with a comment that it is *deliberately* not named
  `compute_program_hash`, so the framework does not detect a custom cache hash; it is reached only
  through a pybind alias. The framework uses the **default reflection hash**. Nothing for the port to
  touch.

- **Device 2.0 (every kernel used): GREEN.** All three bound kernels are structurally Device 2.0 —
  `Noc` from `noc.h`, `DataflowBuffer` wrappers, `AllocatorBank`. Zero hits for broad Device-1.0
  idioms (`InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`, raw `noc_async_read(` /
  `noc_async_write(`, raw `noc_semaphore_*`) and no non-sanctioned CB-index free-function holdovers.
  All three are matmul-owned, so there is no donor kernel to gate on.

  | Kernel (under `device/kernels/`) | Bound at | Device 2.0 evidence |
  |---|---|---|
  | `dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` | `:429` | `Noc` ×6 incl. `async_write_multicast`, `DataflowBuffer` ×2 |
  | `dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` | `:445` | `Noc` ×11, `DataflowBuffer` ×4, `AllocatorBank<DRAM>` |
  | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | `:495` | `DataflowBuffer` ×11, LLK compute APIs |

  The mcast handshake uses ordinary program semaphores (three `SemaphoreDescriptor`s, IDs 0/1/2
  assigned manually at `:306-308`, pushed at `:653-658`) rather than raw semaphore addresses.

- **Feature compatibility: GREEN.** Every Appendix A entry scanned against the factory and its three
  kernels. No entry's recognition signals fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `.global_circular_buffer` field, no `remote_index` / `remote_cb_config`, no `global_cb` parameter anywhere in the file |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset`, no `set_address_offset`, no `UpdateDynamicCircularBufferAddress` in any form, no `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`. Three ordinary `SemaphoreDescriptor`s — sender, receiver, and a **sender-valid** flag, which is one more than the mcast factories carry |

- **Offset base pointers: GREEN.** A scan for `address() +`, `addr + …` and `+ …offset…` across the
  file returns **zero** hits, and the only two address expressions (`:796`, `:797`) are placeholders
  overwritten by tensor references. No Type 1, no Type 2, no Type 3, no Type 4.

  The in1 kernel *does* perform address arithmetic — `in1_tensor_addr + l1_read_addr_in1` (`:100`,
  `:148`), `in1_tensor_addr + source_tile * in1_tile_size_bytes` (`:132`) — but every offset is
  computed **kernel-side** from loop state. This gate asks only about *host*-folded offsets, and
  there are none. The checked-in offset triage lists no matmul row, and my own scan agrees.

- **TensorAccessor 3rd argument: N/A.** No kernel in this factory constructs a `TensorAccessor` at
  all — the DM kernels address memory through `AllocatorBank<DRAM>` bank/address endpoints and
  through borrowed CBs. The subject cannot fire.

---

## Port-work summary  *(mirrors the brief)*

- **Target concept: `ProgramSpecFactoryConcept`** (base). The factory has no
  `override_runtime_arguments`, so the framework refreshes tensor bindings on cache hit and the port
  writes one method, `create_program_artifacts`. **Do not add an override.**

- **Tensor bindings — four, split two ways.**

  | Binding | How the host delivers it | How the kernel consumes it | Classification |
  |---|---|---|---|
  | `in1` | RTA slot [1], rebound to the tensor at `:917` | **raw** — `{.bank_id, .addr = in1_tensor_addr + …}` (in1 kernel `:87`, `:100`, `:132`, `:137`, `:148`) | **Case 2** |
  | `bias` | RTA slot [2], rebound at `:919` (bias present only) | **raw** — `{.bank_id, .addr = in3_tensor_addr}` (in1 kernel `:191`) | **Case 2** |
  | `in0` | **borrowed CB** `c_2` (`cb_desc.tensor = &in0_tensor`, `:576`) | `DataflowBuffer dfb_in2(dfb::cb_in0_sharded); dfb_in2.get_read_ptr()` (in0 kernel `:53`, `:60`, `:71`) | **clean** — borrowed-memory DFB read |
  | `output` | **borrowed CB** `c_6` (`cb_desc.tensor = &out_tensor`, `:633`) | `dfb_out_reshard.get_write_ptr()` (in1 kernel `:218`) | **clean** — borrowed-memory DFB read |

  The two **Case 2** bindings are both in **data-movement** kernels, so the sanctioned
  `TensorAccessor::get_bank_base_address` bridge is available and **the port is not blocked**. (A
  Case 2 binding in a *compute* kernel would have blocked it; neither is.) The raw arithmetic stays
  unchanged. The two **clean** bindings are the causal-link-gate case — the borrowed DFB *is* the
  tensor access, and the port expresses each as `DataflowBufferSpec::borrowed_from` naming the
  corresponding `TensorParameter`, not as a Case 1 or Case 2 binding.

- **CB endpoints — seven CBs, four dispositions.** Census run per node across all three kernels.
  Because `skip_compute` / `skip_in0_mcast` / `skip_write_back` are hardcoded `false` (below), there
  is a single code path and no per-config flip; the only conditionals are bias presence and the
  output/intermediate format branch.

  | CB | Backing | Touchers | Verdict | Disposition |
  |---|---|---|---|---|
  | `c_0` in0 mcast dest | regular | in0 kernel (locked producer, `:145`/`:214`/`:224`/`:233`), compute (locked consumer) | plain 1:1 | no action |
  | `c_1` in1 | regular | in1 kernel (locked producer, `:92`/`:108`/`:122`/`:158`/`:162`/`:178`), compute (locked consumer) | plain 1:1 | no action |
  | `c_3` bias | regular | in1 kernel (locked producer, `:183`/`:206`), compute (locked consumer) | plain 1:1 | conditional binding (bias present) |
  | `c_4` out | regular | compute (locked producer, packs), in1 kernel (locked consumer, `:210`/`:246`) | plain 1:1 | no action |
  | `c_5` interm0 | regular | compute only — packs into it and reads it back | 1 toucher | **compute self-loop** |
  | `c_2` in0 sharded | **borrowed** ← `in0_tensor` | in0 kernel only, `get_read_ptr()` at `:71` — **no FIFO ops at all** | 1 toucher, **sync-free** | **self-loop** |
  | `c_6` out reshard | **borrowed** ← `out_tensor` | in1 kernel only, `get_write_ptr()` at `:218` — **no FIFO ops at all** | 1 toucher, **sync-free** | **self-loop** |

  `c_2` and `c_6` are the interesting pair: both are borrowed views that a single kernel reaches by
  base pointer with no `reserve_back` / `push_back` / `wait_front` / `pop_front` anywhere. On Gen1 the
  self-loop is the sanctioned shape (bind the one kernel PRODUCER **and** CONSUMER); the kernel code
  is untouched and runtime behaviour is identical. **Both are also `LocalTensorAccessor` candidates
  for the post-port sync-free style pass** — sync-free *and* borrowed is exactly that pass's target —
  but that is a separate pass, not this port.

  Note both are genuinely kernel-referenced: `c_2` reaches the in0 kernel as the named CTA
  `cb_in0_sharded` (`:436`) and `c_6` reaches the in1 kernel as `cb_out_reshard` (`:454`). Neither is
  a dead CB.

- **Aliased DFBs — a two-member group, config-dependent.** When
  `(interm0_data_format == output_data_format)` **and** not (`untilize_out` with more than one W
  subblock), a **single `CBDescriptor` carries both `c_4` and `c_5`** (`:605-620`) — two logically
  distinct buffers sharing one L1 region. Port as **two `DataflowBufferSpec`s with mutual
  `advanced_options.alias_with`** (same total size, same bound kernels, strict clique). In the other
  branch (`:581-604`) they are two separate descriptors and no aliasing applies. **Derive per
  instantiation.** Unlike some sibling factories there is no third alias index here — the group is
  exactly two.

- **`opt_level` — absent.** `grep -n opt_level` returns nothing. An unset
  `KernelDescriptor::opt_level` still resolves to the legacy per-kernel-type default — **`O3` for a
  `ComputeConfigDescriptor`**, `O2` for DM — while Metal 2.0's `CompilerOptions` defaults to `O2` for
  both. Set `O3` explicitly on the compute `KernelSpec`. The two DM kernels need nothing.

- **Hardware config — Style A, and nothing dropped.** The factory resolves a TTNN
  `ComputeKernelConfig` via `get_compute_kernel_config_args` (`:1041`); the builder **receives**
  `dst_full_sync_en` (signature `:51`) and **sets** it on the `ComputeConfigDescriptor` (`:526`),
  alongside `math_fidelity`, `fp32_dest_acc_en` and `math_approx_mode` (`:523-527`). All four
  helper-covered knobs are set, so `to_compute_hardware_config(device->arch(), config)` translates
  faithfully with nothing to reapply by hand. `packer_l1_acc` has no Metal 2.0 counterpart; no action.

  **No `unpack_to_dest_mode` is set** by this factory, so there is no legacy `unpack_modes` table to
  reindex. The only reason an entry would be required is the Float32-DFB-with-`enable_32_bit_dest`
  rule; check the resolved config rather than the tensor dtypes.

- **DM configs are custom — replicate, do not use the helpers.** Two explicit
  `DataMovementConfigDescriptor`s: `RISCV_1 / in0_noc` (`:439`) and `RISCV_0 / in1_noc` (`:457`),
  where both NOCs come from `preferred_noc_for_dram_write` / `preferred_noc_for_dram_read` rather
  than the reader/writer defaults. Copy each field verbatim into a `DataMovementGen1Config`;
  `create_reader_datamovement_config` / `create_writer_datamovement_config` would substitute the
  default triple and regress silently. **`in1_noc` is additionally load-bearing**: multi-worker mode
  requires it to be `NOC_0` (`:120`).

- **RTA varargs: none.** No kernel reads arguments in a loop, at a data-computed index, or through a
  running counter — every read is `get_arg_val<uint32_t>(<literal>)` at the top of the kernel. All
  become named RTAs.

- **Two dead named CTAs, and one that cannot simply be dropped.** The factory passes
  `cb_in0_intermediate` (`c_8`) and `cb_in1_intermediate` (`c_9`) to the compute kernel (`:507-508`),
  and the compute kernel **references neither**, under any `#ifdef`; no `CBDescriptor` allocates
  either. The port drops them. `cb_in0_transposed` (`c_10`, `:509`) is different: the compute kernel
  reads it at line 200 inside a **parse-time ternary**
  (`in0_transpose_tile ? get_named_compile_time_arg_val("cb_in0_transposed") : …("cb_in0")`), so both
  operands are name-looked-up regardless of the condition. There is no CB behind `c_10`, so the port
  must use the **conditional-binding pattern** (`#ifdef` gating from
  `KernelSpec::compiler_options.defines`) rather than deleting the CTA.

- **Device-operation-class edits the port forces** — two sanctioned exceptions:
  1. **Remove the pybound factory entry point.** `matmul_nanobind.cpp:1309-1323` is an
     `nb::class_<…MultiCastDRAMShardedProgramFactory>` block whose only member is `create_descriptor`.
     Deletion is mandatory once that method goes; user-visible API change.
  2. **Drop the pybind-hook-only parameter.** `create_descriptor`'s fourth argument,
     `const std::optional<CoreRangeSet>& core_range_set`, exists only for that hook. Drop it.

  Exception 3 does not apply — the op has a proper `program_factory_t` variant.

- **The op's cache key is not an edit.** No framework-visible custom hash exists; the default
  reflection hash is in use.

---

## Heads-ups  *(mirrors the brief)*

- **Only one kernel is shared, and it needs a fork.** This is markedly lighter than the mcast
  factories.

  | Kernel | Copies | Binders | `_metal2` fork? | Rung |
  |---|---|---|---|---|
  | `dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` | 1 | **1** — this factory | no | **convert in place** |
  | `dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` | 1 | **1** — this factory | no | **convert in place** |
  | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | 1 | **6** | **no** | **rung 2 — create the fork** |

  The compute kernel's other five binders are `matmul_multicore_reuse_optimized`,
  `matmul_multicore_reuse_mcast_1d` (a file hosting **two** factories),
  `matmul_multicore_reuse_mcast_2d`, `matmul_multicore_reuse_batched_hs_dram_sharded`, and the sparse
  device-op's `sparse_matmul_multicore_reuse_mcast_1d_optimized`. The rung-1 check was run
  **locationally**: `find` over `matmul/device/kernels/` returns **zero** `*_metal2*` files, so this
  port creates the first fork of it. Whatever binding vocabulary that fork uses becomes the interface
  every later consumer inherits — worth agreeing across the matmul factories before any of them ports.

- **`num_workers_per_dram_bank` is a real configuration axis with hard constraints.** Values above 1
  are **Blackhole-only** (`:117`) and require the in1 DM kernel on **NOC_0** (`:120`), and the weight
  shard width in tiles must divide by it (`:144`). It changes the worker count, the reader-to-bank
  assignment, and `per_core_N_in1_sender` — but it adds no kernel and no CB, so the endpoint census
  above holds across it. Port the config field through; do not attempt to normalise it.

- **The factory picks its own subblock shape, and pads it.** `get_matmul_subblock_params` gives a
  starting point, then a widening pass grows `out_subblock_w` up to the DEST limit if that reduces
  the subblock count. Widening can pad `per_core_N_compute` past what the reader actually pushed, so
  `last_subblock_w_valid` tells compute how many lanes of the final subblock are real. Carry that CTA
  across faithfully — it is easy to read as redundant and is not.

- **Op-level preconditions worth knowing while reading the factory** (enforced in the device-op's
  validator, not this factory, and not the port's to change): in0 `WIDTH_SHARDED` in L1 ROW_MAJOR,
  in1 `WIDTH_SHARDED` in DRAM, output sharded matching in0, `M == per_core_M == 1` tile row, and tile
  height ≥ 16 (the tiny-tile path is unsupported here per issue #42927). These explain why the
  factory's shapes look narrower than the mcast factories'.

---

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean**. No function-call escapes to
  another op's helpers and no file-path escapes — all three kernels are owned and instantiated by
  matmul. **Borrowed kernel files: none.** Nothing outside the op calls into this factory either:
  it has no `override_runtime_arguments` and no build helper exported for CCL fused ops, unlike some
  sibling matmul factories.

- **Relaxation candidates:** none. No custom hash to mine; the sheet's `TensorParameter relaxation`
  is `none`.

- **TTNN factory analysis:** op-owned tensors — none. MeshWorkload need — none (`Execution Model =
  SPMD`, `Concept = descriptor`). Pybind `create_descriptor` — `matmul_nanobind.cpp:1309-1323`.
  Other risky pybind — the device-op class binding at `matmul_nanobind.cpp:1222-1237`, which survives
  the port untouched. Custom hash — none framework-visible. `get_dynamic_runtime_args` — absent.
  `override_runtime_arguments` — absent. Target concept — `ProgramSpecFactoryConcept`.

---

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **Three configuration flags are hardcoded off, making their kernel branches dead.**
  `create_descriptor` sets `skip_compute = false`, `skip_in0_mcast = false`, `skip_write_back = false`
  (`:1036-1038`) and passes them straight through (`:1086-1088`). They gate the `SKIP_COMPUTE`,
  in0-side `SKIP_MCAST`, and `SKIP_WRITE_BACK` defines (`:392-399`), none of which is ever emitted on
  this dispatch path. The parameters look like a debug/bring-up affordance that is now unreachable
  from production. Worth an owner deciding whether to keep them.

  **Do not confuse these with the two defines set unconditionally just above them** — `OUT_SHARDED`
  (`:389`) and the in1-writer's own `SKIP_MCAST` (`:390`) are always on. The name `SKIP_MCAST`
  appearing in both an unconditional and a conditional form, twelve lines apart and for different
  kernels, is a genuine trip hazard.

- **Two named CTAs are passed to a kernel that never reads them.** `cb_in0_intermediate` (`c_8`) and
  `cb_in1_intermediate` (`c_9`) at `:507-508`. Harmless today, dropped by the port; flagged so an
  owner can confirm they are leftovers rather than a half-removed feature.

- **Dead placeholder `.address()` calls.** `:796` and `:797` compute an address that the variant
  rebinding overwrites ~120 lines later. Both already carry a `smuggled-rta-ok` marker and `:796`
  additionally says "rebound", which is good practice — but `:797` omits the "rebound" note even
  though its rebind at `:919` is equally real. Aligning the two comments would remove a small
  inconsistency for the next reader.

- **`create_descriptor`'s `core_range_set` parameter is accepted and ignored.** A Python caller
  passing one has it silently discarded. The port removes the parameter, so this resolves itself.

---

## Questions for the user

1. **Fork vocabulary for `bmm_large_block_zm_fused_bias_activation.cpp`.** This port creates the
   first `_metal2` fork of a kernel bound by six factories in this directory. Whichever factory ports
   first sets binding names the other five inherit and cannot change. Worth one decision, centrally,
   before any matmul port lands — it is not specific to this factory but this port is a plausible
   place for it to be settled by default.

2. **Scope.** Six factories in this directory remain unaudited, plus the sparse device-op's factory.
   Should those follow, and in what order?

## Recipe notes

- **The sync-free / self-loop guidance resolved both borrowed CBs cleanly, and the
  "count touchers, then read roles off the census" framing is what made it quick.** `c_2` and `c_6`
  each have exactly one toucher doing a pure base-pointer read with no FIFO ops — textbook
  one-toucher sync-free — and the recipe's insistence that a raw peek is a *toucher* (so the CB is
  not dead) but *role-free* (so the labels are cosmetic) gave the answer without ambiguity. The
  contrast with a genuinely dead CB is well drawn.

- **Useful confirmation of the "classify per instantiation" rule, in the negative.** This factory has
  three `skip_*` flags that look like configuration axes and are in fact hardcoded off, so the CB
  census does *not* need to be repeated across them. Checking where a flag's value comes from before
  treating it as an axis saved a three-fold census here; the recipe implies this but does not say it
  outright.

- **Friction — the sanctioned-free-function list keeps needing a second read on DFB-era kernels.**
  These kernels are already `DataflowBuffer`-based, so `get_read_ptr()` / `get_write_ptr()` appear as
  *member* calls and are unambiguous. But the recipe's holdover cue is phrased around *free*
  functions with a wrapper in scope, and on a first pass it is easy to mis-flag a member call as the
  holdover shape. A one-line "member calls on a `DataflowBuffer` are never holdovers" would close it.
