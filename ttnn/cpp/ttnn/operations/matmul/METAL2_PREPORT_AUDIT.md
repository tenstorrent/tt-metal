# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/matmul`

**Audit scope: ONE ProgramFactory — `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory`.**
The op directory holds two DeviceOperations and eight ProgramFactories; this audit covers a single
factory by request. The other seven are named below for disambiguation only and were **not**
audited — no statement in this report is a verdict on any of them.

- **`MatmulDeviceOperation`** (`device/matmul_device_operation.hpp`)
  - **`MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory`** ← **audited**
    (`device/factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.{hpp,cpp}`)
  - `MatmulMultiCoreProgramFactory` — not audited
  - `MatmulMultiCoreReuseOptimizedProgramFactory` — not audited
  - `MatmulMultiCoreReuseMcast1DProgramFactory` — not audited
  - `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` — not audited
  - `MatmulMultiCoreReuseMcast2DProgramFactory` — not audited
  - `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` — not audited
- **`SparseMatmulDeviceOperation`** (`device/sparse/sparse_matmul_device_operation.hpp`)
  - `SparseMatmulMultiCoreReuseMcast1DProgramFactory` — not audited

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`

**Readiness sheet:** fetched live this session via the Google Drive connector (486 rows, 28 columns).

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/matmul` |
| **Overall** | **GREEN** (for `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` only) |
| **DOps / Factories** | `MatmulDeviceOperation` → `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` (1 of 8 factories audited) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 3 bound kernels are Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | **Ok** — no donor functions; all includes are `api/*` (LLK/HAL) plus one in-op sibling header |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — GlobalCircularBuffer / `address_offset` / GlobalSemaphore | N/A · N/A · N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD | N/A (not a `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | **No** (framework-visible). Renamed hook at `device/matmul_device_operation.hpp:50` — see Gate detail |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** (absent from the op's code) |
| *TTNN Readiness* — `override_runtime_arguments` | **No** (this factory has none) |
| *TTNN Readiness* — Pybind `create_descriptor` | **Yes** — `matmul_nanobind.cpp:1325-1338` (not a gate; port deletes it) |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (base) |
| *Port work* — Offset base pointer | **none** — no `->address()` anywhere in the factory |
| *Port work* — Tensor bindings (per binding) | 4 bindings, **all Case 2** (raw pointer → bridge). All in DM kernels, so none is blocked |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **N/A** — no kernel constructs a `TensorAccessor` at all |
| *Port work* — CB endpoints | **2 dead-CB drops**, **1 compute self-loop**, 4 plain 1:1, plus an aliased-DFB pair |

---

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, alongside this file).

Every gate cleared. The port is nonetheless **not a trivial one** — four findings carry real work and
two of them are the kind that go wrong quietly:

1. **Every tensor binding is Case 2** (raw base pointer, no `TensorAccessor` anywhere in the op's
   kernels). All four sit in data-movement kernels, so the `get_bank_base_address` bridge is
   available and the port is not blocked — but this is the less common classification and the
   factory has no Case 1 site to pattern-match against.
2. **Two CBs have zero endpoints** and must be dropped; a bindingless DFB cannot be expressed.
3. **The compute kernel is shared by six factories in this directory**, so it cannot be converted in
   place — the port creates the first `_metal2` fork of it.
4. **Two aliasing patterns of different kinds** appear on the same CB pair, and the recipe warns
   they are correctness-critical to distinguish.

There is no blocked code path within this factory and therefore no subset to carve.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.** The sheet's row for
  (`matmul`, `MatmulDeviceOperation`, `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory`)
  reads **`yes`**, with `Diego validation = yes`, `Op Classification = PD Op (pointer-patching)`,
  `Execution Model = SPMD`, `Porting Target = ProgramSpecFactoryConcept`. Lightweight cross-check
  against the code came back clean on every primary column:

  | Column | Sheet | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor` returning `ProgramDescriptor` is the factory's only member (`…batched_hs_dram_sharded_program_factory.hpp:14`) | ✓ |
  | `Custom hash` | `no` | No `compute_program_hash` override on the device-op — see below | ✓ |
  | `Backdoor custom hash` | `no` | No `attribute_values` / `to_hash` anywhere in the op | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | Zero hits across the op's `.cpp` / `.hpp` | ✓ |
  | `Override runtime args method?` | `no` | Absent from the factory header and `.cpp` | ✓ |
  | `Pybind descriptor` | `PR` | Present at `matmul_nanobind.cpp:1325-1338` | ✓ (in-flight PR; still on this checkout) |
  | Factory-set match | 8 rows | 8 factory structs in code, 1:1 — no phantom row, no missing row | ✓ |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no` on a `descriptor` row, and
  `Op-owned tensors?` is empty (it is only ever `yes` on `WorkloadDescriptor`).

  **On the custom hash — the code looks misleading and the sheet is right.** The device-op declares
  `compute_descriptor_program_hash` (`device/matmul_device_operation.hpp:50`) carrying a comment
  that it is *deliberately* not named `compute_program_hash`, so the framework does **not** detect a
  custom program-cache hash; it is reached only through a pybind alias. The framework therefore uses
  the **default reflection hash**, which is what the sheet's `no` records. The sheet's
  `Formerly custom hashed? = yes` corroborates the history. **Nothing here for the port to touch.**

  Two non-gating cells worth carrying forward: `Pointer patching perf issue? = "suspect perf
  regression (+ fixed latent bug)"` and `Formerly custom hashed? = "yes"`. Both refer to the earlier
  ProgramDescriptor migration, not to a Metal 2.0 port. Recorded so a post-port performance
  comparison is not misread — a measured regression against an older baseline may predate the port.

- **Device 2.0 (every kernel used): GREEN.** All three bound kernels are structurally Device 2.0 —
  `Noc` from `noc.h`, `DataflowBuffer` wrappers, `UnicastEndpoint` / `AllocatorBank` endpoints. Zero
  hits for broad Device-1.0 idioms (`InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`,
  raw `noc_async_read(` / `noc_async_write(`, raw `noc_semaphore_*`), and no isolated CB-index
  free-function holdovers. All three live in matmul's own `device/kernels/` tree, so there is no
  donor kernel to gate on.

  | Kernel (under `device/kernels/`) | Role | Device 2.0 |
  |---|---|---|
  | `dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp` | in0 reader (RISCV_1) | ✓ `Noc::async_read`, `DataflowBuffer`, `UnicastEndpoint` |
  | `dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp` | in1 reader + output writer (RISCV_0) | ✓ `Noc::async_read/write`, `DataflowBuffer`, `AllocatorBank<DRAM>`, `UnicastEndpoint` |
  | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | compute | ✓ `DataflowBuffer`, LLK compute APIs |

  **One sanctioned free-function site, recorded because the shape cue false-fires.** The compute
  kernel reads `get_local_cb_interface(mm_partials_dfb_id).fifo_rd_ptr` at line 116.
  `get_local_cb_interface` is on the recipe's **sanctioned** list, and that list "does not turn on
  what object is in scope" — so it stays sanctioned even though a `DataflowBuffer` is in scope. Not
  a Device 2.0 violation. It *is* Metal 2.0 port-stage work (a cursor read maps to the DFB's public
  `get_read_ptr()` peek). Note the site sits behind `if (mm_partials_reload_dfb_id !=
  mm_partials_dfb_id)` (line 112), a condition the caller makes always-false by construction
  (line 212 assigns them equal) — so it is compiled but unreachable.

  The two DM kernels additionally use `get_tile_size(dfb_id)` in `constexpr` initialisers
  (in1 kernel lines 59-60) — also sanctioned, and the `constexpr` declaration determines its port
  form (see Port-work summary).

- **Feature compatibility: GREEN.** Every Appendix A entry scanned against the factory and its three
  kernels. No entry's recognition signals fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `.global_circular_buffer` field, no `remote_index` / `remote_cb_config`, no `global_cb` parameter. This factory never reads `MatmulParams::global_cb` |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset`, no `set_address_offset`, no `UpdateDynamicCircularBufferAddress` in any form, no `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | The factory creates **no semaphores at all** — there is no cross-core synchronisation to express |

- **Offset base pointers: GREEN.** The factory contains **no `->address()` / `.address()` expression
  at all**, so there is no host-side arithmetic that could fold an offset into a base. Addresses
  reach the kernels as tensor objects pushed into the runtime-arg lists (lines 563, 569, 571, 579),
  which the framework resolves. No Type 1, no Type 2, no Type 3, no Type 4.

  The kernels *do* perform address arithmetic — `input_shard_l1_addr + read_offset` (in0 reader
  line 61), `in1_tensor_addr + in1_batch_offset + curr_dram_offset` (in1 kernel line 95),
  `output_shard_l1_addr + out_batch_offset` (line 133) — but every one of those offsets is computed
  **kernel-side** from loop variables. This gate asks only about *host*-folded offsets, and there
  are none. The checked-in offset triage lists no matmul row, and my own scan agrees.

- **TensorAccessor 3rd argument: N/A.** No kernel in this factory constructs a `TensorAccessor` at
  all — the DM kernels address memory through `UnicastEndpoint` and `AllocatorBank<DRAM>` directly.
  The subject cannot fire.

---

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings — four, and all four are Case 2.** In every case the host delivers the tensor
  object through the runtime-arg list (the `Buffer*`-binding form, which the framework patches on
  cache hit — this is *not* the silent-wrong smuggled-pointer hazard, and the sheet agrees:
  `Smuggled pointer = no`), and the kernel consumes the resulting base address **raw**, with its own
  arithmetic, never through a `TensorAccessor`.

  | Binding | Host site | Kernel use | Case |
  |---|---|---|---|
  | `in0` | factory:563 (`in0_reader_args.push_back(in0_tensor)`) | in0 reader RTA 3 → `{.noc_x, .noc_y, .addr = input_shard_l1_addr + read_offset}` | **2** |
  | `in1` | factory:569 | in1 kernel RTA 1 → `{.bank_id, .addr = in1_tensor_addr + …}` | **2** |
  | `bias` (optional) | factory:571 | in1 kernel RTA 2 → `{.bank_id, .addr = in3_tensor_addr}` | **2** |
  | `output` | factory:579 | in1 kernel RTA 7 → `{.noc_x, .noc_y, .addr = output_shard_l1_addr + out_batch_offset}` | **2** |

  All four are in **data-movement** kernels, so the sanctioned `TensorAccessor::get_bank_base_address`
  bridge is available and **the port is not blocked**. (A Case 2 binding in a *compute* kernel would
  have blocked it — there is no bridge there. None of these is.) The raw arithmetic stays unchanged;
  only the address delivery moves onto the typed channel.

- **CB endpoints — seven CBs, four dispositions.** Census run per node across all three kernels.

  | CB | Backing / cores | Touchers | Verdict | Disposition |
  |---|---|---|---|---|
  | `c_0` in0 | regular, rect grid | in0 reader (locked producer), compute (locked consumer) | plain 1:1 | no action |
  | `c_1` in1 | regular, rect grid | in1 kernel (locked producer), compute (locked consumer) | plain 1:1 | no action |
  | `c_3` bias | regular, rect grid | in1 kernel (locked producer), compute (locked consumer) | plain 1:1 | no action — bound only when bias is present, so the binding is **conditional** |
  | `c_4` out | regular, workers | compute (locked producer), in1 kernel (locked consumer) | plain 1:1 | no action |
  | `c_5` interm0 | regular, workers | **compute only** — packs into it and reads it back for accumulate / bias-add | **1 toucher** | **compute self-loop** — bind compute PRODUCER **and** CONSUMER |
  | `c_2` in2 | **borrowed** (`cb_desc.tensor = &in0_tensor`), input storage cores | **none** | **dead CB (0 endpoints)** | **drop** — see below |
  | `c_6` out reshard | **borrowed** (`cb_desc.tensor = &out_tensor`), output storage cores | **none** | **dead CB (0 endpoints)** | **drop** — see below |

  **The two dead CBs, and why I am reporting them as dead.** The recipe rightly says a `(0,0)`
  result is more likely a gap in the auditor's analysis than a real dead CB, so I ran the extra
  checks it asks for:
  - Neither `tt::CBIndex::c_2` nor `c_6` appears anywhere in the factory except its own
    `CBFormatDescriptor` construction (factory:253 and factory:325 respectively) — one hit each.
  - No **named CTA** carries either index. The factory's three `named_compile_time_args` lists
    (factory:455, 470, 487) enumerate `cb_in0`/`c_0`, `cb_in1`/`c_1`, `cb_bias`/`c_3`,
    `cb_out`/`c_4`, `cb_intermed0`/`c_5`, `cb_in0_intermediate`/`c_8`,
    `cb_in1_intermediate`/`c_9`, `cb_in0_transposed`/`c_10` — and neither `c_2` nor `c_6`.
  - No bound kernel body references them, directly or through a helper.
  - There is one code path (plus the bias conditional), so the verdict does not flip per config.

  What they *are*: borrowed views onto the in0 and output tensors' L1 shards, declared on the
  storage cores. The kernels reach that same memory by explicit NOC address instead — the host
  passes the tensor objects as runtime args (factory:563, 579) and the kernels address
  `{.noc_x, .noc_y, .addr}` directly. So the CBs are anchors nothing consumes. Because the storage
  cores and the worker cores are asserted to be the same ordered set (factory:104-113), these are
  even same-core accesses.

  **A dead CB cannot be carried into Metal 2.0** — a DFB with no producer and no consumer binding is
  rejected by the spec validator — so the drop is the only expressible outcome, and it is
  zero-functional-change (a CB nothing touches has no behavior). **But see the question below:** the
  borrowed-CB-over-a-resident-tensor idiom usually exists for a reason, and I could not find this
  one's. The safety net runs one way only — the validator catches a dead DFB loudly, while nothing
  catches a wrongly-dropped live one — so an owner should confirm intent before the drop lands.

- **Two *different* aliasing patterns, on the same CB pair — do not conflate them.** The recipe
  flags this distinction as correctness-critical, and this factory has one of each:
  - **Aliased DFBs (host side).** When `interm0_data_format == output_data_format` (factory:301-317),
    a **single `CBDescriptor` carries two `format_descriptors`** — `c_4` and `c_5` — i.e. two
    logically distinct buffers sharing one L1 region. Port as **two `DataflowBufferSpec`s with
    mutual `advanced_options.alias_with`**. In the other branch (factory:277-300) they are two
    separate descriptors and no aliasing applies — so this is **config-dependent**.
  - **Same-FIFO aliasing (kernel side).** The compute kernel does
    `constexpr uint32_t mm_out_dfb_id = mm_partials_dfb_id;` (line 228) — **one** CB under two
    names, sharing one set of FIFO pointers. Port as **one binding plus a `constexpr` handle alias**;
    do **not** add a second `DFBBinding` and do **not** model it with `alias_with`.

- **Dead named CTAs — `cb_in0_intermediate` (`c_8`) and `cb_in1_intermediate` (`c_9`).** The factory
  passes both to the compute kernel (factory:493-494), but the compute kernel **never reads
  either** — they appear nowhere in its source, under any `#ifdef` — and **no `CBDescriptor`
  allocates them**. Dead plumbing on both ends. The port drops them.

- **`cb_in0_transposed` (`c_10`) is a named CTA for a CB that does not exist, and it cannot simply be
  dropped.** The compute kernel selects its in0 handle with a **parse-time ternary** at line 200:
  `in0_transpose_tile ? get_named_compile_time_arg_val("cb_in0_transposed") :
  get_named_compile_time_arg_val("cb_in0")`. This factory always passes `in0_transpose_tile = 0`
  (positional CTA index 17, hardcoded `0u` in the compute arg list at factory:415-433), so the
  transpose branch is never taken — but **both operands of the ternary are name-looked-up
  regardless**, which is exactly why the factory must supply the CTA today even though no CB backs
  it. In Metal 2.0 a named CB index becomes a `DFBBinding`, and there is no DFB to bind here, so the
  port must use the **conditional-binding pattern**: gate the alias with `#ifdef` from
  `KernelSpec::compiler_options.defines` so the unused branch never enters name lookup.

- **`opt_level` — absent, and this is the failure mode with no diff to read.** `grep -n opt_level`
  on the factory returns **nothing**. An unset `KernelDescriptor::opt_level` still resolves to the
  legacy per-kernel-type default — **`O3` for a `ComputeConfigDescriptor`**, `O2` for DM — while
  Metal 2.0's `CompilerOptions` defaults to `O2` for both. The port must set
  `compiler_options.opt_level = KernelBuildOptLevel::O3` explicitly on the compute `KernelSpec`.
  The two DM kernels need nothing.

- **Hardware config — Style A, no dropped field.** The factory resolves a TTNN `ComputeKernelConfig`
  via `get_compute_kernel_config_args` (factory:673-674), so translate with
  `to_compute_hardware_config(device->arch(), config)`. All four helper-covered knobs are set on the
  compute descriptor (factory:513-516). `packer_l1_acc` is genuinely *consumed* — it derives
  `packer_l1_acc_en` (factory:161) and hence `interm0_data_format` (factory:163-165) — but it has no
  Metal 2.0 counterpart, so no action. `unpack_modes` needs an entry **only if** `interm0` resolves
  to `Float32` with `enable_32_bit_dest` on; that happens when `fp32_dest_acc_en` is set, so the
  porter must check the resolved config rather than the tensor dtypes. The DM kernels use explicit
  custom configs (`RISCV_1` for in0, `RISCV_0` + `in1_noc` for in1, factory:459-476), so replicate
  them field-for-field rather than reaching for a reader/writer helper.

- **Device-operation-class edits the port forces** — two of the three sanctioned exceptions apply:
  1. **Remove the pybound factory entry point.** `matmul_nanobind.cpp:1325-1338` is an
     `nb::class_<ttnn::prim::MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory>` block whose
     only member is `create_descriptor`. That method vanishes at port time, so the block must be
     deleted. User-visible API surface change — record under Handoff points.
  2. **Drop the pybind-hook-only factory parameter.** `create_descriptor` takes a fourth argument,
     `const std::optional<CoreRangeSet>& core_range_set`, which the factory body **ignores** —
     spelled `/*core_range_set*/` at `…batched_hs_dram_sharded_program_factory.cpp:603`. It exists
     only for that hook. Drop it; there is no production default to inline, as nothing reads it.

  Exception 3 does not apply — the op has a proper `program_factory_t` variant.

- **The op's cache key is not an edit.** No framework-visible custom hash exists; the default
  reflection hash is in use. The renamed helper and its pybind alias stay exactly as they are.

---

## Heads-ups  *(mirrors the brief)*

- **Shared kernel — the compute kernel is bound by SIX factories in this directory.** This is the
  intra-op shared-kernel case and it is the largest single constraint on the port.

  | Kernel | Binders | Copies | `_metal2` fork beside it? | Rung |
  |---|---|---|---|---|
  | `dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp` | **1** (this factory) | 1 | no | convert in place |
  | `dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp` | **1** (this factory) | 1 | no | convert in place |
  | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | **6** | 1 | **no** | **rung 2 — create the fork** |

  The six binders of the compute kernel are this factory plus
  `matmul_multicore_reuse_optimized_program_factory.cpp`,
  `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`,
  `matmul_multicore_reuse_mcast_2d_program_factory.cpp`,
  `matmul_multicore_reuse_mcast_1d_program_factory.cpp`, and
  `sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp` — all binding the same
  path, so all are genuine consumers. Converting it in place would break the five that are not
  porting in this change. The rung-1 check was run **locationally**: `ls` of
  `device/kernels/compute/` shows no `bmm_large_block_zm_fused_bias_activation_metal2.cpp`, so this
  port is the first to reach this kernel and **creates the fork**. The remaining five are the
  sunset / coordination list — **not** authorization to convert in place.

  The fork's `#include` closure pulls in `bmm_fused_activation.hpp`, an in-op sibling in the same
  directory. It takes **no** CB-id parameters, so it needs no conversion and no fork of its own —
  the forked kernel keeps including it, resolving to the same file.

- **Name the fork's bindings for the kernel, not for this factory.** Whatever names this first fork
  uses become the interface the other five inherit when they port. Take them from the kernel's own
  role vocabulary, not from this factory's locals.

- **RTA varargs: none.** Both DM kernels read their arguments as distinct fields at constant indices
  in a block at the top (in0 reader slots 0-3; in1 kernel slots 0-7, with slot 2 gated by
  `FUSE_BIAS`). No loop-indexed read, no data-selected index, no sentinel scan. All become named
  RTAs.

- **The compute kernel reads its CB indices as *named* CTAs already** (`get_named_compile_time_arg_val`),
  which become `DFBBinding`s rather than named args. Its dimension CTAs are positional
  (`get_compile_time_arg_val(0..17)`) and become named.

- **`interm0` is a compute self-loop, not sync-free.** The compute kernel runs genuine FIFO
  machinery against `c_5` — it packs into it and reads it back for accumulation and the bias add.
  Bind compute as both PRODUCER and CONSUMER. Do not mistake this for the sync-free case; a
  self-loop is a statement about endpoints, not about synchronisation.

- **Storage cores and worker cores are the same cores.** The factory asserts the input storage,
  output storage, and DRAM-bank worker orderings are element-wise identical (factory:104-113). Any
  reasoning about "remote" storage-core access should account for that — the NOC reads and writes
  are same-core.

---

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean**. No function-call escapes and no
  file-path escapes. Every include across the three kernels is either `api/*` /
  `hostdevcommon/*` / `internal/*` (LLK, HAL and firmware headers — donor class 1, no concern) or
  the in-op sibling `bmm_fused_activation.hpp`. No donor takes a `Semaphore`, `TensorAccessor`,
  `TensorAccessorArgs<N>`, CB id, or `CircularBuffer&`, so the recipe's boundary assumption is not
  violated. **Borrowed kernel files: none** — all three kernels are owned and instantiated by matmul.

  (`bmm_fused_activation.hpp` is itself included by `fused_swiglu.cpp` under
  `experimental/deepseek_prefill/`, i.e. it is shared beyond matmul — but it exposes no CB-id
  surface, so the port neither converts nor forks it. Recorded for completeness.)

- **Relaxation candidates:** none. There is no custom hash to mine, and the sheet's
  `TensorParameter relaxation` is `none`.

- **TTNN factory analysis:** op-owned tensors — none. MeshWorkload need — none (`Execution Model =
  SPMD`, `Concept = descriptor`). Pybind `create_descriptor` — `matmul_nanobind.cpp:1325-1338`.
  Other risky pybind — the device-op class binding at `matmul_nanobind.cpp:1222-1237`, which exposes
  `create_output_tensors`, `compute_output_specs` and `compute_program_hash` (aliased to the renamed
  helper); it survives the port untouched. Custom hash — none framework-visible.
  `get_dynamic_runtime_args` — absent. `override_runtime_arguments` — absent. Target concept —
  `ProgramSpecFactoryConcept`.

---

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **Two named CTAs are passed to a kernel that never reads them.** `cb_in0_intermediate` (`c_8`) and
  `cb_in1_intermediate` (`c_9`) are handed to the compute kernel at factory:493-494. The compute
  kernel references neither, and no `CBDescriptor` allocates either index. Harmless today (a CTA
  nothing reads), and the port drops them — flagged so an owner can confirm they are not a
  half-removed feature rather than leftovers.

- **A provably-unreachable branch in the shared compute kernel.** At line 112 the kernel tests
  `if (mm_partials_reload_dfb_id != mm_partials_dfb_id)`, but line 212 defines the former as equal to
  the latter, so the branch is dead in every build. The dead body contains the kernel's only
  `get_local_cb_interface(...)` call (line 116). Not this port's to remove — the file is shared by
  six factories — but worth an owner's eye, since it is also the one line that will read as a
  Device 2.0 concern to a future auditor.

- **Two borrowed CBs with no consumer** (`c_2`, `c_6`) — the substance is in the Port-work section
  and the question below; noting here that if these turn out to be vestigial across the DRAM-sharded
  factories generally, the same pattern is worth checking in the sibling
  `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`, which uses a similar reshard-CB shape.
  That factory was **not** audited and this is a hypothesis, not a finding.

---

## Questions for the user

1. **The two dead borrowed CBs — please confirm intent before the port drops them.** `c_2`
   (`cb_desc.tensor = &in0_tensor`, factory:247-259) and `c_6` (`cb_desc.tensor = &out_tensor`,
   factory:319-331) have **zero endpoints**: no kernel binds them, no named CTA carries their index,
   and the kernels reach the same memory by explicit NOC address from a runtime arg instead. Metal
   2.0 cannot express a bindingless DFB, so the port must drop them. My census is unambiguous, but
   the borrowed-CB-over-a-resident-tensor idiom normally exists for a reason and I could not
   identify this one's. If they are load-bearing for something the census cannot see — an allocator
   or lifetime effect, or a consumer added later — the port needs to know before it drops them.

2. **Scope.** Six factories in this directory remain unaudited, plus the sparse device-op's factory.
   Should those follow, and in what order? Note that whichever of the five other binders of
   `bmm_large_block_zm_fused_bias_activation.cpp` ports next will inherit the `_metal2` fork this
   port creates, so fork naming is worth agreeing before the first one lands.

## Recipe notes

- **The `(0,0)` "distrust a dead CB" guidance did its job, and I would keep it exactly as written.**
  Two dead CBs in one factory is precisely the result the recipe says to disbelieve, and working
  through its checklist — index references, named CTAs, indirect paths, per-config confirmation — is
  what turned a suspicion into a defensible finding plus a scoped question. The one-way safety-net
  framing (the validator catches a kept dead DFB, nothing catches a dropped live one) is what
  decided the report shape.

- **The two-aliasing-patterns warning earned its place too.** This factory has a host-side
  `alias_with` case and a kernel-side same-FIFO case on the *same* CB pair, a few hundred lines
  apart, and the host-side one is config-dependent. Without the recipe's explicit table
  distinguishing them, modelling the kernel-side alias with `alias_with` would have been the natural
  mistake and would have produced two independent FIFOs at one address.

- **Friction — the Case 2 bridge is specified for DRAM tensors; two of these bindings are L1 shards
  reached by explicit NOC endpoint.** The recipe's Case 2 remedy is
  `TensorAccessor::get_bank_base_address()`, described against a kernel that used a raw base address.
  Here `in0` and `output` are L1-sharded tensors whose base the kernel feeds into
  `{.noc_x, .noc_y, .addr}` rather than a bank-relative access. I believe the bridge still applies —
  and since storage and worker cores coincide, the address is local — but the recipe does not cover
  the L1-shard-via-endpoint shape explicitly, and a porter could reasonably wonder whether
  `get_bank_base_address()` returns what that call site needs. Worth a sentence in the Case 2
  section if this shape recurs.
