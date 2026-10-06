# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/topk_router_gpt`

> **Re-audit, 2026-10-06.** This supersedes the 2026-10-01 audit. Two op-side commits landed since then:
> - `8e04962ad72` (#58858) swapped both `noc_async_atomic_barrier()` holdovers in `dm1.cpp` for `noc.async_atomic_barrier()`. **This clears the Device 2.0 gate.**
> - `16327cdfae5` (#58472) added a Blackhole P150 configuration (8 DRAM-aligned cores, 1 sender per group) and logical batches 1–32. This adds a host-derived config axis (`cores_per_group` = 3 on WH / 2 on BH), so the census below is recorded per config.
>
> The recipe docs have not changed since the last audit (same provenance hash).

- **`TopkRouterGptDeviceOperation`** (`device/topk_router_gpt_device_operation.hpp:16`)
  - *Direct descriptor.* `create_descriptor` is a static member of the device-op itself (declared `device/topk_router_gpt_device_operation.hpp:26`, defined `device/topk_router_gpt_program_factory.cpp:44`). There is no `program_factory_t` and no factory struct. This shape comes from the PD migration in `f5093e705ae` (#57409, 2026-09-25), which deleted the old `TopkRouterGptProgramFactory` struct and its header `device/topk_router_gpt_program_factory.hpp`.
  - Kernels, all owned by the op and all referenced by the descriptor. Each one runs on `all_cores`, i.e. every DRAM-bank-aligned worker core (`program_factory.cpp:53-56, 234, 245, 256`):
    - `device/kernels/dm0.cpp` — RISCV_1 / NOC_0. Reads weight, input and bias.
    - `device/kernels/dm1.cpp` — RISCV_0 / NOC_1. Moves data between cores, builds the helper tiles and writes the outputs.
    - `device/kernels/compute.cpp` — matmul, partial-sum and bias add, top-k merge and softmax.
  - One program. Each core gets one of three roles from its RTAs: **sender**, **worker**, or **collector** (the collector is also a worker, at ring position `num_senders`). Code paths branch on the `is_sender` / `is_worker` / `is_collector` RTAs.
  - **Config axis (new since #58472):** `cores_per_group = num_cores >= 12 ? 3 : 2` (`program_factory.cpp:63`), with `num_senders = cores_per_group - 1`. Both are host-derived from the device and are not user attributes.

    | Config | Device | Cores used | Senders/group | Senders | Workers (incl. collector) | Collector ring pos |
    |---|---|---|---|---|---|---|
    | **WH** | Wormhole, 12 DRAM-aligned | 12 | 2 | 8 | 4 | 2 |
    | **BH** | Blackhole P150, 8 DRAM-aligned | 8 | 1 | 4 | 4 | 1 |

    Every other axis is pinned by `validate_on_program_cache_miss`: num_experts = 128, padded batch = 32 (logical B in 1..32), and TILE/BF16 inputs. Devices with fewer than 8 DRAM-aligned cores are rejected by the `TT_FATAL` at `program_factory.cpp:66-70`.
  - The op directory has no unreferenced kernel files.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(taken from the `/localdev/edwinlee/Port_Recipe` checkout; the op's checkout carries no recipe tree)*

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/topk_router_gpt` |
| **Overall** | **GREEN (user waiver).** Every code-side gate is clean. The only RED, the stale readiness-sheet row, was waived by the user on 2026-10-06 (see Result). |
| **DOps / Factories** | `TopkRouterGptDeviceOperation` → direct `create_descriptor` (no factory struct) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** `8e04962ad72` cleared both holdovers (`dm1.cpp:164`, `dm1.cpp:260` now call `noc.async_atomic_barrier()`). No free-function NoC, CB, addr-gen or semaphore idioms remain in any kernel. |
| *Prereqs* — Cross-op escapes | Ok — every `#include` is in `tt_metal/*`. No borrowed kernel files. |
| *Feature Support* — overall | GREEN — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok — not used |
| *TTNN Readiness* — `Is able to port?` (the gate) | Sheet: `yes (with PD step)`. **Gate: RED — spreadsheet broken, waived by the user 2026-10-06.** `Concept` conflicts with the code, and the factory row is a phantom. The sheet refresh is still owed to the readiness-sheet owner. |
| *TTNN Readiness* — Concept (current) | Sheet: `legacy device-op`. Code: **`descriptor`** (direct `create_descriptor`, `device_operation.hpp:26`). |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No (sheet `no`, code agrees; no backdoor either) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (sheet `no`, code agrees) |
| *TTNN Readiness* — `override_runtime_arguments` | No. Sheet `n/a`, a leftover from the legacy row. The code has none (see the comment at `device_operation.hpp:22-25`). |
| *TTNN Readiness* — Pybind `create_descriptor` | No (sheet `no`; `topk_router_gpt_nanobind.cpp:29-58` binds only the free function) |
| *TTNN Readiness* — Op-owned tensors | No (sheet `no`; `descriptor` concept) |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (sheet `Porting Target` agrees) |
| *Port work* — Offset base pointer | none — all five address RTAs are clean bases |
| *Port work* — Tensor bindings (per binding) | `input`, `weight`, `bias`, `indices_rm`, `weights_rm` — all **Case 1** (`Buffer*`-binding form → `TensorAccessor`) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **drop (Class 2)** at `dm1.cpp:343` and `dm1.cpp:344`. Do **not** set `dynamic_tensor_shape`. |
| *Port work* — CB endpoints | Same in both configs. 1P+1C: c_0, c_1, c_2, c_3, c_4, c_6, c_8, c_9, c_12, c_15, c_16 · self-loop: c_5 (DM), c_19 (DM), c_10, c_11, c_13, c_14 (compute) · no dead CBs · no multi-binding |

**CB endpoints** are dispositions, not gates. Nothing in them blocks the port. The per-CB census is in Gate detail and Port-work below.

## Result

**GREEN by user waiver → brief issued** (`METAL2_PORT_BRIEF.md`). As audited, this was RED on the TTNN factory concept gate (spreadsheet-broken). On 2026-10-06 the user waived that gate ("If it's the same stale sheet problem, then proceed"): the blocker is only the sheet being out of date, not anything in the op. The sheet refresh is still owed to the readiness-sheet owner as housekeeping; it is no longer a port blocker. The original finding is kept below.

**Original finding (pre-waiver): RED → blocked on one gate: the TTNN factory concept row is spreadsheet-broken.** Routed to the **readiness-sheet owner (Diego)**. This gate needs no op engineering.

- The live sheet (fetched this session, 2026-10-06) still describes the op as it was before #57409. It shows `Concept` = `legacy device-op` and a `TopkRouterGptProgramFactory` factory whose struct and `.hpp` no longer exist. The code is a `descriptor` op with a direct `create_descriptor`.
- A primary-column conflict plus a phantom factory row is a spreadsheet-broken trigger. Every other sheet column that can be checked in code agrees with the code.
- **Path forward:** either the sheet row is reconciled, or the user explicitly waives the gate on the code cross-check. Waivers were granted for other ops from the same #57409 batch (`bcast_to`, `deepseek_moe_post_combine_tilize`, `test/hang_device`, `transformer/concatenate_heads`). A waiver does not carry over, so this audit initially issued **no brief**. *(Superseded: waived 2026-10-06, brief issued.)*

**Cleared since the last audit:** the Device 2.0 gate (`8e04962ad72`). **Every other gate-bearing subject is clear:** feature compatibility, offset base pointers, and the TensorAccessor 3rd argument (Class 2). Once the sheet gate clears, nothing else in this audit blocks the port.

**No portable subset.** The op has one program, and the sheet gate covers all of it. The audit was **RED at op level; no portable subset** before the waiver. The new WH/BH axis does not create a subset, because the sheet gate is not config-scoped.

**Why the informational subjects ran anyway.** The remaining blocker clears **outside** the op (a sheet edit or a waiver). Re-audit would read the same code, so the recipe's exception applies and the full census is recorded below. It is ready to become the brief once the gate clears.

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED — spreadsheet broken → readiness-sheet owner to reconcile.** Sheet row (live fetch, 2026-10-06; unchanged from 2026-10-01):
  - `Op` = `experimental/topk_router_gpt`, `Device operation` = `TopkRouterGptDeviceOperation`, `Factory (variant)` = `TopkRouterGptProgramFactory`
  - `Concept` = `legacy device-op`, `Op Classification` = `Legacy Op`, `Execution Model` = `SPMD`, `Porting Target` = `ProgramSpecFactoryConcept`
  - `Custom hash` = `no`, `Backdoor custom hash` = `no`, `Runtime-args update (get_dynamic_runtime_args)` = `no`, `Override runtime args method? (PD only)` = `n/a`, `Pybind descriptor` = `no`, `Smuggled pointer` = `no`
  - `Known op issues` = *(blank)*, `TensorParameter relaxation` = `none`, `Op-owned tensors?` = `no`, `Diego validation` = `yes`
  - `Is able to port?` = `yes (with PD step)`
  - `Factory definition path` = `…/device/topk_router_gpt_program_factory.hpp` (**this file no longer exists**), `Declared in` = `…/device/topk_router_gpt_device_operation.hpp`

  Cross-check against the code:

  | Column | Sheet | Code | Match? |
  |---|---|---|---|
  | `Concept` | `legacy device-op` | `descriptor`: `static ProgramDescriptor create_descriptor(...)` on the device-op (`device_operation.hpp:26`). No `create()` and no `program_factory_t`. | **✗ conflict** |
  | Factory set | 1 row, `TopkRouterGptProgramFactory` | That struct was deleted in `f5093e705ae`. The one remaining entry point is the direct descriptor. | **✗ phantom row** (and no row for the direct descriptor) |
  | `Custom hash` / backdoor | `no` / `no` | no `compute_program_hash`, `attribute_values` or `to_hash` anywhere in the op | ✓ |
  | `get_dynamic_runtime_args` | `no` | absent | ✓ |
  | `Override runtime args method?` | `n/a` | absent. As a `descriptor` op it should now read `no`. | ✓ in substance (the `n/a` is a legacy-row leftover) |
  | `Pybind descriptor` | `no` | `topk_router_gpt_nanobind.cpp:29-58` binds `ttnn.experimental.topk_router_gpt` only | ✓ |
  | `Smuggled pointer` | `no` | the five tensor addresses are pushed as `Buffer*` (`program_factory.cpp:344-346, 359-360`), so they are framework-patched `BufferBinding`s | ✓ |
  | `Op-owned tensors?` | `no` | the `descriptor` concept can't carry them | ✓ |

  No cross-column invariant is violated. The `yes (with PD step)` verdict reads as "portable once the PD migration lands", and the code shows that it has landed. The gate is blocked only by the stale `Concept` and factory rows, and a sheet edit fixes both. #58472 did not change any sheet-visible property: no hash, no override, no pybind, no op-owned tensors.
- **Device 2.0 (every kernel used): GREEN.** The two isolated holdovers from the last audit are gone. `dm1.cpp:164` and `dm1.cpp:260` now call `noc.async_atomic_barrier()` on the in-scope `Noc noc` (`dm1.cpp:68`), as of `8e04962ad72`. A grep of all three kernels finds:
  - no free `noc_*()` calls
  - no CB-index free functions (`get_read_ptr(cb)` / `get_write_ptr(cb)` / `cb_reserve_back` …)
  - no `InterleavedAddrGen*` / `ShardedAddrGen`
  - no raw semaphore addresses (`get_semaphore`)

  Data movement uses `Noc`, `CircularBuffer` wrappers, `Semaphore<>`, `TensorAccessor`, `CoreLocalMem` and `UnicastEndpoint` throughout. `compute.cpp` uses the normal compute LLK idiom (`uint32_t` CB ids into `matmul_block`, `pack_tile`, `reduce_tile`, …), which is not a DM holdover. No sanctioned free functions are in play.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | — | — | none | — |

- **Feature compatibility: GREEN — no gate fired.**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` / `global_circular_buffer` / `remote_index` anywhere. dm1's cross-core writes into another core's CB are a plain NoC unicast to an L1 address, not a remote CB. |
  | CBDescriptor `address_offset` (non-zero) | N/A | every `CBDescriptor` (`program_factory.cpp:30-38, 192-200`) leaves `address_offset` at its default of 0, and no CB is buffer-backed |
  | GlobalSemaphore | N/A | only plain `SemaphoreDescriptor`s (`program_factory.cpp:271-274`) |
- **CB endpoints (GATE-free): every CB resolves to 1P+1C or a self-loop in both configs. No multi-binding, no dead CB.** The WH → BH change touches only `c_2`'s depth (`num_senders` = 2 → 1, `program_factory.cpp:158`) and the role counts. No CB's toucher set changes. Roles: S = sender, W = non-collector worker, C = collector.

  | CB | Allocated on | Touchers per node (role) | Disposition (WH and BH) |
  |---|---|---|---|
  | c_0 `cb_weight` | all | dm0 FIFO-P (`dm0.cpp:75,92`), compute FIFO-C (`compute.cpp:116,132`) — S/W/C | 1P+1C (dm0 P, compute C) |
  | c_1 `cb_input` | all | dm0 FIFO-P (`dm0.cpp:74,91`), compute FIFO-C (`compute.cpp:115,131`) — S/W/C | 1P+1C (dm0 P, compute C) |
  | c_2 `cb_partial_recv` | all | W/C: dm1 FIFO-P (`dm1.cpp:219,226`), compute FIFO-C (`compute.cpp:153,160`). S: dm1 peek only (`dm1.cpp:140`; uses its own local base as the *remote* write address). | 1P+1C (dm1 P, compute C). On S nodes only dm1 touches it, and the dm1 P binding covers that. |
  | c_3 `cb_local_out` | all | S: compute FIFO-P (`compute.cpp:142,146`), dm1 FIFO-C (`dm1.cpp:146,166`). W/C: compute names it only in `compute_kernel_hw_startup` (`compute.cpp:104`, pack-format init); no FIFO use. | 1P+1C (compute P, dm1 C) |
  | c_4 `cb_bias` | workers | dm0 FIFO-P (`dm0.cpp:101,105`), compute FIFO-C (`compute.cpp:163,165`) | 1P+1C (dm0 P, compute C) |
  | c_5 `cb_index` | workers | **dm1 only**: FIFO-P (`dm1.cpp:176,182`) **and** FIFO-C (`get_read_ptr` `:234/:274`, `pop_front` `:263/:295`). Compute declares `cb_index` (`compute.cpp:84`) but never uses it. | **self-loop (DM)**: dm1 PRODUCER + CONSUMER. Do **not** bind compute. |
  | c_6 `cb_topk_val` | workers | compute FIFO-P (`compute.cpp:169,173`), dm1 FIFO-C (`dm1.cpp:229, 262/294`) | 1P+1C (compute P, dm1 C) |
  | c_8 `cb_gathered_val` | workers | C: dm1 FIFO-P with a raw fill of slot 0 (`dm1.cpp:276,279,303`), compute FIFO-C (`compute.cpp:186,233`). W: dm1 peek only (`dm1.cpp:237`; its local base is the *remote* destination on C). | 1P+1C (dm1 P, compute C) |
  | c_9 `cb_gathered_ind` | workers | same shape as c_8 (`dm1.cpp:246,277,280,304`; `compute.cpp:187,234`) | 1P+1C (dm1 P, compute C) |
  | c_10 `cb_intermed_val` | collector | compute only, P and C (`compute.cpp:224/230`, `237/254`, `257/264`, `325/344`) | self-loop (compute) |
  | c_11 `cb_intermed_ind` | collector | compute only (`compute.cpp:225/231`, `238/255`) | self-loop (compute) |
  | c_12 `cb_softmax_mask` | collector | dm1 FIFO-P (`dm1.cpp:187,203`), compute FIFO-C (`compute.cpp:239,346`) | 1P+1C (dm1 P, compute C) |
  | c_13 `cb_softmax_tmp` | collector | compute only (`compute.cpp:256/263`, `271/295`, `296/300`, `305`, `323/342`) | self-loop (compute) |
  | c_14 `cb_reduce_scalar` | collector | compute only (`compute.cpp:272/283`, `286/302`, `306/320`, `324/343`) | self-loop (compute) |
  | c_15 `cb_bcast_scaler` | collector | dm1 FIFO-P (`dm1.cpp:206,215`), compute FIFO-C (`compute.cpp:240,347`) | 1P+1C (dm1 P, compute C) |
  | c_16 `cb_final_out` | collector | compute FIFO-P (`compute.cpp:326,341`), dm1 FIFO-C (`dm1.cpp:307,355`) | 1P+1C (compute P, dm1 C) |
  | c_19 `cb_dispatch` | collector | dm1 only: reserve/push (`dm1.cpp:314,339`), wait/pop (`dm1.cpp:352,353`) | self-loop (DM) |

  *Hidden-second-writer hunt (face a):* none on any node, in either config. Two cross-core raw writes exist:
  - senders write into their worker's c_2 (`dm1.cpp:151-158`)
  - non-collector workers write into the collector's c_8/c_9 (`dm1.cpp:237-253`)

  Both are **remote** NoC writes, coordinated by `sem_partial_ready` / `sem_topk_ready`. On the receiving node they land in a slot that the local dm1 has reserved or will reserve, and that dm1 is the CB's only local producer. No co-resident second kernel writes, so no CB needs the multi-binding flag.

  *Face (b)/(c):* there is no dual-instance kernel and no multi-reader CB.
- **Offset base pointers: GREEN.** All five address RTAs are bare `Buffer*` pushes with no host arithmetic: `[2]` weight, `[3]` input, `[4]` bias, `[17]` indices_rm, `[18]` weights_rm (`program_factory.cpp:313-317, 344-346, 359-360`). Each one feeds a `TensorAccessor` base unmodified (`dm0.cpp:61,62,99`, `dm1.cpp:343,344`). All per-tile offsets are page ids. The op is not in the 2026-07-19 triage tables, and the scan agrees.
- **TensorAccessor 3rd argument: GREEN — 2 sites, both Class 2 (redundant → drop).**
  - The sites:
    - `dm1.cpp:343` `TensorAccessor(indices_rm_accessor_args, indices_rm_addr, aligned_page_size)`
    - `dm1.cpp:344` `TensorAccessor(weights_rm_accessor_args, weights_rm_addr, aligned_page_size)`
  - Q1, sharding: **interleaved.** Both outputs are `INTERLEAVED` / L1 row-major (`device_operation.cpp:107`).
  - Q2, magnitude: **correct.** The value is RTA `[19]` = `indices_rm.buffer()->aligned_page_size()` (`program_factory.cpp:296-297, 361`), which is exactly `aligned_page_size` for `indices_rm`. `weights_rm` reuses the indices value. That is safe because both outputs have padded shape `[32, k_padded]` with a 2-byte dtype (UINT16 / BF16, `device_operation.cpp:111-121`) in the same buffer type, so their page sizes match: `k_padded × 2` bytes, a multiple of 16.
  - **Still constant per compiled program after #58472.** Logical B now varies (1..32), but the padded shape stays `[32, k_padded]`, so the RM page (one padded row) does not depend on B. `k` is a hashed attribute. This still matches the 2026-07-06 triage row (`topk_router_gpt | 2 — Redundant`, ‡ note). The note's justification "pinned by `B==32`" now reads "pinned by padded batch = 32", which leaves the class unchanged. Drop the arg, and **do not** set `dynamic_tensor_shape`.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding). Every address is currently delivered as a `Buffer*` RTA, which the framework patches on a cache hit, so none of this is a correctness hazard. All five are routine Case 1:
  - `input` — **Case 1.** RTA `[3]` → `dm0.cpp:61` `TensorAccessor(input_accessor_args, input_addr)`; CTA block 0.
  - `weight` — **Case 1.** RTA `[2]` → `dm0.cpp:62`; CTA block 1.
  - `bias` — **Case 1.** RTA `[4]` → `dm0.cpp:99` (worker only); CTA block 2.
  - `indices_rm` (output 0) — **Case 1.** RTA `[17]` → `dm1.cpp:343`; CTA block 3.
  - `weights_rm` (output 1) — **Case 1.** RTA `[18]` → `dm1.cpp:344`; CTA block 4.
  - The `TensorAccessorArgs` order is input, weight, bias, indices_rm, weights_rm (`program_factory.cpp:203-212`). That is **not** the RTA order (weight comes first there). Binding by name removes the mismatch.
  - Only the kernels that use a tensor need it bound. dm1 and compute read the input/weight/bias address RTAs but never use them, so those three tensors need binding only on dm0. The two outputs need binding only on dm1.
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:**
  - Drop the redundant page-size arg at `dm1.cpp:343` and `dm1.cpp:344`. Do not set `dynamic_tensor_shape`.
  - After the drop, RTA `[19]` `aligned_page_size` has no consumer in any kernel. Its host source is `program_factory.cpp:296-297, 361`.
- **CB endpoints** (identical in WH and BH configs):
  - 1P+1C on c_0, c_1, c_2, c_3, c_4, c_6, c_8, c_9, c_12, c_15, c_16, with roles as in the census table.
  - Self-loop on c_5 (dm1), c_19 (dm1), c_10, c_11, c_13, c_14 (compute).
  - Nothing is dead and nothing needs the multi-binding flag.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** none. All three kernels belong to the op, and every include is `tt_metal/*` (`api/dataflow/*`, `api/compute/*`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`).
- **RTA varargs:** none. All three kernels read RTAs `[0..19]` through one fixed `argidx++` run (`dm0.cpp:30-50`, `dm1.cpp:88-108`, `compute.cpp:39-59`). That is ordinary positional plumbing, so name every field. The new `num_senders` loop in compute (`compute.cpp:156-158`) iterates over CB tiles, not args.
- **New config axis: WH vs BH.**
  - `cores_per_group` / `num_senders` are picked from the device's DRAM-aligned core count (`program_factory.cpp:63-64`) and reach the kernels as the named CTA `num_senders` (`program_factory.cpp:220`; read at `dm1.cpp:73`, `compute.cpp:35`).
  - They also size `c_2` (`program_factory.cpp:158`) and move the collector to ring position `num_senders` (`:130`).
  - The port must keep all three of these derived from that one host value, and must test on **both** WH (12 cores, 2 senders/group) and BH P150 (8 cores, 1 sender/group). Validation at `16327cdfae5` covered both.
- **Uniform L1 offset across nodes is load-bearing (`dm1.cpp:135-140`, `:237`, `:246`; `program_factory.cpp:85-87`).** dm1 reads its *own* write pointer for a CB and uses that address as the NoC destination on *another* core:
  - c_2: sender → worker (`dm1.cpp:140,151`).
  - c_8/c_9: non-collector worker → collector (`dm1.cpp:237-238`, `246-247`).

  This works only because c_2 sits at the same L1 offset on every core, and c_8/c_9 sit at the same offset on every worker. Legacy guarantees that by allocation order: CB0–3 on `all_cores` first, then c_4–c_9 on workers, then the collector-only CBs. In Metal 2.0 a DFB's placement is *derived* from its bound kernels (`migration_guide.md`, Troubleshooting: "DFB placement is derived, not specified"). Every kernel here runs on all cores, so a straight one-`WorkUnitSpec` port allocates **every** DFB on all 12 (WH) or 8 (BH) nodes. That keeps the layout uniform, which is what the trick needs. The cost is roughly 13 extra tiles plus the c_19 scratch of L1 on each sender core. That is a footprint change only.

  Do **not** split the kernels into per-role `WorkUnitSpec`s to recover the legacy per-subset placement unless you can show the DFB offsets stay identical across the role sets. A split can break this address trick silently: wrong data, no assert. See Questions.
- **Semaphore IDs travel through RTAs.**
  - `sem_partial_ready` = id 0 (RTA `[5]`) and `sem_topk_ready` = id 1 (RTA `[16]`), set at `program_factory.cpp:269-274, 347, 358`.
  - dm1 consumes them as `Semaphore<> x(rta_value)` (`dm1.cpp:162,222,258,299`).
  - Port both as `SemaphoreSpec`s bound to dm1 (`sem::name`). Each is used both remotely (`.up(noc, x, y, 1)`) and locally (`.wait` / `.set`), and both are allocated on all cores. Neither dm0 nor compute uses them.
  - The worker's wait count is now `num_senders` (`dm1.cpp:223`), not a literal 2.
- **Physical-coordinate args stay as plain values.** `collector_physical_x/y` are named CTAs (`program_factory.cpp:221-222`) and `worker_phys_x/y` are RTAs `[12]`/`[13]`. These are NoC coordinates, not addresses, so carry them over unchanged.
- **Kernel-side includes.** All three kernels include `api/dataflow/circular_buffer.h` (`dm0.cpp:13`, `dm1.cpp:23`, `compute.cpp:24`) and declare `CircularBuffer` objects. These are the standard CB→DFB swap sites. Compute declares 16 `CircularBuffer` objects but only uses their FIFO methods; the LLK calls take the raw `*_id` constants, which then come from the `dfb::name` tokens.
- **Every named RTA must be set on every node** (`migration_guide.md`). The descriptor emplaces RTAs only for the `required_cores` ring positions (`program_factory.cpp:319`), while the kernels are placed on all `num_cores` DRAM-aligned cores (`:56, :234, :245, :256`). The two sets are the same on WH (12 = 12) and BH P150 (8 = 8). On any device where `num_cores` is not exactly 8 or 12, Metal 2.0 would reject the run args. Keep the placement and RTA node sets identical, and don't invent values for unused cores. See Misc anomalies.

## Team-only

- **Out-of-directory coupling & donor shape: ✓ clean.**
  - Op-level roll-up: no function-call escapes outside `tt_metal/*`, and no file-path kernel borrowing.
  - Summary table:

    | Op kernel | Donor file(s) | Class | Status |
    |---|---|---|---|
    | `dm0.cpp` | `api/dataflow/{dataflow_api,noc,circular_buffer}.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | 1 (`tt_metal/*`) | ✓ |
    | `dm1.cpp` | the dm0 set plus `api/dataflow/{noc_semaphore,endpoints}.h` | 1 | ✓ |
    | `compute.cpp` | `api/compute/*` (topk, matmul, eltwise_binary, transpose, reduce, softmax, bcast, pack, …), `api/dataflow/circular_buffer.h` | 1 | ✓ |
  - Per-call detail: omitted (everything rolls up ✓).
  - Borrowed kernel files: none. There is no `_metal2` fork question.
- **Relaxation candidates:** none. The op has no custom hash to mine.
- **TTNN factory analysis:**
  - `descriptor` concept, direct `create_descriptor` (`device_operation.hpp:26`).
  - No op-owned tensors, no MeshWorkload, no pybound `create_descriptor` (`nanobind.cpp:29-58`).
  - No custom hash or backdoor, no `get_dynamic_runtime_args`, no `override_runtime_arguments`. The header comment at `device_operation.hpp:22-25` explains why the override was dropped: all non-address per-core state derives from the specs and the DRAM bank assignment.
  - The WH/BH choice derives from the device, which is part of the program-cache context. Logical B is in the input spec, so each B compiles its own program, but nothing per-B reaches the kernels.
  - Target: `ProgramSpecFactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Possible un-arg'd cores (scope widened by #58472).** Kernels are placed on every DRAM-aligned core (`program_factory.cpp:53-56`; `all_cores` at `:234, :245, :256`). RTAs are emplaced only for the first `required_cores` ring positions (`:319`), which is 12 or 8. The `TT_FATAL` at `:66-70` checks `>=`, not `==`. So any device whose count is 9–11 or more than 12 would run all three kernels with unset RTAs on the extra cores. This is latent on WH (12) and BH P150 (8).
- **Dead RTAs.**
  - `[0] dram_bank_id` and `[1] vchannel` are read by all three kernels and used by none (`dm0.cpp:31-32`, `dm1.cpp:89-90`, `compute.cpp:40-41`). The host still computes the vchannel conflict-avoidance table for them (`program_factory.cpp:277-292`).
  - Most of the other 20 RTAs are read but unused by at least one kernel. For example, compute uses only `is_sender`, `is_collector` and `num_k_tiles`.
- **Dead named CTAs.** `num_cores` and `cores_per_group` (`program_factory.cpp:217, 219`) are read by no kernel. (`num_senders`, added by #58472, is live.)
- **Unused `CircularBuffer` declarations.**
  - `compute.cpp:84` `cb_index` is never used in compute.
  - dm0 and dm1 also build wrapper objects for CBs that are not allocated on every node they run on (e.g. `dm1.cpp:126-131` on sender cores). This is harmless today, since construction doesn't touch the CB.
- **Shared page-size RTA.** `weights_rm`'s accessor uses `indices_rm`'s `aligned_page_size` (`dm1.cpp:344`, `program_factory.cpp:296-297`). The values match today only because both outputs have a 2-byte dtype and the same padded shape.
- **Stale comment.** The comment at `dm1.cpp:341-342` says the RTA override exists because `AlignedPageSize` "may be stale on program cache hits". The host comment at `program_factory.cpp:294-295` contradicts it, correctly: the page size is fixed by the hashed output spec. The override is a no-op (Class 2, above).
- **Padding rows are computed and written.** With logical B < 32, dm1 still emits all 32 rows (`dm1.cpp:326, 345`) into the padded output buffers. This is intended per the #58472 description ("Both buffers contain 32 physical rows", `nanobind.cpp:48-49`). It is noted only so a reader of the port doesn't mistake it for a bug.

## Questions for the user

1. ~~**Waive the spreadsheet-broken gate?**~~ **Answered 2026-10-06: waived.** You waived it for `bcast_to`, `deepseek_moe_post_combine_tilize`, `test/hang_device` and `transformer/concatenate_heads`, all from the same #57409 batch, on the code cross-check alone. The same evidence holds here: the code is `descriptor`, `Known op issues` is empty, the relaxation is `none`, and there is no `get_dynamic_runtime_args`. With Device 2.0 now clear, **a waiver would make this audit GREEN** and the brief could be written straight from the sections above.
2. **DFB placement for the per-subset CBs.** Legacy allocates c_4–c_9 on the 4 workers and c_10–c_19 on 1 core (`program_factory.cpp:164-200`). A Metal 2.0 port that keeps one kernel per role across all cores puts every DFB on all 12 (WH) or 8 (BH) cores. That adds L1 footprint on the sender cores but changes no behavior. Is that acceptable as zero-functional-change? The alternative, per-role `WorkUnitSpec`s, puts the cross-core uniform-offset trick at risk (see Heads-ups).

## Recipe notes

- **Re-audits are not covered.** The recipe assumes a fresh audit. It says nothing about what a re-audit should carry over, such as the previous gate verdicts, a `superseded` marker, or which subjects must be redone when only one gate was expected to clear. Here an unrelated feature commit (#58472) landed between the two audits and added a config axis. A line like "re-audit runs every subject against current `HEAD`; diff the op's history since the last audit before trusting carried-forward detail" would make that explicit.
- **Host-derived config axes.** "Classify per instantiation" in CB endpoints is written around user-facing configs (sharding, split-reader). This op's axis is picked by the **device** (`num_cores >= 12`), so the same op build instantiates differently on WH and BH. The recipe could say explicitly that device-derived branches count as configs for the census and for the brief's test matrix.
- **Derived DFB placement vs. legacy per-subset CBs.** This is still open from the last audit. The audit has no subject for a legacy CB whose `core_ranges` is narrower than the kernels that construct it.
- **The recipe tree is in a separate checkout.** The provenance hash comes from `/localdev/edwinlee/Port_Recipe`.
