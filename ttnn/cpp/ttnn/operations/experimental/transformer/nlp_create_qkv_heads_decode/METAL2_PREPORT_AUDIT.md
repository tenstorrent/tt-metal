# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/`

Single device-operation directory:

- **`NLPCreateQKVHeadsDecodeDeviceOperation`** (`device/nlp_create_qkv_heads_decode_device_operation.{hpp,cpp}`)
  - `NLPCreateQKVHeadsDecodeInterleavedProgramFactory` (`device/nlp_create_qkv_heads_decode_interleaved_program_factory.cpp`)
    - kernel: `device/kernels/reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` (instantiated twice — RISC0 reader / RISC1 writer)
  - `NLPCreateQKVHeadsDecodeShardedProgramFactory` (`device/nlp_create_qkv_heads_decode_sharded_program_factory.cpp`)
    - kernel: `device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` (instantiated as q_reader/q_writer, and k_reader/k_writer when `!overlap_qk_coregrid`)
  - `NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory` (`device/nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp`)
    - kernel: `device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp` (same RISC pairing)

All three kernels are data-movement (reader-class) kernels; there are no compute kernels. Each kernel `.cpp` is instantiated on two RISCs (a "reader" on NOC0 and a "writer" on NOC1) — the two RISCs each read one sub-tile phase of the same tile and write into the same output CBs.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

**Recipe docs:** `776151aeca8 2026-06-24 docs(metal2): clarify SPSC face-(b) producer-as-consumer, aliased-vs-same-FIFO, no-portable-subset`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/` |
| **Overall** | **RED** |
| **DOps / Factories** | `NLPCreateQKVHeadsDecodeDeviceOperation` → Interleaved · Sharded · ShardedSubcoregrid |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok (one shared-lib donor: `data_movement/common/kernels/common.hpp` `tt_memmove`, uint32-L1-addr shape — clean) |
| *Feature Support* — overall | GREEN (no UNSUPPORTED feature in use) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *Ops readiness* — Sync-free CBs (address-only) | Present, but they exceed SPSC ceiling — see Result/SPSC: the output CBs `c_16/c_17/c_18` are sync-free yet carry **2 raw writers per node** (SPSC violation, not a self-loop case). The `batch_offset` reader CB `c_15` is a single-ended FIFO producer (self-loop candidate) in the subcoregrid factory; in the **sharded** factory it is shared by both RISCs (SPSC). |

**Sync-free CBs** = CBs used purely as an address source (kernel grabs the base pointer and walks the memory, no FIFO ops). The output CBs here are sync-free, but the gating issue is *endpoint multiplicity*, not the sync axis — see SPSC below.

## Result

**RED** at op level — blocked on **DFB endpoint legality (SPSC)**, routed to the **op owner** as a pre-port functional change.

Every program factory instantiates the *same* dataflow kernel `.cpp` on **two RISCs on the same node** (a NOC0 "reader" reading sub-tile phase 1 and a NOC1 "writer" reading phase 2), and **both RISCs write into the same output CBs** `c_16` (Q), `c_17` (K), `c_18` (V) by raw base pointer (`cb_*_out.get_write_ptr()` + offset). That is **two PRODUCER (writer) endpoints per node** on each output CB → an SPSC violation (the *visible* two-writers face, here both writers being raw, sync-free co-fillers rather than FIFO producers). The self-loop workaround **cannot** absorb a second co-resident kernel touching the CB, so this is not port-fixable — the op owner must collapse the two-RISC split write into a single endpoint per output CB before the port (e.g. a Gen2-conditional that has one DM fill the whole tile, demoting the split-RISC L1-bandwidth optimization to a deferred optimization).

**This SPSC shape is unconditional / structural** — the two-RISC split read+write is *always on* in all three factories, not one branch among portable siblings. **There is no portable subset**: the op is RED whole, and the op-owner functional fix is the only path forward.

Additional (non-gating) endpoint findings, see SPSC detail:
- **Sharded factory only** — when `batch_offset` is provided, both the reader and writer RISCs use the *same* batch-offset CB index `c_15` (the writer's compile-time arg for the CB index is **not** overridden, unlike the subcoregrid factory), so `c_15` also takes **2 FIFO producers per node** → a *second* SPSC violation in that factory/config.
- **Sharded factory only** — `c_14` (`batch_offset_cb_index_writer`) is **allocated but its index is threaded to no kernel** → a **dead CB (0,0)** to drop pre-port.

Reassuring framing: this is not a permanent blocker and not a framework gap — all prereqs and feature gates are clean. Once the op owner collapses the per-output-CB second writer (and fixes the sharded batch-offset CB-index plumbing), the op is otherwise structurally ready for the Metal 2.0 port. The path forward is an op-owner functional change, not a wait-for-feature.

## Gate detail

- **ProgramDescriptor:** GREEN. All three factories populate a `tt::tt_metal::ProgramDescriptor` with `CBDescriptor` / `KernelDescriptor` / `TensorAccessorArgs` and use `emplace_runtime_args` — no imperative `host_api.hpp` (`CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs`). The device-operation is on the new TTNN device-operation API (`select_program_factory`, `program_factory_t` variant, `ttnn::device_operation::launch<>`), with the factory exposing `create_descriptor`.

- **Device 2.0 (every kernel used):** GREEN. All three kernels use Device 2.0 idioms throughout: `Noc noc;` + `noc.async_read(...)` / `noc.async_read_barrier()`, `CircularBuffer cb_*(...)` wrapper objects with member-form `cb.get_write_ptr()` / `cb.reserve_back()` / `cb.push_back()`, `TensorAccessor(args, addr)`, `CoreLocalMem<uint32_t>(...)`, `UnicastEndpoint`. No raw `noc_async_read(addr,...)` on tensor memory, no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast`, no CB-index free-function holdovers (`get_write_ptr(cb_id)` etc.), no raw semaphore addresses. The shared-lib donor `tt_memmove` (`data_movement/common/kernels/common.hpp`) is called only on local uint32 L1 addresses and is Device-2.0-shaped.

  Because Device 2.0 is GREEN, the DFB endpoint legality (SPSC) subject is run (not deferred).

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `.global_circular_buffer` field, no `remote_index` / remote-CB idiom. |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | Output CBs `c_16/c_17/c_18` set `CBDescriptor::buffer = output[N].buffer()` in all three factories — borrowed-memory CBs. Port maps via `DataflowBufferSpec::borrowed_from` naming the Q/K/V output `TensorParameter`s. (See SPSC: these same CBs are the SPSC blocker on the *count* axis.) |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set anywhere (default 0). The interleaved scratch CBs round their base up *inside the kernel*; no host-side `address_offset`. |
  | Aliased Circular Buffers | N/A | Every `format_descriptors` initializer is single-element (`{{CBFormatDescriptor{...}}}`). |
  | GlobalSemaphore | N/A | Op uses no semaphores at all (`GlobalSemaphore` or otherwise). |
  | Non-zero semaphore initial value | N/A | No semaphores. |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | `TensorAccessorArgs(buffer)` single-arg form only — no `ArgConfig::Runtime*` token. |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBuffer*` calls (no `override_runtime_arguments` hook either). |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Fixed-shape CTA lists; `tensor_args_t` carries a fixed `Tensor input_tensor` + `std::optional<Tensor> batch_offset`, not a variable-count container. Kernels read CTAs by fixed literal index. |

  Subject verdict: **GREEN — no gate fired.** The only LANDED feature actually in use is the borrowed-memory DFB (output CBs), which is routine port work via `borrowed_from`.

- **DFB endpoint legality (SPSC):** **RED — structural, no portable subset.** Device 2.0 is GREEN so this subject is evaluated (not deferred).

  Endpoint census, per CB, per node:

  | CB (index) | Factory / config | Producers (writers) per node | Consumers per node | Verdict |
  |---|---|---|---|---|
  | `c_16` Q out / `c_17` K out / `c_18` V out | **all 3 factories**, always | **2** (NOC0 reader + NOC1 writer, both raw `get_write_ptr()`+offset writes) | 0 | **SPSC violation** (2 producers) — sync-free, two raw co-writers |
  | `c_15` batch_offset reader CB | Interleaved | n/a (not allocated) | — | — |
  | `c_15` batch_offset reader CB | **Sharded**, `batch_offset` present | **2** (both reader & writer RISCs use `c_15`; writer CTA index not overridden) | 0 | **SPSC violation** (2 FIFO producers `reserve_back`/`push_back`) |
  | `c_14` batch_offset writer CB | **Sharded**, `batch_offset` present | 0 | 0 | **Dead CB** — allocated, index threaded to no kernel |
  | `c_15` batch_offset reader CB | Subcoregrid, `batch_offset` present | 1 (reader RISC: `reserve_back`/`push_back`, no consumer) | 0 | Single-ended FIFO producer → self-loop (FYI-P) |
  | `c_14` batch_offset writer CB | Subcoregrid, `batch_offset` present | 1 (writer RISC: `reserve_back`/`push_back`, no consumer) | 0 | Single-ended FIFO producer → self-loop (FYI-P) |
  | `c_0` reader scratch / `c_1` writer scratch | Interleaved, DRAM aligned-path only | 1 each (one RISC: `get_write_ptr` raw scratch) | 0 each | Single-ended / sync-free scratch → self-loop (FYI-P) |

  **Primary SPSC blocker (gating):** output CBs `c_16/c_17/c_18` carry **two writer endpoints per node** in all three factories — `device/kernels/reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:151,164,177,198,236,272` (write-ptr grabs) instantiated on two RISCs from `nlp_create_qkv_heads_decode_interleaved_program_factory.cpp:190-191`; `device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:98,157,216` instantiated on two RISCs from `nlp_create_qkv_heads_decode_sharded_program_factory.cpp:280-281,283-284`; `device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp:95,148,201` from `nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp:276-277,279-280`. Op-owner pre-port fix: collapse the two-RISC split write so each output CB has a single producer endpoint per node (output-preserving; the per-RISC phase split is an L1/NOC-bandwidth optimization that can be deferred). **Unconditional → no portable subset.**

  **Secondary SPSC blocker (sharded factory, `batch_offset` config):** `c_15` takes 2 FIFO producers per node because `nlp_create_qkv_heads_decode_sharded_program_factory.cpp` builds `q_writer_compile_time_args` from `q_reader_compile_time_args` and only overrides index `[9]` (phase), leaving `cb_batch_offset_id` (CTA 16) = `c_15` for the writer too. Contrast `nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.cpp:198,229` which *does* override the writer's CB index to `c_14`. Op-owner fix: same one-line override in the sharded factory (this also retires the dead `c_14`).

  **Dead CB:** `c_14` (`batch_offset_cb_index_writer`) at `nlp_create_qkv_heads_decode_sharded_program_factory.cpp:36,79-87` is allocated but its index reaches no kernel (the writer never reads CTA 16 = `c_14`). Op owner drops it pre-port (functional cleanup, not port work) — or it disappears naturally when the secondary SPSC fix wires the writer to `c_14`.

## Port-work summary  *(mirrors the brief — no brief issued; RED)*

- **Tensor bindings** (per binding):
  - `input_tensor` (Q/K/V source):
    - **Interleaved factory** — **Case 1** (via `TensorAccessor`). Kernel builds `TensorAccessor(qkv_args, q_start_addr)` and reads through it (`noc.async_read(qkv_reader, ...)`); host smuggles the base via the `Buffer*`-binding RTA form (`reader_desc.emplace_runtime_args(core, {in_tile_offset_by_batch, in_buffer})`, `interleaved_program_factory.cpp:186-187`). Port: express as a `TensorParameter`/`TensorBinding`; kernel uses `TensorAccessor(ta::name)`; the address-via-RTA + `TensorAccessorArgs` plumbing disappear. Low-risk.
    - **Sharded & Subcoregrid factories** — **Case 2** (raw pointer). The kernel does **not** wrap `q_start_addr` in a `TensorAccessor`; it reads the input via explicit NOC-coord walks (`noc.async_read(src_ep, ..., {.noc_x=..., .noc_y=..., .addr = qkv_read_addr})` where `qkv_read_addr = q_start_addr + ...`). Host passes the base as a `Buffer*` RTA (`rt.push_back(in_buffer)`, sharded `:254,269`, subcoregrid `:252,266`). Port: bind as a `TensorParameter`, pull the base via the `TensorAccessor::get_bank_base_address` bridge, keep the raw NOC walk unchanged. **These are dataflow (reader) kernels, not compute** — so Case 2 is *not* blocked by the compute-kernel `TensorBinding` gap. Low-risk.
  - `batch_offset` (sharded & subcoregrid factories, optional): **Case 1** (via `TensorAccessor`). Kernel builds `TensorAccessor(index_args, batch_offset_tensor_addr)` and reads page 0 through it. Host passes the base as a `Buffer*` RTA (`rt.push_back(batch_offset_buffer)`), or `uint32_t{0}` when absent. Port: optional `TensorParameter`/`TensorBinding` (conditional binding — see patterns "Conditional / optional DFB bindings"); kernel uses `TensorAccessor(ta::name)`.
- **Custom hash:** none — no custom `compute_program_hash`; default TTNN hash already in use.

> Note: the above tensor-binding port work is **described for completeness**, but the port cannot start until the SPSC gate clears. No `METAL2_PORT_BRIEF.md` is issued (RED).

## Heads-ups  *(would mirror a brief; recorded for the team — no brief issued)*

- **Notable LANDED constructs:** borrowed-memory DFB on output CBs `c_16/c_17/c_18` (all factories) → `DataflowBufferSpec::borrowed_from`. No aliased CBs, no dynamic TA, no non-zero sem init.
- **Sync-free / single-ended CBs (self-loop workaround):** interleaved scratch CBs `c_0`/`c_1` (DRAM-aligned path only, `interleaved_program_factory.cpp:115-132`; kernel `:76,87` raw `get_write_ptr`); subcoregrid batch-offset CBs `c_15`/`c_14` (single-ended FIFO producers). These do **not** gate — the port self-loops them — *but they are downstream of the SPSC gate and irrelevant until it clears.* (The output CBs are also sync-free but are SPSC violations, handled above, **not** self-loop cases.)
- **Dead CB:** `c_14` in the sharded factory (see SPSC).
- **Cross-op / shared kernels:** the interleaved kernel `#include`s `ttnn/operations/data_movement/common/kernels/common.hpp` and calls `tt::data_movement::common::tt_memmove<...>` (shared-lib pool, lib team owns) — clean shape (uint32 L1 addresses only). No cross-family kernel borrows; all four instantiated kernel `.cpp`s are op-owned.
- **RTA varargs:** none. The sharded/subcoregrid kernels read a *fixed* set of positional RTAs plus two NOC-coord arrays via `get_arg_addr` pointer arithmetic (`in0_mcast_noc_x/y`) — these are routine count-driven array reads (the count is a CTA, `num_x`/`num_y`/`in_num_cores`), not runtime-varying-index varargs. Routine port work (named RTAs for the scalars; the coord arrays port as named RTA arrays or remain pointer-walked).
- **TTNN factory analysis (porter-relevant):** no pybind `create_descriptor`, no other risky pybind, no custom `override_runtime_arguments`. Nothing to delete on the device-op class.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ✓ clean.

| Op kernel | Donor file | Donor class | Shape |
|---|---|---|---|
| `reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp` | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` (`tt_memmove`) | shared-lib (data_movement common) | uint32 L1 addresses (+ optional `Noc`); no CB-index / semaphore / TensorAccessor handle in signature → ✓ excellent |

No per-call detail needed (all ✓). **Borrowed kernel files (file-path instantiation):** none — all four kernel `.cpp`s are owned by this op directory. The sharded/subcoregrid factories include `<tt-metalium/tensor_accessor_args.hpp>`, `<tt-metalium/work_split.hpp>`, `<tt-metalium/program_descriptors.hpp>` (framework, no concern).

### Relaxation candidates (mined from custom hash)

N/A — no custom hash to mine.

### TTNN factory analysis — six questions

1. **Op-owned tensors?** **No.** `create_output_tensors` allocates the three declared output tensors (Q/K/V) via `create_device_tensor` (`device_operation.cpp:185-194`); these are `tensor_return_value`, not intermediate/scratch. No factory allocates an op-owned device tensor. The interleaved scratch CBs (`c_0/c_1`) are CBs, not device tensors.
2. **MeshWorkload concept needed?** **No.** No `create_mesh_workload` / `cached_mesh_workload_t`; each factory returns a single `ProgramDescriptor`. No cross-program/cross-device coordination. (And per Q1 there are no op-owned tensors, so there is no op-owned-tensor MeshWorkload artifact either.)
3. **Pybind `create_descriptor`?** **No.** `nlp_create_qkv_heads_decode_nanobind.cpp` binds only the user-facing function via `ttnn::bind_function<"nlp_create_qkv_heads_decode">`. No `nb::class_<...ProgramFactory>`.
4. **Other migration-risky pybind?** **None.** No `DeviceOperation`/factory/param class exposed to Python; no `compute_program_hash`/`create_output_tensors`/`compute_output_specs`/`select_program_factory` bound.
5. **Custom hash?** **No** — see Custom program hash subject (nothing to delete).
6. **Custom override-runtime-args?** **No** — no `override_runtime_arguments` on any factory (factories expose only `create_descriptor`).

Q1/Q2 FYI-U; Q3/Q4/Q6 are all "none," so nothing mirrors to a (non-existent) brief.

## Misc anomalies  *(team-only, non-gating)*

- **Sharded factory: `batch_offset_cb_index_writer` (c_14) is effectively dead** — allocated (`sharded_program_factory.cpp:79-87`) and the *writer* kernel was clearly intended to use it, but the writer's compile-time arg for the CB index is never overridden (only `[9]` phase is), so the writer reads `c_15` instead. Result: `c_14` is unreferenced and both RISCs share `c_15`. This looks like a latent bug (the subcoregrid factory does the override at `:198,229`). Routed to the op owner; the SPSC pre-port fix should address it. `file:line` — `sharded_program_factory.cpp:36,79-87,200-201,230-231`.
- **`v_cores` derived from `q_shard_spec.grid` rather than `v_shard_spec.grid`** in the interleaved factory (`interleaved_program_factory.cpp:74`) and subcoregrid factory (`subcoregrid_program_factory.cpp:115`). Likely intentional (V shares Q's grid), and `v_shard_spec.grid == q_shard_spec.grid` per `compute_output_specs` (`device_operation.cpp:152`), but the inconsistency with the sharded factory (which uses `v_shard_spec.grid`, `sharded_program_factory.cpp:115`) is worth an op-owner glance. Non-gating.
- **Subcoregrid factory: `v_shard_spec` reads `output[0].shard_spec()`** (`subcoregrid_program_factory.cpp:114`) rather than `output[2]`. Again harmless if Q and V shapes match, but inconsistent with the other factories' `output[2]`. Non-gating.

## Questions for the user  *(none — the verdict is unambiguous RED on SPSC)*

## Recipe notes

- The recipe's SPSC subject is written largely around *FIFO* producers/consumers and the "hidden second writer" being a raw co-fill *alongside* a FIFO producer. This op's primary SPSC shape is different but cleanly covered by the endpoint-census rule ("*any* access counts ... raw pointer ... all count"): **two raw, sync-free writers** (no FIFO producer at all) on the same output CB across two RISCs. It is neither the "hidden writer" face (no FIFO producer to hide behind) nor exactly the "multiple readers" face (these are writers into output, not readers of a resident tensor-view). It is most naturally read as the *visible* `(≥2 producers)` row of the census table. The recipe's two named "faces" ((a) hidden writer, (b) multiple readers) didn't obviously enumerate "two co-resident raw writers into a borrowed output CB"; a third illustrative face (split-RISC co-writers into output) would have made the match more immediate. Flagging per the recipe's invitation to log friction.

---

## ⚠️ Post-port-attempt correction (2026-06-25) — second, framework-level blocker after the SPSC fix

The SPSC prerequisite was resolved (single-producer collapse, **PR #48151**, validated 205/39/0). The port was then attempted and **gated** on a **second blocker this audit missed**: after the collapse, the op is **compute-less** — the single DM reader writes the borrowed output CBs **`c_16`/`c_17`/`c_18`** via raw `get_write_ptr()`, **producer-only, with no consumer kernel**. A DM-producer-only CB cannot pair (no consumer) and cannot self-loop (self-loop is compute-kernel-only). 

**Corrected status: RED, framework-blocked** (wait-for-feature: sync-free/scratch DFB or DM-kernel self-loop). See [[metal2-port-portability-predictor]]. The #48151 SPSC prereq cleared the producer-count gate but the op is still not portable until the DM-output-CB feature lands.

---

## 🔄 Revision (2026-06-25, supersedes the correction above) — workaround found; NOT framework-blocked

The "framework-blocked / wait-for-feature" verdict above is **overstated**. A workaround exists with **no framework change**: the **cross-kernel DFB bridge**. Only a DM-kernel *self*-loop FATALs; a DM kernel paired *cross-kernel* with a different co-located kernel (DM↔DM or DM↔compute) is fully legal. **Proven in shipped code:** the landed JointSDPA port (PR #48175, 160 passed/0 failed) binds `mask`/`scale`/`col_identity` as PRODUCER on the **writer (DM)** → CONSUMER on **compute** (`joint_sdpa_program_factory.cpp:359-451`); the SPSC validator accepts and runs them.

**This op — partial exception:** after the #48151 SPSC collapse it is **single-kernel (reader-only)** — no co-located compute/writer kernel exists to host the missing bridge endpoint for the producer-only borrowed output CBs `c_16`/`c_17`/`c_18`. The cross-kernel bridge therefore needs a **trivial second kernel added** (more invasive than the other ops). This is the one DM-sync-free case where a framework feature (or a second-kernel restructure) is still warranted. Not as cleanly portable as sampling/embedding.
