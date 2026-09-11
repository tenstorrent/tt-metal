# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded`

- **`InterleavedToShardedDeviceOperation`** (`device/interleaved_to_sharded_op.hpp:17`)
  - `InterleavedToShardedProgramFactory` (`device/interleaved_to_sharded_program_factory.cpp:58`) — the **only** factory; `program_factory_t` is a single-alternative variant (`device/interleaved_to_sharded_op.hpp:23`).

One device-operation, one factory, no bundling. The op directory contains **no kernel files** — all six kernels are file-path-instantiated from the in-family shared pool `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/` (inventoried under *Team-only → Out-of-directory coupling*); there are therefore no unreferenced kernels in the op's own tree.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Audited at:** `e3ca937f19c` (`origin/main`, 2026-09-11), in worktree `/localdev/iwrosz/tt-metal-i2s-audit` on local branch `iwrosz/i2s-metal2-audit`. Two changes landed on this op in the week before the audit and both matter below: `e1426777c4d` (#55407, 2026-09-04, *Resolve I2S and Slice Issues for Metal 2.0*) and `37fb77a975d` (#55495, 2026-09-08, *Hash Tensor Specs in I2S*).

**Recipe docs:** the provenance command prints **nothing** in this checkout — `main` carries no `docs/.../metal_2.0/` directory, so the version cannot be pinned from the audited tree. The recipe was read out of the doc branch instead: `origin/akertesz/op-porting-recipe` @ `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`.

**Readiness sheet:** *Operations analysis* (Diego), fetched fresh 2026-09-11 via the Drive connector, per `analyses/ttnn_op_porting_readiness.md`. Columns below are quoted by header name and cell value verbatim.

**Config vocabulary used throughout.** Three axes gate the factory's structure, and several findings are per-config:

| axis | values | decided by |
|---|---|---|
| layout | `TILE` / `RM` | `input.layout()` — factory:98 / :120 |
| output buffer | `dst-L1` / `dst-DRAM` | `dst_buffer->buffer_type()` — factory:94, branched at :174, :221, :303, :401 |
| format conversion | `convert_df` / plain | `input dtype != output dtype` — factory:90; **TILE-only** (rejected on RM at `device/interleaved_to_sharded_op.cpp:92-96`) |

Six reachable combinations: `TILE·{plain,convert_df}·{dst-L1,dst-DRAM}`, `RM·{dst-L1,dst-DRAM}`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded` |
| **Overall** | **RED** — one gate fails: the readiness sheet disagrees with the code on a cross-checked primary column (*spreadsheet-broken*). Every other gate is clear. |
| **DOps / Factories** | `InterleavedToShardedDeviceOperation` → `InterleavedToShardedProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all six kernels are structurally Device 2.0 (`Noc`, `DataflowBuffer`, `TensorAccessor`); no holdovers, only sanctioned CB-index free functions |
| *Prereqs* — Cross-op escapes | Ok — no ttnn-side donor function calls at all; only `tt_metal` `api/*` includes |
| *Feature Support* — overall | GREEN — every Appendix A entry `N/A` |
| *Feature Support* — GlobalCircularBuffer / `address_offset` / GlobalSemaphore | N/A / N/A / N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | `yes` — **but** the gate fails on the cross-check, not on this cell (see *Gate detail*) |
| *TTNN Readiness* — Concept (current) | `descriptor` — confirmed: `create_descriptor()` returning `ProgramDescriptor` (factory:58) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — not a `WorkloadDescriptor` op |
| *TTNN Readiness* — Custom hash | **Code: Yes** (`device/interleaved_to_sharded_op.hpp:35`, defined `device/interleaved_to_sharded_op.cpp:144-162`) · **Sheet: `no`** → **conflict = the failing gate**. Not a gate in itself; the port leaves the hash alone |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — sheet `no`, grep of the device-op clean |
| *TTNN Readiness* — `override_runtime_arguments` | No — sheet `no`, grep of the op directory clean |
| *TTNN Readiness* — Pybind `create_descriptor` | No — sheet `no`; no `nb::class_` of the device-op in `interleaved_to_sharded_nanobind.cpp` |
| *TTNN Readiness* — Op-owned tensors | No — sheet blank; `descriptor` concept cannot carry them (invariant holds) |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (`Override runtime args method? == no`; sheet's `Porting Target` agrees) |
| *Port work* — Offset base pointer | **none** — no `->address()` anywhere in the factory; per-core column shift already rides as a separate scalar |
| *Port work* — Tensor bindings (per binding) | `input` Case 1 · `output` Case 1 (dst-DRAM) / clean borrowed-DFB (dst-L1) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) — no analysis doc needed or expected |
| *Port work* — TensorAccessor 3rd arg | **none — no accessor in the op passes a 3rd argument** (4 construction sites, all 2-arg) |
| *Port work* — CB endpoints | legal 1:1 for `c_0` / `c_16` in every config · `c_1` **self-loop** under `RM` · `c_1` **conditional DFB** (dead under `TILE`, live under `RM`) |

**CB endpoints** are dispositions, not gates: every out-of-window CB here has a port-time resolution. Recorded per `(CB, config)` below.

## Result

**RED at op level; no portable subset** — but the blocker is a **data-reconciliation on the readiness sheet, not a defect in the op**, and it clears with the op's code untouched.

The failing gate is the [TTNN factory concept prerequisite]'s cross-check: the op's device-operation declares a custom `compute_program_hash` (`device/interleaved_to_sharded_op.hpp:35`, `device/interleaved_to_sharded_op.cpp:144-162`, landed 2026-09-08 in #55495) while the sheet's `Custom hash (compute_program_hash)` cell reads `no`. That is a primary-column conflict backed by independent evidence (the code), which the recipe routes as *"spreadsheet is broken" → GATE → readiness-sheet owner to reconcile*. The sheet's own `Is able to port?` cell reads `yes`; whether that derived verdict depends on the stale input is not visible from the CSV, which is precisely why the recipe stops here rather than proceeding on it.

Nothing else blocks. **Device 2.0 ✓ · Feature compatibility ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (no sites)**, and the substantive readiness signals (`descriptor` concept, no `get_dynamic_runtime_args`, no override, relaxation `none`) are all clear and code-confirmed. So the expected path forward is a one-cell sheet update (or a confirmation from the sheet owner that the column is stale and the verdict unaffected), after which this op is GREEN with **no code change on any team's part**.

Per the recipe's *Red-outcome scoping rule*, I judged this a blocker that **clears elsewhere than the op's code** (readiness-sheet side), so the seven purely-informational subjects were **run in full** rather than deferred — a re-audit will read the same code, and today's detail survives intact. The port-work and heads-up detail below is therefore complete and ready to be lifted into a brief the moment the gate clears.

**No `METAL2_PORT_BRIEF.md` is emitted** — the audit is not fully GREEN.

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED — spreadsheet-broken (primary-column conflict).**

  The sheet row read for this op (fetched 2026-09-11), verbatim:

  | column | cell |
  |---|---|
  | `Op` | `data_movement/sharded/interleaved_to_sharded` |
  | `Device operation` / `Factory (variant)` | `InterleavedToShardedDeviceOperation` / `InterleavedToShardedProgramFactory` |
  | `Concept` | `descriptor` |
  | `Op Classification` | `PD Op (pointer-patching)` |
  | `Porting Target` | `ProgramSpecFactoryConcept` |
  | `Custom hash (compute_program_hash)` | `no` ← **conflicts with the code** |
  | `Backdoor custom hash (attribute_values / to_hash)` | `no` |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` |
  | `Override runtime args method? (PD only)` | `no` |
  | `Pybind descriptor (nb::class_ of device op)` | `no` |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` |
  | `Known op issues` | *(empty)* |
  | `Is able to port?` | `yes` |
  | `TensorParameter relaxation` | `none` |
  | `Provisional relaxation finding (Edwin)` | `fix merged, then match_padded_shape` |
  | `Op-owned tensors?` / `Secretly SPMD Workload?` | *(empty)* / *(empty)* |
  | `Diego validation` | `yes` |

  **Cross-check results, column by column:**

  | column | sheet | code | verdict |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor()` → `ProgramDescriptor` (factory:58; decl `device/interleaved_to_sharded_program_factory.hpp:37`) | ✓ agrees |
  | `Custom hash` | `no` | **declared** `device/interleaved_to_sharded_op.hpp:35`; defined `device/interleaved_to_sharded_op.cpp:144-162` (`hash_operation<…>(output_mem_config, output_dtype, input_tensor.tensor_spec())`, plus the output spec when pre-allocated) | ✗ **conflict** |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | no such hook on the device-op (grep of the op directory clean) | ✓ agrees |
  | `Override runtime args method?` | `no` | no `override_runtime_arguments` anywhere in the op directory | ✓ agrees |
  | `Pybind descriptor` | `no` | `interleaved_to_sharded_nanobind.cpp` binds only the two `ttnn::interleaved_to_sharded` overloads (`:57-95`); no `nb::class_`, no `create_descriptor` binding | ✓ agrees |
  | `Secretly SPMD Workload?` | *(empty)* | N/A — not a `WorkloadDescriptor` concept | ✓ n/a |
  | factory-set match | 1 row | 1 factory (`program_factory_t`, op.hpp:23) | ✓ one-to-one, no phantom/missing row |
  | cross-column invariants | — | `get_dynamic_runtime_args == no` on a `descriptor` row ✓; `Op-owned tensors?` not `yes` on a `descriptor` row ✓ | ✓ consistent |

  **The pybind-rename escape does not apply.** The recipe's one explanation for a `Custom hash` grep-vs-sheet mismatch is a pybound-`create_descriptor` op that renames the hook. This op has no pybind of device-op internals (`Pybind descriptor == no`, confirmed by grep), so the mismatch has no benign reading available; it is a stale cell.

  **Most likely cause, offered as context, not as a verdict:** the hash landed on 2026-09-08 in #55495, three days before this fetch, and the sheet's own `Provisional relaxation finding (Edwin)` cell already reads `fix merged, then match_padded_shape` — i.e. the sheet has absorbed *some* of #55495 while the `Custom hash` column has not. That is a routine lag, not an allegation about the sheet's maintenance.

  **Routing:** → **readiness-sheet owner** (Diego) to reconcile the `Custom hash` cell for this row, and to confirm whether the derived `Is able to port?` value depends on it. Note for that reader: a custom hash is explicitly **not** a portability question — the port leaves `compute_program_hash` exactly as it is — so reconciliation is expected to be a cell edit with no effect on the verdict, at which point this audit's RED lifts with no code change.

- **Device 2.0 (every kernel used): GREEN.**

  All six kernels the factory instantiates are structurally Device 2.0 — `Noc` for every transfer, `DataflowBuffer` for every buffer handle, `TensorAccessor` for every tensor walk. No `noc_async_read`/`noc_async_write` free calls, no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, no raw semaphore addresses (the op declares no semaphores at all), no manual CB index management. Migration history for this family: #46483 (i2s → Device 2.0), #47364 (finish `data_movement/sharded` kernels), #49392 (CircularBuffer → DataflowBuffer).

  | kernel (all under `data_movement/sharded/device/kernels/`) | reached in | Device 2.0 evidence |
  |---|---|---|
  | `dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp` | `TILE` | `Noc` :38, `DataflowBuffer` :39, `TensorAccessor` :40, `noc.async_read(s, dfb_in, …)` :50 |
  | `dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `RM` | `Noc` :29, DFBs :30-31, `TensorAccessor` :36, TRID-tagged `noc.async_read` :91, :113 |
  | `dataflow/writer_unary_sharded.cpp` | `dst-L1` | DFB :28, `wait_front`/`pop_front` :30/:33 (no NoC — output is already in place) |
  | `dataflow/writer_unary_sharded_blocks_start_id.cpp` | `TILE·dst-DRAM` | `TensorAccessor` :29, `Noc` :31, DFB :32, `noc.async_write(dfb_out, s, …)` :40 |
  | `dataflow/writer_unary_sharded_stick_layout_start_id.cpp` | `RM·dst-DRAM` | `TensorAccessor` :24, `Noc` :26, DFB :27, `noc.async_write` :33 |
  | `compute/eltwise_copy.cpp` | `convert_df` | DFBs :20-21 with FIFO ops :27-41 |

  **Two free-function CB-index sites, both sanctioned — not flagged:**

  | File | Line | Call | Wrapper in scope | Disposition |
  |---|---|---|---|---|
  | `…/dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp` | 36 | `get_tile_size(cb_id_in0)` | `DataflowBuffer dfb_in` (:39) | sanctioned — kept by Device 2.0 itself |
  | `…/dataflow/writer_unary_sharded_blocks_start_id.cpp` | 27 | `get_tile_size(cb_id_out)` | `DataflowBuffer dfb_out` (:32) | sanctioned |
  | `…/dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | 63 | `get_local_cb_interface(cb_id_in1).fifo_page_size` | `DataflowBuffer dfb_in1` (:31) | sanctioned |

  Per the Green bullet the sanctioned list is the whole test and does not turn on what object is in scope, so the DFB's own `get_tile_size()` existing does not make these holdovers. They do become port-time work (move onto the DFB object, kernel-side whitelist rule 7) — carried as a breadcrumb in *Heads-ups*.

  **Compute-side CB-index arguments are not Device 2.0 holdovers.** `compute/eltwise_copy.cpp` passes buffer ids to `compute_kernel_hw_startup` (:23), `copy_init` (:24), `copy_tile` (:30) and `pack_tile` (:38). These are compute LLK primitives, not data-movement free functions, and the in-tree Metal 2.0 fork beside the file keeps every one of them, substituting `dfb::in` / `dfb::out` tokens for the raw indices (`compute/eltwise_copy_metal2.cpp:27-42`). Nothing here needs a Device 2.0 change.

- **Feature compatibility: GREEN — every Appendix A entry `N/A`.**

  Scan covered the op directory (host + nanobind) and all six instantiated kernels. Grep signals: `GlobalCircularBuffer`, `global_circular_buffer`, `CreateGlobalCircularBuffer`, `remote_index`, `remote_cb`, `GlobalSemaphore`, `global_semaphore`, `address_offset`, `set_address_offset`, `UpdateDynamicCircularBufferAddress`, `cb_descriptor_from_sharded_tensor` — **zero hits**.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CBDescriptor::global_circular_buffer` field set, no `remote_*` CB idiom. The op's only CB-attachment field is `cb.buffer = <Buffer*>` (factory:46, set at :174) — plain borrowed memory, which is a mechanical porting-recipe translation (`DataflowBufferSpec::borrowed_from`), not this entry. |
  | CBDescriptor `address_offset` (non-zero) | N/A | Every CB in the op is built by one anonymous-namespace helper (`push_i2s_cb_pair`, factory:30-48) which sets only `total_size`, `core_ranges`, `format_descriptors` and `buffer`. `address_offset` is never assigned → statically zero at all three call sites (:155, :167, :184). |
  | GlobalSemaphore | N/A | The op declares no semaphores of any kind (grep for `semaphore` in the op directory: no hits). |

- **CB endpoints (GATE-free): every CB resolvable at port time.** Census is per CB, per node, per config; the op's CBs are all allocated over the single `all_cores` range (factory:87), so every node sees the same instance shape. Touchers counted by any access — FIFO or raw pointer.

  | CB | config | distinct touchers on a node | verdict | port-time disposition |
  |---|---|---|---|---|
  | `c_0` (input CB; allocated only when `convert_df`, factory:151-163) | `TILE·convert_df` (both dst types) | reader **locked producer** (`reserve_back` :46 / `push_back` :62) · compute **locked consumer** (`wait_front` :27 / `pop_front` :33) | plain 1:1 | legal — bind 1P + 1C, no action |
  | `c_0` (doubles as the **output** CB when `!convert_df`: `out_cb_index = input_cb_index`, factory:145) | `TILE·plain·dst-L1`, `RM·dst-L1` | reader **locked producer** · `writer_unary_sharded` **locked consumer** (:30/:33) | plain 1:1 | legal; the DFB is **borrowed** from the output tensor (`cb.buffer = dst_buffer`, factory:174) → `DataflowBufferSpec::borrowed_from` |
  | `c_0` | `TILE·plain·dst-DRAM` | reader producer · `writer_unary_sharded_blocks_start_id` consumer (`wait_front` :35 / `pop_front` :49) | plain 1:1 | legal; plain (unborrowed) DFB |
  | `c_0` | `RM·dst-DRAM` | RM reader producer (`reserve_back` :38 / `push_back` :148) · `writer_unary_sharded_stick_layout_start_id` consumer (:30/:39) | plain 1:1 | legal; plain DFB |
  | `c_16` (output CB when `convert_df`, factory:152 + :167-174) | `TILE·convert_df·dst-L1` | compute **locked producer** (`reserve_back` :35 / `push_back` :41) · `writer_unary_sharded` **locked consumer** | plain 1:1 | legal; borrowed from the output tensor |
  | `c_16` | `TILE·convert_df·dst-DRAM` | compute producer · `writer_unary_sharded_blocks_start_id` consumer | plain 1:1 | legal; plain DFB |
  | `c_1` (alignment scratchpad, factory:176-192) | `RM` (both dst types) | **1** — the RM reader only: `reserve_back` :62, `push_back` :137, raw peek `get_write_ptr()` :79, `get_local_cb_interface(...)` :63 | single-ended / sync-free | **self-loop** — bind the RM reader **PRODUCER and CONSUMER** (legal on Gen1 for DM) |
  | `c_1` | `TILE` (all three variants) | **0** — no kernel receives the index | **dead in this config only** | **conditional DFB** — gate the `DataflowBufferSpec` on `input.layout() == Layout::ROW_MAJOR`. **Do not drop it**: it is live under `RM`. |

  **How the `c_1` `(0, 0)` result was established** (the recipe rightly distrusts a dead-CB claim): the allocation condition at factory:179 is layout-independent and its last disjunct is the literal `keep_l1_aligned`, hardcoded `true` at factory:65 — so the scratch CB is allocated in **every** config, `TILE` included. Its index reaches a kernel only through the `RM` reader's CTA list (`{input_cb_index, scratch_cb_index, num_trids}`, factory:207). The `TILE` reader's CTA list is `{input_cb_index, all_cores.num_cores()}` (factory:200) — no scratch index; the two `TILE` writers take `{out_cb_index}` (+ accessor args, factory:220/:231); `compute/eltwise_copy.cpp` hardcodes `c_0`/`c_16` (:20-21). There is no indirection to hide behind: no helper receives a CB index, no index is computed or aliased, and the op has no `#ifdef`-gated kernel variants. Hence 0 touchers under `TILE`, 1 under `RM` — a *conditional* DFB, not a drop.

  **Hidden-second-writer hunt: negative.** The only raw-pointer accesses in the op are the `RM` reader's own `dfb_in1.get_write_ptr()` (:79) and the local L1 loopback read addressed from it (:113-121) — the same kernel peeking at the buffer it already binds, i.e. one toucher, not two. No `fifo_wr_ptr` / `fifo_rd_ptr` writes, no `evil_set_*_ptr` cursor drivers, and no semaphores anywhere in the op, so there is no semaphore-gated co-fill to miss. No multi-reader face either: each CB's read sites sit in exactly one kernel per config. **Nothing in this op needs the multi-binding advanced option.**

- **Offset base pointers: GREEN — no fold anywhere; the op is not in the triage tables (the *"no fold, not listed"* outcome).**

  The factory contains **no `->address()` / `.address()` expression at all** (grep clean over the op directory). Both tensor bases are delivered by pushing the `Buffer*` itself into `KernelDescriptor::RTArgList` (`reader_rt.push_back(src_buffer)` factory:291 and :388; `writer_rt.push_back(dst_buffer)` factory:306 and :406), which `emplace_runtime_args` auto-registers as a `BufferBinding` (`tt_metal/api/tt-metalium/program_descriptors.hpp:177`, :192). That delivery form carries the **base only** — there is no host expression into which an offset could be folded — so the op is GREEN by construction, not merely by inspection.

  Every per-core displacement travels as its **own scalar arg** and is applied kernel-side:

  | config | displacement args | applied |
  |---|---|---|
  | `RM` reader | `aligned_width_offset` (factory:394), `aligned_offset` (:396) | as the *source* `.offset_bytes` of each accessor read (reader :46, :95) and as an addend on the scratch L1 address (reader :119) |
  | `TILE` reader | `curr_idx_h + curr_idx_w` (factory:297), `starting_idx_h` (:298) | as `page_id` arithmetic on the accessor (reader :30, :44-59) |
  | `dst-DRAM` writers | `start_id` / `start_id_offset` + `start_id_base` (factory:310-312, :410) | as `page_id` arithmetic (writer :34-46, :29-35) |

  The `RM` reader even documents the invariant in place (`reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:33-35`): *"The accessor base must stay the unshifted buffer base: Metal 2.0 supplies it from the tensor binding and offers no seam for a pre-offset base."* This op's split-out predates the audit — `6abdf94214d` (#51747, *[Cleanup] Fix offset pointers in I2S and S2I*) did it. Types 3 and 4 do not arise (no `address_offset`, no `ttnn::narrow`).

- **TensorAccessor 3rd argument: N/A — no accessor in the op passes a 3rd argument.** All four construction sites use the 2-arg form: `reader_unary_sharded_blocks_interleaved_start_id.cpp:40`, `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:36`, `writer_unary_sharded_blocks_start_id.cpp:29`, `writer_unary_sharded_stick_layout_start_id.cpp:24`. The subject never fires — this is *no sites*, not *sites found and classified redundant*. The op is likewise absent from the dated 3rd-arg triage table, consistent with the scan. (#55407's companion change removed the analogous overrides from `slice`; nothing of that kind was left in i2s.)

## Port-work summary  *(would mirror the brief)*

- **Tensor bindings** (per binding; the classification splits by config, so it is recorded per config rather than flattened):

  | binding | config | how the base reaches the kernel today | what the kernel does with it | case |
  |---|---|---|---|---|
  | `input` | all six | `Buffer*` in the reader's `RTArgList` (factory:291 `TILE`, :388 `RM`) → framework `BufferBinding`, patched on cache hits | fed straight into `TensorAccessor(src_args, src_addr)` (reader `TILE`:40, `RM`:36); all addressing through the accessor | **Case 1** — declare a `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::…)`; the address RTA and its `TensorAccessorArgs` CTAs (factory:201, :208) both disappear |
  | `output` | `dst-DRAM` (`TILE` and `RM`) | `Buffer*` in the writer's `RTArgList` (factory:306, :406) | fed into `TensorAccessor(dst_args, dst_addr)` (writer `:29`, `:24`) | **Case 1** — same treatment; drops the accessor-args CTAs at factory:231 |
  | `output` | `dst-L1` (`TILE` and `RM`) | borrowed-memory CB: `cb.buffer = dst_buffer` (factory:46, set :174) | writer only `wait_front`/`pop_front` — the DFB *is* the tensor access | **clean** — no Case 1/2 work; ports via `DataflowBufferSpec::borrowed_from` (causal-link gate) |

  Neither Case 1 is the silent-wrong hazard: this op never smuggled a raw `->address()`, it is already on the `Buffer*` delivery form the framework patches. Both are routine port work.

- **TensorParameter relaxation:** `none` — the port applies no relaxation, and no `analyses/relaxations/…` doc is expected or needed. Context worth carrying: the op declares **no `TensorParameter` today**, so `ValidateTensorArgs` never runs on it; after the port it runs, comparing `tensor_layout()` (alignment included) exactly. #55495 pre-hardened the program hash for precisely that transition (its rationale is recorded in the hash comment, `device/interleaved_to_sharded_op.cpp:147-155`).

- **TensorAccessor 3rd arg:** none — no sites.

- **CB endpoints:**
  - `c_1` — **self-loop** under `RM·dst-L1` and `RM·dst-DRAM` (bind the RM reader PRODUCER *and* CONSUMER).
  - `c_1` — **conditional DFB**: dead under all three `TILE` variants, live under both `RM` variants → make the `DataflowBufferSpec` conditional on `input.layout() == Layout::ROW_MAJOR`. This is **new host structure**: the legacy factory allocates it unconditionally (factory:176-192) and gates only the kernel-side use, so there is no existing conditional to translate.
  - `c_0` and `c_16` — legal 1:1 in every config; bind 1P + 1C as the census shows, with `borrowed_from` on whichever of them is the output CB in the `dst-L1` configs.

## Heads-ups  *(would mirror the brief; recorded here since no brief is issued)*

- **CB endpoints (multi-binding shapes to watch):** none — the hunt came back negative in every config (see *Gate detail*). The only non-trivial dispositions are `c_1`'s self-loop and its conditional spec.

- **Cross-op / shared kernels — the op owns no kernels; all six are borrowed from the in-family pool.** Rung per file (vocabulary per `port_patterns.md` → *Caution: Porting a shared kernel*):

  | borrowed kernel | rung | detail |
  |---|---|---|
  | `dataflow/writer_unary_sharded.cpp` | **1 — reuse the existing fork** | `dataflow/writer_unary_sharded_metal2.cpp` sits beside it. Its header states the interface (`:18-20`): `dfb::out` bound **CONSUMER**, `args::num_units`. Already bound by production factories — `tilize_with_val_padding_multi_core_sharded_program_factory.cpp:173`, `transpose_wh_sharded_program_factory.cpp:100`, `untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp:39`, `reduce_op_multi_core_h_program_factory.cpp:513`, `reduce_op_multi_core_w_program_factory.cpp:391`. **Read-only to this port**; adopt its names. |
  | `compute/eltwise_copy.cpp` | **1 — reuse the sibling fork, and mind *which* fork** | Two forks of this kernel exist in the tree and **their interfaces differ**. The sibling (`compute/eltwise_copy_metal2.cpp:22`) reads `per_core_tile_cnt` as a **runtime** named arg; the shared-pool one (`ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp:23`) reads it as a **`constexpr`** (compile-time) named arg. i2s emits this count **per core** (`compute_desc.emplace_runtime_args(core, {curr_num_units_per_shard})`, factory:425 — it differs on the end core), so only the **sibling, runtime-arg fork fits**. Its current consumer: `copy_default_tilized_program_factory.cpp:45`. |
  | `dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp` | **2 — create the first fork** | No `_metal2` sibling; none anywhere in the tree for this stem. Fork beside the original, in `data_movement/sharded/device/kernels/dataflow/`. |
  | `dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | **2** | Same. |
  | `dataflow/writer_unary_sharded_blocks_start_id.cpp` | **2** | Same. Note the near-miss neighbour: `writer_unary_sharded_blocks_interleaved_start_id_metal2.cpp` **is** a different kernel (sharded_to_interleaved's writer) — not a fork of this file. |
  | `dataflow/writer_unary_sharded_stick_layout_start_id.cpp` | **2** | Same, with the same near-miss (`writer_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp` is s2i's). |

  **Sunset / coordination lists** (other binders of each legacy file — *a sunset list, not authorization to convert anything in place*):

  | legacy kernel | other binders |
  |---|---|
  | both readers, `writer_unary_sharded_blocks_start_id.cpp`, `writer_unary_sharded_stick_layout_start_id.cpp` | `interleaved_to_sharded_partial` (`…/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp:197, :204, :218, :222`) · the tt-metal DM microbenchmark `tests/tt_metal/tt_metal/data_movement/interleaved_to_sharded_hardcoded/test_interleaved_to_sharded_hardcoded.cpp:51, :105, :159, :213, :267, :330` (a non-op binder — it keeps the legacy file alive past the op ports) |
  | `writer_unary_sharded.cpp` | `interleaved_to_sharded_partial` · `tilize_multi_core_sharded_program_factory.cpp` · `tilize_multi_core_sharded_retile_program_factory.cpp` · `untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp` · `experimental/padded_slice/…/padded_slice_rm_program_factory.cpp` · `experimental/transformer/nlp_kv_cache_load_slice/…` |
  | `compute/eltwise_copy.cpp` (the sharded-family copy — a *second* legacy copy of the same kernel lives at `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` with its own consumers) | `interleaved_to_sharded_partial` (`…_partial_program_factory.cpp:234-235`) only |

  **`interleaved_to_sharded_partial` shares all six kernels and is itself blocked** (`Is able to port? = no`; `TensorParameter relaxation = (legality - pending analysis)`, `Custom hash = yes`, `Override runtime args method? = yes`). So the two ops cannot co-migrate today, which is what forces rung 2 rather than in-place conversion on the four unforked files.

  **Negative pointer:** `ttnn/cpp/ttnn/operations/experimental/quasar/interleaved_to_sharded/` holds a hacky pre-port copy of this op with its own kernel tree. It is out of bounds — not a template, not a fork to reuse, not a naming source. It also appears in filename greps for the shared kernels above; discount those hits.

- **Nearest in-tree precedent (use as an interface reference, not a template):** the mirror op `sharded_to_interleaved` — same family, same kernel pool — is **already on Metal 2.0 on `main`** (`…/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp`: `DataflowBufferSpec`s :107-133, reader/writer `KernelSpec`s :135-183, compute :190-215, `ProgramSpec` + `TensorParameter`s :284-291). Two details there are directly reusable knowledge:
  - It binds **one DFB from two kernels under different accessor names** (`.dfb_spec_name = convert_df ? OUT_DFB : IN_DFB, .accessor_name = "out"`, :156-162) — exactly the shape i2s needs for `c_0` when `!convert_df`, where the reader calls it `in` and the writer calls it `out`.
  - Its compute `KernelSpec` states `compiler_options = {.opt_level = KernelBuildOptLevel::O3}` with the reason inline (:196-198): a legacy all-default `ComputeConfigDescriptor` resolves to **O3**, while Metal 2.0's `CompilerOptions` defaults to **O2**. i2s's compute descriptor is also all-default (factory:245), so the same explicit `O3` is needed to keep the port behaviour-neutral.

- **RTA varargs:** none. Every kernel reads its args at fixed constant indices (`reader TILE` :22-29, `reader RM` :14-22, `writer sharded` :24, `writer blocks` :13-20, `writer stick` :13-18, `compute` :18) — no counted loop over an arg index, no `arg_index++` block, no data-selected element, and no `get_compile_time_arg_val(i)` at a varying index. All args are nameable; nothing needs the vararg mechanism.

- **Two sanctioned free-function lookups become port work** (kernel-side whitelist rule 7 — move onto the DFB object, do not swap blind): `get_tile_size(cb_id_in0)` (`reader_unary_sharded_blocks_interleaved_start_id.cpp:36`, feeding a `constexpr` used as a template argument at :42 — confirm the DFB accessor is usable in that position before swapping) and `get_local_cb_interface(cb_id_in1).fifo_page_size` (`reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:63`). Also `get_tile_size(cb_id_out)` at `writer_unary_sharded_blocks_start_id.cpp:27`.

- **The `RM` reader is the delicate file in this port.** It carries TRID-tagged multi-slot pipelining (`:85-133`), a local L1 loopback read whose address is built from `dfb_in1.get_write_ptr()` (`:79`, :119), and a sticky-register reset at the end (`:141-147`). The port is a binding-layer change only: nothing in that control flow should move.

## Team-only

- **Out-of-directory coupling & donor shape.**
  - **Op-level roll-up: ✓ clean** for function-call escape. Every include in every one of the six kernels resolves to `tt_metal` framework headers — `api/dataflow/dataflow_api.h`, `api/dataflow/dataflow_buffer.h`, `api/dataflow/noc.h`, `api/dataflow/endpoints.h`, `api/tensor/noc_traits.h`, `api/compute/{common,tile_move_copy,eltwise_unary/eltwise_unary}.h`, `tensix_types.h` — i.e. donor class 1 (LLK/HAL/firmware, no concern). **No ttnn-side donor headers, and no donor functions called at all**, so the per-call shape analysis has no rows and the per-call detail section is omitted.
  - **Borrowed kernel files (file-path instantiation): all six.** Inventory, owning pool, co-borrowers and fork status are tabled under *Heads-ups* above. Independent of the clean function-call roll-up, this is a real cross-op coordination cost: four new `_metal2` forks land in a peer-shared directory, and their sunset waits on `interleaved_to_sharded_partial` (blocked) plus a tt-metal microbenchmark that binds the legacy files directly.
  - **Host-side coupling (not a kernel escape; noted for completeness):** the factory draws on `ttnn/operations/data_movement/sharded/sharded_common.hpp` (`calculate_starting_idx_h`, factory:249), `ttnn/api/ttnn/tensor/tensor_utils.hpp` (`get_optimal_worker_cores_for_sharded_tensor`, factory:86) and `ttnn/operations/math.hpp`. None of these is affected by the port.

- **Relaxation candidates (FALLIBLE — candidates to verify; default strict):** **none offered.** The custom hash keys on the *whole* `TensorSpec` of the input (and of the output when pre-allocated) plus `output_mem_config` and `output_dtype` (`device/interleaved_to_sharded_op.cpp:153-160`) — i.e. it is at least as strict as the default `TensorParameter` match, so it reveals no property the op could be shown not to depend on. Recorded for the relaxation roadmap: the sheet's `Provisional relaxation finding (Edwin)` cell reads verbatim `fix merged, then match_padded_shape`, i.e. a `match_padded_shape` relaxation may be proposed for this op later; the gate column (`TensorParameter relaxation`) currently reads `none` and that is what this audit acted on.

- **TTNN factory analysis (sheet-derived facts, with code evidence):**
  - Current concept `descriptor` → **target `ProgramSpecFactoryConcept`** (base concept, not Custom: no `override_runtime_arguments` exists to translate).
  - **Op-owned tensors:** none — and structurally impossible on this concept.
  - **Custom hash: present** (`device/interleaved_to_sharded_op.hpp:35`, `…op.cpp:144-162`) — the port leaves it exactly as is. This is the cell the sheet disagrees on; see *Gate detail*.
  - **`get_dynamic_runtime_args`:** absent (the deprecated hook) — gate conjunct, clear.
  - **`override_runtime_arguments`:** absent — hence the base target concept.
  - **Pybound `create_descriptor` / other risky pybind:** none; `interleaved_to_sharded_nanobind.cpp` exposes only the two public overloads, so the port has no user-visible pybind deletion to report.
  - **`Op Classification` reads `PD Op (pointer-patching)`**, which matches the code exactly: both tensor bases ride `Buffer*` RTArgs that the framework patches on cache hits (factory:291, :306, :388, :406).

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **Dead RTA in the `RM` reader.** The factory pushes ten reader args (factory:387-398) but the kernel reads indices 0 and 2-9 only: arg **1** (`num_units_per_row`, pushed at factory:389) is never read (`reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:14-22`). Dead plumbing.
- **The public `keep_l1_aligned` argument is inert.** It is a documented Python kwarg on both overloads (`interleaved_to_sharded_nanobind.cpp:79`, :94, default `False`) and is plumbed into `InterleavedToShardedParams` (`…_op_types.hpp:15`), but the factory hardcodes `bool keep_l1_aligned = true;` with the attribute read commented out (factory:64-65) and never consults the attribute. It is also deliberately excluded from the program hash (`…op.cpp:147-148`). A caller passing `keep_l1_aligned=False` silently gets the aligned behaviour.
- **The alignment scratch CB is allocated in `TILE` configs that never touch it.** factory:179's condition is layout-independent and its last disjunct is the hardcoded `true` above, so every `TILE` program allocates `c_1` at `num_trids * align(input_unit_size + dram_alignment, dram_alignment)` (factory:183-190) — for a bf16 tile on Blackhole that is 4 × 2112 ≈ 8.4 KB of L1 per core, burned for nothing. (The port's answer is the conditional DFB; the waste itself is the ops team's to fix if they care.)
- **`starting_idx_h` is structurally always zero.** `num_slices = 1` / `slice_index = 0` are hardcoded for backward compatibility (factory:55-56, issue #32752), and `calculate_starting_idx_h` returns `0` whenever `num_slices <= 1` (`sharded_common.cpp:17-19`). So the `TILE` reader's arg 7 and the `TILE·dst-DRAM` writer's arg 7 are constant zero, and the kernel-side `start_id_base + start_id_offset` additions (reader :30, writer :34) are dead arithmetic.
- **A standing TODO in the `aligned` computation.** For an L1 source on non-Blackhole/Quasar the factory sets `aligned = true` unconditionally (factory:361-363) while the Blackhole/Quasar path checks `curr_idx_w` and `padded_offset_bytes` against the L1 alignment; the code itself flags the asymmetry as unverified (`// TODO: is this right, leaving non BH case the same for now, should investigate`, factory:368-369).
- **Probably-stale include.** `#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"` (factory:14) appears unused — no `shard_builder`/addrgen-helper symbol is referenced, and `get_optimal_worker_cores_for_sharded_tensor` comes from `ttnn/tensor/tensor_utils.hpp` (included at factory:16).
- **Misnamed variable, self-flagged.** In the `RM` branch `num_units_per_shard_width_last` holds a byte size, not a page count, with its own TODO (factory:132-134).

## Questions for the user  *(two, both routing rather than technical)*

1. **The failing gate is a sheet cell, not code — who takes it?** The `Custom hash` cell for `data_movement/sharded/interleaved_to_sharded` needs reconciling with #55495 (`device/interleaved_to_sharded_op.cpp:144-162`), and the sheet owner should confirm whether the derived `Is able to port? = yes` depends on that input. If you would rather not block on a spreadsheet round-trip, say so and I will re-run the gate against an owner-confirmed value and issue the brief — every other gate is already clear and the port-work detail above is complete.
2. **Confirm the port scope is i2s-only.** `interleaved_to_sharded_partial` binds all six of the same kernels and is blocked on its own relaxation analysis, so this port takes rung 2 (four new `_metal2` forks beside the originals) rather than converting the shared files in place. If a bundled i2s + i2s_partial port is ever intended, that changes the rung — and it would need the partial op's gate cleared first.

## Recipe notes

1. **A stale *non-gating* column produces a hard GATE with no brief, and the recipe gives the auditor no room to say so.** The *TTNN factory concept prerequisite* is explicit that `Custom hash` "is not a portability question" and that the port leaves the hash alone — yet a code-vs-sheet mismatch **on that same column** routes through the *spreadsheet-broken* rule to a full GATE, suppressing the porter brief for an op whose every substantive gate is green. The stop is defensible in the abstract (don't proceed on data we can't trust), but the outcome here is a RED whose entire remediation is one cell edit, and a second full audit pass to undo. Consider either (a) a carve-out: a conflict on a column the recipe itself classifies as non-gating is recorded and routed to the sheet owner, but does not gate; or (b) an explicit third outcome — *"RED (data reconciliation)"* — that still issues the brief with the conflict stamped on it, since no port decision in the brief depends on the disputed value. Worth noting that this failure mode will recur: pre-port hardening PRs (#55407, #55495 here) *add* custom hashes to exactly the ops that are about to be audited, so the column is stale precisely when the audit runs.
2. **The scoping-rule exception worked, and the judgment call was easy to make** — the blocker clears sheet-side, the op's code is untouched, so the seven informational subjects ran. Recording it here because the recipe asks which side I judged it on: **elsewhere → run them**.
3. **The `Buffer*`-binding delivery form makes the Offset-base scan vacuous, and the recipe could say so.** *Offset base pointers* is written over "address RTAs" that you "resolve to their host computation" — but an op entirely on the `Buffer*` push form (`RTArgList::push_back(Buffer*)`, auto-registered as a `BufferBinding`) has **no host expression at all** to inspect: the framework substitutes the base at enqueue time, so a fold is structurally impossible, not merely absent. One sentence in the recognition bullets — *"a `Buffer*`-form RTA cannot carry a fold; the subject is GREEN by construction for those bindings"* — would let an auditor close the subject with certainty rather than by exhaustive grep. (Same reasoning covers *TensorParameter analysis*, which already handles the form, and it is arguably why this op is clean today: #51747 converted the folds out and the form makes regression impossible.)
4. **The fork-existence check's locational rule is right, but the choice can hinge on *which legacy copy* you bind.** `eltwise_copy.cpp` exists as **two** legacy copies (`ttnn/cpp/ttnn/kernel/compute/` and `data_movement/sharded/device/kernels/compute/`), each with its own `_metal2` fork, and the two forks have **incompatible interfaces** — one reads `per_core_tile_cnt` as a `constexpr` compile-time arg, the other as a runtime arg. The locational check picks the right one here, but only because it is anchored to the copy this op binds; an auditor who greps by stem finds two equally plausible forks and no rule for choosing. Suggest adding: *"if the stem appears in more than one directory, the fork that counts is the sibling of the copy your factory actually binds — and run the fit check, since duplicate forks can diverge in arg kind, not just names."*
5. **Minor, for the *Feature compatibility* `address_offset` entry.** This op builds every CB through one local helper that constructs a `CBDescriptor` field-by-field (factory:30-48). Establishing "`address_offset` is statically zero" therefore means reading the helper, not grepping the call sites — a one-line recognition hint ("where a factory funnels CB construction through a helper, the field set is the helper's, not the call site's") would save the next auditor the detour.
