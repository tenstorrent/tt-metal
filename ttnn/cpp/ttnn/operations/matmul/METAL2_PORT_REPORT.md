# Port Report — `matmul` / `MatmulMultiCoreReuseMcast1DProgramFactory`

## Outcome

**`CAPITULATED`** — on a framework gap, not on the conversion. The conversion itself is **complete
and green**: `MatmulMultiCoreReuseMcast1DProgramFactory` is on Metal 2.0
(`ProgramSpecFactoryConcept`), both descriptor paths (`mcast_in0` and `mcast_in1`), together with the
six kernel entry points it binds — one reused `_metal2` fork and five created. **483 pytest passed /
0 failed** across four matmul suites and **23 of 24** matmul gtests pass, all with the Metal 2.0
legality checks forced on.

**One reachable configuration cannot be expressed today** and fails at spec validation, so the
factory is **not landable as-is**: `mcast_in0` with **in0 block-sharded over more cores than have
output work** (`in0_is_sharded && in0_sender_num_cores > num_cores_with_work`). The factory then puts
an in0 sender on nodes where compute does not run; that sender self-loops `in0` (it pushes, then pops
its own tiles to keep the write pointer in lockstep, because it multicasts *from its own write
pointer*). So `in0`'s CONSUMER is `compute` on some nodes and a data-movement kernel on others — and
Metal 2.0 requires one kernel *kind* per DFB role. This is the **same blocker already documented on
Port 5** (`MatmulMultiCoreReuseMcast2DProgramFactory`), with a written-up analysis and a proposed
fix; details and why there is no workaround are under *Handoff points → 7*. Caught by
`MatmulSmoke.WidthSharded1DAuto`.

Everything else in the diff is finished work and reviewable now; it lands unchanged the moment the
two rules are arch-gated.

The other six factories in this device-op (`MatmulMeshWorkloadMultiCoreReuseMcast1D`,
`MultiCore`, `MultiCoreReuseOptimized`, `MultiCoreReuseMcast2D`, `MultiCastDRAMSharded`,
`BatchedHSDRAMSharded`) and the sparse device-op's factory are untouched and stay on their current
concepts; four of them are already ported. The variant dispatches per factory at runtime, so the op
builds and runs throughout.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
  (read out of `origin/akertesz/op-porting-recipe`; the docs are not on this branch, so the command
  in the recipe prints nothing from this checkout.)
- **Audit docs (inherited):** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`
- **Base commit:** rebased onto `origin/main` @ `061086bbdec`. The audit was written against
  `beb2ea8f08a` (619 commits earlier); the factory `.cpp`/`.hpp` are byte-identical between the two,
  so every line reference in the audit still resolves.

## TTNN ProgramFactory

### Concept realized

**`ProgramSpecFactoryConcept`** (base) — **not** the `CustomProgramSpecFactoryConcept` the readiness
sheet names as `Porting Target`. The deviation was surfaced to the invoker before it was taken and
explicitly approved; it is argued in full in `METAL2_PORT_PLAN.md` under *TTNN ProgramFactory*.
Summary:

- The sheet's target follows the recipe's selector — *does the ported-from factory have an
  `override_runtime_arguments`?* The struct does. Reading the method rather than its presence
  changes the answer: its second parameter is `shared_variables_t`, which `create_descriptor` never
  produced, so **nothing on the descriptor path could ever call it.** It exists for the sibling
  MeshWorkload factory in the same `.cpp` and for `all_gather_matmul_async`.
- Walking both legacy override helpers statement by statement, **every write is a tensor address**
  and there is no non-tensor refresh at all. A `ProgramRunArgs`-returning translation would have
  carried `tensor_args = {in0, in1, output, [bias]}` and an empty `kernel_run_args` — byte-for-byte
  what the base concept's framework hit path does for free. **Cache-hit behaviour is identical**;
  no refresh was dropped to reach the simpler concept.
- The custom concept is keyed on `&T::override_runtime_arguments` (`operation_concepts.hpp:111-115`),
  so an overload is ill-formed and would leave the factory on the base concept *silently*. Taking it
  therefore required **deleting** the void method and editing its two callers — one of them outside
  this op's directory, i.e. across the port's scope boundary. The base concept keeps the entire diff
  inside `operations/matmul/`.
- `operation_concepts.hpp:108` anticipates this exact shape: *"the legacy void-returning
  `override_runtime_arguments` (some matmul factories) doesn't match"*.

**→ Sheet action requested (see Handoff points):** flip this row's `Porting Target` to
`ProgramSpecFactoryConcept`, and consider whether the column should be derived from the override's
*signature and reachability* rather than its presence — the same reading likely applies to the
Mcast2D row, whose override has the identical shape.

### Device-op-class edits

- **Pybind entry points removed:** `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:1239-1253` —
  the `nb::class_<ttnn::prim::MatmulMultiCoreReuseMcast1DProgramFactory>` block, whose only member
  was `create_descriptor`. Mandatory: the method no longer exists. See Handoff points.
- **Factory parameter dropped:** `create_descriptor`'s fourth argument,
  `const std::optional<CoreRangeSet>& core_range_set`, was accepted and ignored (spelled
  `/*core_range_set*/`). The fixed `create_program_artifacts` signature cannot carry it and nothing
  read it, so it is gone with the method (device-op-class exception 2).
- **Custom `compute_program_hash`:** none. The device-op declares `compute_descriptor_program_hash`
  (`device/matmul_device_operation.hpp:50`), deliberately *not* named `compute_program_hash`, so the
  framework uses its default reflection hash. Untouched.
- Nothing else in the device-operation class was edited.

### Open items

- **Relaxation candidates:** none identified. `TensorParameter` matching left strict.
- **`get_dynamic_runtime_args`:** absent, as the sheet records.

## Handoff points

1. **Readiness sheet — `Porting Target` for this row.** *(owner: sheet owner / Diego)* The cell says
   `CustomProgramSpecFactoryConcept`; the port landed on `ProgramSpecFactoryConcept` for the reasons
   above. Also worth noting the same sheet had `Is able to port? = no` / `Smuggled pointer = yes` at
   audit time (2026-09-01) and now reads `yes` / `no` — the audit's disagreement box was resolved in
   the code's favour, which is what unblocked this port. Thank you.

2. **Removed pybind surface.** *(owner: TTNN / downstream Python consumers)*
   `matmul_nanobind.cpp` — `MatmulMultiCoreReuseMcast1DProgramFactory.create_descriptor` is gone from
   the Python module. It exposed the legacy factory's `ProgramDescriptor` construction for
   introspection/testing; the Metal 2.0 factory has no equivalent entry point, and its
   `create_program_artifacts` has a different signature and return type, so it was **not**
   retargeted. The sibling `MatmulMultiCoreReuseMcast2DProgramFactory.create_descriptor` binding is
   untouched. A repo-wide grep found no Python caller, but a notebook or internal tool could have one.

3. **`ENABLE_GLOBAL_CB` is not carried into the in1 sender/writer fork.** *(owner: Metal 2.0 /
   whoever ports a GlobalCB consumer)* `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` holds a
   global-CB ("remote CB") path — `api/remote_circular_buffer.h`, `remote_cb_id = c_31`,
   `remote_cb_wait_front` / `remote_cb_pop_front` / `update_remote_cb_config_in_l1` — entirely behind
   `#ifdef ENABLE_GLOBAL_CB`. A GlobalCircularBuffer is **not** a `DataflowBuffer` and
   `GlobalDataflowBuffer` is not implemented, so the region has no Metal 2.0 spelling; carrying it
   would mean carrying a raw CB index into a kernel that has none. The fork omits it. Nothing
   regresses: this factory never defines the macro (factory selection routes GCB configs away from
   it, `matmul_device_operation.cpp:2204`), and the only setters are the legacy mcast-1d MeshWorkload
   paths and `llama_all_gather_matmul_async`, neither of which can bind a Metal 2.0 fork. The legacy
   original keeps serving them unchanged.

4. **The DRAM-sharded in1 readers are not carried into the fork either.** *(owner: whoever ports
   `MatmulMultiCoreReuseMcast2DProgramFactory`)* `IN1_DRAM_WIDTH_SHARDED` /
   `IN1_DRAM_HEIGHT_SHARDED` are set **only** by the mcast-2d factory, which has not ported. Both
   regions use the in1 base address as a raw pointer with explicit bank arithmetic (a Case 2 binding
   needing `TensorAccessor::get_bank_base_address()`) and walk per-bank stride/id lists that become
   runtime varargs. Converting them with no consumer able to exercise them would ship untested
   address arithmetic, so they are left out for the mcast-2d porter to add — at which point the fork
   gains a second consumer and becomes read-only to everyone.

5. **CCL fused-op call sites cannot cross to named arguments.** *(owner: kernel-lib / CCL)*
   `MatmulOpReceiver` and `OpSignaler` (`ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp`,
   **outside** this op's directory) consume **positional** runtime args through a `uint32_t&` index
   they advance. Metal 2.0's named-argument channel cannot feed that, and the fix is upstream of the
   porter. This is not a blocker here — `create_program_artifacts` passes
   `fused_op_signaler = std::nullopt` (inherited from `create_descriptor`), so the path is
   unreachable and the forks `#ifdef`-gate it, with a `TT_FATAL(!fuse_op, …)` on the host so it
   cannot be reached silently. It **is** a blocker for anyone porting a CCL-fused matmul consumer,
   which is why it is recorded here.

7. **PORT CAPITULATION — a DFB whose consumer *kind* varies per node cannot be expressed.**
   *(owner: Metal 2.0 runtime — this is the blocker that makes the port unlandable)*

   - **Op / factory:** `ttnn/cpp/ttnn/operations/matmul`,
     `MatmulMultiCoreReuseMcast1DProgramFactory`, `create_program_mcast_in0_artifacts`.
   - **Trigger:** `in0_is_sharded && in0_sender_num_cores > num_cores_with_work` — in0's shard grid
     covers more cores than the output's work grid, so the factory instantiates
     `in0_no_work_in_receiver` / `in0_no_work_not_in_receiver` senders on cores that run neither
     compute nor the writer. Reachable and covered: `MatmulSmoke.WidthSharded1DAuto`.
   - **Why the conversion cannot express it.** On those nodes the in0 sender is the *only* toucher and
     does both halves — `reserve_back` / `push_back`, then `pop_front` under
     `if constexpr (!core_has_output_block_work)`. That is a textbook self-loop, and the per-node
     census is a clean 1P+1C on **every** node. But `in0`'s CONSUMER role then holds `compute` (on
     with-work nodes) and a DM kernel (on no-work nodes), and
     `program_spec.cpp:1366-1377` requires per-role kernel-kind uniformity — *"the DFB's hardware
     config carries a single processor mask per role."* This is not validator overreach:
     `dataflow_buffer_spec.hpp:41-51` states the same three conditions for binding one role from
     several KernelSpecs (non-overlapping nodes, **same kernel kind**, identical binding-site params).
     The topology is legal on Gen1, where a DFB lowers to a plain circular buffer with no mask at all.
   - **Why there is no workaround, and why `allow_instance_multi_binding` is the *wrong* tool.**
     (a) A private DFB for the no-work cores decouples the address, and the multicast writes to the
     *sender's own* `get_write_ptr()` as the destination address on every receiver — so a separate
     allocation breaks the mcast. (b) `alias_with` is rejected too: alias members must share a node
     set (`advanced_options.hpp:170`), and these don't. (c) The flag does suppress rule 1, but only
     as a side effect of disabling per-role config — it would assert a per-node multiplicity that
     does not exist and forfeit the FIFO guarantees this kernel's lockstep-write-pointer argument
     depends on. And it does not even work: the self-loop set-equality rule
     (`program_spec.cpp:1507-1522`, `producer_kernels == consumer_kernels`) sits **outside** the
     `if (!allow_multi)` guard at `:1380` and rejects the program anyway.
   - **Suggested fix (unchanged from Port 5):** arch-gate both rules to Gen2, exactly as the DM
     self-loop check at `:1494` already is — a few lines — or evaluate the kind condition per node.
     Rule 2's exclusion from the `allow_multi` guard looks like an oversight given it shares rule 1's
     rationale.
   - **Prior art:** identical shape hit on `MatmulMultiCoreReuseMcast2DProgramFactory` (Port 5); full
     write-up at `/localdev/iwrosz/metal2-dfb-endpoint-rules-bug.md` and a proposal at
     `/localdev/iwrosz/metal2-dfb-endpoint-rules-proposal.md`. **Two independent matmul factories now
     block on this one rule pair**, which is worth weighing when prioritising it.

8. **`./build_metal.sh` exits 0 on a failed ninja build.** *(owner: build tooling)* A compile error
   left `ninja: build stopped: subcommand failed.` in the log while the wrapper returned **exit 0**.
   A porter following the recipe's "run it in the background, read the exit code" pattern would read
   that as a green build and go on to test a stale binary. Detected here only because the log was
   grepped for `error:` as well. Suggest the recipe's log-reader prompt add "`ninja: build stopped`"
   to what it looks for, independent of exit status.

## Successes

- **Forcing the legality checks paid for itself, immediately.** The recipe's
  *Ensure the Metal 2.0 host-side legality checks are enabled* step caught a real spec bug on the
  first hardware run: `Aliased DFBs 'out' and 'intermed0' have inconsistent borrowed_from`
  (`program_spec.cpp:1713`). In the legacy shared-buffer branch a **single** `CBDescriptor` carries
  `c_4`, `c_5` and optionally `c_7` with one `tensor` field backing the whole region, so when the
  output is sharded all three are backed by the output tensor — but translating index-by-index makes
  it natural to put `borrowed_from` only on the output's own spec. The validator rejected it in the
  one configuration that exercises it. Without the force this would have shipped as a wrong backing
  address for the partials buffer on sharded-output configs.

- **The two-toucher / self-loop census caught a dead CB the audit had listed as live.** The audit
  flagged `c_6` ("Local L1 to store temp vars") as *"the one CB here most likely to fall outside the
  plain 1:1 shape"* and asked for a toucher census. The census came back **zero**:
  `grep -rn cb_l1_array …/kernels/` returns nothing, so no kernel ever read the index the host was
  handing it. Per the recipe that is a **dead-CB drop**, not a self-loop — no spec built, dead CTA
  dropped, 64 bytes per core reclaimed with no behaviour attached.

- **`Caution: Porting a shared kernel`, rung 1, worked exactly as designed.** The compute kernel
  already had a `_metal2` fork (created by #55961 three weeks after the audit was written, which is
  why the audit says none existed). Its `dfb::` vocabulary — including
  `dfb::intermed0_reload_alias` and the `MM_PARTIALS_RELOAD_ALIAS` flag define — turned out to be an
  *exact* fit for what this factory needed, down to the conditional-binding pattern the audit had
  predicted this port would have to invent. The fork's names became the constraint and the port
  renamed on its own side. Zero friction; the convention paid off across two independent ports.

- **The `Anti-pattern: Demoting per-group CTA to RTA` entry fired correctly.** The block-sharded in0
  reader is instantiated three times over disjoint core sets, differing only on two `if constexpr`
  flags. Reaching for one `KernelSpec` with those flags demoted to RTAs was tempting and would have
  cost the kernel its compile-time specialization. Three `KernelSpec`s in three `WorkUnitSpec`s over
  disjoint nodes is a legal single-role binding, not a multi-binding — the catalog says so
  explicitly and it is correct.

## Friction

### Gaps

- **The recipe has no procedure for a port whose audit came back RED and was later unblocked
  sheet-side.** The precondition is "the audit produced an **overall GREEN** result", and this one
  says RED in its Status summary — while its own disagreement box predicted, correctly, that the
  gate would clear *with the op's code untouched*, which is exactly what happened. So the artifact
  on disk fails the precondition while the condition it gates on is satisfied. No brief was issued
  either (briefs are only issued on a clear), so the port's designated actionable input did not
  exist. What made this recoverable is that the auditor ran the seven informational subjects anyway
  under the Red-outcome scoping exception, so the *Port-work summary* stood in for the brief almost
  perfectly. Suggest the recipe say what to do here: re-fetch the sheet, confirm the gate has
  lifted, record the before/after cells, and proceed from the audit's port-work section — rather
  than leaving the porter to decide whether a RED artifact still binds.

- **Nothing tells the porter that a `_metal2` fork may have appeared *since* the audit.** The audit
  is explicit that none of the six kernels had a fork and that the port therefore "creates six
  forks"; by the time the port ran, one existed. The rung-1 check is defined as a locational `ls`,
  which found it — but only because the check was run rather than the audit trusted. Suggest the
  shared-kernel Caution add a line: *the audit's fork inventory is a snapshot; re-run the rung-1
  check per kernel at port time.* Cheap, and it is the difference between reusing a fork and
  shipping a duplicate.

- **"While a test run is in flight, kernel sources are frozen" — the warning is correct, prominent,
  and I walked into it anyway.** During a broad pytest run I applied the self-audit's `cb_*` → `dfb_*`
  rename, which touched a kernel fork's named argument (`in0_reuse_in_CB` → `in0_reuse_in_dfb`). The
  host binary in the running process still emitted the old name; the JIT compiled the just-edited
  source; every program constructed after that point died with
  `'in0_reuse_in_dfb' is not a member of 'args'` — 442 occurrences, presenting as a wall of failing
  tests that looked exactly like a broken port. Diagnosis took a couple of minutes only because the
  recipe describes this failure precisely.

  What made it easy to fall into: the recipe frames the hazard around *converting* kernels ("park the
  kernel edits until the run exits"), and by that point conversion was long finished — the edit was a
  cosmetic rename from the verification checklist, which does not feel like kernel work. Suggest the
  freeze rule say so explicitly: *this includes cosmetic renames from the anti-pattern self-audit —
  any edit to a kernel source or a header it includes, for any reason.* The self-audit section could
  carry a one-line pointer back, since that is the step most likely to produce a late kernel edit.

### Confusion

- **"Aliased DFBs must be bound to the same set of kernels" (catalog) vs "all members must target
  the same node set (derived from their bound kernels' WorkUnitSpecs)" (`advanced_options.hpp:170`).**
  The alias group here is {`out`, `intermed0`}(+`intermed0_reload_alias`) in the shared-buffer
  branch, and those are *not* bound to the same kernels — `out` is produced by compute and consumed
  by the in1 writer, while `intermed0` is touched only by compute. Under the catalog's wording the
  port would have had to bind the partials buffer to the writer as well, purely to satisfy the rule,
  which would be a fabricated endpoint. The header's wording is satisfied as written, because both
  kernels' WorkUnitSpecs cover the same nodes. Went with the header per *go to the headers first*.
  Suggest aligning the catalog's phrasing to the header's.

- **`SemaphoreSpec` has no `initial_value`, and the legacy descriptors set `INVALID`.** A moment's
  worry that the port could not express the initial value, until `INVALID == 0`
  (`hostdevcommon/common_values.hpp:13`) and Metal 2.0's default is 0 — so the faithful translation
  is to set nothing, and the `initial_value` advanced option is both deprecated and unnecessary
  here. Worth one sentence in the migration guide: *legacy `INVALID` is zero; a semaphore initialized
  to `INVALID` needs no advanced option.*

## Open items for downstream

### Shared kernel touches

All six kernel sources this factory binds are shared and none could be converted in place. Five live
in matmul's own directory and are bound by the mcast-2d factory, the sparse factory, and/or the
MeshWorkload sibling in the same `.cpp`; the sixth is bound by five other matmul factories. No peer
op's directory was written — every fork landed beside its original.

| kernel (under `device/kernels/`) | rung | fork path | remaining unmigrated consumers |
|---|---|---|---|
| `compute/bmm_large_block_zm_fused_bias_activation.cpp` | **1 — reused** (no new file) | `…_metal2.cpp` (created by #55961) | Mcast2D, Sparse, MeshWorkload sibling. *(Optimized / McastDRAMSharded / BatchedHS already bind the fork.)* |
| `dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` | **2 — created** | `…_in0_sender_padding_metal2.cpp` | Mcast2D, Sparse, MeshWorkload sibling |
| `dataflow/…_in0_sender_receiver_padding_block_sharded.cpp` | **2 — created** | `…_block_sharded_metal2.cpp` | Mcast2D, MeshWorkload sibling |
| `dataflow/reader_bmm_tile_layout_in0_receiver.cpp` | **2 — created** | `…_in0_receiver_metal2.cpp` | Mcast2D, Sparse, MeshWorkload sibling |
| `dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | **2 — created** | `…_in1_sender_writer_padding_metal2.cpp` | Mcast2D, Sparse, MeshWorkload sibling |
| `dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | **2 — created** | `…_in1_receiver_writer_padding_metal2.cpp` | Mcast2D, MeshWorkload sibling |

The pointer comment landed in all five legacy originals. **The five new forks are now read-only to
later porters** — the next factory onto them adopts their `dfb::` / `tensor::` / `args::` names and
renames on its own side. Note in particular that the in1 sender/writer fork is missing two regions
the mcast-2d factory will need (handoff points 3 and 4); adding them is that porter's work, and at
that moment the fork stops being read-only for exactly that change.

### Notes for the next matmul porter

- **The binding vocabulary is now set for six kernels.** `dfb::{in0, in1, in0_sharded, out,
  intermed0, intermed0_reload_alias, bias, in0_transposed, sparsity}`,
  `tensor::{in0, in1, out, bias, sparsity}`, `sem::{in0_mcast_sender, in0_mcast_receiver,
  in1_mcast_sender, in1_mcast_receiver}`. These were chosen from each kernel's own role vocabulary
  rather than from this factory's locals, per the catalog. The audit's question 2 — *"fork vocabulary
  should be agreed before any matmul factory ports"* — is answered de facto by this port plus #55961;
  worth a look before Mcast2D ports, since it inherits all six.

- **Conditional bindings are pervasive here and all follow the `#ifdef` pattern**: `SPARSITY`,
  `FUSE_BIAS`, `IN0_SHARDED`, `EXTRACT_SHARD_SUB_BLOCKS`, `IN1_SHARDED`, `BIAS_SHARDED`,
  `OUT_SHARDED`, `SKIP_MCAST`, `IN0_TRANSPOSE_TILE`, `MM_PARTIALS_RELOAD_ALIAS`, `FUSE_OP*`. Two of
  these were `if constexpr` gates on a CTA in the legacy kernel and had to be **promoted** to the
  preprocessor because the discarded branch still name-looks-up an unbound `dfb::`/`tensor::` token:
  `extract_shard_sub_blocks` (which selects between the in0 and in0_sharded buffers) and
  `in0_transpose_tile` (in the compute fork, already done there). Expect more of the same in Mcast2D.

- **One genuine vararg retained.** The block-sharded in0 sender's per-sender mcast NOC coordinate
  lists (`in0_mcast_noc_x`, `in0_mcast_noc_y`) are indexed collections whose lengths are compile-time
  args rather than source literals, so they stay `advanced_options.num_runtime_varargs`, laid out
  x-list then y-list. Every other argument on all six kernels is named — including the long
  `rt_args_idx++` runs at the top of the in0/in1 readers, which are distinct fields read once each,
  not varargs.

### Findings — behaviour preserved, reported not repaired

- **A resolved compute-config field is silently dropped, and the port preserves that.**
  `create_descriptor` resolves the full five-tuple including `dst_full_sync_en`, but **neither**
  descriptor builder took it as a parameter or set it on the `ComputeConfigDescriptor`, so the
  descriptor default applied and this op has always ignored the knob whatever a caller passes.
  `to_compute_hardware_config` reads the *resolved* config and would have handed the caller's value
  back — changing behaviour — so the port explicitly pins `double_buffer_dest = true` (the
  `dst_full_sync_en = false` legacy-default result). **The ops team may want to know this knob is
  inert on this factory**; fixing it is a separate change with its own review.

- **Dangling CB indices handed to kernels that never read them.** Beyond the dead `c_6` above, the
  host passed `cb_in0_sharded` (`c_2`) to the *interleaved* in0 reader where no such CB exists, and
  `cb_sparsity` as `c_6` (in0 sender) and `c_7` (in1 sender/writer) — neither of which is a sparsity
  buffer; `c_7` is the partials alias. All were legacy placeholders for a disabled feature; they are
  gone with the port, with no behaviour attached.

- **Dead placeholder `.address()` calls.** The descriptor builders computed buffer addresses that
  were immediately overwritten by the variant rebinding a few dozen lines below. Harmless, but they
  are what a smuggled-pointer sweep finds first, and they cost this audit a full disagreement box
  and this port a re-verification before the sheet was corrected. They are gone with the port on
  this factory; the **sibling** factory in the same `.cpp` still has 21 of them.

### Test coverage notes

- `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_1d_2d.py` is the primary gate and
  exercises both paths plus sharded/bias/fp32 variants. `test_matmul_block_sharded_1d_grid.py`,
  `test_matmul.py`, `test_matmul2.py` and `test_bert_matmuls.py` cover the op more broadly.
- **The in0-sharded `EXTRACT_SHARD_SUB_BLOCKS` sub-path of the `mcast_in1` builder has thin
  coverage** — it needs `in0_shard_width_in_tiles / in0_block_w > 1` or a 2-D block split, which
  few configurations in the suite hit. Worth a targeted case before the next factory leans on that
  fork.
