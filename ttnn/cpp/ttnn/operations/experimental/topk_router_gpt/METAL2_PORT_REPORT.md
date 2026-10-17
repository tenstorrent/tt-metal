# Port Report — topk_router_gpt

## Outcome

**PORTED — NOT YET VERIFIED.** The single factory (`TopkRouterGptProgramFactory`, new) and all three kernels are converted. The user asked for the commit while the post-port build was still running (281/963). So this commit has **not been compiled, and the op's tests have not run against it**. Verification still owed:

- `./build_metal.sh --build-tests` green.
- `tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_router_gpt.py` on n150 (WH) with Watcher on, plus `METAL2_CHECKS_FORCED` markers in the log.
- A bit-exact diff of raw outputs against the pre-port capture: 8 cases, DRAM and L1, k = 3/4/8, B = 1..32, each run twice so the second call is a cache hit with shifted allocations.

**Pre-port baseline (captured):** 41 passed, 0 failed, on n150, Watcher on; capture run reports 8 program-cache entries for 8 cases.

BH P150 is not available in this workspace (one n150 only), so the BH config (8 cores, 1 sender/group) is unverified. See Open items.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` (from the `/localdev/edwinlee/Port_Recipe` checkout; this checkout has no recipe tree)
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. No `override_runtime_arguments`; the framework refreshes the five tensor bindings on a cache hit.

### Device-op-class edits
- **Exception 3 (direct-descriptor op):** `device/topk_router_gpt_device_operation.hpp`. The device-op-level `create_descriptor` is replaced by a nested `TopkRouterGptProgramFactory::create_program_artifacts` plus `using program_factory_t = std::variant<TopkRouterGptProgramFactory>`. Include swap: `program_descriptors.hpp` → `metal_v2_artifacts.hpp` and `<variant>`. The header comment is reworded from "runtime-arg bindings in create_descriptor" to "tensor bindings in create_program_artifacts".
- Pybind entry points removed: none (`topk_router_gpt_nanobind.cpp` binds only the user function).
- Custom `compute_program_hash`: none.

### Open items
- Relaxation candidates: none. Logical B varies 1..32 while the padded shape is fixed at 32, so each B compiles its own program today. A `match_padded_shape_only` relaxation on `input` / `bias` *might* be tolerated by the kernels, which never read logical B; that is for the op owner, not the port.

## Handoff points

- **Op arrived in the direct-descriptor shape** (from PD batch #57409). Converted to a conventional factory struct per ttnn_factory.md §3.
- **Readiness-sheet gate waived (user, 2026-10-06).** The sheet row still shows `legacy device-op` and a phantom `TopkRouterGptProgramFactory` at a deleted `.hpp`. After this port the struct exists again, with the same name, as a `ProgramSpecFactoryConcept` factory in `device_operation.hpp`. The sheet refresh is still owed to the readiness-sheet owner.

## Successes

- **Brief's uniform-L1-offset warning.** The brief flagged that dm1 uses its own `partial_recv` / `gathered_val` / `gathered_ind` write pointer as a remote NoC destination. I confirmed in `program_spec.cpp` that DFBs are created in `spec.dataflow_buffers` order over the bound kernels' node set. So one work unit on `all_cores` plus legacy CB order gives an identical layout on every core. Without that warning, per-role work units would have looked like the "faithful" way to keep the legacy narrower placement.
- **Census re-derivation matched the brief** exactly: 11 × 1P+1C; self-loops on `index` and `dispatch` (dm1) and on `intermed_val`, `intermed_ind`, `softmax_tmp`, `reduce_scalar` (compute); no multi-binding.
- **compiler_options rule.** The factory set no `opt_level` anywhere, so compute would have silently dropped to O2. It is set explicitly: O2/O2/O3 (`program_factory.cpp:265, 307, 407`).
- **LLK handle pass-through.** I checked `compute_kernel_hw_startup.h` before relying on it: only `uint32_t` overloads exist, so passing `dfb::` tokens selects the same overload as the legacy ids.

## Friction

**Gaps**
- **Headers vs recipe naming.** The recipe names `ComputeGen1Config` / `DataMovementGen1Config` and `std::get<ComputeGen1Config>(compute_hw)`. The headers have flat `ComputeHardwareConfig` / `DataMovementHardwareConfig` with optional `config_1xx` / `config_2xx`, and the TTNN DM helpers (`create_reader_datamovement_config`) take no `arch` argument. I resolved this from the headers: Style B compute config is `ComputeHardwareConfig{HiFi2, Precise, enable_32_bit_dest = true, double_buffer_dest = true}`, with `config_1xx` unset because its default `bfp_pack_precision_mode = Approximate` equals legacy `bfp8_pack_precise = false`.
- **Unbound dead CB wrapper in a kernel.** `compute.cpp` declared `CircularBuffer cb_index(cb_index_id)` and never used it. Compute has no binding to `index` (binding it would make three touchers), so there is no `dfb::` token to swap to. I deleted the declaration and its constant. Rule 1 doesn't say what to do with a wrapper whose CB the kernel never touches.
- **Shared checkout, two porters.** First, a Debug `build_metal.sh` started from another shell mid-port, and my Release configure failed at the same moment on a missing umd source (likely a submodule refresh race). I waited, per the user. Second, a parallel session committed `67f73b4c768` [concatenate_heads] to this branch at 20:00, and around then my uncommitted `skip_validation` force and markers in `tt_metal/impl/metal2_host_api/` disappeared from the working tree. I reapplied them before the post-port build. The recipe's force-and-prove step assumes one porter per checkout; with two, one porter's scaffolding cleanup silently un-forces the other's checks.

**Confusion**
- **Orphaned `aligned_page_size` RTA.** The brief said to remove it "only if the recipe's rules on orphaned args allow it". Dropped Plumbing covers it: "a `page_size` value emitted solely to feed a `TensorAccessor`'s third constructor argument" is dropped. Its only consumer was the 3rd arg at `dm1.cpp:343-344`, so the RTA, its host computation, and the stale "may be stale on program cache hits" comment are gone.
- **Environment.** The shell had `TT_METAL_DPRINT_CORES=all` exported. I unset it for both baseline and post-port runs.

## Open items for downstream

- **Verification above is outstanding** (build, pytest, bit-exact diff, checks-forced markers).
- **BH P150 untested here.** The config differs only in values (`num_senders` = 1 → `partial_recv` depth 1, collector at ring position 1). It should be run on a P150 before merge (brief: #58472 validated 41 tests per arch).
- **L1 footprint change (audit Q2, user not ruled).** Every DFB is now on every core. Sender cores gain bias, index, topk_val, gathered_val ×4, gathered_ind ×4, intermed_val ×2, intermed_ind, softmax_mask, softmax_tmp, reduce_scalar, bcast_scaler, final_out ×2 (19 tiles × 2 KiB = 38 KiB) plus `dispatch` (`64·k_padded·2` B, 1 KiB at k ≤ 8). Non-collector workers gain the 10 collector tiles + dispatch. The per-program high-water mark is unchanged: the collector already held all 17 at the same offsets, and that max is what the L1-buffer clash check sees.
- **Host role sets dropped.** `sender_cores` / `worker_cores_vec` / `collector_cores_vec` and the derived `CoreRangeSet`s (legacy `program_factory.cpp:136-151`) existed only to size the CB core ranges. With derived placement they had no consumer, so they went with the CBs. The ring sort, collector selection, k-split and vchannel table are unchanged.
- **Carried-over anomalies (not fixed):**
  - read-but-unused RTAs (`dram_bank_id`, `vchannel`, and most role fields in compute);
  - dead named CTAs `num_cores` and `cores_per_group`;
  - RTA node set (`required_cores`) vs placement (`all_cores`) equal only on 12- and 8-core devices;
  - `weights_rm` relied on `indices_rm`'s page size (now moot: each accessor carries its own).
- **Gen2 debt, self-documenting:** DM self-loops on `index` and `dispatch` (dm1).
- **Sibling ops:** none shared; all three kernels are private to this op.
