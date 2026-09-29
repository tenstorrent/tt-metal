# Self-Reflection: high_bw_all_reduce

## Summary
- **Final blind: 400/400 pass.** 384 golden cells (192 bf16 + 192 fp32) + 16 regression cells. 0 fail, 0 hang, 0 `supported_fail`, 0 `xpass_drift`. There was no translated suite in the blind dir; it holds only `test_golden` + `test_regression`.
- SUPPORTED equals TARGET except for the per-axis Ring cells. Those are in EXCLUSIONS and are also not collected on the 2×2 box, so none were run.
- There is no failure cluster to explain. The most important finding is a **coverage blind spot, not a bug**: every golden cell ran on a 2×2 mesh, so each per-axis group had `G = 2`. The `middle` role and the `G ≥ 3` split-port path on axis lines were never exercised, yet `SUPPORTED` claims `cluster_axis ∈ {0,1}` without that condition. Nothing in the axis model can express that facet.
- The perf phase ran: 2 ideas, both null, nothing graduated. The helper-bypass table has 2 rows (one `missing`, one `too-hard`), and I found no unrecorded bypass in the op.
- The remaining problems are framework-level: fabric teardown under `--dev`, no fabric-bandwidth ceiling reference, and regression tests that never reach the snake/ring routes.

## 1. Golden coverage → `eval/golden_tests/high_bw_all_reduce/feature_spec.py`

No blind failures, so there are no failure clusters. The findings below are gaps found by absence.

**1a. The exact-sum regression tests never reach the routes with the most routing logic.**
- **What:** `test_rank_identity` and `test_single_contributor` are the only exact checks that catch a dropped, doubled or misrouted contribution. They are pinned to `Linear`, `num_links=1` and `cluster_axis ∈ {0,1}`, which is `G = 2`, head + tail only. So `cluster_axis=None` (snake line/ring, with middles, slices, split ports and credit-batch gates) and 2-link lanes get only the random-data PCC check. A chunk-level misroute touching a few percent of elements can stay above the bf16 PCC floor of 0.995.
- **Evidence:** `test_regression.py:33` `_AXES = [a for a in (0, 1) ...]`; `:39` `topology=ttnn.Topology.Linear, num_links=1`. Refinement 1 had to fix a real ring routing issue: breadcrumb `ttnn-implementer#7` "ring fixed via per-reducer port service".
- **Recommendation:** add `cluster_axis=None × {Linear, Ring} × num_links {1,2}` to `test_rank_identity` and `test_single_contributor`, filtered by `topology_feasible`. This is 8 extra cells per shape; the exact-integer check gives the snake/ring routes the proof of which device got which data that PCC does not.
- **Confidence:** med

**1b. Group size / mesh shape is an axis-blind facet (device-derived, not shape-derived).**
- **What:** on the 2×2 box every per-axis cell has `G = 2`. The kernels branch on `G`: the `middle` role, and `_split_ports` returning 0 unless `G ≥ 3`. So cells tagged "cluster_axis=0 / 1" look covered, but the `G ≥ 3` axis-line path has never run. No tagger or observed axis records `G`.
- **Evidence:** `high_bw_all_reduce_program_descriptor.py:209` `return SPLIT_PORTS if group_size >= 3 else 0`. `verification_report.md:3` "Every group has `G = 2` (head + tail)". `op_design.md:182` "R1 with `G ≥ 3` (middle role) runs on T3K / Galaxy".
- **Recommendation:** have `axes.py`'s observe wrapper record a derived `group_size` (or `mesh_shape`) observed facet. Consider promoting it to an observed-only axis so `verify_supported` can report which `(cluster_axis, G)` pairs are evidence-backed. A LOOSE_CASE cannot fix this on a 2×2, so it has to come from the harness.
- **Confidence:** med

**1c. No small or sub-chunk shapes in golden.**
- **What:** every `INPUTS` shape has ≥ 4M elements. TARGET has no size axis, so tiny tensors are in-TARGET. These include shapes with fewer tiles than `lanes × REDUCERS_PER_LANE` (reducers with zero blocks) and a single tile. They were exercised only by the op's own unit tests (Perf 1 guard set "single tile 14.1 µs"), not by golden.
- **Evidence:** `feature_spec.py` INPUTS comment "Per-device shapes, 4M-32M elements each". The file has no `LOOSE_CASES`.
- **Recommendation:** propose `LOOSE_CASES` boundary entries, since they do not multiply the cartesian: `(1,1,32,32)`, `(1,1,32,64)` (fewer tiles than reducers) and `(1,1,33,31)` (both dims ragged inside one tile). Run each at `cluster_axis=None, Ring, num_links=2`, the deepest lane/slice fan-out.
- **Confidence:** med

**1d. No perf-flagged LOOSE_CASE.** Because of this, the perf coordinator free-selected its focus cell (`changelog.md:251` "`feature_spec.py` has no `LOOSE_CASES` perf flag"). Consider adding one `attention:` entry that states the intended bandwidth target, e.g. bf16 `(1,1,4096,4096)` None-Ring 2 links. Confidence: low (this is a product choice).

## 2. SUPPORTED honesty → `high_bw_all_reduce.py` `SUPPORTED` / `EXCLUSIONS`

- **Counts (blind `verifier_report.json`):** `supported_pass` 384, `supported_fail` 0, `xpass_drift` 0. `no_axes_found` 16 (all `test_regression.py`, all pass). No fix, demote or promote is indicated.
- **EXCLUSIONS `{Ring, 0}` / `{Ring, 1}` are honest.** They are refused because no torus cluster was available to verify them (`high_bw_all_reduce.py:60-65`), and the 2×2 never emits them. Keep them.

**2a. `cluster_axis ∈ {0,1}` is claimed unconditionally, but only verified at `G = 2`.**
- **What:** see 1b. The verifier disclosed this honestly (`verification_report.md:235` "Middle role is untested on this box"). `SUPPORTED` itself cannot carry the condition, so a Galaxy user sees a clean claim.
- **Recommendation:** no demotion (the `None` snake does exercise `middle` at G=4 on the same kernels). Consider a registry-level "verified-on" annotation (mesh shape) next to `SUPPORTED`, or at minimum a comment on the `cluster_axis` line pointing at `verification_report.md:235`.
- **Confidence:** med

**2b. Harness nit: `test_regression.py` ignores the registry, so refused cells count as failures.** In phase 0 and R1, 12 of the 16 regression cells show `failed` with `UnsupportedAxisValue` (fp32 / `4096x2050`) instead of xfail (`golden_phase0/test_results.json`). This inflates FAILED counts in early phases (the "PASSED=52 FAILED=12" headline) with registry-correct refusals. Consider converting `UnsupportedAxisValue`/`ExcludedCell` to xfail in regression tests. Confidence: high.

## 3. Helper / reference docs

**3a. The fabric-link ceiling is undocumented, and the missing rule cost a 1.8× perf regression in the design.**
- **What:** the design defaulted to fused write + atomic-inc per packet. Per the fabric golden CSVs that runs ~6× slower than plain writes. No reference or skill points at a fabric ceiling: `/perf-ceiling-dm` covers NoC links only.
- **Evidence:** breadcrumb `ttnn-implementer#6` "NOC_FUSED_UNICAST_ATOMIC_INC packets ~6x slower than plain writes, which the design picked as the per-packet default and cost the first 1.8x". Commit `e3db9b327a1` "chunk-granular fused atomic inc (plain writes + inc on last packet)". `grep -i fabric .claude/skills/perf-ceiling-dm/SKILL.md` gives no fabric section (only NoC `link_BW`, line 251).
- **Recommendation:** add a "Fabric link ceiling" section to `perf-ceiling-dm/SKILL.md` that cites `tests/tt_metal/tt_fabric/test_infra/golden/golden_bandwidth_summary_*.csv` per arch and packet type. Add one line: "fused write+atomic-inc per packet caps a BH link at ~6 GB/s; inc once per chunk".
- **Confidence:** high

**3b. `run_safe_pytest.sh` leaves the fabric dead after a passing `--dev` run (reported independently by two agents).**
- **Evidence:** `scripts/run_safe_pytest.sh:806` resets only `if [[ "$IS_HANG" == true ]]`. Breadcrumbs `ttnn-implementer#5` and `incremental-verifier#1` say "the next run ... fail[s] at mesh open with an eth-core timeout". Commit `9162cfc08cf` "note one-fabric-module-per-session test constraint".
- **Recommendation:** reset after `--dev` runs that opened a fabric, or when the log contains "Timed out waiting for active ethernet core". Document the `/tmp/tt-device.dirty` workaround in `CLAUDE.md` § Hang triage.
- **Confidence:** high

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `compute_kernel_lib::binary_sfpu<AddBinary>` / `output(..., L1Accumulation)` (fp32 add via packer L1-acc) | capability | **missing** | L1-acc is pinned to one output tile. `chain.inl:955-957`: `walk` requires `L1AccumulationMode == Disabled`. `chain.inl:1046`: "L1 accumulation also has to stay pinned to one output tile". `chain.inl:961-965`: L1-acc requires `(OneUpfront, OneAtEnd)` or caller-managed. A per-tile, block-wide two-pass overlay cannot be expressed. Raw 2977 ns vs helper 5232 ns (1.76×) in isolation, but only 0.2–0.6% end to end, so it was not graduated. | Add a walking L1-acc mode (`L1Accumulation::AddToExisting` + `Upfront/AtEnd` window, `out_idx = i_flat`) so a block can be packed onto an already-packed block. Low priority until fp32 compute is on a critical path. |
| `compute_kernel_lib::add` / `copy` convenience wrappers (per-call init) | ergonomics ("no way to keep the init across calls") | **too-hard** (the "no way" claim is inaccurate) | `eltwise_chain<InitReconfigOwner::Caller>` exists and is documented (`api/chain.hpp:149-167`), but using it means (a) dropping from `convenience.hpp:44-86` (no `Owner` parameter) to raw `eltwise_chain`, (b) hand-writing the raw `*_init` yourself, (c) disabling operand reconfig (`chain.inl:3036`), and (d) tracking head-copy ↔ add role flips yourself. That is exactly the raw kernel. 832→748 ns bf16 add (−10%), flat end to end. | Expose `Owner` on the convenience wrappers and add an "init-if-changed" mode so alternating chains re-init only on a role change. |

```cpp
// confidence: low — derived from perf_experiments/compute_add_fast/kernels/compute_l1acc.cpp pass 2
constexpr auto out_acc = output(cb_reduced, ReservePolicy::None, PushPolicy::None,
                                DataFormatReconfig::Enabled, TileAddressing::Direct,
                                DestAccumulation::Disabled, L1Accumulation::AddToExistingWalk);
copy<in_local,   out_plain>(block);   // pass 1: seed the chunk
copy<in_partial, out_acc>(block);     // pass 2: packer adds onto tile i_flat

// too-hard row: what the caller should not own (compute_raw.cpp tracks `Mode` + raw add_init/copy_init)
InitCache cache;                                   // helper-owned "last init emitted"
if (head) copy<in_local, out_reduced>(block, cache);
else      add<in_partial, in_local, out_reduced>(block, cache);   // re-inits only when role flips
```

- **Unrecorded bypasses:** none. The op's compute kernel is helper-only. Its raw `cb_*` calls are inside `#ifdef HBAR_ABLATE_COMPUTE` (`kernels/high_bw_all_reduce_compute.cpp:59-72`), which is off by default. The dataflow/port kernels use raw NoC/fabric APIs, but the kernel library has no fabric helper, so there was nothing to bypass.

## 4. Agent prompts → `.claude/agents/*.md`

**4a. Planner/verifier: add an explicit coverage line when the box cannot reach a regime.** The planner specified regime-pinned tests for `G ≥ 3` "where the mesh permits" (`op_design.md:338`). The verifier disclosed the gap, but the 4 refinements then tuned `G ≥ 3` paths (`_split_ports`, the R4 "split ports (G>=3)" breadcrumb `ttnn-implementer#12`) that were validated only via the `None` snake. Propose that `incremental-verifier.md` require a "regimes not reachable on this box → SUPPORTED claims they back" table in `verification_report.md`, fed into the registry "verified-on" note (2a). Confidence: low-med.

**4b. `--dev` guidance for fabric ops.** The verifier breadcrumb says "the verifier prompt mandates --dev exclusively". I could not find a literal `--dev` mandate in `incremental-verifier.md`, so it probably comes from a shared reference. Wherever it lives, add: "for FABRIC ops, one fabric test module per `run_safe_pytest` session until 3b is fixed". Confidence: low (I could not locate the source of the mandate).
