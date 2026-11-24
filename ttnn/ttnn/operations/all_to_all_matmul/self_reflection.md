# Self-reflection: all_to_all_matmul (run 1)

## Summary
- **Final blind:** 155/156 pass. The 148 golden cells are all `supported_pass`. Of the 8 regression tests, 7 pass. There is no `supported_fail`, no `xpass_drift`, and no hang. SUPPORTED equals TARGET on every axis, and all 5 refinements plus 2 perf rounds are `[x]`.
- **The one failure is not the op.** `test_regression.py::test_small_packet_size` fails in **every** graded phase, from phase 0 through the blind pass, with the same harness error. Because of that, the op's "packets sized from the active payload" contract has never been graded below 7232 B.
- **Most important finding (framework level):** packet size is an axis-blind facet. The perf graduations branch on it, but golden grades only two values (14400 B and 7232 B), and only the 4 LOOSE_CASES run at 7232 B.
- **Second finding (framework level):** both perf rounds wrote "Helper bypasses — none" while the op's whole fabric sender is raw code alongside `kernel_lib/ccl`. The bypass table's definition covers only raw LLK.
- The op itself looks healthy. Every finding below is about the harness, the prompts or the docs.

## 1. Golden coverage → `eval/golden_tests/all_to_all_matmul/`

**1a. The small-packet regression test has never run in a graded pass. (high)**
- **What:** In graded mode the test errors before reaching the op. It asks for `_fabric_mesh_cache` directly, but the L1-profiling plugin requires a `device`/`mesh_device` fixture. In agent (non-graded) mode the 4160 B router firmware does not build at all.
- **Evidence:**
  - `test_regression.py:146`: `def test_small_packet_size(_fabric_mesh_cache):`
  - `eval/l1_profiling.py:31-35`: "has no device or mesh_device fixture".
  - Same `failure_category: other` in `golden_phase0` … `golden_blind_final` (9/9 phases).
  - `verification_report.md:37`: "`Program size (26848) too large for kernel config buffer (26624) on ACTIVE_ETH`".
- **Recommendation:** Propose two fixes.
  1. Give the test a fixture named `mesh_device` that is parametrized with a payload, reusing `conftest.py:117-126 pinned_payload`, or let `l1_profiling._device_from_item` accept `_fabric_mesh_cache`.
  2. Lower-bound the payload at the smallest router payload that builds on Blackhole, which is ≥ 2 tiles. 4160 B does not build today.

**1b. Packet size is an axis-blind facet that the op's code paths branch on. (med)**
- **What:**
  - The host derives `packet_tiles = payload // tile_bytes` and gates three things on it: packing across entries (`all_to_all_matmul_program_descriptor.py:479`, `packet_tiles <= slot_tiles`), the M-split (`:604`, a piece smaller than one packet → off) and the Perf-2 read window (`:482`, `ceil(40 KiB / packet)`).
  - Golden grades only two packet sizes: 7 tiles (14400 B, every cell) and 3 tiles (7232 B, the 4 LOOSE_CASES only).
  - Those LOOSE_CASES are all `tile_aligned` with a bf16 weight. No `m_non_aligned` (retile + packing) cell and no `bfloat8_b` cell is graded at the production payload.
- **Evidence:**
  - `feature_spec.py` LOOSE_CASES: all `_KIMI`/`_DSV32`/`_GLM`, with `fabric_payload_bytes` pinned only via `_PLACEMENT`.
  - The perf subagent had to cover this ad hoc: `changelog.md:250` "72/72 GOLDEN-mode cells", and the breadcrumb "18 INPUTS x payload 7232/max x model/phase0".
- **Recommendation:**
  - Propose two correctness `LOOSE_CASES` pinned at `fabric_payload_bytes=7232` with no perf extras:
    - `((1, 8, 48, 64), (1, 32, 64, 128), 4)` with `_MODEL` axes (`m_non_aligned`, input 1.5 tiles).
    - `((1, 24, 160, 128), (1, 96, 128, 512), 4)` with `weight_dtype=bfloat8_b`.
  - Consider a `packet_tiles` tagger (in the conftest, since it is a mesh facet rather than an input facet) so that coverage across packet sizes becomes visible in the axes.

## 2. SUPPORTED honesty → `all_to_all_matmul.py`
No finding. In `golden_blind_final/verifier_report.json`: `supported_pass` 148, `supported_fail` 0, `xpass_drift` 0, `xfail_wrong_mode` 0, `no_axes_found` 8 (the regression tests). `registry_snapshot.json` SUPPORTED equals TARGET on all 10 axes, and `EXCLUSIONS = []`. The over-claim and under-claim signals were also zero in every intermediate phase.

## 3. Helper / reference docs

**3a. Reference §12 tells authors the CCL helper is unmeasured but not what it cannot do. (med)**
- **What:** The planner had to work out from the `.inl` that `FabricStreamSender` binds one route per stream. It then fell back to the raw sender, citing the reference's "[unmeasured]" tag. Neither the reference nor the `.hpp` scope note lists what the helper does not cover.
- **Evidence:**
  - `op_design.md:306`: "this op needs per-packet 2D destination routing to 1-hop and 2-hop chips on one connection".
  - `.claude/references/blackhole-ccl-kernels.md:426`: "[unmeasured]: no example here uses it".
- **Recommendation:** Add a "Not covered → use the raw sender" line to §12: several destinations on one connection, routes by chip id, and fused scatter-write + atomic-inc.

**3b. Measured fabric costs this run produced are still marked `[unmeasured]` in the reference. (med)**
- **Evidence:**
  - Setting a route on an L1 header: `changelog.md:118` "`fabric_set_unicast_route` costs ~0.33 us per L1 header". This is 4.3 us of sender startup for 12 headers. Building the route once in a local image and storing only the route words got it to 1.79 us.
  - Fused scatter-write + atomic-inc: `blackhole-ccl-kernels.md:125` still says **[unmeasured]**. Refinement 4 used it to pack packets across entries, going from 168 to 160 packets per link, as part of 60.4 → 55.8 us.
- **Recommendation:** Add both as measured rows to §3.3 and §3.2: "route L1 headers once per destination from a local image; ~0.33 us/header otherwise". Mark them `[measured: all_to_all_matmul Perf 1 / Refinement 4]`.

**3c. Growing the per-core runtime-arg count made back-to-back calls hang. (low)**
- **What:** Single calls were correct, and only back-to-back or traced calls hung. The root cause was not isolated: the subagent worked around it by packing the new fields into existing argument words.
- **Evidence:**
  - `changelog.md:162`: "adding per-core runtime arguments made back-to-back calls hang".
  - Part-optimizer breadcrumb (19:55): "Watcher: reader forward with garbage RT args".
- **Recommendation:** Root-cause this before documenting it. If it reproduces, add a line to the program-descriptor reference: "changing a kernel's runtime-arg count requires a back-to-back test".

#### Helper gaps (perf)
| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `FabricStreamSender` / `FabricStream` (`kernel_lib/ccl`) | "none — no raw LLK" (`changelog.md:169,256`). The bypass is real: the port sender is raw `PacketHeaderPool` + `fabric_set_unicast_route` + `to_noc_*` (`all_to_all_matmul_port_sender.cpp:46,111,255`). Perf 1 rewrote exactly this route setup and the split open. | **missing** | `ccl_helpers_dataflow.hpp:505-508`: "bind the stream's unicast route … Every unicast arm_* reuses this route". `.inl:197-232` sets one `line_unicast_route_info_t` per stream; a sender opens once (hpp:519 "Never call both"). The op sends to 1-hop and 2-hop chips over one connection. | Allow per-arm routes, or several streams over one connection. Accept a destination node id on 2D fabrics. |
| `FabricStream` fused channels | not recorded | **missing** | hpp:36-42 offers `arm_scatter_write` and `arm_fused_write_inc` but no fused scatter + inc. The op uses `to_noc_fused_unicast_scatter_write_atomic_inc` (`port_sender.cpp:255`); Refinement 4 measured it as a win. | Add `arm_fused_scatter_write_inc(chunk0_bytes, val)`. |
| Split open (`open_start`/`open_finish`) | used through the raw adapter (`port_sender.cpp:90,122`) | not a gap | The helper already offers it (hpp:510-527). It could not be adopted only because of row 1. | none |
| *(record)* bypass table | "none" in both rounds | **unrecorded bypass** | Perf 1 `sender_startup` graduated changes to a raw fabric path. Its numbers are on record (header setup 4.28 → 1.79 us), but it is not in the table. | See §4b. |

API sketch (confidence: low — derived from one op's sender, `port_sender.cpp:100-122,239-266`):
```cpp
// one connection, N destinations, each routed once (host passes node ids)
FabricMultiStream<DirectConn, /*MaxDst=*/4> s = sender.open_multi(dst_nodes, n_dst);  // route images built once
auto ch = s.arm_fused_scatter_write_inc(/*dst_idx=*/d, chunk0_bytes, /*val=*/1);
ch.write_fused_scatter({dst0, dst1}, src_l1, arrival_sem_noc);   // packs a packet across two entries
```

## 4. Agent prompts

**4a. The perf prompts name an instrumentation header that does not exist. (high)**
- **Evidence:**
  - `.claude/agents/perf-coordinator.md:49-50` and `.claude/eval/prompts/perf_tournament_prompt.txt:9` both say to use `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`. No such file exists in the tree.
  - The coordinator logged this in both rounds (`changelog.md:94,174`) and fell back to the op-local `A2AMM_ZONE`.
- **Recommendation:** Land the header (it is the `DeviceZoneScopedN` alias described in `references/device-zone-scope-attribution.md:11`), or point both prompts at that reference until it lands.

**4b. The bypass-table definition covers only raw LLK, so fused-CCL ops record "none". (high)**
- **Evidence:** `perf-coordinator.md:318`: "Every graduated **raw-LLK** path states … why the helper wasn't used". The coordinator wrote "No raw LLK" (`changelog.md:170,257`) for a round that rewrote the raw fabric sender (see §3).
- **Recommendation:** Change the wording to "every graduated path that bypasses any `ttnn/cpp/ttnn/kernel_lib` helper, including dataflow/fabric (`kernel_lib/ccl`)". Also require bypasses inherited from earlier phases to be re-listed once.

**4c. The perf coordinator reported "Golden 156/156", which contradicts every graded phase. (med)**
- **Evidence:**
  - `changelog.md:161,250` and the breadcrumbs (`"golden": "156/156"`).
  - Every graded phase, including `golden_perf_1` and `golden_perf_2`, shows `PASSED=155 FAILED=1`.
  - The verifier had already documented `test_small_packet_size` as unrunnable (`verification_report.md:37`).
- **Recommendation:** Have `perf-coordinator.md` quote the harness's own `golden_results.txt` summary line verbatim, and name any known-failing nodeids, instead of restating a count.

**4d. A verifier "please fix" request about the read-only golden suite had no consumer. (med)**
- **Evidence:**
  - `verification_report.md:37`: "**Please fix via `/golden-tests`** (give the test a `mesh_device`-named fixture)".
  - Phase 0 is commit `e6d7d6f442d`. The same failure then persisted through 8 more graded phases.
- **Recommendation:** Have the verifier prompt write such items to a machine-readable "harness findings" list (for example in `verifier_report.json`) that the pipeline surfaces on the dashboard, rather than leaving them in free text.
