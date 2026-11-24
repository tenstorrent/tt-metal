# Self-Reflection: all_gather_rms_norm

## Summary
- **Final blind pass:** 1457 passed, 1 failed, 0 hangs, 139 skipped, out of 1720 (golden 1425 pass / 1 fail / 123 xfail / 11 INFEASIBLE_L1 skip; regression 32/32; ring-mock 128 skipped because it runs only in the exhaustive tier). There were **no translated cells** in this blind pass.
- **The one failure** is a `supported_fail` OOM: `128x8192` bf8b HEIGHT_SHARDED LayerNorm with γ+β, cluster_axis 0. It failed the same way in **all 8 measurements from Refinement 1 to blind**, was called "pre-existing" or "known" every time, and was never filed as a refinement or demoted.
- **Most important finding (framework level):** the reduce helper's `.hpp` says AccumulateViaAdd streaming is "aligned only". The `.inl` does support streaming with a ROW partial mask. That doc gap is why `cb_ln_sq` was sized to a full tile-row instead of the designed 2 tiles, and that full-row buffer is about half of the overflow.
- Two more helper-contract gaps (the mask CB's format, and a to-dest input CB) each cost the implementer a debug cycle. The process gap is that no one owns a SUPPORTED cell that has never passed.
- Perf ran two rounds: Perf 1 graduated one idea (2 bypass rows, both dataflow); Perf 2 graduated nothing.

## 1. Golden coverage (`eval/golden_tests/all_gather_rms_norm/feature_spec.py`)

**G1 — HEIGHT_SHARDED LayerNorm has no runnable cell between W_local 2048 (passes) and 8192 (fails). This is a gap the axes can't see.**
- **What:** Among HEIGHT_SHARDED LN+β cells, every W_local ≤ 2048 passes (`32x2048`, `1x1x32x2048`, `1x1x32x1792`, …). Every W_local ≥ 4096 cell is either skipped `INFEASIBLE_L1` (all on cluster_axis 1, G=4, where the gathered output shard is bigger than a bank) or is the one failure. The discriminating facet is per-device row width, i.e. the op's per-core CB footprint, and no `INPUT_TAGGERS` axis captures it. The failing axes tuple has no other shape in golden.
- **Evidence:** blind `test_results.json`: `1x1x32x4096`/`1x1x32x8192`/`1x1x128x4096`/`1x32x4096` HS LN, all `skipped` with "INFEASIBLE_L1: op: … bank_manager.cpp:495"; `pytest_stdout.log:2046` "Not enough space to allocate 2097152 B L1 buffer across 1 banks".
- **Recommendation:** propose a `LOOSE_CASES` boundary probe whose output shard fits (G=2):
  `{"inputs": ((1, 1, 32, 4096),), **{**_PHASE0, "norm_type": "layer_norm", "beta_mode": "beta", "memory_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED, "cluster_axis": 0}}`.
  It would give any fix for the over-claim in §2 a second, cheaper cell to test against. Don't add a tagger: there is no separate kernel code path for wide rows, so the verifier rule against inventing taggers to bucket failures applies.
- **Confidence:** low. I estimated by hand that the shard fits; I did not measure it.

No other in-TARGET gaps: the one failing cell is already a golden cell, so golden caught it.

## 2. SUPPORTED honesty (`all_gather_rms_norm.py` SUPPORTED / EXCLUSIONS)

Blind `verifier_report.json`: `supported_fail` 1, `xpass_drift` 0, `xfail_wrong_mode` 0. Both EXCLUSIONS (`{float32, fp32_dest_acc_en=False}` and `{bfloat8_b, w_non_aligned}`) are backed by measurements and still match the evidence.

**S1 — `supported_fail` (OOM): `128x8192` bf8b HEIGHT_SHARDED LN γ+β, cluster_axis 0 → recommend FIX, not demote.**
- **What:** The static CBs collide with the caller's L1 shards. Per R4b's own measurement, the input shard (272 KB) and output shard (544 KB) leave about 540 KB for CBs, and the op asks for about 1.2 MB: `cb_ln_sq` 512 KB plus `cb_gamma`/`cb_beta` 256 KB each, all sized by `cw_max` = 256 tiles. EXCLUSIONS can't express row width, so demoting would drop the whole axes tuple, and the verifier rule says OOM cells stay failing and get named in a refinement.
- **Evidence:** blind nodeid `test_op[128x8192-…-dtype=BFLOAT8_B-…-memory_layout=HEIGHT_SHARDED-norm_type=layer_norm-…]`: "Statically allocated circular buffers … clash with L1 buffers … static circular buffer region ends at 1389248". Sizing is at `all_gather_rms_norm_program_descriptor.py:1011-1022` (`_cb(CB_LN_SQ, row_chunk * cw_max, …)`).
- **Recommendation:** file a refinement for wide-row sharded LN in two steps.
  1. Stream the T reduce: `WaitAndPopPerTile` + `only_partial` is implemented for ROW (see H1), so `cb_ln_sq` can go back to the 2-tile depth the design specified (`op_design.md:342`). That saves about 510 KB.
  2. Chunk γ/β by column in the normalize, which R4b already proposed.

  Step 1 alone may not be enough. Note also that R1 misdiagnosed the cause: it blamed the output shard, while the CB clash message points at the op's CBs.
- **Confidence:** high on the diagnosis, med on the fix sizing.

## 3. Helper / reference docs

**H1 — `reduce_helpers_compute.hpp` says AccumulateViaAdd streaming is "aligned only", but the `.inl` implements streaming + ROW partial. This drove the OOM.**
- **What:** The header's RESTRICTED list says `WaitAndPopPerTile (streaming …) — contiguous row/scalar, aligned only` and puts partial "under BulkWaitBulkPop". A stale `.inl` comment repeats "and no partial". But the `.inl` validation block says "Streaming (WaitAndPopPerTile) + partial is supported for ROW", and the streaming loop folds the masked last tile. The op followed the header: every reduce in the compute kernel uses `BulkWaitBulkPop`, and the ledger explains the deviation from the 2-tile design as "no streaming with a partial mask".
- **Evidence:** `reduce_helpers_compute.hpp:142-145`; `reduce_helpers_compute.inl:849-851` (stale) vs `.inl:856` and `.inl:383-386`; `l1_ledger.md:108` "The masked AccumulateViaAdd reduce is BulkWaitBulkPop over a whole tile-row (no streaming with a partial mask)"; `all_gather_rms_norm_compute.cpp:379,413,431,660,707,776`.
- **Recommendation:** change `.hpp:142` to "WaitAndPopPerTile … row/scalar; ROW partial supported (masked last tile folds as the final streamed op)", and fix the comment at `.inl:849-851`.
- **Confidence:** high.

**H2 — AccumulateViaAdd's partial fold reads the mask CB as srcB in the *input* CB's format, and this is undocumented.**
- **What:** The per-call reconfig sets both operands to the input CB (`reconfig_data_format(input_dfb_id, input_dfb_id)`). `fold_partial_last` then unpacks the mask from `scaler_dfb_id` on srcB without reconfiguring. So the mask CB must have the input CB's data format. Neither `ReducePartialScaler::only_partial()` nor `prepare_reduce_mask` says this. A bf16 mask was read as fp32 and corrupted the LN W-tail.
- **Evidence:** `reduce_helpers_compute.inl:296`, `.inl:353-354`; `.inl:494` says this only in passing ("SrcB … == input …, which the partial fold needs"); breadcrumb `ttnn-implementer_breadcrumbs.jsonl:1` "a bf16 mask read as fp32 corrupted the LN W-tail mask".
- **Recommendation:** add to `prepare_reduce_mask` (`reduce_helpers_dataflow.hpp:62`) and `only_partial()` (`.hpp:311`): "the mask CB must share the input CB's data format (unpacked as srcB with no reconfig)". Alternatively, reconfigure srcB inside `fold_partial_last`.
- **Confidence:** high.

**H3 — An UnpackToDestFp32-tagged *input* CB is silently invalid for AccumulateViaAdd. The guard and the doc exist only for the accumulator CB.**
- **What:** AccumulateViaAdd sums the input tiles with `add_tiles`, which reads through srcA/srcB. The header documents the to-dest restriction, and the `.inl` asserts it, only for `acc_cb` under FoldViaAdd. The implementer hit this three times, getting wrong sums each time.
- **Evidence:** `.inl:326` `ASSERT(… || !dfb_unpacks_to_dest(acc_cb))`, with no equivalent check on the input CB; `changelog.md` R1 "an UnpackToDestFp32-tagged CB cannot be read through srcA — hit twice"; R1b "AccumulateViaAdd over tagged `cb_ln_tmp` with ≥ 2-row chunks gave PCC 0.82".
- **Recommendation:** add `UNPACK(ASSERT(!dfb_unpacks_to_dest(input_dfb_id)))` to the AccumulateViaAdd path, plus a RESTRICTED bullet at `.hpp:137`: "input CB must be UnpackToDestMode::Default (use ReduceTile for to-dest CBs)".
- **Confidence:** high.

**H4 — When a partial reduce has only the masked tile (no full tiles, `full_cnt == 0`) and several outputs, the result was wrong. That contradicts the inline claim that a freshly acquired DEST reads 0.**
- **What:** When a row's only tile is the masked one, there is no seed copy, so `fold_partial_last` accumulates straight onto DEST. The implementer measured rel RMS 0.14 with crt > 1 and worked around it with a mask-block multiply.
- **Evidence:** `.inl:362-366` / `.inl:550-565` "a freshly-acquired DST reads 0 on its first write"; `changelog.md` R1 issue (2) "`only_partial` fold of a lone tail tile with several tile-rows accumulates onto an unseeded DEST (rel RMS 0.14)".
- **Recommendation:** propose a helper unit test for `full_cnt == 0 ∧ n_out > 1`. If it reproduces, use a non-accumulating masked mul when `full_cnt == 0`.
- **Confidence:** med. The root cause is not isolated.

**H5 — The reference never warns that a logical 1×N ring opened on a 2D box turns at every hop, so 2D mcast ranges along it are not expressible.**
- **What:** The perf subagent's first mcast attempt hung. The reference describes `MeshMcastRange` (one N/S spine plus E/W branches) but not what that means for a snake ring.
- **Evidence:** `perf-part-optimizer_breadcrumbs.jsonl:3-4` "mcast R20xW384 hangs on first call … QB2 1x4 ring is physically a 2x2 square … Straight mcast ranges of 2 cannot follow the ring"; `.claude/references/blackhole-ccl-kernels.md:209`.
- **Recommendation:** add to `blackhole-ccl-kernels.md:209`: "A ring opened over a 2D box (snake) is not physically straight: a range must be an N/S spine then E/W; other paths hang. Compute routes from physical directions."
- **Confidence:** med.

(Minor: the CB ring-wrap rule is documented in `ttnn-cb-memory-fundamentals.md:85`, but R1b still hit it ("block-indexed helper cannot read a block that wraps the ring", PCC 0/inf). Consider a one-line pointer at the `BulkWaitBulkPop` description, `reduce_helpers_compute.hpp:97`. Confidence: low.)

#### Helper gaps (perf)
| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| (none) on-chip "scatter pages to remote L1 homes + one completion counter" | capability | **missing** | `agrn_worker_args.hpp` `stage_dist_block`: raw `noc_async_write` → `noc_async_write_barrier` → `noc_semaphore_inc` → `noc_async_atomic_barrier`. `kernel_lib/local_copy_helpers_dataflow.hpp:9` covers only self-aimed copies; `ccl_helpers_dataflow.hpp:378` "unicast-write + atomic-inc" is fabric-only. | Consider an on-chip analogue of the fabric write+inc channel (sketch below). The helper and raw paths have no separate timings, so this is a capability gap, not an ergonomics one. |
| `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid` (port reader's trid-pipelined chunk reads) | capability ("no kernel_lib counterpart") | **undocumented** for the trid pipeline; **missing** only for the op-specific chunk-home address math | The port reader uses raw `noc_async_read_set_trid` / `noc_async_read_barrier_with_trid` (`all_gather_rms_norm_port_reader.cpp:200,203,280,345`). The helper is a 1:1 wrapper that works for any `async_read` on that NoC (`local_copy_helpers_dataflow.inl:110-114`), and its example at `.inl:96-108` is exactly this pipeline. The file header calls it "Local L1 -> L1 … self-aimed", so a caller doing remote reads would not look there. | Move the trid helpers, or mention them, outside the "local copy" file, e.g. a `noc_pipeline_helpers` header or a pointer in `ccl_helpers_dataflow.hpp`. Doc-only change, cheap. |

No unrecorded perf-phase bypass: the Perf 1 graduation touched only dataflow files (`git diff aa2dec92dde~1 aa2dec92dde`: `agrn_worker_args.hpp`, port reader, writer), and Perf 2 changed only an ablation `#if`. The compute kernel's raw SFPI bodies (`all_gather_rms_norm_compute.cpp:70-149`, `_llk_math_eltwise_unary_sfpu_params_`) come from Phase 0 and R4, not perf, so they are outside the bypass table's scope.

```cpp
// confidence: low — derived from the single stage_dist_block call site
// wanted: scatter N pages from a CB to (core, l1_addr) homes, then one counter bump per consumer
template <typename HomeFn>  // HomeFn(page) -> {noc_xy, l1_addr}
void scatter_pages_then_signal(DataflowBuffer& src, uint32_t n_pages, uint32_t page_bytes,
                               HomeFn home, std::span<const uint64_t> counters, uint32_t inc = 1);
// call site:
scatter_pages_then_signal(cb_out_block, n_pages, out_page_bytes,
                          [&](uint32_t t) { return stage_home(t); }, port_counters);
```

## 4. Agent prompts

**P1 — Nobody owns a SUPPORTED cell that fails but never passed. It went unfiled through 8 phases.**
- **What:** R1 lifted HEIGHT_SHARDED into SUPPORTED with one cell failing. It ticked `[x]` even though its Done-when asked for "every gate-tier cell … moves … to supported_pass", and it gave a wrong reason ("cannot fit its 1.1 MB output shard"). The verifier rule says OOM `supported_fail` cells must be "named in a refinement", but the verifier ran only at Phase 0, before HEIGHT was supported. Every later gate checks only cells that used to pass, so this cell was invisible to them. R1b, R4b, Perf 1 and Perf 2 all reported it as "pre-existing".
- **Evidence:** `op_requirements.md` R1 Outcome "71/72 runnable gate cells; 128×8192 bf8b LN cannot fit its 1.1 MB output shard"; `incremental-verifier.md:220` "leave them failing and name them in a refinement"; `ttnn-implementer_breadcrumbs.jsonl:11` `"status":"fail_preexisting"`; `perf-coordinator_breadcrumbs.jsonl:5` "1 pre-existing fail".
- **Recommendation:** in `ttnn-implementer.md`, add: "If a refinement widens SUPPORTED and any newly claimed cell ends `supported_fail`, mark `[~]` and append a follow-up refinement entry naming the cell and its failure category. Quote the allocator line when the category is OOM." Optionally, have the harness completion gate list cells that are `supported_fail` and never passed, as a warning that doesn't block.
- **Confidence:** high.

**P2 — The gate-violation note gives only a count, so the debug agent couldn't reconcile it.**
- **What:** R4b received "responsible cells 1424/1437" (13 regressed). With the same seed and tier the implementer could reproduce only 2, and wrote "I couldn't account for the other 11".
- **Evidence:** `op_requirements.md` R4b Verifier notes "Bullet 3 FAIL: REGRESSION … (responsible cells 1424/1437)"; `refinement_4_debug_output.json` result "the gate reported 13 failing cells … I only ever saw 2".
- **Recommendation:** have the completion gate's notes generator list the regressed nodeids and their status (failed / hung / never ran) in the debug refinement's Verifier notes. Strictly this is the harness that writes those notes, not a prompt.
- **Confidence:** med.
