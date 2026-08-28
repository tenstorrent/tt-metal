
## Area by area, on silicon

Each row names the log. `runs` is how many fresh processes the claim got; a claim
with one run is *observed*, not qualified, and says so.

### Area 1 — paged KV

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Prefill and decode page tables have the layouts D-C1 assumes | `a2_01b_page_table_placement`, `a2_s34_placement_run2`, `a2_s35_placement_run3` | **3** | **PASS.** decode global `(32, 64)` → device-local `(8, 64)`; prefill global `(32, 64)` → device-local `(32, 64)`; ratio 4; **both DRAM-interleaved**. Identical output all three runs |
| A cache bound after construction serves a request | `a2_02_llama_late_capacity` | 1 | **FAIL** on `assert all(spec.paged_attention_config is None …)`. Not a model defect — **D-C4**: `from_pretrained` substitutes the default pool for `None`. Test rewritten to the reachable claim and re-queued |
| Paged fill then decode, PCC ≥ 0.99 against the contiguous path | `a2_03_llama_paged_vs_contig` | 0 | **STOPPED at 4 min, deliberately** (`rc=143`). D-C4 makes both arms the same 2048-block pool, so the case was a tautology. Rewritten as `*_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable` and re-queued. The gate line as written is **not expressible at this adaptor API** |
| No cross-slot contamination in the blocks | — | 0 | **NOT REACHED** |
| Transactional unbind, failed bind leaves no partial state | host suite only (attempt 1, 39 tests) | — | host PASS; no device case was reached |

### Area 2 — concat-32 physical prefill

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Concat-32 prefill agrees with sequential prefill, Llama | `a2_g10_llama_demo_concat32` | 1 | **FAIL — L1 address clash, and a new detail.** `program 1552` clashes on `[0-0 - 6-9]` — the **whole 7×10 grid**, not the four sender cores of the other L1 failures. The test runs `run_direct_demo` twice, so the second prefill follows a decode |
| Concat-32 prefill agrees with sequential prefill, Qwen | `a2_g22_qwen_demo_concat32` | 1 | **FAIL, and not the Llama failure.** `Statically allocated circular buffers on core range [0-0 - 2-3] grow to 1669312 B which is beyond max L1 size of 1499136 B`, from `validate_circular_buffer_region` at `direct_runner.py:484` (`prefill_batched`). A **capacity** overflow, not an address clash. **Finding D-C6** |
| Active batches 16, 31, 32 write no KV and return no logits for inactive slots | — | 0 | **NOT REACHED** |
| Lengths 128 → 2048 in the padded lengths the policy supports | — | 0 | **NOT REACHED** on device. The host recipe suite covers all five Llama lengths |

### Area 3 — prefix-cached and chunked prefill

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Prefix-cached prefill matches uncached, Llama | `a2_g2_llama_prefix` | 1 | **PASS** — two 128-token chunks vs one 256-token prefill, same argmax and PCC ≥ 0.99 |
| Prefix-cached prefill matches uncached, Qwen | `a2_g13_qwen_prefix` | 1 | **PASS** |
| Chunked prefill matches a single uncached prefill | — | 0 | **NOT REACHED** |
| A prefix-cached request then a normal one | — | 0 | **NOT REACHED** |
| A mix of both in one batch | — | 0 | **NOT REACHED** (and the Qwen test did not exist; attempt 3 wrote it) |

### Area 4 — device sampling

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Device greedy sampling equals host argmax, Llama, through the demo | `a2_g11_llama_demo_sampling` | 1 | **FAIL — L1, `program 100`.** The demo runs twice (host policy, then device policy), so the second prefill follows a decode and never reaches the sampler. The claim itself is untested by this log |
| Device greedy sampling equals host argmax, Qwen, through the demo | `a2_g23_qwen_demo_sampling` | 1 | **FAIL, and not L1 at all.** `MatmulMultiCoreProgramConfig: Input B memory layout must be INTERLEAVED, got: TensorMemoryLayout::WIDTH_SHARDED` at `collectives.py:445`, `GalaxyColumnUserSelector.__call__`, reached from `model.sample_decode` → `select_decode_column_users`. The host-sampling half ran first and passed. **Finding D-C5** |
| Seeded slot stability, padded vocabulary, near-zero temperature (D4), per-slot heterogeneous controls | — | 0 | **NOT REACHED.** All four cases were written (the padded-vocabulary and temperature cases *by* attempt 2) and queued, and the host was withdrawn before they ran |

### Area 5 — long context

| Geometry | Llama | Qwen |
| --- | --- | --- |
| 4K | **PASS** (`a2_g3`, ~7 min, 2 chunks of 2048) | **PASS** (`a2_g14`, ~3 min) |
| 32K | **PASS** (`a2_g4`, ~11 min, 16 chunks) | **PASS** (`a2_g15`, ~3 min) |
| 128K | **PASS** (`a2_g5`, ~13 min, 64 chunks, then a decode at position 131072) | **PASS** (`a2_g16`, ~5 min) |

One run each. Attempt 1's accounting predicted ~5.2 GiB per device for Llama at
128K against 12 GB and named fragmentation as the risk; it fits. **Qwen3-32B's
`max_position_embeddings` is 40960**, so its 128K smoke runs three times past the
trained context and nothing in the stack refuses it — `max_context_len` is carried
on the runtime config and never checked against `max_seq_len`. Functional, as the
brief defines it; not a quality statement.

Attempt 1's capacity accounting for these three geometries (blocks per user,
pool size, KV bytes per device, RoPE table size, chunk count) is in area 5 above
this section and was not re-derived; what attempt 2 adds is whether each one
actually runs.

### Repeat and cleanup

| Shape | Llama | Qwen |
| --- | --- | --- |
| `*_repeated_requests_and_deterministic_cleanup` — the same request twice through two runners on one live model | **FAIL 2/2**, deterministic (`a2_g6`, `a2_L1_llama_repeat_run2`): `program 100` clashes on `[0-0 - 0-3]`, L1 buffer at 544832, static CB region ends at 630080 | **PASS 3/3** (`a2_g17`, `a2_L1_qwen_repeat_run2/3`) |
| `*_batch32_slots_are_isolated` — slot 0 alone, then slot 0 inside a full batch | **FAIL 1/1**, same signature (`a2_g7`) | **PASS 3/3** (`a2_g18`, `a2_L1_qwen_batch32_run2/3`) |
| Repeated model construction and teardown in one process (`test_two_models_in_one_process`) | **NOT REACHED** | not applicable — the bringup file is Llama-only |

The two Qwen run-3 logs (`a2_L1_qwen_repeat_run3`, `a2_L1_qwen_batch32_run3`,
both `exit=0`) landed after `RESULTS_A2.md`'s last row was written and are
recorded here for the first time; attempt 3 re-read them off disk to confirm it.
`a2_L1_llama_repeat_run3` was in flight when the host went away and has no
verdict.
