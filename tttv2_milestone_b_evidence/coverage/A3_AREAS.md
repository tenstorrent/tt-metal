
## Area by area, on silicon — attempt 3

`runs` is how many **fresh processes** the claim got *at this tree*, counting
attempt 2's logs where §A3's section head proved their source byte-identical to
`HEAD`. A claim with one run is **observed**, not qualified, and says so. A claim
with three identical *failures* is qualified in the other direction: three of
Milestone A's four defects presented as intermittent passes, so a failure that
repeats to the byte across fresh processes is not a race.

*This table is written as results land; a row marked `IN FLIGHT` was still
running at the timestamp in §A3's head.*

### Area 1 — paged KV

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Prefill and decode page tables have the layouts D-C1 assumes | `a2_01b`, `a2_s34_placement_run2`, `a2_s35_placement_run3` | **3** | **PASS.** decode global `(32, 64)` → device-local `(8, 64)`; prefill `(32, 64)` → `(32, 64)`; ratio 4; both DRAM-interleaved. Identical all three runs |
| A prefill-shaped page table fed to decode is **rejected** | as above | **3** | **FAIL by design — D-C1.** `32 % 8 == 0` and both tables are interleaved, so `_validate_decode_page_table` cannot separate the prefill layout from a legitimate L1-sharded repeat. Needs a 2D-module expectation changed, so it needs a decision, not a patch |
| Paged fill then decode, PCC ≥ 0.99 **against the contiguous path** | — | — | **NOT EXPRESSIBLE — D-C4.** `from_pretrained(paged_attention_config=None)` installs the default 2048-block pool, not a contiguous cache. The brief's wording has no reachable form at this adaptor API |
| …its nearest reachable form: two *different* paged pools agree | `a3_q_pool_default`, `a3_q_pool_default_run2`, `a3_q_pool_explicit`, `logs3/a3_h12_pool_compare_committed_tree` | **2** per pool arm | **PASS for Qwen.** 2048-block against 4096-block, `[pool] all 32 slots agree at PCC >= 0.99 for prefill and decode`. Guard exercised: with either recording absent the comparison **fails** (`logs3/a3_h10_pool_compare_missing_guard`), so the pass is a comparison and not a no-op |
| …the same, in **one** process | `a3_q_two_pools` | 1 | **FAIL — D-C7.** The second model's `activate("decode")` cannot create its global circular buffer: 923776 of 1393472 B per L1 bank still allocated after the first model's `close()` and an explicit `gc.collect()`. This is what forced the cross-process split above |
| Llama half of both pool claims | `a3_l_pool_default`, `a3_l_pool_explicit`, `a3_l_two_pools` | — | IN FLIGHT |
| Late capacity resolution — a cache bound after construction | `a2_02` (superseded), `a3_q_late_capacity`, `a3_l_late_capacity` | — | IN FLIGHT. `a2_02`'s failure was **D-C4**, not the model: it asserted `paged_attention_config is None` after construction, which the adaptor never leaves true. The case was rewritten to the reachable claim and re-queued |
| No cross-slot contamination in the blocks | `a3_q_cross_slot`, `a3_l_cross_slot`; and both demos' `*_batch32_has_no_cross_slot_contamination` | 1 per model for the demo | Demo form **PASS** both models (`a2_g9`, `a2_g21`). Block-level form IN FLIGHT |
| Transactional unbind, and a failed bind leaves no partial state | host suite (`G/test_step7_paged_kv.py`) | — | host **PASS**. The unwind is pure Python; no device case is needed and none was written |

### Area 2 — concat-32 physical prefill

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Concat-32 agrees with sequential prefill, Llama, through the demo | `a2_g10` | 1 | **FAIL — L1 address clash**, `program 1552` on `[0-0 - 6-9]`, the whole 7×10 grid. The demo prefills, decodes, then prefills again |
| Concat-32 agrees with sequential prefill, Qwen, through the demo | `a2_g22` | 1 | **FAIL — D-C6**, and not the clash: static circular buffers on `[0-0 - 2-3]` sum to 1669312 B against 1499136 B of L1. A **capacity** overflow, 11% over, raised by `validate_circular_buffer_region` from `direct_runner.py:484` |
| Concat-32 agrees with sequential prefill, step-7 form, lengths 128 → 2048 | `a3_{q,l}_concat_len*` | — | IN FLIGHT. The step-7 form builds a model and prefills **once**, with no preceding decode, so it is the case that can distinguish D-C6 (which should still fire) from the L1 clash (which should not) |
| Padded rows change no active row's logits, active 16 / 31 / 32 | `a3_{q,l}_concat_active*` | — | IN FLIGHT |
| Active batches 16 and 31 are not expressible as a smaller allocation | — | — | **G-C1**, host, unchanged from attempt 1 |

### Area 3 — prefix-cached and chunked prefill

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Prefix-cached prefill matches uncached, Llama | `a2_g2` | 1 | **PASS** — two 128-token chunks against one 256-token prefill, same argmax and PCC ≥ 0.99 |
| Prefix-cached prefill matches uncached, Qwen | `a2_g13` | 1 | **PASS** |
| Chunked prefill matches a single uncached prefill, and the decode after it reads what the chunks wrote | `a3_{q,l}_chunked` | — | IN FLIGHT |
| A prefix-cached request then a normal one | `a3_{q,l}_prefix_then_plain` | — | IN FLIGHT |
| A mix of both in one batch | `a3_{q,l}_mixed_slots` | — | IN FLIGHT. The Qwen case did not exist before attempt 3 |
| The `chunk_page_table` guard is unreachable | — | — | **G-C3**, host, unchanged |

### Area 4 — device sampling

**BLOCKED for both models, and measured rather than unmeasured.** Two stacked
defects in shared Galaxy code, the second only visible once the first is removed:

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Device greedy sampling equals the host argmax, Qwen | `a2_g23` (demo), `a3_q_greedy` (step-7) | 2 | **FAIL — D-C5.** `collectives.py:445`, `Input B memory layout must be INTERLEAVED, got WIDTH_SHARDED` |
| Device greedy sampling equals the host argmax, Llama | `a2_g11` (demo, died earlier on L1), `a3_l_greedy` (step-7) | 1 for the sampler | **FAIL — D-C5, same frame, same assertion.** So the defect is not Qwen-specific and not an artefact of the demo path |
| …with D-C5 removed at the call site: greedy, padded vocabulary, D4's near-zero reciprocal temperature, seed repetition, per-slot heterogeneous controls | `a3_q_dc5`, `a3_q_dc5_run2`, `a3_q_dc5_run3` | **3** | **FAIL — D-C8.** The relocation works (`WIDTH_SHARDED → INTERLEAVED`, width 19200) and the same line then raises `Kernel group cores do not match sub device cores`. **None of the five claims could be evaluated**, because all five are behind the selector |
| The same diagnostic, Llama | `a3_l_dc5` | — | IN FLIGHT |
| Seeded slot **stability across slots** | host (`G/test_step7_sampling.py`) | — | **FAIL by design — D-C2.** `_seed_digest` mixes the slot in, so moving a request changes its stream. A product decision |
| Llama pads its vocabulary, so the padded-vocab gate is live | host, `recipes.galaxy_padded_vocab_size` | — | **F-C1 superseded.** 128256 → 129024 (768 ids); Qwen 151936 → 153600 (1664) |
| D4's reciprocal-temperature pairing, **on the host, by inspection**, since the device cannot reach it | source read at `HEAD` | — | **CORRECT.** `sampling_2d.py:213` writes `1.0 / call.temperature[index]` into the buffer and passes it as `temp=self._temperature` (line 384), so the module performs the inversion exactly once. Both host references divide: `sampling_2d.py:260` and `direct_runner.py:570` compute `torch.topk(row / T, k=k)`. And `direct_runner.py:531` hands the module the **raw** `policy.temperature`. Raw T in, one inversion inside, division on the host reference — the pairing the brief asked to be verified rather than assumed. This is a code reading, **not** the device measurement the brief wanted; that one is behind D-C5 and D-C8, and `test_*_a_near_zero_temperature_collapses_onto_the_host_argmax` at `T = 0.02` is written, committed and queued for the day the selector works |
| The composition has a device test that cannot see either defect | `G/test_column_user_selector_wh_galaxy.py` | — | It builds its input `DRAM_MEMORY_CONFIG` — the one layout the real model never produces — and loads no sub-device manager. Every module in the chain is green in its own suite; the chain does not run |

### Area 5 — long context

| Geometry | Llama | Qwen |
| --- | --- | --- |
| 4K | **PASS** `a2_g3`, 357.81s | **PASS** `a2_g14`, 117.91s |
| 32K | **PASS** `a2_g4`, 641.17s | **PASS** `a2_g15`, 136.29s |
| 128K | **PASS** `a2_g5`, 721.70s | **PASS** `a2_g16`, 245.76s |

One run each, commit `718997518ab`, which §A3's head proves is byte-identical to
`HEAD` under `models/`. Where the capacity goes: attempt 1's accounting (blocks
per user, pool size, KV bytes per device, RoPE table size, chunk count) predicted
~5.2 GiB per device for Llama at 128K against 12 GB and named fragmentation as
the risk; it fits, at 64 chunks of 2048 followed by a decode at position 131072.
**Qwen3-32B's `max_position_embeddings` is 40960**, so its 128K smoke runs three
times past the trained context and nothing in the stack refuses it —
`max_context_len` rides on the runtime config and is never checked against
`max_seq_len`. Functional, as the brief defines it; not a quality statement.

### Repeat and cleanup

| Shape | Llama | Qwen |
| --- | --- | --- |
| repeated requests, two runners, one live model | **FAIL 3/3**, byte-identical (`a2_g6`, `a2_L1_llama_repeat_run2`, `a3_L1_llama_repeat_run3`) — L1 address clash | **PASS 3/3** (`a2_g17`, `a2_L1_qwen_repeat_run2/3`) |
| `*_batch32_slots_are_isolated` | **FAIL 1/1**, same signature (`a2_g7`) | **PASS 3/3** (`a2_g18`, `a2_L1_qwen_batch32_run2/3`) |
| **two model constructions in one process** | `a3_l_two_models`, `a3_l_two_pools` IN FLIGHT | **FAIL** (`a3_q_two_pools`) — **D-C7**, and this is the shape the brief warned about |

See "L1, corrected" below: the address clash is Llama-only at this tree, the
capacity residue is not, and only the first of the two could yield to the
teardown ordering the brief suggests.
