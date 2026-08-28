
## Which device case covers which of the brief's five areas

`L` = `models/common/tests/models/llama33_70b_galaxy/`,
`Q` = `models/common/tests/models/qwen3_32b_galaxy/`,
`G` = `models/common/tests/models/galaxy/`,
`step7` = `test_step7_coverage_wh_galaxy.py`,
`full` = `test_full_model_wh_galaxy.py`.

| Brief area | Claim it asks for | Device case |
| --- | --- | --- |
| 1 paged KV | paged fill then decode, PCC ≥ 0.99 vs contiguous | `{L,Q}/step7::*_paged_and_contiguous_caches_agree` |
| 1 | late capacity resolution | `{L,Q}/step7::*_paged_capacity_resolved_after_construction_serves_a_request` |
| 1 | transactional bind/unbind, failed bind leaves no partial state | host only (`G/test_step7_paged_kv.py`) — no device case needs one, the unwind is pure Python |
| 1 | no cross-slot contamination | `{L,Q}/step7::*_a_write_for_one_user_never_appears_in_another_users_blocks`, and `{L,Q}/demo.py::*_batch32_has_no_cross_slot_contamination` |
| 1 | a prefill-shaped table fed to decode is **rejected** | **not satisfiable at this contract** — D-C1. Pinned on the host and now on silicon: `G/test_step7_page_table_placement_wh_galaxy.py` |
| 2 concat-32 | concat-32 agrees with sequential prefill, 128 → 2048 ascending | `L/step7::*_concat32_matches_sequential_prefill_at_each_length[len128..len2048]`, `Q/…[len128..len512]` |
| 2 | padded rows change no active row's logits, active 16/31/32 | `{L,Q}/step7::*_concat32_padded_rows_change_no_active_rows_logits[active16,31,32]` |
| 3 prefix cache | prefix-cached output matches uncached | `{L,Q}/full::*_prefix_cached_prefill_matches_uncached` and `{L,Q}/step7::*_chunked_prefill_matches_a_single_uncached_prefill` (the second also decodes, so the cache the chunks *wrote* is read) |
| 3 | a prefix-cached request then a normal one | `{L,Q}/step7::*_a_prefix_cached_request_then_a_normal_one` |
| 3 | a mix of both in one batch | `L/step7::test_llama_prefix_cached_and_plain_requests_mixed_across_slots` (Llama only) |
| 4 sampling | greedy equals host argmax, every slot | `{L,Q}/step7::*_device_greedy_sampling_equals_host_argmax`, `{L,Q}/demo.py::*_device_sampling_matches_host_greedy` |
| 4 | seeded slot stability across runs | `{L,Q}/step7::*_a_seeded_slot_repeats_across_runs` |
| 4 | a padded id can never be sampled | `Q/step7::test_qwen_no_padded_vocabulary_id_is_ever_sampled`, and **new in attempt 2** `L/step7::test_llama_no_padded_vocabulary_id_is_ever_sampled` |
| 4 | per-slot heterogeneous top-k/top-p/temperature | `L/step7::test_llama_per_slot_heterogeneous_sampling_controls` (Llama only) |
| 5 long context | batch-1 4K / 32K / 128K functional smokes | `{L,Q}/full::*_long_context_smoke[4k,32k,128k]` |
| repeat/cleanup | repeated requests, deterministic | `{L,Q}/full::*_repeated_requests_and_deterministic_cleanup` |
| repeat/cleanup | two model constructions in one process | `G/test_step7_repeat_and_cleanup.py` on host; **no device case** — see L1 |
