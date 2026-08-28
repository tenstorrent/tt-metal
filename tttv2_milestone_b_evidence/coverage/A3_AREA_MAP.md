
## Corrections to §A2's area map

Four rows of §A2's "which device case covers which area" table are wrong at this
commit and attempt 3 corrects them rather than reprinting the table:

| §A2 row | At `af589dff4d5` |
| --- | --- |
| area 1, "paged fill then decode, PCC ≥ 0.99 vs contiguous" → `*_paged_and_contiguous_caches_agree` | that test no longer exists. D-C4 made it a tautology and attempt 2 replaced it with `*_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable`. **The brief's claim as written has no device case, because the contiguous path is unreachable through `from_pretrained`** |
| area 3, "a mix of both in one batch" → Llama only | **both models.** Attempt 3 wrote `test_qwen_prefix_cached_and_plain_requests_mixed_across_slots` |
| area 4, "per-slot heterogeneous top-k/top-p/temperature" → Llama only | **both models.** Attempt 3 wrote `test_qwen_per_slot_heterogeneous_sampling_controls` |
| repeat/cleanup, "two model constructions in one process" → "no device case" | there is one: `L/test_bringup_wh_galaxy.py::test_two_models_in_one_process`. Attempt 3 queued it |
