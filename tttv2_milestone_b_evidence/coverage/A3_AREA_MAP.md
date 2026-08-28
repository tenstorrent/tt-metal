
## Corrections to §A2's area map

Four rows of §A2's "which device case covers which area" table are wrong at this
commit and attempt 3 corrects them rather than reprinting the table:

| §A2 row | At `af589dff4d5` |
| --- | --- |
| area 1, "paged fill then decode, PCC ≥ 0.99 vs contiguous" → `*_paged_and_contiguous_caches_agree` | that test no longer exists. D-C4 made it a tautology and attempt 2 replaced it with `*_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable`. **The brief's claim as written has no device case, because the contiguous path is unreachable through `from_pretrained`** |
| area 3, "a mix of both in one batch" → Llama only | **both models.** Attempt 3 wrote `test_qwen_prefix_cached_and_plain_requests_mixed_across_slots` |
| area 4, "per-slot heterogeneous top-k/top-p/temperature" → Llama only | **both models.** Attempt 3 wrote `test_qwen_per_slot_heterogeneous_sampling_controls` |
| repeat/cleanup, "two model constructions in one process" → "no device case" | there is one: `L/test_bringup_wh_galaxy.py::test_two_models_in_one_process`. Attempt 3 queued it |

### Three device cases attempt 3's second invocation added to the map

| Brief area | Claim it asks for | Device case, and why it is new |
| --- | --- | --- |
| 1 paged KV | paged fill then decode, PCC ≥ 0.99 vs the contiguous path | `{L,Q}/step7::*_paged_pool_logits_are_recorded_for_cross_process_comparison[default2048\|explicit4096]` plus the host-only `{L,Q}/step7::*_two_paged_pools_agree_across_processes`. **The claim as worded has no device case (D-C4) and its nearest reachable form has no *single-process* device case (D-C7)**, so the recording and the comparison are separate node ids. Same PCC threshold, same claim, one model per process |
| 4 sampling | all five area-4 claims at once, with D-C5 removed at the call site | `{L,Q}/step7::*_device_sampling_claims_behind_dc5_with_interleaved_logits`. A **diagnostic**, not a substitute gate: area 4 stays BLOCKED whatever it reports. It is what distinguishes "one memory-layout precondition" from "a memory-layout precondition and a sub-device core-set violation behind it" (D-C8) |
| regression | `models/common/tests/llm_runtime`, the third directory of the brief's regression command | no new test; the directory had simply never been run. Host-only, verified device-free first. See the gate table |
