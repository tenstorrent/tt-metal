
## Findings you need for the modularity scorecard

Attempt 1's seven stand except F-C1. Attempt 2 adds two and corrects one.

| ID | Severity | Where | What |
| --- | --- | --- | --- |
| **D-C1** | correctness | `attention_2d._validate_decode_page_table` | A prefill-shaped page table fed to decode is accepted. **Premise now confirmed on silicon**: the decode table's device-local view is 8 rows, the prefill table's is 32, `32 % 8 == 0`, and both are DRAM-interleaved. Unchanged verdict; the fix needs a 2D-module expectation changed, so it needs a decision |
| **D-C2** | contract conflict | `sampling_2d._seed_digest` | Moving a seeded request to another slot changes its stream. Product decision: is a seed per-request or per-(request, slot)? |
| **D-C3** | test-infra, expensive | `modules/lazy_weight.py` | The weight-cache fingerprint contains `MeshDevice.id()`, so every test after the first in one pytest process re-stages **every** weight: 965 tensors, 138 GB, 26 min for Llama. One node id per process is mandatory on this stack. One-line fix (fingerprint the mesh *shape*), outside this job's mandate |
| **D-C4** | contract gap | both `hf_adaptor.from_pretrained` | `paged_attention_config=None` installs the *default* pool, not a contiguous cache, so area 1's "PCC vs the contiguous path" gate is not expressible through the adaptor — and the committed test for it was comparing a pool against itself |
| **G-C1** | limitation | `direct_runner.prefill_batched` | Concat-32 needs all 32 slots active; it cannot combine with the `active_slots < 32` sink-block mechanism |
| **G-C2** | minor | `direct_runner.prefill_batched` | An empty row is rejected one call too late |
| **G-C3** | dead code | `attention_2d._validate_prefill` | An unreachable guard |
| **F-C1** | **superseded** | `recipes.galaxy_padded_vocab_size` | Attempt 1: "Llama has no vocabulary padding, its padded-vocab gate is vacuous". **False.** Llama pads 768 ids (129024), Qwen 1664 (153600). The gate is live for both and now has a device case for both |
| **F-C2** | test-infra | `tests/models/galaxy/test_plans.py` | Looks host-only, needs a cluster. On a live mesh its 13 failures should disappear — worth re-checking now that one exists |

## Suggested order for your night

1. Read `RESULTS_A2.md` first, not the report: it is one row per run with the log
   name, written as each finished, and it says how many fresh processes each
   claim got.
2. Take the exit-gate table from `REPORT.md` §A2. Every row has its command.
3. The verdict is **not** "infrastructure-blocked" any more. Say which lines pass,
   which fail, and which are *not expressible* — D-C4 makes one gate line
   unmeasurable at this API, which is a different thing from unmeasured.
4. D-C1, D-C2 and D-C4 are the three that should reach a human. D-C3 should reach
   whoever owns `lazy_weight.py`.
