# `mb-coverage` attempt 2 — run-by-run, in the order it happened

Every line here is a log in `logs2/`. Times are UTC on 2026-08-27/28.
Written as each run finished, so it survives a timeout.

| # | log | what | result |
| --- | --- | --- | --- |
| 00 | `a2_00_mesh_health.log` | `test_partition_wh_galaxy.py`, the 13-second mesh check | **5 passed in 12.32s** — the mesh is alive; attempt 1's "the mesh never came back" is superseded |
| H1 | `a2_h1_1d_contract_gate.log` | the 22 1D `test_demo_contract.py` / `test_hf_adaptor.py` files, host only (`cov_1d_contract_gate.sh`), 23:45–23:47 | **5 failed, 296 passed in 107.96s** — the same five test ids attempt 1 recorded, re-measured at this commit |
| H2 | *(grep, no log)* | boundary greps at `bc6ad03bfc2..HEAD` | 338 changed paths, **0** matching `_1d.py`, **0** matching `llm_runtime`; none of the five failing 1D packages appears in the diff at all |
| H3 | *(grep, no log)* | model-named import gate over `models/common/{models/galaxy,modules,models/*_galaxy}` | **0 matches** for `models.demos.*` and for the non-galaxy `models.common.models.{llama33_70b,qwen3_32b}` |
| H4 | `a2_h2_step7_host_suite.log` | attempt 1's six host step-7 files, re-run at this commit | **162 passed in 41.50s** — unchanged |
| 01 | `a2_01_llama_full_model_file.log` | Llama `test_full_model_wh_galaxy.py`, all 8 node ids, one process, started 23:23 | see the rows below — this one log carries five gate lines |
| 01a | ″ | `test_llama33_70b_galaxy_full_model_prefill_and_first_decode_token` | **PASSED**, ~6 min (80-layer build from the warm ring cache) |
| 01b | ″ | `test_llama33_70b_galaxy_teacher_forced_accuracy_batch1` — **the Llama accuracy gate** | **PASSED. top-1 501/511 = 98.04%** (gate ≥ 91%), **top-5 511/511 = 100.00%** (gate ≥ 99%). Re-measured at this commit, not quoted: bit-identical to `mb-llama` attempt 3 and to `mb-qwen` attempt 2's re-run. 26 min of that was first-time weight staging (965 tensors, 138 GB) |
| 01c | ″ | `test_llama33_70b_galaxy_batch32_slots_are_isolated` and the five node ids after it | **NOT RUN in this cycle — the cycle was stopped deliberately at 00:18** (`exit=143`). Reason in the row below; re-queued one node id per process |
| 01b | `a2_01b_page_table_placement.log` | new `test_step7_page_table_placement_wh_galaxy.py`, 3 cases | **3 passed in 9.26s.** `[placement] decode table global=(32, 64) device-local=(8, 64)`; `[placement] prefill table global=(32, 64) device-local=(32, 64)`; both DRAM-interleaved, ratio 4. **D-C1's premise confirmed on silicon** |
| 02 | `a2_02_llama_late_capacity.log` | `test_llama_paged_capacity_resolved_after_construction_serves_a_request` | **1 failed in 237.75s** — `assert all(spec.paged_attention_config is None ...)`. Not a model defect: `from_pretrained` substitutes `default_paged_attention_config` when passed `None`. **Finding D-C4.** 240 cache hits, 0 misses — a single-node-id process is fully warm |
| 03 | `a2_03_llama_paged_vs_contig.log` | `test_llama_paged_and_contiguous_caches_agree` | **stopped at 4 min (`exit=143`), deliberately.** Its "contiguous" arm passes `paged_attention_config=None`, which D-C4 turns into the *same* 2048-block pool as its "paged" arm: `default_paged_attention_config` at `max_seq_len=2048`, batch 32, block 32 is 2048 blocks, and `_paged_config(context=2048, active_slots=32)` is 2048 blocks. The case was a tautology. Rewritten as `test_*_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable` and re-queued |
