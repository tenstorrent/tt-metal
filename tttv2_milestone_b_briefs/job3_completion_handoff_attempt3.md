# Job 3 (`mb-coverage`) attempt 3 → `mb-signoff`: completion handoff

**Written progressively, as each result landed, so it survives a kill.** Attempt 2
died with no handoff of its own and cost the next attempt an hour of
archaeology; this file is rewritten in place at every checkpoint and the
"Last updated" line below is authoritative. Anything marked `IN FLIGHT` was still
running when that line was stamped.

Last updated: **2026-08-28 08:58Z**. Status: **IN PROGRESS**.

Branch `apbernal/tttv2_wh_glx_2d_modules_milestone_b`. Full account:
`tttv2_milestone_b_evidence/coverage/REPORT.md` §A3. Run-by-run index, one row
per run written as it finished: `.../coverage/RESULTS_A3.md`. Machine-written
verdicts, extracted from the logs rather than typed:
`.../coverage/VERDICTS_A3.txt`. Environment, costs and the harness:
`.../coverage/ENVIRONMENT.md`.

## Read these four paragraphs before anything else

**1. Attempt 3 ran as two agent invocations inside one driver run.** The first
started `07:37:58Z` at `af589dff4d5` and ended `08:16:43Z`; the driver relaunched
at once. The detached device queue (`cov_queue.sh`, reparented to init) never
stopped — it dequeued the next item one second after the relaunch — so the mesh
was continuously busy and nothing was lost or paid for twice. The second
invocation adopted the running queue instead of killing it. **The Llama build it
would have killed is the run that closed D-C5.**

**2. The mesh is alive and has been all night.** 32 boards on the bus, 32 device
nodes, a real 8×4 cluster opens in ~13 s. Attempt 1's "the mesh never came back"
is two days stale; do not plan from it.

**3. Device sampling does not work on this hardware at this tree, and it is two
defects deep.** This is the single most important thing this job found.
**D-C5**: `GalaxyColumnUserSelector.__call__` is a bare `ttnn.matmul` whose
default program config requires an INTERLEAVED input B, and the *shared* Galaxy
recipe makes both models' decode logits WIDTH_SHARDED — measured on silicon for
Qwen (`a3_q_greedy`) and for Llama (`a3_l_greedy`), same frame, same assertion.
**D-C8**: with that satisfied at the call site, the same line then fails
`TT_FATAL @ program.cpp:2205, Kernel group cores do not match sub device cores` —
the matmul builds its program over cores outside the loaded decode sub-device.
The brief's whole area 4 is behind these two.

**4. L1 has two signatures, and only one of them is the ordering problem the
brief describes.** The address clash is Llama-only at this tree (4/4 Llama, 0/6
Qwen, byte-identical across two commits and three fresh processes). The other —
**D-C7** — is that the L1 a *closed, dereferenced, garbage-collected* model held
is not returned, so the second model in one process cannot create its global
circular buffer: 923776 of 1393472 bytes per bank still allocated. Measured on
**Qwen**, the model that does not clash. No teardown ordering fixes that one.

## Exit-gate verdict

The measured table with a command behind every row is `REPORT.md` §A3, section
"The Milestone B exit gate — final table, measured". Summary:

| Gate line | Verdict |
| --- | --- |
| Llama teacher-forced 512/511, top-1 ≥ 91% / top-5 ≥ 99% | **PASS** — 98.04% / 100.00% |
| Qwen teacher-forced 512, top-1 ≥ 89% / top-5 ≥ 97% | **PASS** — 97.46% / 100.00% |
| Batch-32 direct demos valid, no cross-slot contamination | **PASS**, both models |
| Batch-1 4K / 32K / 128K functional smokes | **PASS**, both models, all three geometries |
| Prefix-cached output matches uncached execution | **PASS**, both models |
| No dependency imports from a model-named implementation package | **PASS** for Milestone B; one pre-existing exception, finding F-C3 |
| Zero changes to 1D module implementation files | **PASS** — 0 of 384 changed paths |
| Zero changes to `llm_runtime` | **PASS** — 0 of 384 |
| Existing 1D contract/demo-contract host tests green, expectations unchanged | **FAIL**, 5 of 301, and not owned by Milestone B |

**On "re-measure at this tree, do not quote".** `git diff --name-only
718997518ab..HEAD -- models/` returns exactly one file, and it is a step-7 *test*
file that `test_full_model_wh_galaxy.py` and `demo.py` do not import. So the gate
logs are not older measurements of a changed thing — they are measurements of a
byte-identical thing, and §A3 states the commit for every row.

## Findings, and which need a human

`REPORT.md` §A3 "Findings, attempt 3" has the full write-up for each.

| ID | Needs | What |
| --- | --- | --- |
| **D-C5** | a fix in shared Galaxy code | selector matmul rejects both models' decode logits (WIDTH_SHARDED) |
| **D-C8** | a fix, and it is the harder half | with D-C5 removed, the same matmul violates the loaded decode sub-device's core set. The selector accepts no `program_config` and knows nothing about sub-devices |
| **D-C7** | the Milestone C L1 redesign | a closed model does not return its L1; one model per process |
| **D-C1** | a **decision** | decode's page-table validator cannot separate a prefill-shaped table from a legitimate L1-sharded repeat. Three attempts have declined the fix as a boundary violation |
| **D-C4** | a decision | `paged_attention_config=None` is the default pool, not a contiguous cache, so area 1's gate as *worded* is unreachable. Attempt 3 measured the reachable form instead |
| **D-C2** | a product decision | is a sampling seed per-request or per-(request, slot)? |
| **D-C3** | whoever owns `lazy_weight.py` | weight-cache fingerprint contains `MeshDevice.id()`; 138 GB and 26 min per extra model in a process |
| **F-C3** | `mb-signoff` wording | one pre-existing `models.demos` import under `models/common/tests/modules/moe/` |
| **D-C6, G-C1, G-C2, G-C3, F-C1, F-C2** | as §A2 leaves them | |

## Status of the five areas

Filled in as runs land; `IN FLIGHT` means exactly that.

| Area | Llama | Qwen |
| --- | --- | --- |
| 1 paged KV | IN FLIGHT | **PASS** for the two-pool PCC claim (cross-process), rest IN FLIGHT |
| 2 concat-32 | IN FLIGHT | IN FLIGHT |
| 3 prefix / chunked | **PASS** for prefix-vs-uncached; rest IN FLIGHT | **PASS** for prefix-vs-uncached; rest IN FLIGHT |
| 4 device sampling | **BLOCKED** by D-C5 then D-C8, measured | **BLOCKED** by D-C5 then D-C8, measured |
| 5 long context | **PASS** 4K/32K/128K | **PASS** 4K/32K/128K |
| repeat & cleanup | **FAIL 3/3**, L1 address clash, deterministic | **PASS 3/3** on one live model; **FAIL** on two models in one process (D-C7) |

## If you are attempt 4 rather than `mb-signoff`

Read `RESULTS_A3.md` first, not the report: one row per run, the log name, and how
many fresh processes each claim got. `queue.txt` is the resume point and is
consumed line by line by `cov_queue.sh`; anything still in it has not run.
Re-running what `RESULTS_A3.md` records is the only way to waste a Galaxy night.
