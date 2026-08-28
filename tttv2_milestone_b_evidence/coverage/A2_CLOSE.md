
## What attempt 2 committed

Tests, evidence and two docstring corrections. **No implementation file, in any
package.** Both boundary greps stay empty and the model-named import gate stays
at zero.

```text
models/common/tests/models/galaxy/test_step7_page_table_placement_wh_galaxy.py   new, 3 device cases
models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py   +1 case (x3 policies), docstring
models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py     docstring, `_distinct_rows` fallback
tttv2_milestone_b_evidence/coverage/                                            logs2/, RESULTS_A2.md, this section
```

Three test-level changes, and the reason for each:

1. **`test_llama_no_padded_vocabulary_id_is_ever_sampled`** — the case F-C1 said
   was vacuous. It is not; Llama pads 768 ids.
2. **`test_step7_page_table_placement_wh_galaxy.py`** — the one host assumption
   attempt 1 flagged as needing a mesh, as a test rather than a one-off script,
   because D-C1's write-up depends on it.
3. **`_distinct_rows` cyclic fallback** — the reference file holds 1024 tokens, so
   the straight window walk *skipped* every concat-32 length ≥ 1024, which are
   exactly the lengths the brief asks for last. A skip is not a result. The
   exact-window path is untouched, so results taken before the change are
   comparable.

None of these relaxes a threshold, a tolerance or a parametrization; (3) widens
one.

## What Milestone C inherits from this job

* **L1's remaining half — prefill after a decode — is now costed.** Five step-7
  cases cannot be measured behind it, and the list is in §A2's L1 section with
  the one untried hypothesis (confine the prefill mode plan to worker cores) and
  the one new fact that narrows it (the prefill matmuls are already
  worker-confined, so the clashing program is a collective or the MLP ring form).
* **D-C1** — decode's page-table validator cannot separate the prefill layout
  from a legitimate L1 repeat, and the premise is now confirmed on silicon. The
  fix requires changing a 2D-module expectation, which two attempts have now
  declined as a boundary violation. It needs a decision, not a patch.
* **D-C2** — is a sampling seed per-request or per-(request, slot)? A product
  decision about the serving contract.
* **G-C1, G-C2, G-C3, F-C2** — unchanged from attempt 1.
* **The device weight cache is unbounded.** Staging Llama's full interleaved and
  ring weight sets at this commit wrote **138 GB** in 26 minutes, on a filesystem
  with 1.0 TB free and 95% used. A step-7 sweep that resolves many recipes is a
  disk-capacity question as much as a device-time one.
