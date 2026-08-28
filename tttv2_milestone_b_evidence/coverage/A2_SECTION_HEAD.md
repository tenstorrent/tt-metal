
---

# §A2 — attempt 2, on a live mesh

Written 2026-08-27/28 by `mb-coverage` **attempt 2**, unattended, at commit
`b1e824537a4` (`mb-qwen` attempt 2's tip) on
`apbernal/tttv2_wh_glx_2d_modules_milestone_b`.

Everything above this line is attempt 1's report and is left untouched. It was
written with the mesh down and is a host-only document; where it and this section
disagree about the machine, this section was measured on silicon and is later.

## The three premises of attempt 1 that are false at this tree

| Attempt 1 said | At this tree |
| --- | --- |
| "The mesh never came back … `ttnn` cannot open a cluster at all." | **Alive.** `ls /sys/class/tenstorrent \| wc -l` = 32, and `test_partition_wh_galaxy.py` opens a real 8×4 cluster: `5 passed in 12.32s` (`logs2/a2_00_mesh_health.log`). Established before planning anything. |
| "Three consecutive device jobs have produced zero numerical results from silicon, for either model." | **False.** `mb-qwen` attempt 2 (17:53–22:51 UTC, after attempt 1) qualified both models end to end: Llama 501/511 top-1, Qwen 498/511, PCC 0.999+ per block for both. Its handoff is `job2_completion_handoff_attempt2.md`. |
| "Qwen's weights are not on this machine." | **Present**, under `HF_HOME=/localdev/ctr-apbernal/hf_data` — *not* `/proj_sw/user_dev/hf_data`, which reaches Llama only. |

Attempt 1 was not wrong about what it saw at 03:00 UTC. It was superseded by a
mesh repair at ~17:00 and by a job that ran after it. This is the same failure
mode its own handoff warned about ("evidence collected at a tree that has since
moved is not evidence") applied to the *machine* rather than the tree.

## F-C1 is superseded: Llama does pad its vocabulary, by 768 ids

Attempt 1's finding F-C1 reads: *"**Llama has no vocabulary padding.** 128256 is
already a multiple of `8 * 32`. Its padded-vocab gate is vacuous; only Qwen pads
(128 ids)."* Both halves are false at this tree, and the tree already knew:

```python
>>> from models.common.models.galaxy.recipes import galaxy_padded_vocab_size
>>> galaxy_padded_vocab_size(128256), galaxy_padded_vocab_size(151936)
(129024, 153600)
```

The width is not rounded to `8 * 32`; it is rounded so that the **per-device**
width is a whole number of 24-core ring rows — `(padded // 8) % (24 * 32) == 0` —
which is the invariant D-B19 was named for. `128256 // 8 = 16032` is 501 tiles,
which no usable core count divides, so Llama pads to 129024 and carries **768**
invalid ids; Qwen pads to 153600 and carries **1664**, not 128.

`test_step7_sampling.py` was corrected for this in `60fdec0c09e` (after attempt
1's commit), so the host suite is right. What was left wrong was the *device*
coverage: `test_step7_coverage_wh_galaxy.py` for Llama said in its module
docstring that the padded-vocabulary case is "not applicable" and omitted it.
Attempt 2 added `test_llama_no_padded_vocabulary_id_is_ever_sampled` at three
policies (greedy, T=1.5, T=0.5 — never T=1.0, which is its own reciprocal and
hides D4) and corrected both files' docstrings.

**Why it matters beyond bookkeeping:** an invalid id winning is a correctness
bug, and for Llama the gate was recorded as vacuous — i.e. nobody would ever
measure it.
