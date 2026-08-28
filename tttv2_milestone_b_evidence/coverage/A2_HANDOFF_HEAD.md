# Job 3 (`mb-coverage`) attempt 2 → `mb-signoff`: completion handoff

Written 2026-08-27/28 by `mb-coverage` **attempt 2**, unattended, on
`apbernal/tttv2_wh_glx_2d_modules_milestone_b`.

Full account: `tttv2_milestone_b_evidence/coverage/REPORT.md` §A2.
Run-by-run index: `tttv2_milestone_b_evidence/coverage/RESULTS_A2.md`.
Machine and mesh facts, costs, the exact harness:
`tttv2_milestone_b_evidence/coverage/ENVIRONMENT.md`.

## Read this paragraph first

**Do not plan around `job3_completion_handoff.md` (attempt 1).** Its headline —
*"the mesh never came back … three consecutive device jobs have produced zero
numerical results from silicon, for either model"* — was true when written at
03:31 UTC and is false now. The mesh was repaired, `mb-qwen` attempt 2 then
qualified **both** models end to end on silicon (17:53–22:51 UTC), and attempt 2
of this job measured step 7 on a live 8×4 mesh. Attempt 1's host analysis is
still good and is still worth reading; its verdict is not.

Three of its statements are simply wrong at this tree, and one of them changes a
gate:

* the mesh is alive (`ls /sys/class/tenstorrent | wc -l` = 32, a real cluster
  opens in 12 s);
* Qwen's weights are on this machine, under
  `HF_HOME=/localdev/ctr-apbernal/hf_data`;
* **Llama does pad its vocabulary** — by 768 ids, `galaxy_padded_vocab_size(128256)
  = 129024`. Attempt 1's finding F-C1 says the opposite and calls Llama's
  padded-vocab gate vacuous. It is live, and attempt 2 added the device case.
