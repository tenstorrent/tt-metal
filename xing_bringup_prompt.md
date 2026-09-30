# Xing4.0-29B-A4B prefill bring-up (LoudBox, 8 chips)

## 0. One-time setup on the machine (do this before starting Claude Code)

The `/bringup` skill lives in the repo and is installed as links in `~/.claude/skills`. From the tt-metal checkout:

```bash
cd <your tt-metal checkout>
git fetch origin && git checkout dnijemcevic/ernie45_prefill && git pull
./build_metal.sh                     # needed once per checkout / after pulling C++ changes (the bring-up forks)
mkdir -p ~/.claude/skills
for s in bringup bringup-fork-op bringup-fork-tests; do
  ln -sfn "$PWD/models/demos/common/bringup/skill/$s" ~/.claude/skills/$s
done
ls -l ~/.claude/skills               # the three links must point into this checkout
```

Then start Claude Code from the checkout (for a long run, inside tmux: `tmux new -s xing`; detach with Ctrl-b d;
re-attach with `tmux attach -t xing`) and paste everything below the line.

---

/bringup

Bring up chunked prefill of Xing4.0-29B-A4B from Hugging Face with the gated bring-up framework in
`models/demos/common/bringup`. This machine is a LoudBox with 8 chips. Repo: this checkout, branch
dnijemcevic/ernie45_prefill (pushed at 1e84de37e97 or later). It is forked from llk_helper_library: compare against
origin/llk_helper_library, never main.

Before any device work, check that no other job uses the device (`ps -ef | grep -E "run_safe_pytest|tt-probe"`).
One device job at a time. Check the box yourself (`ls /dev/tenstorrent | wc -l`, `tt-smi -ls` for the chip type) and
tell me what you find.

Read first:
- `models/demos/common/bringup/README.md` and `docs/pipeline_design.html` (the framework).
- `models/demos/common/bringup/docs/test_speedup_notes.md` (swap tests now check themselves and freeze without a review
  agent; `agents.swap_review` brings the review back).
- `models/demos/hy4_preview_d_p/bringup/supervision.md` and `models/demos/mimo_v2_6_d_p_2x2/bringup/supervision.md`
  (the last two supervised runs, including what went wrong).
- `models/demos/common/bringup/knowledge/known_issues.md` and `repo_map.md`.
- `ttnn/ttnn/bringup/INDEX.md` (the bring-up forks: reuse or extend them; never edit an existing TTNN op).

## Intake: find out, then propose defaults to me

- The exact HF repo id. Pin the revision (commit sha). Size, gated or not, license.
- Architecture from config.json and the HF modeling code: layers, hidden size, attention type per layer, heads and
  KV heads, experts, top-k and shared experts, anything unusual. Base or instruct (raw vs model_turn input); any
  thinking / reasoning template kwargs.
- Whether python_env's transformers has this model. If not, vendor the newer modeling code into the artifacts folder;
  never upgrade the shared python_env.
- Fit: at about 29B parameters the whole model should fit 8 chips. Compute it; prefer all layers over a subset.
- Mesh for 8 chips (1x8 or 2x4) and fabric. My standing rule is FABRIC_2D; if the box cannot do it, stop and ask me.
- Target 56320 tokens in 5120-token chunks, the default ladder. HiFi4 everywhere (agent rule 7); bfp8 expert weights
  unless the checkpoint itself is 4-bit.
- Dashboards: both styles (standard + teletext); give me both links at the start and at the end.
- Owner rules for `agents.rules`: always FABRIC_2D; text decoder only (no vision, audio or MTP); point the agents at
  the closest existing repo models you found; be aware of the ttnn.bringup forks and use them as they see fit.

Show me the spec and wait for my explicit "yes". Then approve the intake and launch as the skill says.

## Lessons from earlier runs

- Check that the HF model itself works. In Hy4 the HF port had the wrong RoPE layout: the smoke test passed ("Paris")
  but next-token accuracy was 0.24 until it was fixed. Low accuracy means find the bug; never lower the floor
  yourself.
- The whole-model HF sanity check needs the whole checkpoint. Check free disk before downloading. For a layer subset,
  task R.4 trims the checkpoint afterwards.
- Performance picks are MY call: show me the opportunity list and wait. A vague "etc" from me never covers them.
- Review every gate commit for cheating (loosened thresholds, edited frozen tests, CPU work in device code, reading
  the golden inside the model), and every op-gen deferral the same way.
- Never edit files in the tree while an agent step runs. Export the dashboards outside the tree during agent steps
  (`dashboard.export --out <scratchpad>`) and republish both after every gate. Dashboards over 500 KB fail the repo's
  large-file hook: publish them, do not commit them.
- If an agent says the fix is outside its allowed files (a gate setting, a framework change), pause the run and fix
  it yourself instead of letting the orchestrator retry.
- Commit each verified piece right away, with explicit paths. Never push unless I ask. Never run `tt-smi -r` unless I
  say so. Run `run_safe_pytest.sh` / `tt-probe.sh` in the foreground. Never wrap orchestrator runs in `timeout`.
- Decide routine things yourself. Ask me only for: the spec, the plan, perf picks, op-gen requests, board resets and
  pushes.
- Keep your answers to me short and in plain language. No walls of text.

At the end, report: per-layer PCC, the 50k->55k chunk device time and the 0->55k prefill time, which bring-up forks
were reused, extended or created, and whether any step was deferred to op-gen.
