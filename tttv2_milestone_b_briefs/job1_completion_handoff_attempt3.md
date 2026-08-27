# Job 1 (`mb-llama`), attempt 3 → whoever runs next: completion handoff

Written 2026-08-27 by `mb-llama` attempt 3, unattended.
Full account: `tttv2_milestone_b_evidence/llama/REPORT.md` §"Attempt 3", run-by-run
in `.../ATTEMPT3.md`, environment in `.../ENVIRONMENT.md` §"Addendum — attempt 3",
logs in `.../logs3/`.

`job1_completion_handoff.md` is attempt 1's and
`job1_completion_handoff_attempt2.md` is attempt 2's. **Where any of the three
disagree, this one is later.** Everything they assert about the code they describe
was true when written; what changed is that the numbers now exist.

## The headline

**The step-2 gate is MET.** One Llama block is qualified in decode *and* prefill
against an independent Hugging Face reference, with the KV cache checked on both
sides, three runs in three fresh processes, bit-identical:

```text
prefill 128 logits                       0.999584002863212
prefill 128 cache K user 0/8/16/24       0.9999347766610057
prefill 128 cache V user 0/8/16/24       0.9997498179150203
prefill 2048 logits                      0.9996201066107949
prefill 2048 cache K user 0/8/16/24      0.9999333561149281
prefill 2048 cache V user 0/8/16/24      0.9997581361217711
decode position 128 logits u0/8/16/24    0.9997463458407887
decode 128 cache K user 0/8/16/24        0.9999342257320987
decode 128 cache V user 0/8/16/24        0.9997493345003990
```

Attempt 2's handoff opened by saying "there still is not" a Llama baseline for
Qwen to be compared against. **There is one now, and this is it.**

Step 3's status - the 80-layer model, the demo text and the teacher-forced
accuracy number - is in `REPORT.md` §A3.6. Read that section, not this summary,
before assuming anything about it.

## Read this first: the sentence attempt 3 would put on the wall

Attempt 2's generalisation was about *placement*: a decode program on cores the
sub-device manager does not own, or not told which sub-device it runs under. That
is still true and still the single most useful thing to know. Attempt 3's addition
is about *measurement*:

> **On this mesh the apparatus fails quietly more often than the graph does.**

Three of the defects across attempts 2 and 3 produced *no error at all*:

* `load_reference_tokens` returned a length-1 sequence, so the accuracy gate could
  only **skip** (attempt 2);
* `to_torch_auto_compose` composed the logits along the wrong mesh axis and the
  runner then sliced `[:, :vocab_size]`, which **narrows a 64128-wide tensor
  without raising** - every step-3 logit, argmax, demo token and accuracy number
  would have been computed from four copies of one mesh row's vocabulary slice,
  with nothing anywhere to say so (D-B23);
* the MLP read the attention's prefetched weights and returned confident garbage
  (D-B25a).

So: **before you trust a number, make the thing that produces it prove itself.**
Attempt 3's three cheapest and most valuable diagnostics were all of that kind, and
none of them needed a device run to design:

1. apply the *reference's own module* to the *device's own intermediate* - "HF's
   MLP on the device's MLP input" turned "the MLP is 0.096" into "the MLP is a
   wrong function, 0.085, and its input is not the problem" in one line;
2. compare a tensor at more than one window - splitting the KV comparison into
   prefix / full / appended-row turned "K is 0.0002" into "prefill's keys are
   intact and the decode write is `inf`";
3. report *every* user, not user 0 - four column-local users that prefill filled
   identically reported four *different* attention numbers, which said "placement
   across columns" before any op was suspected.

## Two things that are the same shape and must not be confused

Attempt 3 spent runs on both, and they look alike:

**A width that is over-covered is fine for a matmul and fatal for a CCL.** The ring
matmul's in0 is 2048 logical columns in a 24 x 96 = 2304 spec and that is correct -
production does the same, and the gathered tiles arrive in logical order. The LM
head's reduced logits were 501 tiles in a 42 x 12 = 504 spec and that **hung
forever**, because `all_reduce_async`'s reduction kernel does

```cpp
cb_in.wait_front(num_blocks * block_num_tiles);   // ring_size * shard, on EVERY core
```

so the last core waits for a shard the fabric will never fill. **Rule: a tensor
handed to `all_reduce_async` must have `logical_width == cores * shard_width`
exactly. A tensor handed to a matmul need not.**

**A tensor's declared topology is right when a mapper set it and wrong when an op
produced it.** `to_torch_auto_compose` is safe for a weight or a staged activation
and unsafe for anything that contracted a sharded axis - a matmul output carries
its *activation's* placements, not its weight's. Use `compose_galaxy_logits`, or
an explicit `ConcatMesh2dToTensor`, for anything downstream of a matmul.

## What Qwen inherits, on top of attempts 1 and 2's lists

Everything here is already in the shared tree; Qwen gets it by construction and
must **not** re-derive it.

1. **`galaxy_padded_vocab_size` pads to `GALAXY_ROWS * RING_ALIGNMENT`.** Qwen:
   151936 -> **153600**, i.e. 19200 per device, 600 tiles, 50 reduce cores x 12
   tiles. Exact. Attempt 2 predicted 50 cores from the old padding and it happens
   to be the same number, but the *tensor* is what changed. Without this Qwen hangs
   in the LM head all-reduce exactly as Llama did, with no abort and no traceback.
2. **`use_qk_fused_rotary` defaults to True.** Do not turn it off. The non-fused
   decode pair wrote a K of `|max| = inf` for Llama; Qwen's 64-head geometry makes
   the head-row asymmetry that exposes it *larger*.
3. **Only the MLP's projections are prefetched** (`("w1", "w3", "w2")`), and
   attention gets an `_UnprefetchedContext` so it still knows its worker
   sub-device. Qwen's attention decode matmuls are confined for the same L3 reason,
   so it has the same unconsumed-global-CB-entry problem. **Qwen must make the same
   change in its own model file** - `LLAMA33_70B_PREFETCHED_WEIGHT_NAMES` is
   Llama's; Qwen has its own list.
4. **`Prefetcher2DConfig.defer_global_cb`**, defaulted True by
   `build_galaxy_prefetcher_config`. Qwen inherits it free and needs it: without it
   the *first op of prefill* is unplaceable.
5. **`compose_galaxy_logits`** in `galaxy/collectives.py`, used by
   `GalaxyDirectRunner`. Qwen inherits the runner fix free; if Qwen's own device
   test composes logits itself, use this and not auto-compose.
6. **`LMHead2D` and `Sampling2D` accept a non-minimal padded vocabulary.** Both
   validations used to demand exactly the minimum. Qwen's ring-exact width is
   *further* from the minimum than Llama's (1536 columns vs 768), so Qwen would
   have hit both.
7. **`rope_2d.py`'s prefill table is tilized and its prefill transformation matrix
   is `TILE_SIZE`-square.** Shared module, so Qwen inherits both.
8. **`from_pretrained` takes a `load_hf_model` seam** and the step-3 tests inject
   `load_layer_subset_causal_lm` for layer subsets. Use it for iteration; the gates
   ignore it.
9. **`_assert_pcc` prints every PCC**, and the accuracy gate prints raw counts, and
   the demo prints its text. Copy that: a passing gate that records no number is
   not evidence.

### The one Qwen-specific warning attempt 2 gave, now half-answered

Attempt 2 wrote that Llama's invalid-logits mask is identically zero and Qwen's is
not, so "the mask placement is the one piece of the LM head change that Llama
cannot have qualified for you". **That is no longer true.** With the ring-exact
padding, Llama's `padded_vocab_size` (129024) exceeds its `vocab_size` (128256) by
768 columns, all of them in mesh row 7's shard, so the `-inf` mask add is
load-bearing for Llama too and it is exercised by every run above. The
`decode_stage_mask` path - `interleaved_to_sharded` of the mask into the sharded
output placement, then a broadcast add over 32 rows against a one-row mask - is
qualified.

## Harness — what attempt 3 added, and what it cost to learn

`run3.sh` (one cycle, CCL trace on, settable pytest deadline, resets into
`logs3/`) and `run3_sequence.sh` (a manifest of cycles, strictly serial) sit beside
attempt 1's and attempt 2's scripts. `device_run.sh` and `after_device_run.sh`
gained two backward-compatible knobs (`MB_PYTEST_TIMEOUT`, `MB_RESET_DIR`) and one
real behaviour change.

**1. A decode-mode `AssertionError` hangs the mesh exactly like a `TT_FATAL`.**
Attempt 2 knew the `TT_FATAL` case. The hang is in the `mesh_device` *fixture*
teardown, so **any** decode-mode failure leaves a process holding all 64
`/dev/tenstorrent` fds after its verdict is already in the log. `device_run.sh`
now starts a 90 s grace when a per-test verdict appears and then reaps, which saves
about ten minutes a run.

Keyed on the **per-test verdict**, not the session summary: the teardown hangs
*before* pytest writes the summary, so these logs can end at a bare `FAILED` with
no `=== N failed ===` line at all. If you are parsing them, do not assume the
summary exists.

**2. Never launch a cycle until the previous `run3.sh` has exited.** Attempt 3 lost
a whole run to this - `Waiting for lock 'CHIP_IN_USE_22_PCIe' ... held by PID
131554` - because the verdict in the log is not the moment the mesh is free; the
reap and the reset come after it. Worse, the recovery overlapped two
`tt-smi -glx_reset` calls and the second reported

```text
Error when re-initializing chips! It is not currently safe to communicate with ARC
because, another message is queued (0x2c)
```

The mesh survived, but **do not run two resets at once.** Use
`run3_sequence.sh`; it makes this impossible.

**3. `-o faulthandler_timeout=<n>` replaces `gdb`.** pytest's own faulthandler
plugin dumps every thread's Python stack from a watchdog thread and lets the run
continue. Attempt 2 needed `gdb -p` and a recovery attempt to locate a hang; this
is free. Diagnostic only; nothing committed depends on it.

**4. The CCL trace now synchronises per op.** Naming three enqueued ops was not
enough to find D-B19, because enqueues are asynchronous and the block landed on the
collective's own final `synchronize`. `_ccl_trace`'s `trace_step` now waits after
each op when `TTTV2_GALAXY_CCL_TRACE` is set, and prints each tensor's shard spec.
That converted "one of three ops" into a name in one run - and the shard specs it
printed contained the *cause*.

**5. `pgrep -f tt-smi` matches its own caller.** Any shell command whose text
contains `tt-smi` is matched, so a "is a reset still running?" check answers 2
forever. Use `pgrep -x tt-smi`. Same family as the `pgrep -af pytest` self-kill the
house rules warn about.

Two costs worth budgeting, both measured on this machine:

* the **141 GB checkpoint load is only about 60 seconds** (safetensors, 723
  tensors). Attempt 2 expected it to dominate the night; it does not.
* what *does* cost is **staging 80 layers of weights to device the first time**:
  about 2.3 s per tensor of `LazyWeight` cache generation, roughly 400 tensors,
  so ~15 minutes. It is a **one-off** - a second process in the same tree gets
  `[cache hit]` at about 0.05 s per tensor. So the first 80-layer run of a session
  is expensive and every one after it is not. Plan the order accordingly.

## Do not

* **Do not turn `use_qk_fused_rotary` off**, and do not "simplify" the rotary
  adapter to the non-fused pair. See D-B25b: `|max| = inf` in the K cache, silently,
  with V exact beside it.
* **Do not register a weight with the prefetcher unless the matmul that consumes it
  runs on the ring.** An unconsumed global-CB entry does not error; it shifts every
  later consumer by one.
* **Do not use `to_torch_auto_compose` on anything a matmul produced.** It is right
  for mapper-placed tensors and silently wrong for op-produced ones.
* **Do not assume a width that a matmul tolerates is a width a CCL tolerates.**
* **Do not compare a device KV cache against HF's `past_key_values` without
  permuting K.** The device is Meta-interleaved, HF is split; they cancel inside
  `Q.K^T`, so the *logits* will agree while the caches score ~0.04 and it reads as
  a model failure.
* **Do not pass `models/common/tests/modules` as a directory** to pytest - it
  collects the 1D device suites and takes the mesh for ten minutes.
* Do not edit `models/common/modules/MILESTONE_A_STATUS.md` or
  `tttv2_2d_modules_plan.md`; job 0 and job 4 own them. Proposed L3 text is in
  `REPORT.md` §A3.9.
* Do not touch `models/common/modules/**/*_1d.py` or `models/common/llm_runtime/**`.
  Both greps are empty across all three attempts.
* Do not raise `after_device_run.sh`'s reset cap back to 600 s. It is 900 s for a
  measured reason (attempt 2).

## Step 3, and the one thing that is not met

| Step-3 item | Result |
| --- | --- |
| Full-model prefill plus first decode token | **PASS** — 80 layers, both predictions inside the reference top-5 (`logs3/a3_43`) |
| Teacher-forced decode, the accuracy gate | **PASS** — top-1 **501/511 = 0.9804** (gate 0.91), top-5 **511/511 = 1.0000** (gate 0.99) (`logs3/a3_44`) |
| The direct demo producing real text | **PASS** — see below (`logs3/a3_45`) |
| Batch 1 | **PASS** — the demo and the accuracy gate are both batch 1 |
| Batch 32 | **PASS via a single runner**; the two-runner isolation test hits a named limitation |

```text
[demo] slot 0 prompt: 'Explain what a tensor is to a software engineer in two sentences.'
[demo] slot 0 text  : 'A tensor is a multi-dimensional array of numerical values, similar to a matrix,'
```

**The one thing that is not met**, and it is a limitation with a name rather than a
defect. `test_llama33_70b_galaxy_batch32_slots_are_isolated` opens **two runners in
one process** - slot 0 alone, then all 32 - so the second one *prefills after the
first has decoded*, and by then `activate("decode")` has allocated the global
circular buffer, which nothing frees:

```text
TT_THROW ... Statically allocated circular buffers in program 100 clash with L1
             buffers on core range [0-0 - 0-3]
```

Same program, same op, same four sender cores as D-B20. **D-B20's fix narrowed
limitation L1 rather than removing it**: prefill-before-any-decode is now fine,
prefill-after-a-decode is not. Production has the same property. Batch 32 itself is
not blocked - the demo's batch-32 test prefills all 32 slots before any decodes and
passes.

**This is the first thing to do next**, and here is what to know before trying it.
The fix is to release the global CB on `activate("prefill")` and recreate it on the
next `activate("decode")`, behind a config flag, with that test as the oracle. It
was deliberately **not** attempted here: it changes the mode-switching of the one
module every qualified decode path depends on, and turning it on would put every
number in this handoff back in doubt with no budget left to re-take them. Two traps:

* the recreated buffer must land at the **same L1 address**, or the decode programs
  already in the ttnn program cache hold stale addresses - that is a silent
  corruption, not an error;
* **every** reference has to be dropped. Attempt 1 found that `cleanup()` alone
  does not free it because the mode contexts still hold handles; `self._global_cb`
  and `self._contexts["decode"].global_cb` are both live references.

## Where attempt 3 stopped, in one paragraph

The step-2 gate is met and measured three times identically; prefill at 128 and
2048 and decode at batch 32 are all correct against an independent Hugging Face
reference, with the K and V caches checked after both; the full 80-layer model
prefills and decodes real reference tokens inside the reference top-5; the
Milestone B teacher-forced accuracy gate for Llama passes at top-1 98.04% and
top-5 100.00%; and the 80-layer demo produces fluent English. Seven defects were
found and fixed, two of which produced no error of any kind. The one step-3 item
left is two runners in one process, which is limitation L1's remaining half, named
and reproduced rather than worked around.

**If you are `mb-qwen`: there is now a Llama baseline to compare against, and it is
the numbers at the top of this document.** Read `REPORT.md` §A3.5 for the list of
things you inherit for free and §A3.2 for the seven defects, because Qwen has all
seven in front of it and five of them are already fixed in shared code.
