# Llama's state, distilled for `mb-qwen`

Written 2026-08-27, after `mb-llama` attempt 3 met its finish condition.

**This file replaces reading the three `job1_completion_handoff*.md` documents and
`REPORT.md` in full.** Those are the record; this is what you need in order to
work. Every fact below was re-verified against the tree and the machine on the day
it was written, not copied forward from a narrative.

## What to read, and what to leave alone

| Document | Lines | Read it? |
| --- | --- | --- |
| **this file** | ~430 | **yes, in full** |
| `job1_completion_handoff_attempt3.md` | 315 | only if this file sends you there |
| `job1_completion_handoff{,_attempt2}.md` | 708 | **no** — attempts 1 and 2, superseded here |
| `tttv2_milestone_b_evidence/llama/REPORT.md` | 1768 | **by section only**, never whole (index at the end) |
| `.../ATTEMPT3.md`, `ATTEMPT2.md` | 1836 | only to check a specific run's raw numbers |

Reading the whole Llama package costs ~5,500 lines of context; what it carries
that you can act on is below, in a tenth of that. Where anything in those files
disagrees with this one, **this
one is later** — including attempt 2's "you must set the stage masks" and attempt
1's "port `_relocate`", both of which are now done or derived.

## 1. What a passing Galaxy model looks like

Llama's step-2 gate, three fresh processes, bit-identical, against an independent
Hugging Face reference:

```text
prefill 128 logits                       0.999584002863212
prefill 128 cache K / V  (users 0,8,16,24)  0.9999347766610057 / 0.9997498179150203
prefill 2048 logits                      0.9996201066107949
decode position 128 logits (u0,8,16,24)  0.9997463458407887
decode 128 cache K / V   (users 0,8,16,24) 0.9999342257320987 / 0.9997493345003990
```

80 layers, teacher-forced, batch 1, prefill 512 / decode 511, three fresh
processes with **identical counts**: top-1 501/511 = 0.9804, top-5 511/511 =
1.0000. The demo emits fluent English, character-identical across three runs and
identical again when slot 0 is served alongside 31 others.

Two numbers to calibrate against, so they do not alarm you at 3 a.m.:

* the **residual stream** after a layer scores ~0.979 and the final norm ~0.990
  against an fp32 reference while the *logits* score 0.99975. bfloat16 residual
  against fp32 reference is the floor, not the gate — the LM head contracts 8192
  terms per logit and averages the quantisation noise out.
* every gate was **bit-identical across processes**. On this mesh that is a
  property worth testing rather than assuming: a bfloat16 cross-device logit sum
  is order-dependent on ETH ring arrival. `fp32_dest_acc=True` on the LM head
  all-reduce is what buys it. If your three runs *differ*, that is a defect of
  exactly the kind to chase, not to average.

## 2. Before anything else: `HF_HOME`

Qwen3-32B **is** on this machine. Attempt 1 of `mb-qwen` reported it was not, and
that report is wrong — it searched one cache.

```text
/localdev/ctr-apbernal/hf_data/hub/models--Qwen--Qwen3-32B   62 G, 17/17 shards, 0 .incomplete
/proj_sw/user_dev/Qwen/models--Qwen--Qwen3-32B               76 G, same revision
revision 9216db5781bf21249d130ec9da846c4624c16137  (the one QWEN3_32B_GALAXY DEFAULT_HF_REVISION pins)
```

Verified by resolution, offline, both directions:

```text
HF_HOME=/localdev/ctr-apbernal/hf_data     Qwen/Qwen3-32B  -> OK   qwen3, 64 heads, vocab 151936
                                           Llama-3.3-70B   -> OK   (4 KB symlink farm into /proj_sw)
HF_HOME=/proj_sw/user_dev/hf_data          Qwen/Qwen3-32B  -> OSError -> pytest.skip
```

**Use `HF_HOME=/localdev/ctr-apbernal/hf_data`.** It reaches both models.
`run3.sh`, `device_run.sh` and `run_sequence.sh` under
`tttv2_milestone_b_evidence/llama/` all hardcode the `/proj_sw` value, and
`ENVIRONMENT.md` states that either path "reaches the same shards" — true for
Llama, false for Qwen. Fix the export in the first script you copy, and check
`[skipped]` never appears in a run you intend to count.

`hf_config_or_skip` skips on any resolution failure, which is the honest behaviour
and also means **a wrong `HF_HOME` produces a green-looking run that measured
nothing.**

## 3. Six deltas between Llama's qualified model file and yours

Verified in the tree today. These are Llama's fixes that live in *model* code, so
Qwen does not get them by construction — unlike §4, which it does.

**3.1 Your prefetcher registers weights the ring never consumes.** *(defect D-B25a
— cost 4 device runs to find, produced no error of any kind)*

```python
models/common/models/qwen3_32b_galaxy/model.py:126
QWEN3_32B_PREFETCHED_WEIGHT_NAMES = ("wqkv", "wo", "w1", "w3", "w2")   # yours
models/common/models/llama33_70b_galaxy/model.py:138
LLAMA33_70B_PREFETCHED_WEIGHT_NAMES = ("w1", "w3", "w2")               # after the fix
```

A prefetched matmul takes its weight from the global circular buffer **in
registration order**. Only the 24 ring cores receive that buffer. Llama's
attention decode projections are confined to a worker rectangle (Milestone A's L3
constraint), so they never read it — and registering them put two unconsumed
entries per layer into the buffer, shifting every later consumer by one. The MLP's
`w1` read the entry meant for `wqkv`.

The symptom was decode logits at PCC 0.063 with **every configuration field
correct**: the MLP's memory configs, program configs, ring core coordinates and
mesh mappers were all bit-equal to the qualified Milestone A decode fixture. The
probe that settled it in one line was applying *HF's own MLP to the device's own
MLP input* — 0.085, so a wrong function, not a wrong input.

Fix: register **exactly the weights whose matmuls run on the 24-core ring**, and
give attention `_UnprefetchedContext` (`llama33_70b_galaxy/model.py:476`, used at
:869) so it keeps its worker sub-device id — without it a ttnn matmul defaults to
sub-device **zero**, the prefetch senders. Your attention passes the raw
`decode_context` (`qwen3_32b_galaxy/model.py:857`; the MLP's at :871 is correct as
it stands). Check which of your decode matmuls actually
resolve to the ring before you decide the list; do not copy Llama's tuple blind.

**3.2 Your rotary defaults to the non-fused pair, which writes an infinite K.**
*(D-B25b)*

```python
models/common/models/qwen3_32b_galaxy/hf_adaptor.py:275    use_qk_fused_rotary: bool = False
models/common/models/llama33_70b_galaxy/hf_adaptor.py:283  use_qk_fused_rotary: bool = True
```

On a prefetcher mesh, production chooses `rotary_embedding_llama_fused_qk`; the
non-fused pair is the Blackhole fallback and wants a different cos/sin layout. The
observed result on Llama was a decode K cache with `|max| = inf` on user 0 and
`8.773e+37` on user 8 — different garbage per column, i.e. uninitialised memory —
with **V exact beside it**, because V does not pass through RoPE. Attention output
read 0.737/0.669/0.695/0.597 on the four column-local users that prefill had
filled identically.

Set it True. One flag switches `RotarySetup2D` to the expanded table layout and
`GalaxyAttentionCollectives` to the fused call together. Qwen's 64-head geometry
makes the head-row asymmetry that exposes this *larger*, not smaller.

**3.3 Your decode LM head is wired the way Llama's was before attempts 2 and 3.**

```python
models/common/models/qwen3_32b_galaxy/model.py:1056-1073
    decode_input_memcfg=ttnn.L1_MEMORY_CONFIG,      # interleaved
    decode_output_memcfg=ttnn.L1_MEMORY_CONFIG,     # interleaved
    decode_output_dtype=precision.decode_activation_dtype,   # bfloat16
    # no decode_program_configs, no decode_sub_device_id, no prefill_sub_device_id
```

Llama's (`llama33_70b_galaxy/model.py:1099`) uses
`decode.lm_head_input_memcfg` / `lm_head_output_memcfg` / `lm_head_program_config`
— the 24-core gather-in0 ring, which is the only placement whose circular buffers
fit beside the resident decode activations — both sub-device ids as lambdas over
`resources.context(...).worker_sub_device_id`, and
`precision.lm_head_output_dtype` = **bfloat8_b**.

Each omission has a known symptom: an unset sub-device id defaults the ring matmul
to the prefetch senders (`TT_FATAL ... Kernel group cores do not match sub device
cores`); a bfloat16 reduction buffer is `GALAXY_COLUMNS` × the logits width, ~96 kB
per core, and clashes with the ring matmul's CBs on the cores they share — no core
count fixes it, bfloat16 cannot get below ~82 kB. bfloat8_b is the production
value and `fp32_dest_acc=True` keeps the cross-device sum in fp32 regardless.
`recipes.py` resolves all three placements for any geometry; you only have to wire
them.

**3.4 Your padded vocabulary is already correct — do not touch it.**

`qwen3_32b_galaxy/model.py:275` calls the shared `galaxy_padded_vocab_size`, which
attempt 3 changed to pad to `GALAXY_ROWS * RING_ALIGNMENT` (8 × 768):

```text
Qwen3-32B   151936 -> 153600   19200/device, 600 tiles, 50 reduce cores x 12, 1664 masked columns
Llama-3.3   128256 -> 129024   16128/device, 504 tiles, 42 reduce cores x 12,  768 masked columns
```

This is not cosmetic. `all_reduce_async`'s reduction kernel does
`cb_in.wait_front(num_blocks * block_num_tiles)` on **every** output core with one
uniform shard size, so a tensor whose width is not exactly `cores × shard_width`
leaves the last core waiting for tiles the fabric will never send: no abort, no
traceback, the host blocks in `wait_for_outstanding_reads`, mesh reset required.
That was D-B19, and it cost attempt 2's night and part of attempt 3's.

**Rule: a tensor handed to `all_reduce_async` must have
`logical_width == cores * shard_width` exactly. A tensor handed to a matmul need
not** — the ring matmul's in0 is 2048 logical columns in a 24 × 96 = 2304 spec and
that is correct. The two look identical and are not.

**3.5 The invalid-logits mask needs nothing set.** Attempt 2's handoff says "you
must set" `decode_stage_mask` / `prefill_stage_mask`. That is superseded:
`_resolve_lm_head2d_config` (`lm_head_2d.py:368`) **derives** both from
`output_memcfg.is_sharded()` and overwrites whatever you pass. The mask *add* is
unconditional; the flag only decides whether the mask is first
`interleaved_to_sharded` into the logits' own placement, which is what keeps the
add inside the sub-device. Do 3.3 and this follows.

The path is qualified for Llama too now — under the ring-exact padding Llama
carries 768 masked columns in mesh row 7's shard, so attempt 2's warning that
"Llama cannot have qualified the mask placement for you" no longer applies.

**3.6 Two of attempt 1's warnings are already discharged in your tree.** Do not
spend time on them: `_relocate` is ported (`qwen3_32b_galaxy/model.py:1116`, with
the reasoning in its docstring), and your decode embedding already targets
`decode_residual_memcfg` rather than interleaved L1 (`:1420`).

## 4. What you inherit for free, and must not re-derive

All shared, all in the tree, all exercised by Llama's passing runs.

1. **`prefetcher_2d.py: defer_global_cb`**, defaulted True by
   `build_galaxy_prefetcher_config`. Without it the **first op of prefill** is
   unplaceable: `seal()` used to allocate the 792,064-byte global CB eagerly on
   the sender columns, and nothing can free it.
2. **`rope_2d.py`** — the prefill cos/sin table is tilized (decode's
   `ttnn.embedding` needs row-major, prefill's `rotary_embedding_llama` needs
   tiled; one module, two consumers, no single layout serves both), and the
   prefill transformation matrix is `TILE_SIZE`-square, not `head_dim`-square.
   Both are corrections where the module disagreed with the op.
3. **`compose_galaxy_logits`** in `galaxy/collectives.py`, used by
   `GalaxyDirectRunner`. If your device test composes logits itself, use this.
4. **`LMHead2D` and `Sampling2D` accept a non-minimal padded vocabulary.** Both
   validations used to demand exactly the minimum and would have rejected 153600
   outright.
5. **`ring_matmul_program_config` takes `global_cb_receivers`** (default 2, the
   MLP's qualified value; pass 1 for a non-prefetched matmul), and `recipes.py`
   resolves `lm_head_input_memcfg` / `output_memcfg` / `program_config` for any
   geometry.
6. **`rmsnorm_2d.py`'s `_decode_distributed` no longer deallocates its own return
   value.** `to_memory_config` returns the *same* tensor when the config already
   matches, and nanobind hands it back as a new Python wrapper, so `is not` could
   not tell "no copy" from "copy". Latent since Milestone A. Your layer 0 hits it.
7. **`galaxy_checkpoint.load_layer_subset_causal_lm(hf_model, layer_indices=(0,))`**
   reads only the safetensors shards holding the layers you ask for, plus the
   embedding, final norm and LM head, and its tensors were verified bitwise equal
   to the shards. It is model-agnostic and works for Qwen unchanged.
   **The seam that injects it is not shared, though**: `llama33_70b_galaxy/hf_adaptor.py:285`
   gives `from_pretrained` a `load_hf_model` parameter and the tests and demo pass
   the subset loader through it (`test_full_model_wh_galaxy.py:105`, `demo.py:96`).
   Qwen's adaptor has no such parameter — add it. It is a few lines and it is the
   difference between a multi-minute and a ~40-second setup on every iteration,
   with the three-runs rule multiplying that. The gates ignore the knob and run
   all layers.
8. **`load_reference_tokens` returns a 1-D sequence.** It used to return length 1,
   so the accuracy gate could only `pytest.skip` — it failed *open* for a whole
   night.
9. **`galaxy/plans.py: build_galaxy_decode_collectives` takes `residual_dtype`**
   (default bfloat16) and sizes the shared axis-0 all-reduce buffer with it.
   **Check your precision recipe agrees.** A mismatch reads as
   `Cannot set circular buffer size to 65536 ... larger than the ... bank size of
   34816 B` — 34816 is a `[32,1024]` bfloat8_b shard, 65536 the bfloat16 one.

## 5. The sentence to keep in your head

> **On this mesh the apparatus fails quietly more often than the graph does.**

Four defects across the three attempts produced **no error at all**: the reference
loader returning one token so the gate could only skip; logits composed along the
wrong mesh axis and then narrowed by a `[:, :vocab_size]` slice that does not
raise on a too-wide tensor; the MLP reading another op's weights; and the K cache
holding `inf` while the logits still scored plausibly.

The three cheapest diagnostics, none of which needed a device run to design:

1. **apply the reference's own module to the device's own intermediate** —
   separates "wrong function" from "wrong input" in one line;
2. **compare a tensor at more than one window** — prefix / full / appended row
   turned "K is 0.0002" into "prefill's keys are intact and the decode write is
   `inf`";
3. **report every user, not user 0** — four column-local users that prefill filled
   identically reporting four different numbers says "placement across columns"
   before any op is suspected.

Corollaries worth stating as rules:

* **`to_torch_auto_compose` is safe for a mapper-placed tensor and silently wrong
  for anything an op produced** that contracted a sharded axis — a matmul output
  carries its *activation's* topology, not its weight's.
* **Do not compare a device KV cache against HF's `past_key_values` without
  permuting K.** The device is Meta-interleaved, HF is split; they cancel inside
  `Q·Kᵀ`, so the logits agree while the caches score ~0.04 and it reads as a model
  failure. Use `reverse_permute_1d` from `models/common/tests/modules/_hf_reference.py`.
* **Print every PCC, passing or failing**, and print raw counts, not percentages.
  A passing gate that records no number is not evidence.

## 6. The placement rule, and the ops that ignore it

Measured on this mesh — a mocked mesh cannot tell you this:

```text
compute grid      x=0..6, y=0..9                              70 cores
worker_cores()    {[1-0 - 3-9], [5-0 - 6-9]}                  50 cores
prefetch senders  x=0 and x=4                                 12 cores
in NO sub-device  {[0-1 - 0-3], [0-6 - 0-8], [4-3], [4-8]}     8 cores
```

The worker envelope is **not contiguous** (the `x=4` sender column splits it) and
its bounding box includes senders; sender ∪ worker does **not** cover the grid.
Any program built over the full compute grid is illegal under the decode manager,
and the abort leaves the mesh un-drainable.

Ops that are **not** sub-device aware: `ttnn::prim::copy` (i.e. the three-argument
`to_memory_config`), the `ttnn.typecast` fallback, generic reshard, and
`ttnn.reduce_scatter` (uses the bounding box). Safe instead:
`sharded_to_interleaved` / `interleaved_to_sharded` (they run on their sharded
side's cores and take `output_dtype`, so a recast rides along),
`reduce_scatter_minimal_async` and `all_gather_async` (they intersect the real
`CoreRangeSet`), `ttnn.all_gather` *given* `sub_core_grids`, `ttnn.embedding` with
a **sharded** output, and matmul program configs with **`allowed_worker_cores`**
set.

Run this before any model work, every session — no checkpoint, no weights, **13
seconds**, and it is also the cheapest mesh health check you have:

```sh
python -u -m pytest -v -rA --color=no -p no:cacheprovider --timeout=900 \
  models/common/tests/models/galaxy/test_partition_wh_galaxy.py
```

## 7. Limitations with names, so you do not diagnose them twice

**L1's remaining half — prefill after a decode.** Once `activate("decode")` has
allocated the global CB, a later prefill is unplaceable on the sender columns:

```text
TT_THROW ... Statically allocated circular buffers in program 100 clash with L1
             buffers on core range [0-0 - 0-3]
```

Prefill-before-any-decode is fine; two runners in one process is not. Production
has the same property. **The obvious fix is implemented and hardware refuted it**
— `Prefetcher2DConfig.release_global_cb_on_prefill` (default off) releases the
buffer and the L1 base address is *identical*, because a `global_circular_buffer`
has no `deallocate`. Do not spend that run again. The open hypothesis is to
confine the prefill mode plan to the **worker** cores so a full-grid prefill
program never needs the sender columns; it costs 20 of 70 cores for prefill and
re-taking every prefill number behind it. It is not required by any Milestone B
gate.

**L3 is CLOSED** — attention decode executes on the prefetch partition, both
matmuls, numerically correct. `REPORT.md` §A3.9 has the proposed
`MILESTONE_A_STATUS.md` wording; **do not edit that file**, job 0 and job 4 own it.

## 8. Harness — the rules each of which cost a run

* **Never start a device cycle until the previous one has actually exited.** The
  pytest verdict in the log is not that moment; the reap and the reset come after
  it. Overlapping runs give you `Waiting for lock 'CHIP_IN_USE_22_PCIe'` and a
  wasted slot. Use a serial manifest (`run3_sequence.sh`).
* **Never run two `tt-smi -glx_reset` at once** — the second reports `not
  currently safe to communicate with ARC` and you are one bad step from a dead
  board.
* **A decode-mode `AssertionError` hangs the mesh exactly like a `TT_FATAL`.** The
  hang is in the `mesh_device` fixture teardown, so *any* decode-mode failure
  leaves a process holding all 64 `/dev/tenstorrent` fds **after its verdict is
  already in the log**. Start a grace timer on the per-test verdict and reap.
* **Keyed on the per-test verdict, not the session summary** — teardown hangs
  before pytest writes the summary, so a log can end at a bare `FAILED` with no
  `=== N failed ===` line. Do not parse for the summary.
* **`-o faulthandler_timeout=<n>` replaces `gdb`** for locating a hang: it dumps
  every thread's Python stack from a watchdog thread and lets the run continue.
* **`pgrep -f tt-smi` matches its own caller.** Use `pgrep -x tt-smi`. Same family
  as the `pgrep -af pytest` self-kill.
* **Do not pass `models/common/tests/modules` as a directory to pytest** — it
  collects the 1D device suites and takes the mesh for ten minutes. The working
  host gate is `tttv2_milestone_b_evidence/llama/host_gate.sh`: explicit files plus
  `--ignore-glob="*_wh_galaxy*.py"`.
* **Print a stage name before every device call and flush.** When the mesh is left
  un-drainable the session never reaches its failure summary and the traceback
  dies with the process; the last `[stage] enter` line is then the only thing that
  says which call aborted. See `test_bringup_wh_galaxy.py::_stage`.

## 9. Costs, measured on this machine — plan the order with these

| Cost | Measured |
| --- | --- |
| `from_pretrained` on the 141 GB Llama checkpoint | **~60 s** (723 tensors) |
| Staging 80 layers to device, **first time in a tree** | ~1.5-2.3 s × 948 tensors ≈ **24 min** |
| The same staging, every process after | `[cache hit]`, ~0.05 s per tensor |
| One step-2 gate cycle (test + reap + reset) | ~7 min |
| One prefill-2048 cycle | ~9 min |
| The accuracy gate (511 eager decode steps) | 16-22 min |

The checkpoint load does **not** dominate a night; the first device staging does,
once. Order your runs so the cheapest full-scale test pays for the cache. Qwen is
32B against Llama's 70B, so expect less, not more.

Note the accuracy-gate wall time above was measured with `TTTV2_GALAXY_CCL_TRACE=1`
adding three device synchronizes per token. **It is not a decode-latency figure and
must not be reported as one.**

## 10. Llama's gates, if you touch shared code

The brief requires you to re-run these and record both results. They are cheap
once the weight cache is warm.

```sh
T=models/common/tests/models/llama33_70b_galaxy
# step-2 gate: prefill 128 + decode batch 32, logits and both caches   (~7 min)
$T/test_model_wh_galaxy.py::test_llama33_70b_galaxy_one_layer_prefill_and_decode
# single-row prefill at the full 2048 recipe                           (~9 min)
$T/test_model_wh_galaxy.py::test_llama33_70b_galaxy_one_layer_prefill_2048
# 80 layers, prefill + first decode token                              (~29 min cold)
$T/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_full_model_prefill_and_first_decode_token
# the Milestone B accuracy gate                                        (~16-22 min)
$T/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_teacher_forced_accuracy_batch1
# the demo, batch 1 and batch 32
models/common/models/llama33_70b_galaxy/demo.py::test_llama33_70b_galaxy_direct_demo_batch1
models/common/models/llama33_70b_galaxy/demo.py::test_llama33_70b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination
```

The sub-module bisection that located four of the seven defects is
`$T/test_model_wh_galaxy.py::test_llama33_70b_galaxy_decode_bisection` — it reports
a PCC at every boundary inside layer 0 against HF forward hooks and asserts on the
logits. **Write Qwen's equivalent before you need it**, not during the night you
need it.

Driven by `tttv2_milestone_b_evidence/llama/run3.sh <tag> <node-id> [pytest args]`,
with `MB_DEADLINE` / `MB_PYTEST_TIMEOUT` and a `tt-smi -glx_reset` after each. The
exact invocations behind every published number are in `ENVIRONMENT.md`
§"Attempt 3 — the exact invocations behind every number". **Change its `HF_HOME`
before you use it** (§2).

## 11. Where the full record is, by section

Open these only for the detail behind a specific claim.

| Want | Section |
| --- | --- |
| The seven attempt-3 defects, with root causes | `REPORT.md` §A3.2 |
| The shared-module changes, declared with their reductions | §A3.4 |
| What Qwen inherits (attempt 3's own list) | §A3.5 |
| Step 3: accuracy gate, 80-layer run, demo, the unmet item | §A3.6 |
| The L3 verdict and the proposed status-page text | §A3.9 |
| Results table and final verdict with a log per line | §A3.11, §A3.12 |
| Run-by-run narrative with every raw number | `ATTEMPT3.md` |
| Mesh facts, firmware, exact invocations | `ENVIRONMENT.md` §"Addendum — attempt 3" |

`REPORT.md`'s **top** verdict table is attempt 1's, from a night that produced no
numbers, and still says step 2 `NOT REACHED` and L3 `NOT CLOSED`. Both are stale;
§A3.12 is the current one.
