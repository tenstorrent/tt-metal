# `mb-qwen` attempt 2 — run-by-run narrative

Every device run this attempt made, in order, with the numbers it produced.
Logs are `tttv2_milestone_b_evidence/qwen/logs2/<name>.log`; the reset log for a
run that needed one is `reset_<name>.log` beside it.

The harness is `run3_sequence.sh <manifest>` → `run3.sh` → `device_run.sh`,
copied from the Llama package with `HF_HOME` corrected to
`/localdev/ctr-apbernal/hf_data`. One pytest on the mesh at a time, never piped,
reaped by PID, `tt-smi -glx_reset` after any non-clean run.

## 1. The mesh, before anything else

| run | what | result |
| --- | --- | --- |
| `a2_00_partition` | `models/common/tests/models/galaxy/test_partition_wh_galaxy.py` | **5 passed in 12.93 s** |

32/32 boards in `/sys/class/tenstorrent`. The previous attempt of this job found
eleven boards off the PCIe bus and reported `BLOCKED (infra)`; that is no longer
true, and this run is the check that established it.

## 2. The 64-head geometry, stated on the mesh

| run | what | result |
| --- | --- | --- |
| `a2_01_geometry` | `test_..._geometry_is_decoupled_8x4_qwen3_32b` | **1 passed** |

```text
dim=5120  n_heads=64  head_dim=128  attention_dim=8192   (1.60 x dim)
local_dim=1280  local_attention_dim=1024  local_qkv_size=1280  local_hidden_dim=3200
wo is [8192, 5120]; per mesh row [1024, 1280]
wo DRAM shard (local_attention_dim) : 12 cores, shape [1024, 128]
wo DRAM shard if dim were used      : 12 cores, shape [1280, 128]
```

`local_qkv_size == local_dim == 1280`, so a fused-QKV-vs-residual confusion is
shape-invisible on this model; `local_attention_dim` (1024) is the one that
differs and the one `wo`'s placement is built from.

## 3. D-B26 — the per-head Q/K decode norm, three ways of being unplaceable

The brief asks for the Q/K norm to be validated alone, in its own geometry,
before the block runs it. That is the whole reason this defect took one night
rather than one night plus a bisection: each failure named itself.

| run | input placement | result |
| --- | --- | --- |
| `a2_02_qknorm`, `a2_03_qknorm` | interleaved DRAM (the module's post-D2 default) | prefill **0.99998 on all 32 devices**; decode `TT_FATAL: Kernel group cores do not match sub device cores` |
| `a2_04_block` | the created heads' own placement | `TT_FATAL: Height sharded inputs are not supported` |
| `a2_06_block`, `a2_07_block` | block-sharded worker rectangle, shape from the batch | `TT_FATAL: Shard layout requires 2x1 = 2 shards but shard grid has 8 cores`; then `TypeError: ttnn.Shape does not slice` |
| `a2_10_block` | block-sharded, shape derived from the tensor, output to `attention_heads_memcfg` | `TT_FATAL: Q and K must not overlap` |
| `a2_09_qknorm`, `a2_11_qknorm`, `a2_14_qknorm` | no configured placement; kernel on named worker cores, output back to the input's own placement | **passes, both modes, all 32 devices** |

Final numbers, identical across `a2_09`, `a2_11` and `a2_14`, and identical on
every one of the 32 devices:

```text
prefill q_norm  0.9999821268225385      decode q_norm  0.999988294981757
prefill k_norm  0.9999833417066442      decode k_norm  0.9999879678611943
```

Shapes exercised are the ones attention produces: prefill
`[1, local_heads, 128, 128]` interleaved DRAM, decode `[1, 8, 32, 128]`
height-sharded on the slice of the head grid `nlp_create_qkv_heads_decode` would
have used - Q on the first eight cores of that grid, K on the next eight, and the
test asserts the two are disjoint and that the norm hands them back on the same
cores.

## 4. D-B27 — the decode LM head's all-reduce, starved of worker cores

`a2_12_block` carried the whole Qwen decode graph - embedding, per-head Q/K norm,
fused rotary, SDPA, WO, MLP, both distributed norms - through to the LM head's
column all-reduce, and then segmentation-faulted the process:

```text
[ccl] lm_head in:     logical=(1,1,32,19200) tiles=600 shard=(32,800)  cores=24
[ccl] lm_head staged: logical=(1,1,32,19200) tiles=600 shard=(32,384)  cores=50
[ccl] lm_head buffer: logical=(1,1,32,76800) tiles=2400 shard=(32,1536) cores=50
AllGather is being launched on a subdevice with fewer worker cores available
than ideal. Ideally 4 cores (1 per link and 4 links) are made available but only
0 are available.
Fatal Python error: Segmentation fault
```

D-B19's invariant held throughout - 50 x 384 = 19200 and 50 x 1536 = 76800, both
exact - so the reduction would not have hung; it had no cores to run its fabric
links on. `lm_head_reduce_core_count` searched for the largest divisor of the
tile width that fits the worker envelope, and for Qwen's 600 tiles that is 50,
the whole envelope. Llama's 504 tiles have no divisor between 43 and 50, so its
42 leaves eight free by luck. The search now reserves four cores explicitly:
Llama still resolves 42, Qwen resolves 40.

## 5. The step-5 gate

`a2_13_block`, `a2_15_block`, `a2_16_block` — three fresh processes, each a
separate `python -m pytest` with a `tt-smi -glx_reset` between them. All 21 PCC
lines are **bit-identical** across the three (`md5sum` of the `[pcc]` lines:
`7c751ada099943bbc51df1d4c1b3efc8`), all passed, all `exit=0`, 108-115 s each.

```text
prefill 128 logits                        0.999303669584255
prefill 128 cache K (users 0,8,16,24)     0.9998897994661545
prefill 128 cache V (users 0,8,16,24)     0.9998944730661905
decode position 128 logits (u 0,8,16,24)  0.999360219056066
decode 128 cache K (users 0,8,16,24)      0.9998896420783983
decode 128 cache V (users 0,8,16,24)      0.9998939662639094
```

`a2_17_prefill2048` — the full 2048-token single-row recipe, a different
attention program config, SDPA geometry and collective plan:

```text
prefill 2048 logits                       0.9990203192392576
prefill 2048 cache K (users 0,8,16,24)    0.9998918196733165
prefill 2048 cache V (users 0,8,16,24)    0.9998937907368274
```

## 6. The decode bisection

`a2_18_bisection`, every boundary inside layer 0 against HF forward hooks:

```text
probe decode global_cb bound              True
probe decode cos / sin (users 0,8,16,24)  1.0 / 1.0
bisect decode embedding user 0            1.0
bisect decode attention norm user 0       0.9999910897024602
bisect decode attention out (u0,8,16,24)  0.9992340211925125   (identical, all four)
bisect decode cache K prefix / full / appended row
                                          0.9998897994661545 / 0.9998896420783983 / 0.9998437736663295
bisect decode cache V prefix / appended   0.9998944730661905 / 0.9998828181814494
probe appended K |max| device / reference 72 / 73
bisect decode residual after attention    0.9992551949454134
bisect decode ff norm user 0              0.9993495387523257
bisect decode mlp out user 0              0.9995019825585748
probe HF's MLP on the device's MLP input  0.9997485357508168
bisect decode after layer 0 user 0        0.9181927142562529
bisect decode final norm user 0           0.7657172612914792
bisect decode logits user 0               0.999360219056066   (asserted)
```

Two of those were much lower than the rest, and they were **the test's
reference, not the model**. `out.hidden_states[-1]` is the output of the model's
*final norm*, not of the last decoder layer, so the residual was being compared
against a normalized reference and the final norm against the norm applied
twice. `a2_35_bisection` crossed the comparisons and found it; a host check on
the real checkpoint confirmed it (`hidden_states[1]` is PCC 1.0 against
`norm(layer0_output)` and 0.9178 against the layer output). With the labels
corrected, `a2_50/51/52_bisection`, three fresh processes, bit-identical:

```text
probe decode global_cb bound              True
probe decode cos / sin (users 0,8,16,24)  1.0 / 1.0
bisect decode embedding user 0            1.0
bisect decode attention norm user 0       0.9999910897024602
bisect decode attention out (u0,8,16,24)  0.9992340211925125
bisect decode cache K prefix/full/appended 0.9998897994661545 / 0.9998896420783983 / 0.9998437736663295
bisect decode cache V prefix / appended   0.9998944730661905 / 0.9998828181814494
probe appended K |max| device / reference 72 / 73
bisect decode residual after attention    0.9992551949454134
bisect decode ff norm user 0              0.9993495387523257
bisect decode mlp out user 0              0.9995019825585748
probe HF's MLP on the device's MLP input  0.9997485357508168
bisect decode after layer 0 user 0        0.9995262312962566   |ref|max 8.375  |dev|max 8.875
bisect decode final norm user 0           0.9995821444530748   |ref|max 58     |dev|max 58.75
probe cross ref final norm vs dev residual  0.9181927142562529   (the sanity check, now low)
probe cross ref after layer 0 vs dev normed 0.9159249720677421   (likewise)
bisect decode logits user 0               0.999360219056066    (asserted)
```

**Every boundary of the Qwen decode step is >= 0.9992 against Hugging Face.**
There is no residual-stream floor; there was a mislabelled reference.

## 7. Step 6 — the full model

| run | what | result |
| --- | --- | --- |
| `a2_19_fullmodel` | 64 layers, prefill 128 + first decode, cold staging | **1 passed, 9 m 59 s** |
| `a2_23_fullmodel`, `a2_28_fullmodel` | the same, warm | 1 passed, 2 m 08 s |
| `a2_20/33/34_accuracy` | teacher-forced, batch 1, 512 / 511 | **top-1 498/511 = 0.9746, top-5 511/511 = 1.0000**, three times identical |
| `a2_21/29/30_demo_b1` | direct demo, batch 1 | 1 passed x3, identical text |
| `a2_22/31/32_demo_b32` | direct demo, batch 32, cross-slot isolation | 1 passed x3, identical text |

Device weight staging for the 64-layer model cost **one** 10-minute run; every
later process is a cache hit (108 GB under `model_cache/Qwen`).

## 8. Llama's regression gates

| run | result |
| --- | --- |
| `a2_40/41/42_llama_step2` | 1 passed x3, all 21 `[pcc]` lines identical |
| `a2_43_llama_prefill2048` | 1 passed, `prefill 2048: 0.9996201066107949` |
| `a2_44_llama_fullmodel` | 1 passed, 13 m 44 s |
| `a2_45_llama_accuracy` | 1 passed, 21 m 33 s, **top-1 501/511 = 0.9804, top-5 511/511 = 1.0000** |
| `a2_46_llama_demo_b1`, `a2_47_llama_demo_b32` | 1 passed each, identical text |

Every number is bit-identical to the values `job1_llama_state_for_qwen.md` §1
records for `mb-llama` attempt 3.

## 9. Host gates

| run | result |
| --- | --- |
| `a2_60_host_gate` | `host_gate.sh` — **570 passed**, exit 0, no skips |
| `a2_61_qwen_host` | Qwen host suites — **50 passed**, exit 0, no skips |

## 10. Runs that produced nothing, and why they are still here

`a2_02/03_qknorm`, `a2_04/06/07/10/12_block`, `a2_05/08_qknorm` each ended in an
abort or a crash, and each named a defect. They are kept because the failure text
is the evidence: three of this job's four findings are quoted from those logs and
nowhere else. `a2_03` and `a2_05` also carry the harness note in
`ENVIRONMENT.md` — a decode-mode `TT_FATAL` hangs the `mesh_device` fixture
teardown *before* pytest writes any per-test verdict, so the wrapper's
verdict-keyed grace timer never arms and only its full deadline reaps the
process.


## 11. Re-qualification at the final commit

The step-5 gate and the Q/K norm were each run three more times at the tip of
this attempt's work, after the last test-side change, so that the three-run
evidence is unambiguously at one commit rather than spread over the night:

| runs | verdict | `md5sum` of the `[pcc]` lines |
| --- | --- | --- |
| `a2_70/71/72_qknorm` | PASSED x3, 128 `[pcc]` lines each | `c911648c291d8091caef85a2f34db9df` |
| `a2_73/74/75_block` | 1 passed x3, 21 `[pcc]` lines each | `7c751ada099943bbc51df1d4c1b3efc8` |

The block gate's hash is the same one `a2_13/15/16` produced hours earlier, so
nothing between the two moved a number.

`a2_70/71/72` exit **124**, and that is the harness, not the test: pytest writes
`PASSED` and then the `mesh_device` fixture teardown holds the mesh past the
90-second grace, so `device_run.sh` reaps it and resets. The Q/K norm test does
this consistently and the block gate never does; it builds a model, activates
both modes and runs no full graph, so something it allocates is not drained on
close. Cosmetic for the evidence - the verdict and all 128 numbers are already in
the log when the reap happens - and worth knowing before you read `rc=124` as a
failure.
