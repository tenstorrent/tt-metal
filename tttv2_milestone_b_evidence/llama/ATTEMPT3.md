# `mb-llama`, attempt 3 — running log

Written 2026-08-27, unattended, on the mesh attempt 2 left healthy.
Attempt 1's account is `REPORT.md` §1-§9, attempt 2's is `REPORT.md` §"Attempt 2"
and `ATTEMPT2.md`. Attempt 3's logs are in `logs3/`; `logs/` and `logs2/` were
not touched.

## Environment, re-verified before planning around it

The driver's own probe at 11:26 UTC enumerated all 32 Wormhole boards with no
`ARC startup error` (`tttv2_milestone_b_runs/20260827T112601Z/mb-llama_tt_smi_before.log`),
and `ls /dev/tenstorrent | wc -l` was 32. Firmware bundle 18.12.1, KMD 2.4.1,
IOMMU enabled — read off the first mesh open in `logs3/a3_02_step2_gate.log`.

Attempt 2's last run (`a2_23`) hung and was reaped, and **no reset was logged
after it** — there is no `logs/reset_a2_23_decode_step.log`. A `TT_FATAL`/hang
inside a multi-sub-device program leaves the fabric dirty, which presents one run
later as `Timed out waiting for ETH heartbeat`, so attempt 3's first act was a
`tt-smi -glx_reset`: `Re-initialized 32 boards after reset`
(`logs3/a3_00_reset.log`). No claim here rests on the mesh state attempt 2 left.

## Runs

| # | Log | What | Result |
| --- | --- | --- | --- |
| 00 | `logs3/a3_00_reset.log` | `tt-smi -glx_reset` before any work | `Re-initialized 32 boards` |
| 01 | `logs3/a3_01_step2_gate.log` | step-2 gate | **harness error** — my wrapper did not `cd` to the repo root, so the node id did not resolve. No device work. Fixed in `run3.sh`. |
| 02 | `logs3/a3_02_step2_gate.log` | step-2 gate, prefill 128 + decode | **FAILED** — new defect **D-B20**, in *prefill*, at the very first op |
| 03 | `logs3/a3_03_prefetcher_host.log` | `test_prefetcher_2d.py` before the D-B20 fix | 17 passed |
| 04 | `logs3/a3_04_prefetcher_host.log` | `test_prefetcher_2d.py` after it, with a new deferral test | 18 passed |
| 05 | `logs3/a3_05_step2_gate.log` | step-2 gate | **FAILED** — D-B20 fixed, prefill reached layer 0; new defect **D-B21** |
| 06 | `logs3/a3_06_rope_host.log` | `test_rope_2d.py` after D-B21's fix | 13 passed |
| 07 | `logs3/a3_07_rope_host.log` | same, with a new layout test | 14 passed |

## Result 02 — D-B20: the prefetcher's global CB made prefill unplaceable

`logs3/a3_02_step2_gate.log`. The step-2 gate test aborted at the **very first
device op of prefill**, `ttnn.embedding`:

```text
TT_THROW ... Statically allocated circular buffers in program 100 clash with
             L1 buffers on core range [0-0 - 0-3]. L1 buffer allocated at
             579104 and static circular buffer region ends at 630080
  (validate_circular_buffer_region, under ttnn::prim::embeddings,
   from Embedding2D.prefill_forward)
```

`[0-0 - 0-3]` is four cores of the `x = 0` **prefetch sender** column, and
1499136 - 579104 = 920 kB of L1 was already resident there. The resident buffer
is the prefetcher's global circular buffer: `GALAXY_GLOBAL_CB_SIZE` is
`728 * 1088` = 792,064 bytes, which is also the production value.

`Prefetcher2D.seal()` allocated it eagerly, at model build, and nothing can free
it. So every prefill program needing static circular buffers on the sender
columns was unplaceable — and the prefill mode plan is a *single* sub-device
covering the whole grid, so every interleaved prefill op is such a program.

**The production Galaxy prefetcher does not do this, and says why in one line**
(`models/demos/llama3_70b_galaxy/tt/prefetcher_common.py`):

```python
self.global_circular_buffer = None  # Global CB will only be allocated before decode runs
...
def create_global_cb(self):
    if not hasattr(self, "global_circular_buffer") or self.global_circular_buffer is None:
        self.global_circular_buffer = ttnn.create_global_circular_buffer(...)
```

Prefill never reads the buffer — `seal()` already hands the prefill context
`global_cb=None` — so holding it through prefill buys nothing and costs 774 kB
of L1 per sender/receiver core.

**Fixed** by a new `Prefetcher2DConfig.defer_global_cb` flag: `seal()` skips the
allocation and the first `activate("decode")` makes it, before the prefetch
program that reads it is enqueued. The flag defaults to `False` so the Milestone
A qualification of the module is unchanged, and `build_galaxy_prefetcher_config`
defaults it to `True` because every Galaxy model here runs prefill before decode.

One implementation detail is deliberate and is called out in the code: the
binding is an `object.__setattr__` on the frozen `Prefetcher2DContext`. Module
configs capture the context *object* at construction
(`MLP2DConfig.decode_prefetch_context`, read as
`getattr(context, "global_cb", None)` at call time), so publishing a *replacement*
context would leave every already-built module holding `global_cb=None`. The
field is bound exactly once, `None` -> buffer, and never rebound.

Covered by a new host test,
`test_deferred_global_cb_is_allocated_on_first_decode_activation`, which checks
all three properties: nothing allocated at seal, one buffer allocated on the
first decode activation and handed to the prefetch program, and the *sealed*
context object seeing it.

### A limitation this fix does not remove, stated rather than hidden

Once decode has been activated the global CB is resident again, so a **prefill
after a decode** is back in D-B20's position. Production has the same property.
Every path this job needs is prefill-then-decode, but three step-3 tests use two
runners in one process (`repeated_requests`, `batch32_slots_are_isolated`) and
would prefill again after a decode. If they abort with the message above, that is
this limitation and not a new defect.

## Result 05 — D-B21: the prefill RoPE tables were row-major

`logs3/a3_05_step2_gate.log`. With D-B20 fixed, prefill 128 executed the
embedding, the distributed norm and the QKV projection, and stopped in RoPE:

```text
TT_FATAL ... cos tensor to rotary embedding must be tilized
  (rotary_embedding_llama_device_operation.cpp:51, cos.layout() == Layout::TILE)
  from GalaxyAttentionCollectives.rotary, mode="prefill"
```

The two modes read the RoPE table through different ops and each op accepts
exactly one layout:

* **decode** calls `ttnn.embedding(rot_idxs, cos_matrix, layout=TILE)`, which
  requires a **row-major** weight table and produces tilized cos/sin;
* **prefill** slices the table directly and hands the slice to
  `rotary_embedding_llama`, which requires **tilized** cos/sin.

`_materialize_table_copy` built the prefill copy with `layout=table.layout`,
inheriting decode's row-major. **Fixed** to `ttnn.TILE_LAYOUT`. The qualified 1D
reference agrees: `get_prefill_rot_mat` in `models/tt_transformers/tt/common.py`
writes its prefill cos/sin with `layout=ttnn.TILE_LAYOUT`.

This is a *module* defect, not a configuration one — there is no single layout
that serves both consumers — and it is exactly the class of defect
`job1_llama.md` ranked first: "RoPE composed with `Attention2D` is the expected
first failure... Milestone A qualified `RotarySetup2D` standalone; the pairing has
never run." Standalone, nothing ever asked the prefill copy for its layout.

Tilizing in the copy keeps it a host-side write, which is the whole point of
`_materialize_table_copy` (see D-B1): a device-side `ttnn.tilize` would compile a
full-grid program under the decode partition and abort the way the clone it
replaced did.
| 08 | `logs3/a3_08_step2_gate.log` | step-2 gate | **FAILED** — D-B21 fixed; new defect **D-B22** |
| 09 | `logs3/a3_09_rope_host.log` | `test_rope_2d.py` after D-B22's fix | 2 failed — the host assertion encoded the wrong shape |
| 10 | `logs3/a3_10_host_quick.log` | rope + prefetcher host after correcting it | 32 passed |
| 11 | `logs3/a3_11_step2_gate.log` | step-2 gate | **FAILED** — the whole prefill graph ran to the logits; new defect **D-B23** |
| 12 | `logs3/a3_12_host_quick.log` | `test_model_host.py` + `test_lm_head_2d.py` after D-B23's fix | 50 passed |

## Result 08 — D-B22: the prefill transformation matrix was the wrong size

`logs3/a3_08_step2_gate.log`. Tilized cos/sin got past D-B21 and the next
argument of the same call failed:

```text
TT_FATAL ... Transformation matrix must have 4th dim equal to TILE_WIDTH
  (rotary_embedding_llama_device_operation.cpp:194,
   trans_mat.logical_shape()[-1] == TILE_WIDTH)
```

`prefill_trans` was built with `get_rot_transformation_mat(dhead=head_dim)`, i.e.
`[1, 1, 128, 128]`. The op applies the transformation one tile at a time; the
helper's own docstring says "dhead: Matrix dimension. **Must equal TILE_SIZE**";
and the qualified 1D reference opens with `dhead = 32  # ROPE op uses a single
tile`, discarding whatever it was passed. **Fixed** to `TILE_SIZE`.

`test_rope_2d.py` asserted the wrong shape — `(1, 1, 128, 128)` — so the module
and its host test agreed with each other and both disagreed with the op. That
assertion is corrected with the device abort quoted against it. It is worth being
explicit that this is *not* a threshold being relaxed to turn a run green: the
device rejects 128 outright, and the value the test now asserts is the value the
qualified reference forces.

## Result 11 — the prefill graph reached the logits, and D-B23 was waiting there

`logs3/a3_11_step2_gate.log`. **The entire one-layer prefill graph executed on
silicon**: embedding, distributed norm, QKV, RoPE on real Q/K, causal SDPA, `wo`,
the axis-1 QKV all-reduce, the axis-0 output all-reduce, all of the MLP's
reduce-scatter/all-gather chain, the final distributed norm, the prefill LM head
matmul and its column all-reduce. 220 s wall for the process.

Then the comparison itself failed:

```text
RuntimeError: The size of tensor a (16416768) must match the size of tensor b
              (8208384) at non-singleton dimension 0
  from comp_pcc, under _assert_pcc(expected_prefill, actual, "prefill 128")
```

128 x 128256 against 128 x **64128**. `_logits` composed with
`to_torch_auto_compose`, which infers its composer from the tensor's own
`tensor_topology()` — and **a matmul output carries its activation's topology,
not its weight's.** The LM head's in0 is replicated over mesh rows and sharded
over mesh columns on its last axis, so the logits are labelled that way, while
the vocabulary is actually sharded over mesh *rows* by the weight mapper
`[PlacementShard(-1), PlacementShard(-2)]` and replicated over columns by the
column all-reduce. Auto-compose therefore concatenated the four columns along the
vocabulary axis: 4 x 16032 = 64128 columns holding **four copies of mesh row 0's
vocabulary slice**.

**Why this is the most dangerous defect of the night.** The only reason it
surfaced is that `comp_pcc` compares sizes. `GalaxyDirectRunner._compose_rows`
did the same auto-compose and then sliced `[:, : self.vocab_size]`, which on a
64128-wide tensor **narrows without raising**. Every step-3 logit, every argmax,
the demo text and the whole teacher-forced accuracy number would have been
computed from the wrong 64128 tokens, with no error anywhere. A gate that fails
open again, one layer down from the `load_reference_tokens` skip attempt 2 found.

**Fixed** by `compose_galaxy_logits` in `galaxy/collectives.py`, used by both the
runner and the test: an explicit
`ConcatMesh2dToTensor(dims=(3, 0))` — rows on the vocabulary axis, the four
identical columns stacked on the free leading axis — then column 0. This is the
composition the production host reference uses
(`lm_head.py::forward_on_host`, `dims=(0, 3)` then `[:1]`), with the axes swapped
because here it is the rows that carry the vocabulary. The runner now also
*rejects* a composed width that is not the vocabulary, so this cannot fail
silently again.

`_compose_kv` in the same test file already refused auto-compose for the same
reason, one tensor earlier in the same graph, and said so in its docstring. The
lesson generalises: **on this mesh, auto-composition is only safe for a tensor
whose placement was set by a mapper, never for one produced by an op that
contracts a sharded axis.**
| 13 | `logs3/a3_13_step2_gate.log` | step-2 gate | **FAILED** — **prefill 128 logits PCC passed**; KV-cache K PCC 0.0386 (**D-B24**) |
| 14 | `logs3/a3_14_step2_gate.log` | step-2 gate | **prefill half of the gate PASSED**; decode reached D-B19 and hung. Reaped deliberately once the trace had named the hang point. |

## Result 13 — the first PCC number this model has ever produced, and D-B24

`logs3/a3_13_step2_gate.log`. With the composition corrected, `prefill 128`
passed `PCC >= 0.99` against the Hugging Face one-layer reference. The next
assertion, the cache, did not:

```text
AssertionError: prefill 128 cache K user 0 failed PCC>=0.99: 0.038602362629236275
```

0.0386 is uncorrelated, not slightly wrong, and the *logits* had just passed —
which is the whole diagnosis. The device K cache holds post-RoPE keys in **Meta
interleaved** head-dim order `(r0, i0, r1, i1, ...)`; HF's `past_key_values`
holds them in HF's split order `(r0, r1, ..., i0, i1, ...)`. The adaptor converts
`wq`/`wk` with `reverse_permute` and the cos/sin tables with
`permute_hf_rope_to_meta_tables` precisely so the device runs the Meta
convention, and the two conventions **cancel inside `Q . K^T`** — so the logits
agree while the raw caches cannot. `V` is untouched by either side
(`wv_meta = wv_raw`).

This is a defect in the *test*, and it was the kind that reads as a model
failure. **Fixed** by permuting the reference K with `reverse_permute_1d` from
`models/common/tests/modules/_hf_reference.py` — the shared 1D/2D test helper the
brief points at — before comparing. Nothing on the device side changed.

`_assert_pcc` now also *prints* every PCC it computes, passing or failing. A gate
that passes silently records no number, and this job exists to produce numbers.

## Result 14 — the prefill half of the step-2 gate, measured

`logs3/a3_14_step2_gate.log`, one process, real `meta-llama/Llama-3.3-70B-Instruct`
layer-0 weights:

```text
[pcc] prefill 128: 0.9995838243615001 (gate >= 0.99)
[pcc] prefill 128 cache K user 0:  0.9999347766610057
[pcc] prefill 128 cache V user 0:  0.9997498179150203
[pcc] prefill 128 cache K user 8:  0.9999347766610057
[pcc] prefill 128 cache V user 8:  0.9997498179150203
[pcc] prefill 128 cache K user 16: 0.9999347766610057
[pcc] prefill 128 cache V user 16: 0.9997498179150203
[pcc] prefill 128 cache K user 24: 0.9999347766610057
[pcc] prefill 128 cache V user 24: 0.9997498179150203
```

All four column-local users, so no mesh column silently wrote nothing.

The decode half then reached **D-B19**, attempt 2's open hang, and the CCL trace
did its job:

```text
[ccl] lm_head stage input from 24 cores
[ccl] lm_head staged, shape=(1, 1, 32, 16032)
[ccl] lm_head buffer DRAM -> L1
[ccl] lm_head buffer in L1, shape=(1, 1, 32, 64512)
[ccl] lm_head all_reduce_async returned, shape=(1, 1, 32, 16032)
[ccl] lm_head reduced placed back
[ccl] lm_head synchronize            <- last line; no "synchronized"
```

That **rules out a host-side block in any of the three ops** — all three enqueued
and returned — and confirms attempt 2's gdb reading exactly: an enqueued device
program never signalled completion. It does not yet say *which*, because enqueues
are asynchronous, so the trace now synchronises after each step when
`TTTV2_GALAXY_CCL_TRACE` is set, and reports each tensor's shard spec and page
count. One more run converts "one of three" into a name.

The process was reaped deliberately rather than left to its 1800 s deadline: the
trace had already extracted everything that run could give, and 20 minutes of
wall clock is worth more than a tidier exit code. PID confirmed `comm=python`
with 64 `/dev/tenstorrent` fds open before signalling, per the house rules.
| 15 | `logs3/a3_15_dbb19_trace.log` | D-B19 diagnostic, decode only | **D-B19 NAMED**: `all_reduce_async` is the op that never completes |
| 16 | `logs3/a3_16_host_quick.log` | lm_head + llama host after the padding change | 1 failed — a host assertion encoded the unpadded vocabulary |
| 17 | `logs3/a3_17_host_gate.log` | (abandoned) `models/common/tests/modules` as a directory | pulled in the 1D device suites and took the mesh; killed |
| 18 | `logs3/a3_18_host_quick.log` | targeted 2D host set after the padding change | **116 passed**, 2 skipped |

## Result 15 — D-B19 named, with its mechanism, and it is a width

`logs3/a3_15_dbb19_trace.log`. With a synchronize after each enqueued op:

```text
[ccl] lm_head in:     logical=(1, 1, 32, 16032) shard=(32, 672)  cores=24
[ccl] lm_head staged, shape=(1, 1, 32, 16032) -- completed on device
[ccl] lm_head staged: logical=(1, 1, 32, 16032) shard=(32, 384)  cores=42
[ccl] lm_head buffer in L1, shape=(1, 1, 32, 64512) -- completed on device
[ccl] lm_head buffer: logical=(1, 1, 32, 64512) shard=(32, 1536) cores=42
[ccl] lm_head reduced: logical=(1, 1, 32, 16032) shard=(32, 384) cores=42
[ccl] lm_head all_reduce_async returned, shape=(1, 1, 32, 16032)
                                       <- no "completed on device"
```

The DRAM->L1 buffer materialisation and the ring->42-core relocation both
**completed on device**. `all_reduce_async` did not. So D-B19 is
`ttnn.experimental.all_reduce_async`, and attempt 2's three candidates are down to
one.

The same trace lines carry the cause. Read the widths:

```text
buffer   64512 = 42 cores x 1536 = 42 x 48 tiles = 2016   exact
staged   16032 = 501 tiles, in a 42 x 12 = 504-tile spec  three tiles short
```

The buffer's page arithmetic is exact; the reduced tensor's is not. 501 tiles over
42 cores at 12 tiles per shard fills 41 cores and leaves the 42nd holding **9 of
12**.

The reduction compute kernel does not tolerate that. From
`.../all_reduce_async/device/kernels/compute/reduction.cpp`:

```cpp
const uint32_t num_blocks      = get_arg_val<uint32_t>(rt_args_idx++);  // ring_size = 4
const uint32_t block_num_tiles = get_arg_val<uint32_t>(rt_args_idx++);  // 12
...
cb_in.wait_front(num_blocks * block_num_tiles);                         // 48
```

and the host hands *every* output core the same `block_num_tiles`:

```cpp
SetRuntimeArgs(program, reduction_kernel_id, output_tensor_cores,
               {1, ring_size, output_tensor_shard_num_pages});
```

So the 42nd core waits for 48 tiles, receives 36, and waits forever. The program
never signals completion, the completion-queue reader spins, and the host blocks
in `FDMeshCommandQueue::wait_for_outstanding_reads` — which is exactly the stack
attempt 2's `gdb` dump captured. **No abort, no traceback, mesh reset required.**

### Why no core count could have fixed it

`lm_head_reduce_core_count` searched for a divisor of the *ring-padded* 504 tiles
and found 42. The tensor has 501. And 501 = 3 x 167, so its only divisors below 50
are 1 and 3 — a 3-core staging would put 167 tiles (~182 kB) per core in L1 and
four times that in the buffer, which is D-B15 and D-B18 all over again. **There is
no usable width-sharded placement of a 501-tile tensor on this mesh.** The tensor
itself has to change.

### The fix: pad the vocabulary to a ring-exact width

`galaxy_padded_vocab_size` now pads to a multiple of `GALAXY_ROWS *
RING_ALIGNMENT` (8 x 768) instead of `GALAXY_ROWS * TILE` (8 x 32):

```text
Llama-3.3-70B  128256 -> 129024   16128/device, 504 tiles, 42 cores x 12
Qwen3-32B      151936 -> 153600   19200/device, 600 tiles, 50 cores x 12
```

Every placement is unchanged — `pad_ring_width(16128) == 16128`, the ring's
`per_core_N` is still 21 and the reduce staging is still 42 x 12 — but the tensor
now *fills* them.

This is what production does, by a different route: it pads to `16 * 1024` per
device (131072 total) and hard-codes `num_cores_after_lm_head = 32`, which divides
512 tiles exactly. Padding is how that geometry is made to work either.

`LMHead2D`'s validation demanded the *minimal* padding and had to be loosened to
"a multiple of the vocabulary-shard tile, at least the minimum, and not more than
one extra shard per mesh row". That is a shared-module change and it is declared
as one; it is a validation that forbade a legal width, not a threshold.

**A consequence worth naming: Llama's invalid-logits mask is no longer
identically zero.** Under the old rule `padded_vocab_size == vocab_size` and the
mask was vacuous, which is why attempt 2 could not qualify the mask placement for
Qwen. It is now load-bearing for Llama too — 768 columns of `-inf`, all of them in
mesh row 7's shard.
| 19 | `logs3/a3_19_step2_gate.log` | step-2 gate with the padded vocabulary | **D-B19 CLOSED on silicon**; prefill gate passed again; **decode logits PCC -0.0215** (D-B25) |
| 20 | `logs3/a3_20_bisect.log` | decode bisection | **VOID** — launched while run 19 still held the mesh; see the harness note below |

## Result 19 — D-B19 is closed, and the decode graph is numerically wrong

`logs3/a3_19_step2_gate.log`. The padded vocabulary works, and the trace says so
in the words that were missing before:

```text
[ccl] lm_head staged: logical=(1, 1, 32, 16128) shard=(32, 384) cores=42
[ccl] lm_head all_reduce_async returned, shape=(1, 1, 32, 16128)
[ccl] lm_head all_reduce_async returned, shape=(1, 1, 32, 16128) -- completed on device
[ccl] lm_head reduced placed back -- completed on device
[ccl] lm_head synchronize
[ccl] lm_head synchronized
```

16128 = 42 x 384 exactly. **D-B19 is CLOSED**, and the diagnosis was right: it was
a width, not a flag, not a topology and not a semaphore count.

The prefill half of the gate passed again, unchanged, with the wider vocabulary
(prefill 128 0.99958, cache K 0.99993, V 0.99975). So the padding cost nothing
numerically, as a masked pad should not.

**And the decode logits are uncorrelated:**

```text
[pcc] decode position 128 user 0: -0.02154469920244183 (gate >= 0.99)
```

That is **D-B25**, and it is a new frontier rather than a regression: no decode
logit had ever been compared to anything. The decode graph now runs end to end and
produces the wrong numbers, which is a strictly better place to be than a hang,
and it is the first time this failure mode has been *reachable*.

## A harness rule attempt 3 had to learn the hard way

Run 20 was launched about six minutes after run 19's pytest reported its failure —
and run 19's process was **still holding all 64 `/dev/tenstorrent` fds**, hung in
mesh teardown. Run 20 opened, blocked, and produced nothing:

```text
warning | UMD | Waiting for lock 'CHIP_IN_USE_22_PCIe' which is currently held by
                thread TID: 131554, PID: 131554
```

Attempt 2 recorded that a `TT_FATAL` leaves the mesh un-drainable. **A plain
`AssertionError` does too**, if it is raised after a decode step: the teardown
hang is in the `mesh_device` fixture, not in the failing op, so any decode-mode
failure leaves a holder behind. Attempt 2's `[stage] leave close model` warning
generalises further than it was written.

The rule, for whoever runs next: **never start a device cycle until the previous
`cycle.sh`/`run3.sh` has actually exited** — the pytest verdict appearing in the
log is not that moment, because the reap and the reset come after it. Both
processes were reaped by PID after confirming `comm=python` and 64 open device
fds, and the mesh was reset before continuing.
| 22 | `logs3/a3_22_bisect.log` | decode bisection, model-level boundaries | embedding **1.0**, after layer 0 **0.0718** — the break is inside the layer |

## Result 22 — the bisection localises D-B25 to inside the decoder layer

`logs3/a3_22_bisect.log`:

```text
[bisect] bisect decode embedding user 0:      1.0
[bisect] bisect decode after layer 0 user 0:  0.0717891518669863
[bisect] bisect decode final norm user 0:     0.060994034769210365
[pcc]    bisect decode logits user 0:         0.06410317912613331
```

The decode embedding is **exact** — PCC 1.0, not 0.999 — which rules out the token
staging, the embedding table, the residual placement and the composition helper in
one line. Everything from "after layer 0" onward is uncorrelated, and the final
norm and the logits are simply carrying that forward, so there is exactly one
suspect region: **the decoder layer's decode path**.

That is the region `job1_llama.md` ranked first and second:

> 1. RoPE composed with `Attention2D` is the expected first failure...
> 4. Fused decode norm at real scale. Job 0 fixed the placement defect on paper.
>    This job runs it.

The bisection now walks inside layer 0 — attention norm, attention output,
residual, FF norm, MLP output — against HF forward hooks on the same four modules,
so one more run separates those two candidates and the three others (decode SDPA,
the QKV create-heads collective, the user gather).
| 23 | `logs3/a3_23_bisect_layer.log` | decode bisection, inside layer 0 | attention out **0.737**, MLP out **0.096** — two divergences, not one |

## Result 23 — D-B25 is two defects, and the norms are not among them

`logs3/a3_23_bisect_layer.log`, one process, in graph order:

```text
[bisect] bisect decode embedding user 0:                  1.0
[bisect] bisect decode attention norm user 0:             0.9999956953474292
[bisect] bisect decode attention out user 0:             *0.7372848194843996
[bisect] bisect decode residual after attention user 0:   0.9435188057220959
[bisect] bisect decode ff norm user 0:                    0.9311189676749084
[bisect] bisect decode mlp out user 0:                   *0.0960404663734013
[bisect] bisect decode after layer 0 user 0:              0.0701357106207587
[bisect] bisect decode final norm user 0:                 0.0598278833917375
[pcc]    bisect decode logits user 0:                     0.0626954356537722
```

Read the two starred rows and what surrounds them:

* **the fused decode norm is correct.** `attention norm` is 0.99999 and `ff norm`
  is 0.9311 *on a 0.9435 input*, so both norms are faithful. Job 0's C1 fix holds
  on silicon, and the brief's risk 4 ("fused decode norm at real scale... this job
  runs it") is discharged as a numerical question. That is worth stating plainly
  because C1 was described as making every decode fail.
* **attention is partly wrong** — 0.737 out of a 0.99999 input. Correlated but
  wrong, which is the signature of a subset being wrong (some heads, some users,
  or a rotation) rather than of garbage.
* **the MLP is badly wrong** — 0.096 out of a 0.931 input. A correct function of a
  0.93-correlated input would return roughly 0.9. This is a second, independent
  defect.

The residual dilutes the attention error (0.737 -> 0.9435) because the residual
stream is the embedding plus the attention output, and the embedding is exact.

Three probes were added rather than guessed at, all of them separating "wrong
function" from "wrong input", and two of them free:

1. **the decode RoPE tables**, composed and compared against the Meta-layout
   tables the adaptor built, at the position all 32 users share. Host-side. This
   is `job1_llama.md`'s ranked-first risk and it costs nothing to eliminate.
2. **the prefetcher's global circular buffer**, checked to be bound for decode.
   Every prefetched weight matmul reads it - `wqkv`, `wo`, `w1`, `w3`, `w2`, which
   is exactly the set of matmuls inside the two broken regions - and attempt 3
   made its creation lazy, so this had to be ruled out explicitly rather than
   assumed. (`_prefetch_kwargs` is read per call, so the lazy binding should
   reach them; "should" is not evidence.)
3. **HF's own MLP re-applied to the device's own MLP input**, compared against the
   device's MLP output. This is the decisive one for the 0.096.
| 24 | `logs3/a3_24_bisect_probes.log` | bisection + the MLP-as-a-function probe | **MLP wrong as a function**: 0.0846 on its own input |
| 25 | `logs3/a3_25_host_quick.log` | host after dropping the attention weights from the prefetcher | 1 failed — a host assertion pinned the old registration |
| 26 | `logs3/a3_26_host_quick.log` | host after updating it | **127 passed** |
| 27 | `logs3/a3_27_bisect_noprefetch.log` | bisection with attention unprefetched | **D-B25a fixed**: MLP 0.096 -> 0.939, as a function 0.085 -> **0.99985**; RoPE tables **1.0** |

## Result 24-27 — D-B25a: the MLP was reading the attention's weights

The probe that settled it, from `logs3/a3_24_bisect_probes.log`:

```text
[bisect] bisect decode mlp out user 0:              0.0960404663734013
[bisect] probe mlp on the device's own input:       0.0846090164672149
```

HF's own MLP applied to the device's own MLP input disagrees with the device's
MLP output. So the MLP is a **wrong function**, and its 0.93-correlated input is
not the explanation.

Everything about the MLP's configuration is nonetheless right. Checked on host,
field by field, against the *qualified Milestone A* decode configuration
(`models/common/tests/modules/_mlp_2d_galaxy.py::decode_ring_config`):

```text
decode_input_memcfg        WIDTH_SHARDED L1 shard=(32, 96)  cores=24   equal
decode_w1_w3_output_memcfg WIDTH_SHARDED L1 shard=(32, 160) cores=24   equal
decode_w2_input_memcfg     WIDTH_SHARDED L1 shard=(32, 160) cores=24   equal
decode_w2_output_memcfg    WIDTH_SHARDED L1 shard=(32, 96)  cores=24   equal
decode_w1_w3_prg_config                                                equal
decode_w2_prg_config                                                   equal
```

The 24 ring core coordinates and the 24 receiver coordinates are identical to the
qualified test's, in the same order, and the weight mesh mappers match too
(`[Shard(-1), Shard(-2)]` for w1/w3, `[Shard(-2), Shard(-1)]` for w2). The ring
program configs also match the *production* `FF1_3_TG_RING_PROGCFG` and
`FF2_TG_RING_PROGCFG` argument for argument.

**So the defect was not in the module or its config. It was the prefetcher.**

The global circular buffer is received by the 24 ring cores
(`galaxy_sender_receiver_mapping`), and a prefetched matmul takes its weight from
that buffer in registration order. The MLP's three projections run on the ring.
The attention decode projections **do not**: Milestone A limitation L3 forced them
onto a confined three-column worker rectangle so they would not straddle the
prefetch sub-device partition. So registering `wqkv` and `wo` put two entries per
layer into the buffer that nothing on the ring ever consumed, and the MLP's `w1`
read the entry meant for `wqkv`.

**Fixed** by registering only `("w1", "w3", "w2")`. Attention keeps its worker
sub-device id through a new `_UnprefetchedContext` - without it a ttnn matmul
defaults to sub-device *zero*, the prefetch senders (D-B13) - and reads its
weights straight from DRAM.

`logs3/a3_27_bisect_noprefetch.log`:

```text
[bisect] bisect decode mlp out user 0:          0.096 -> 0.9392486006493238
[bisect] probe mlp on the device's own input:   0.085 -> 0.9998512706499623
[pcc]    bisect decode logits user 0:           0.063 -> 0.9303782778911268
```

**The confirmation that the mechanism is understood and not merely correlated
with the fix**: the attention PCC is *bit-identical* across the two runs -
`0.7372848194843996` before and after. The attention matmuls never read the
global CB at all; they only desynchronised it for the ones that did.

### Two things this run eliminated for free

```text
[probe] decode global_cb bound: True
[bisect] probe decode cos user 0:  1.0     probe decode sin user 0:  1.0
[bisect] probe decode cos user 8:  1.0     probe decode sin user 8:  1.0
[bisect] probe decode cos user 16: 1.0     probe decode sin user 16: 1.0
[bisect] probe decode cos user 24: 1.0     probe decode sin user 24: 1.0
```

* the deferred global CB (D-B20's fix) **is** bound for decode;
* **the decode RoPE tables are exact** on every column-local user. `RotarySetup2D`
  produces exactly the Meta-layout table row the adaptor built. `job1_llama.md`
  ranked "RoPE composed with `Attention2D`" as the expected first failure; the
  *tables* are not it, so if RoPE is implicated at all it is the application, not
  the data.

D-B25b remains: **decode attention output PCC 0.737**, with an exact norm before
it and exact tables beside it.
| 28 | `logs3/a3_28_bisect_attn.log` | per-user attention + decode cache | attention out differs per user; **decode cache K 0.0002, V 0.9997** |
| 29 | `logs3/a3_29_bisect_kwrite.log` | KV split into prefix / window / appended row | **the appended K row is `inf`**; the prefix is intact |
| 30 | `logs3/a3_30_host_quick.log` | host after defaulting `use_qk_fused_rotary` on | 44 passed |
| 31 | `logs3/a3_31_bisect_fusedqk.log` | decode bisection with the fused QK rotary | **1 passed** — decode logits **0.99975** |

## Result 28-31 — D-B25b: the non-fused decode RoPE wrote an infinite K

Two observations from `logs3/a3_28`, neither of which a single number would have
given:

```text
[bisect] bisect decode attention out user 0:  0.7372848194843996
[bisect] bisect decode attention out user 8:  0.6695424031325465
[bisect] bisect decode attention out user 16: 0.6954387356505118
[bisect] bisect decode attention out user 24: 0.5970376158004123
[bisect] bisect decode cache K user 0: 0.000181252848628809
[bisect] bisect decode cache V user 0: 0.999749334500399
```

Prefill filled local user 0 of every mesh column *identically*, so four different
attention numbers mean the four columns saw four different inputs. And the cache
says which: **K is uncorrelated while V is exact.**

Splitting the window into prefix, full and appended row (`logs3/a3_29`) put it
beyond doubt:

```text
decode cache K prefix (0..127)      0.99993     prefill's keys are intact
decode cache K window (0..128)      0.00018
decode cache K appended row (128)   0.00166     device |max| = inf
                                                reference |max| = 4.438
decode cache V prefix               0.99975
decode cache V appended row         0.99973
```

`inf` on user 0 and `8.773e+37` on user 8 - **different garbage per column** - is
uninitialised memory, not an arithmetic error. And V, which does not pass through
RoPE, is exact. Q carries eight real head rows in its 32-row shard and K carries
one; only K was corrupted.

**The op was the wrong one for this mesh.** Production's Galaxy attention chooses
between the two on exactly the condition `(8, 4)` satisfies:

```python
if self.use_prefetcher:
    q_heads_1BQD, k_heads_1BKD = ttnn.experimental.rotary_embedding_llama_fused_qk(
        q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1],
        self.transformation_mats["decode"])
else:
    # No-prefetcher decode still requires HEIGHT_SHARDED cos/sin for RoPE.
    # get_rot_mats returns [1, 1, local_batch, head_dim], which is the format
    # expected by the non-fused decode rotary op. get_rm_rot_mats expands to
    # [1, expanded_batch, heads, head_dim] for the fused path; ...
    ... rotary_embedding_llama(q, ...); rotary_embedding_llama(k, ...)
```

On a prefetcher mesh the non-fused pair is the **Blackhole fallback**, and it wants
a different cos/sin layout - production says so itself. `use_qk_fused_rotary` now
defaults to True; it is one flag that switches `RotarySetup2D` to the expanded
table layout and `GalaxyAttentionCollectives` to the fused call together.

`logs3/a3_31_bisect_fusedqk.log`, **1 passed**:

```text
[bisect] bisect decode embedding user 0:            1.0
[bisect] bisect decode attention norm user 0:       0.9999956953474292
[bisect] bisect decode attention out user 0:        0.9997521295439139
[bisect] bisect decode attention out user 8:        0.9997521295439139
[bisect] bisect decode attention out user 16:       0.9997521295439139
[bisect] bisect decode attention out user 24:       0.9997521295439139
[bisect] bisect decode cache K appended row:        0.9999321818286975  (|max| 4.5)
[bisect] bisect decode cache V appended row:        0.9997323200088775
[bisect] bisect decode mlp out user 0:              0.9997991570968087
[bisect] bisect decode after layer 0 user 0:        0.9792873560408584
[bisect] bisect decode final norm user 0:           0.9896548517784418
[pcc]    bisect decode logits user 0:               0.9997463458407887
```

All four users now agree exactly, which is the property the four-user check exists
to test.

**This closes `job1_llama.md`'s ranked-first risk.** "RoPE composed with
`Attention2D` is the expected first failure... the pairing has never run." The risk
was real, and it was in the *pairing*: the tables `RotarySetup2D` produces are
exact (PCC 1.0) and `rotary_embedding_llama` is a correct op; the composition
picked the variant meant for a mesh without a prefetcher.

### One number to read carefully

`after layer 0` is 0.9793 and `final norm` 0.9897, both below the 0.99 the *logits*
clear at 0.99975. That is not a defect being hidden: the residual stream is
bfloat16 and the comparison is against an fp32 Hugging Face reference on a single
1x8192 row, while the LM head contracts those 8192 terms into each logit and
averages the quantisation noise out. It is recorded because it is the kind of
number that would look alarming in isolation, and because it sets the expectation
for 80 layers: the residual's PCC against fp32 is the *floor*, not the gate.
| 32 | `logs3/a3_32_step2_gate_run1.log` | **step-2 gate, run 1/3** | **1 passed** |
| 33 | `logs3/a3_33_step2_gate_run2.log` | step-2 gate, run 2/3 | **1 passed**, bit-identical |
| 34 | `logs3/a3_34_step2_gate_run3.log` | step-2 gate, run 3/3 | **1 passed**, bit-identical |

## Result 32-34 — the step-2 gate, three times, identically

```text
[pcc] prefill 128:                            0.999584002863212
[pcc] prefill 128 cache K user 0/8/16/24:     0.9999347766610057
[pcc] prefill 128 cache V user 0/8/16/24:     0.9997498179150203
[pcc] decode position 128 user 0/8/16/24:     0.9997463458407887
[pcc] decode position 128 cache K u0/8/16/24: 0.9999342257320987
[pcc] decode position 128 cache V u0/8/16/24: 0.9997493345003990
```

Identical across three fresh processes to the last digit. See `REPORT.md`
§A3.1 for why that particular property is the one worth reporting.
