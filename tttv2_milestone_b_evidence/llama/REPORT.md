# `mb-llama` — Milestone B steps 1–3, Llama-3.3-70B on WH Galaxy `(8, 4)`

Written 2026-08-27 by the `mb-llama` job, unattended.
Environment, exact commands and mesh facts: `ENVIRONMENT.md`.
Every log from every attempt: `logs/` (105 files, none overwritten).

## Verdict up front

| Plan step | Gate | Result |
| --- | --- | --- |
| 1 — adaptor, host | layout + Llama 3 scaled RoPE vs HF | **PASS**, 9 tests, 3 fresh processes |
| 1 — one-layer model on the mesh | construction, prefetcher sealing, CCL resolution, clean teardown | **PASS**, real layer-0 weights |
| 2 — one block, decode PCC ≥ 0.99 | PCC vs HF, KV-cache PCC | **NOT REACHED** — no PCC number was produced |
| 2 — one block, prefill 128 / 2048 | PCC vs HF | **NOT REACHED** |
| 3 — 80 layers, demo, teacher-forced accuracy | top-1 ≥ 91%, top-5 ≥ 99% | **NOT REACHED** |
| L3 verdict (attention decode on the prefetch partition) | — | **NOT CLOSED — still live.** See §3 |
| L1 verdict (global-CB ownership over two constructions) | — | **NOT MEASURED** |

**No PCC, accuracy or demo number exists in this package, because none was
produced.** The night was spent getting the decode graph to *execute* at all.
That was not the plan, and §2 explains exactly how far it got and why.

Two things ended the session:

* **nine defects** in the decode path, each blocking the next, all of the same
  family (§3). Eight are fixed and committed; the ninth has a candidate fix that
  is host-green and **never ran on hardware**;
* the mesh then **died** (§5): board 7 dropped off the PCIe bus and
  `tt-smi -glx_reset` cannot recover it because it needs that node. Recorded
  `BLOCKED (infra)` after the two recovery attempts the brief allows.

## 1. What is genuinely proven

### 1.1 Step 1, host — the adaptor is numerically correct

`models/common/tests/models/llama33_70b_galaxy/test_hf_conversion_host.py`, new,
9 tests, **passed in 3 fresh processes** (`logs/04`, `02b`, `03b`). No device.

The pre-existing host suite checked that every `weight_utils` conversion returns
the contracted *shape*. A Q/K permutation that pairs the wrong axes returns the
right shape and the wrong numbers, so shape was not the property that needed
proving. What is now proven:

* the Meta cos/sin tables are the HF rotary's **scaled** frequencies,
  pair-duplicated — and llama3 scaling really is applied (checked by showing the
  table moves away from the unscaled-theta table, so the check is not vacuous);
* **`reverse_permute` composed with the interleaved rotation the device kernel
  performs is the same operator as the HF weight layout composed with
  `rotate_half`**, at `head_dim = 128` against the real Llama-3.3 rotary. The
  rotation is not restated by hand: it is the `trans_mat` from
  `tensor_utils.get_rot_transformation_mat`, the matrix the kernel is handed, so
  the host reference cannot drift from the device one;
* `weight_utils.reverse_permute` is bit-identical to the qualified 1D suites'
  `_hf_reference.reverse_permute`;
* the fused row-major QKV packing is invertible per mesh row;
* converted attention, MLP and LM-head weights reproduce the **unmodified** HF
  modules at PCC ≥ 0.9999, with the LM-head padding columns proven inert;
* against the **real checkpoint**: the RoPE tables match Llama 3 scaling
  recomputed from its definition (`rope_theta`, `factor`, both frequency factors,
  original context length — no HF rope helper involved), and layer 0 converts to
  the contracted shapes.

**This closes the RoPE-convention half of the author's ranked risk #1.** The
other half — that the model wires those tables into `Attention2D` correctly on
the mesh — is device work, and §2 reports how far that got.

### 1.2 Step 1, device — the one-layer model builds and tears down

`test_bringup_wh_galaxy.py::test_one_layer_model_constructs_and_closes`
— **PASSED in 109 s** (`logs/10_construct_run1.log`), real layer-0 weights from
the real checkpoint. This is the first Milestone B code ever to run on hardware.

Proven: construction, prefetcher **sealing**, both operation-boundary CCL
contexts resolving, KV-cache bind/unbind, `close()`, and
`prefetcher.owned_resources == ()` afterwards. **The C1/D1 fused-norm statistics
placement holds at real scale** — it is a hard `ValueError` at construction if
re-introduced, and it did not fire.

It was run **once**. It is reported as a single pass, not as evidence under the
three-run rule.

### 1.3 The decode sub-device partition, measured

`models/common/tests/models/galaxy/test_partition_wh_galaxy.py`, new — **5
passed in 12.8 s**, no checkpoint (`logs/39_partition_probe_run4.log`):

```text
compute grid      x=0..6, y=0..9                              70 cores
worker_cores()    {[1-0 - 3-9], [5-0 - 6-9]}                  50 cores
prefetch senders  x=0 and x=4                                 12 cores
in NO sub-device  {[0-1 - 0-3], [0-6 - 0-8], [4-3], [4-8]}     8 cores
```

Two facts here caused most of §3, and neither is visible from a mocked mesh:

1. the worker envelope is **not contiguous** — the `x=4` sender column splits it
   — so its bounding box (`x=1..6`) is not a safe stand-in for it, and several
   ttnn ops use exactly that bounding box;
2. sender ∪ worker does **not** cover the compute grid, so any program built over
   the full grid touches cores owned by no sub-device.

This file is cheap and needs no checkpoint. **Run it first** when a decode
program aborts on placement.

## 2. How far the decode step got, and where it stopped

One decode step, batch 32, one real layer. 25 device processes opened the mesh
(`logs/` 06–89). Ordered by how far each got:

| Stage of `decode_forward` | Status | First proven in |
| --- | --- | --- |
| build / seal / CCL resolve / teardown | **runs** | `logs/10` |
| `activate("decode")`, persistent DRAM prefetch starts | **runs** | `logs/22` |
| `RotarySetup2D.decode_forward` (RoPE tables) | **runs** | `logs/22` |
| `Embedding2D.decode_forward` | **runs** | `logs/30` |
| layer-0 distributed RMS norm | **runs** | `logs/33` |
| QKV `ttnn.linear` | **runs** | `logs/40` |
| RoPE applied to real Q/K, SDPA, `wo` projection | **runs** | `logs/40` |
| attention output all-reduce | **runs** | `logs/80` |
| MLP ring `w1`/`w3`/`w2` matmuls | **runs** | `logs/80` |
| MLP shared axis-0 all-reduce | **blocked** → fixed (D-B8) | `logs/80` |
| back to the attention matmul: its CBs no longer fit L1 | **blocked** (D-B9) | `logs/82` |
| final norm, LM head, logits | **never reached** | — |

So: **RoPE composed with `Attention2D` runs on real silicon** — the pairing
Milestone A never qualified, and the author's predicted first failure. It failed
first, repeatedly, but for *placement* reasons, never for the numerical
convention, which §1.1 had already settled on host.

The last two rows are a coupled pair and are the honest stopping point:

* fixing **D-B8** meant widening the shared axis-0 all-reduce buffer from
  bfloat8_b to bfloat16 (34816 → 65536 B per core, on all 50 worker cores);
* that extra L1 is on `x=1..3`, which is precisely where **D-B5** had just
  confined the attention dense matmuls, and their in1 circular buffer then
  overflowed by about 20 kB.

The candidate fix for D-B9 — halving `in0_block_w` — is committed and host-green
but **was never executed on hardware**, because the mesh died first. It is
flagged in §3 and in the handoff.

**No case flipped across processes.** Every failure reproduced identically at the
same stage until it was fixed. The three-run rule was therefore never in
tension with anything: there is no intermittent result in this package, because
there is no *passing* device result beyond §1.2 and §1.3 to average.

## 3. Defects

All nine are first-silicon findings in Milestone B code or in shared 2D / Galaxy
plumbing. **None is in `**/*_1d.py` and none is in `llm_runtime/**`** (§6).

They share one root cause, worth stating once: **the Galaxy decode sub-device
manager owns only part of the compute grid, and several core ttnn ops choose
their cores from `device->compute_with_storage_grid_size()` or from a sub-device
*bounding box* — neither of which is the worker set.** tt-metal then rejects the
program:

```text
TT_FATAL ... Kernel group cores do not match sub device cores
             for programmable core type TENSIX
```

and, because the abort happens inside a multi-sub-device program, the mesh is
left un-drainable: teardown blocks forever in
`FDMeshCommandQueue::~FDMeshCommandQueue` and the process keeps the per-chip UMD
locks. Every one of these cost a kill and a `tt-smi -glx_reset`.

### D-B1 — `RotarySetup2D` built its prefill tables with an on-device clone

`models/common/modules/rope/rope_2d.py` (**shared 2D module**).
`load_device_weights` used `ttnn.clone` for the prefill cos/sin copies. It is
lazy, so on the decode path it first ran inside `decode_forward`, after the
partition was loaded, and `ttnn.clone` compiles over the full grid. Sealing the
prefetcher loads the decode partition *and leaves it loaded*, so there was no
later safe moment either — an eager call at construction aborted too
(`logs/18`).

Fix: write both copies from the same host source. Numerically identical — same
tensor, dtype, layout, DRAM placement — but a host-to-device write compiles no
program and is legal under any manager. This was the **only** lazy
device-weight loader among the 2D modules that ran a compute op; the other five
were audited and only write.

Evidence: `logs/17` (located it, `rope_2d.py:88`), `logs/20` (cleared it).

### D-B2 — the decode RoPE shards were placed on prefetch sender cores

`models/common/models/galaxy/recipes.py::rope_core_grids` (**shared Galaxy
plumbing**). It took the first `rows` cores of the *whole* compute grid, putting
the decode cos/sin shards on `(0,0)` and `(4,0)` — both prefetch senders — plus
one core outside every sub-device. `ttnn.embedding` then aborted.

Fix: delegate to `_subgrid_cores`, the already-qualified helper the attention KV,
SDPA and reduce-scatter placements use, which anchors at the first worker core
`(1,0)` and never leaves `worker_cores()`.

This is the **same defect shape as Milestone A D1/C1**: a grid named
independently of the partition that has to contain it. Every other placement
`resolve_galaxy_decode_placements` returns was audited — all worker-confined;
rope was the only exception.

Guarded on host: `test_recipes.py::test_rope_batch_grid_lies_inside_the_worker_sub_device`.
The old expression put **3 cores** outside; the new one puts 0.

### D-B3 — the Llama embedding decode output was L1-interleaved

`models/common/models/llama33_70b_galaxy/model.py`.
`decode_output_memcfg=ttnn.L1_MEMORY_CONFIG`. L1 *interleaved* round-robins over
the whole grid including the sender columns, where the ~55 MB global circular
buffer lives, so `ttnn.embedding` could not place its own static CBs:

```text
TT_THROW ... Statically allocated circular buffers in program 100 clash with
             L1 buffers on core range [0-0 - 0-0]
```

DRAM did **not** help (`logs/26`): `ttnn.embedding` takes its program grid from a
*sharded* output's shard grid and only from there. Fix: name
`decode.residual_memcfg`, which confines the program to the 16 worker cores that
placement already occupies and makes the following relocation a no-op. Tilized
width sharding is legal for this op — shard `[32, 128]` is tile-aligned and
divides the local hidden width.

### D-B4 — `RMSNorm2D` deallocated its own return value (latent, newly reachable)

`models/common/modules/rmsnorm/rmsnorm_2d.py` (**shared 2D module**).
`ttnn.to_memory_config` returns *the same* tt_metal tensor when the requested
config already matches, and nanobind hands that back as a **new Python
wrapper** — so `if tt_out is not unplaced_output: unplaced_output.deallocate(True)`
freed the buffer it was about to return:

```text
TT_THROW ... Tensor is not allocated     (in ShardedToInterleaved::validate_inputs)
```

`rms_norm_post_all_gather` already returns the tensor in `decode_output_memcfg`,
so the short-circuit always fired. Both call sites (input at line 273, output at
311) now guard on the memory config first — the idiom
`attention_2d.py::_place_qk` and the models' `_relocate` already use, and which a
docstring in `moe/tt_moe_decode.py` already warns about.

Pre-existing, not caused by this job; `_decode_distributed` had simply never run
on hardware. Evidence: `logs/30`.

### D-B5 — L3 is not closed: the attention decode matmuls still use the dense `(7,1)` grid

`models/common/models/galaxy/recipes.py::dense_matmul_program_config` (**shared
Galaxy plumbing**).

The brief states that "the Milestone B recipes build the partition-compatible
ring/`gather_in0` form instead" of the grid Milestone A recorded as terminal.
**That is true of the MLP and false of attention.** `attention_qkv_program_config`
and `attention_wo_program_config` are still `dense_matmul_program_config`, which
built a `(7, 1)` grid — *the exact grid L3 names* — anchored at `(0,0)`, spanning
both sender columns. On this build it aborts as:

```text
TT_FATAL ... Illegal kernel placement for bmm_large_block_zm_fused_bias_activation,
             Kernels cannot be placed on dispatch cores!
```

**So the L3 verdict is: still live, now with a precise diagnosis.**

Fix applied: populate `allowed_worker_cores`. ttnn grew that field for exactly
this, deprecating `compute_with_storage_grid_size`, and *warns* when a config
that supports it leaves it unset ("will become a hard error in a future
release"). The largest rectangle inside `worker_cores()` is searched for rather
than named, and is three columns wide.

**Cost: three worker columns instead of seven** for these two matmuls — and that
cost is what produced D-B9. The real fix is moving attention to the 24-core
ring/`gather_in0` form the MLP already uses; see §4.

Guarded on host and device:
`test_partition_wh_galaxy.py::test_dense_matmul_work_grid_stays_inside_the_worker_partition`,
four shapes, all confirming 0 non-worker cores (`logs/39`).

### D-B6 — the attention all-reduce used an op that cannot run on this partition

`models/common/models/galaxy/collectives.py::_all_reduce`.
It called `ttnn.reduce_scatter` + `ttnn.all_gather`, having first *validated*
that the keyed resource has a persistent buffer and then never passing it
anywhere.

`ttnn.reduce_scatter` cannot run here at all. Its program factory takes
`worker_cores(TENSIX, sub_device_id).bounding_box()` and lays its workers out
from that rectangle's origin — and our worker bounding box spans `x=1..6`,
straight across the `x=4` sender column. The file carries its own comment:
`// interaction with subdevice needs to be investigated`.

Fix: mode-split, mirroring the **qualified** `MLP2D._all_reduce_tg`.

* decode → `ttnn.experimental.all_reduce_async` with the resource's persistent
  buffer as `buffer_tensor`. That is the same call MLP2D makes for the same
  shared axis-0 resource — `build_galaxy_decode_collectives` says so in a comment
  — and the buffer's `(8, 4, TILE, W)` row-sharded shape *is* a `buffer_tensor`,
  not a scatter output, which is why the old code could validate it and use
  nothing. It rejects an interleaved input, so the interleaved `wo` output is
  width-sharded first with `interleaved_to_sharded`.
* prefill → keeps the plain pair, which is what MLP2D's prefill branch uses.

### D-B7 — `_relocate` reached a full-grid factory three different ways

`models/common/models/llama33_70b_galaxy/model.py`. All three obvious spellings
are unsafe under the partition:

| spelling | factory it reaches |
| --- | --- |
| `to_memory_config(t, memcfg, dtype)` | `ttnn::prim::copy` → `compute_with_storage_grid_size` (with a standing TODO to use worker cores) |
| `ttnn.typecast(t, dtype)` | same full-grid split |
| `to_memory_config(t, memcfg)`, shard specs differing in grid **and** width | `reshard_program_factory_generic` → full grid |

The residual grid (16 cores, 128 wide) → MLP ring grid (24 cores, 96 wide) hop
hits the third.

Fix: the explicit pair. `sharded_to_interleaved` runs on its input's
`shard_spec.grid`; `interleaved_to_sharded` runs on its output shard's cores;
both are worker-confined here, and both accept `output_dtype`, so the recast
rides along instead of needing an op of its own. **Cost: one DRAM round trip per
placement hop** — a real decode-latency cost, on the follow-up list in §4.

### D-B8 — the shared all-reduce buffer was allocated at the wrong dtype

`models/common/models/galaxy/plans.py`. The shared axis-0 all-reduce buffer took
`_spec`'s **bfloat8_b** default, while both Galaxy models set `MLP2D`'s
`decode_ccl_dtype` to their `decode_residual_dtype` — **bfloat16**, deliberately,
so an 80-layer residual sum is never re-quantized. `all_reduce_async` sizes its
circular buffer from the data and checks it against the buffer's L1 bank:

```text
TT_FATAL ... Cannot set circular buffer size to 65536. This is larger than the
             associated dynamically allocated L1 buffer bank size of 34816 B
```

34816 B is exactly a `[32, 1024]` bfloat8_b shard; 65536 B is the bfloat16 one.

Fix: `build_galaxy_decode_collectives` takes a `residual_dtype` parameter,
defaulting to `ttnn.bfloat16`, and sizes the buffer with it — a parameter rather
than a literal so a model with a different residual dtype can say so instead of
silently mismatching. Attention's reduction now runs at the same dtype, so both
consumers of the shared resource agree with it and with each other.

This one is worth reading twice: the two consumers of a *deliberately shared*
resource disagreed with the resource, and nothing on host could see it.

### D-B9 — OPEN: the confined attention matmul's circular buffers no longer fit

**Status: diagnosed, candidate fix committed, NEVER RUN ON HARDWARE.**

After D-B8 widened the shared buffer to bfloat16 (+30 kB of L1 on each of 50
worker cores), the attention dense matmul — confined by D-B5 to the three
columns `x=1..3`, which raised `per_core_N` by the same factor the grid narrowed
— overflows L1 on those cores by about 20 kB:

```text
TT_THROW ... Statically allocated circular buffers in program 320 clash with
             L1 buffers on core range [1-0 - 3-0].
             L1 buffer allocated at 546432, static CB region ends at 566464
```

Candidate fix in the tree: `in0_block_w` from `gcd(k_tiles, 8)` to
`gcd(k_tiles, 4)`, halving the in1 circular buffer (far more than the ~20 kB
shortfall) at the cost of more K iterations. **Host gate green
(`logs/92`, 390 passed). Device: unverified — the mesh died before it could
run.** Treat it as a hypothesis.

The structural answer is §4.1, not this.

## 4. Recommendations, in priority order

### 4.1 Move the attention decode matmuls to the ring/`gather_in0` form

This closes L3 properly and dissolves D-B9. The MLP already does it
(`ring_matmul_program_config`, 24 cores, `hop_cores`, `gather_in0=True`), and the
recipes already contain `attention_qkv_collective_input_memcfg` shaped for
exactly those 24 ring cores — so the design clearly anticipated it and the
attention matmuls were simply left behind. Spreading the work over 24 cores makes
`per_core_N` small and the circular buffers small, which removes both the
three-column penalty and the L1 clash.

Not attempted here: it changes the QKV matmul's input/output shard specs and how
the fused create-QKV-heads collective consumes them, and there was neither the
device time nor a working mesh to qualify it.

### 4.2 Populate `allowed_worker_cores` on the ring matmul config too

`logs/57` shows eight warnings per step:

```text
matmul_multi_core_reuse_mcast_1d_optimized_helper: program_config.allowed_worker_cores
not populated; auto-populating from compute_with_storage_grid_size. ...
This will become a hard error in a future release.
```

It currently works, so it was **not** changed — a qualified path should not be
altered on a night with no way to re-qualify it. But it is auto-populating from
the full grid, which is the exact hazard behind D-B2/D-B5, and it will stop being
a warning.

### 4.3 Reclaim the DRAM round trips D-B7 introduced

Every sharded→sharded relocation now goes through DRAM. Correct, and slower than
a reshard would be. Either the shard specs on either side of each hop should be
made compatible (so `reshard_program_factory_same_width` applies), or
`reshard_program_factory_generic` needs to respect the loaded sub-device.

### 4.4 Upstream: three ops are not sub-device aware

Worth filing against tt-metal, with these exact sites:

* `copy_default_tilized_program_factory.cpp:44` — uses
  `device->compute_with_storage_grid_size()`, with a TODO already in place;
* `reshard_program_factory_generic.cpp:80` — same;
* `reduce_scatter_program_factory.cpp:107` — uses the sub-device **bounding
  box**, which is wrong for any non-contiguous sub-device, and already carries
  `// interaction with subdevice needs to be investigated`;
* `typecast_program_factory.cpp:109` — full grid (the *sharded* typecast factory
  is fine; the fallback is not).

A non-contiguous worker sub-device is a normal Galaxy configuration, not an edge
case, so "bounding box" is a latent bug for every model that partitions its grid.

### 4.5 Proposed text for `MILESTONE_A_STATUS.md`

This job may not edit that file, so the proposal is here:

> **L3 — attention decode on the prefetch subdevice partition. Still open.**
> First tested on silicon by `mb-llama` (2026-08-26). The Milestone B recipes
> moved the *MLP* to the ring/`gather_in0` form but left both attention decode
> matmuls on `dense_matmul_program_config`, i.e. on the same `(7, 1)` grid this
> limitation names. On the current build it aborts as "Illegal kernel placement
> ... Kernels cannot be placed on dispatch cores" rather than as a sub-device
> mismatch. Confining it with `allowed_worker_cores` makes it legal but leaves
> only three worker columns, and its circular buffers then clash with the decode
> activations resident there. The remaining fix is the ring form.

## 5. `BLOCKED (infra)` — the mesh died

Full record: `logs/90_BLOCKED_infra_pcie_board7.log`.

```text
1. recurring:  Timed out waiting for ETH heartbeat on device
               ASIC ID 87032054158471220, ETH core e9-0 / e8-0
               - at mesh open, in TopologyDiscovery. Same ASIC every time.
               Cleared by tt-smi -glx_reset; returned after the next abort.
               Once it hung discovery outright instead of erroring (logs/70).
2. then:       Read 0xffffffff over PCIe ID 17: the board should be reset
               - in TTDevice::init_tt_device (logs/84)
3. then:       tt-smi -ls aborts inside tt_umd and lists zero boards (logs/87)
4. then:       tt-smi -glx_reset: Error in resetting galaxy 6u trays!
               [Errno 6] No such device or address: '/dev/tenstorrent/7'
```

The reset tool cannot recover the mesh because the node it needs is the one that
is gone. `/dev/tenstorrent/7` is still present as a character device but is
unreadable, and `dmesg` shows a kernel oops with `irqs disabled`.

**Recovery attempts used: 2 of 2** (`logs/85`, `logs/88`). Both failed. This
needs an IPMI power cycle of the tray or a host reboot, which is outside what an
unattended job may do. No device work was attempted afterwards.

The ETH instability correlated with a preceding `TT_FATAL` inside a
multi-sub-device program — which is what leaves the mesh un-drainable — so the
working theory is a fabric left dirty by aborted programs that the reset did not
always fully restore on that one board. Whether the ~23 resets this session
required *caused* the PCIe failure or merely preceded it is not something this
job can determine, and it is not claimed either way.

## 6. Regression gates and boundaries

Host gate, final code state, **driver-free selection** (`logs/92`):

```text
390 passed, 3 warnings in 86.57s     0 driver errors
```

Host gate, final code state, **standard selection** (`logs/91`):

```text
13 failed, 385 passed
```

All 13 failures are in `models/common/tests/models/galaxy/test_plans.py`, and all
13 are the dead driver: that file opens the UMD driver for all 32 chips when run
as a whole file (job 0 recorded this), so it cannot run at all now. 26 driver
error lines in that log. It was green at **398 passed** at 00:07, one code change
earlier, on a working mesh (`logs/81`). `test_plans.py` is therefore
**NOT RUN at the final code state**, by infrastructure failure, and is called out
rather than folded into a number.

Baseline at job start: **395 passed** (`logs/01`). Three tests were added to the
standard selection and nine to the driver-free one. **No test was deleted,
`xfail`ed, skipped, or had a threshold, tolerance or parametrization relaxed.**

Boundaries, `b350e51554470414d5a8b08f5ea9775c986145a4..HEAD`:

```text
$ git diff --name-only ... | grep '_1d\.py'       (empty - PASS)
$ git diff --name-only ... | grep 'llm_runtime'   (empty - PASS)
$ git diff --name-only ... | grep -i qwen         (empty - PASS)
```

Files changed (11): `galaxy/collectives.py`, `galaxy/recipes.py`,
`galaxy/plans.py`, `llama33_70b_galaxy/model.py`, `modules/rmsnorm/rmsnorm_2d.py`,
`modules/rope/rope_2d.py`, and five test files.

### Shared modules changed — declared, as the brief requires

| Module | Change | Why config alone could not express it |
| --- | --- | --- |
| `modules/rope/rope_2d.py` | prefill tables written from host instead of `ttnn.clone` | config has no say over whether materialization runs a device program |
| `modules/rmsnorm/rmsnorm_2d.py` | guard the two `to_memory_config` calls on the memory config before the identity test | a self-deallocation bug; no config value changes it |
| `models/galaxy/recipes.py` | rope `batch_grid` from `_subgrid_cores`; `allowed_worker_cores` on the dense matmul; smaller `in0_block_w` | the grids were computed, not configured; nothing upstream could pass a different one |
| `models/galaxy/collectives.py` | mode-split `_all_reduce`; decode uses `all_reduce_async` | the op itself was wrong for a non-contiguous sub-device |
| `models/galaxy/plans.py` | `residual_dtype` parameter for the shared all-reduce buffer | this *is* the config fix — the value was previously unreachable |

Extension-discipline order was followed: config first (D-B3, D-B8), frozen
config value second (D-B8's default), mechanical delegation to an existing
qualified helper third (D-B2, D-B5, D-B6). Nothing larger was attempted; §4.1,
which would be larger, is left as a recommendation and not half-done.

## 7. Commits

```text
3c1759ff20e  Make the Galaxy decode graph partition-safe: norm, matmul, all-reduce, relocation
1d4ded04592  Fix three Galaxy decode placement defects found on first silicon
e4c1adfa61f  Qualify the Llama Galaxy weight conversion and Llama 3 scaled RoPE on host
```

plus this evidence package and the work-log checkpoint. The `in0_block_w` change
of D-B9 is in the tree and is the one change here that hardware has never seen.

## 8. What this job did not do

Stated plainly so nothing is inferred from silence:

* **no PCC number** for a block, in either mode — decode never reached the LM head;
* **no KV-cache PCC**, no prefill 128 or 2048 result;
* **no 80-layer construction**, so **L1 (global-CB ownership) was not measured**;
  `test_two_models_in_one_process` exists but never ran;
* **no demo text**, no teacher-forced accuracy, no top-1/top-5 measurement —
  the accuracy gate this job exists to produce **was not measured**;
* `test_one_prefill_executes` was written but never ran: the decode step was
  ahead of it in the queue every time, and prefill shares the graph it was
  blocking on;
* the 1D device matrix was not run (Milestone A P4, separate hardware, not this
  job's).

Nothing in §8 is a judgement about whether those gates would pass. They were
not measured.

## 9. Out-of-scope items confirmed untouched

Paged KV, prefix-cached/chunked prefill, concat-32 physical batching, device
sampling and long-context smokes are plan step 7 and belong to `mb-coverage`.
Nothing was built for them. Every device test here ran with
`paged_attention_config=None` and `enable_device_sampling=False`.

**A dependency for `mb-coverage` to note:** step 3 was not blocked by anything in
step 7 — it was blocked by the decode graph not executing. No step-7 feature is
needed to reach the step-3 gates.

Qwen was not touched: no file under `models/common/models/qwen3_32b_galaxy/` or
its tests appears in the diff. Everything learned that applies to both models is
in `tttv2_milestone_b_briefs/job1_completion_handoff.md`, not in Qwen's code.
