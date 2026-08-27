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
