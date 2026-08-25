# Attention2D `nlp_concat_heads_decode` C++ Audit

Date: 2026-08-19
Workspace revision: `bc8b7d73b7650450b1c07bf93a4c813b1dc93a9f`
Scope: read-only source investigation; no TT hardware used

## Conclusion

The `bad optional access` failure is caused by an internally inconsistent
sub-core invocation of `nlp_concat_heads_decode`:

1. The Attention decode tensor is height-sharded on a non-origin/sub-core
   grid.
2. `nlp_concat_heads_decode` therefore sets `on_subcoregrids = true`.
3. The failing call did not provide `sub_core_grids`.
4. Output-spec construction unconditionally evaluates
   `args.sub_core_grids.value()` when `on_subcoregrids` is true, producing the
   generic C++ `std::bad_optional_access` text exposed as `bad optional access`.

This is downstream of the fused QKV CCL and SDPA. It is not a CCL hang or a
hardware-health issue.

## Exact Recommended Configuration

Use the 32-core containing head/worker set already constructed by
`_decode_all_reduce_config` as the concat compute domain:

```python
Attention2DConfig(
    # Existing decode settings...
    decode_concat_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    decode_concat_sub_core_grids=decode_all_reduce["head_cores"],
)
```

The resulting call must be equivalent to:

```python
concat = ttnn.experimental.nlp_concat_heads_decode(
    attention,
    num_heads=8,  # 64 global Q heads / 8 mesh rows
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    sub_core_grids=decode_all_reduce["head_cores"],
)
```

`head_cores` is an appropriate containing domain because:

- it is inside the active decode worker domain;
- it contains at least the eight output cores required for eight local Q
  heads;
- it contains the effective eight-core Q shard generated for
  `slice_size=8` by `all_reduce_create_qkv_heads`;
- it is the same domain supplied to decode SDPA.

The operation's native output is nevertheless **width-sharded L1**, with one
output core per local head and shard shape `[padded_batch, head_dim]`. The
`memory_config` argument is ignored by the current primitive implementation
unless an `output_tensor` is explicitly preallocated. Therefore retaining
`decode_concat_memory_config=DRAM` is valid only because `Attention2D` checks
the returned memory config and performs an explicit `ttnn.to_memory_config`
transition before the WO matmul.

Do not attempt to fix this exception by changing only
`decode_concat_memory_config`; it cannot supply the missing optional core set.

## C++ Evidence

### Trigger and optional access

`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp`:

- Lines 133-139 set `on_subcoregrids=true` when the input shard grid has more
  than one range, does not start at `(0,0)`, or an explicit sub-core grid is
  supplied.
- Lines 93-100 then call `args.sub_core_grids.value()` while constructing the
  output grid.
- Validation at lines 61-66 also states that a sub-core invocation requires a
  value and that the set must contain at least `num_heads` cores. Output-spec
  computation occurs early enough that the raw optional exception can surface
  instead of the intended `TT_FATAL` diagnostic.

### Input/output layout contract

The same file requires:

- device-resident BF16 or FP32 tiled input (lines 28-40);
- height-sharded input (lines 43-48);
- one input shard core per user (lines 49-69);
- output shape `[1, 1, max(batch, 32), num_heads * head_dim]` (lines 81-90);
- native width-sharded L1 output on `num_heads` cores (lines 92-110).

The primitive signature receives `memory_config`, but the implementation
comments out its name and does not use it (lines 125-149). Only a supplied
preallocated output bypasses native output-spec construction (lines 74-76 and
113-118).

### Fused Q grid is eight cores

`ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/all_reduce_create_qkv_heads_device_operation.cpp` lines 201-245:

- `slice_size=8` becomes the local batch;
- Q shape becomes `[1, 8, 8, 128]` for this model;
- Q's shard grid is selected as exactly `batch` cores from the supplied final
  memory-config grid;
- its per-core shard is `[32, 128]` because eight heads are tile-padded to 32.

Thus the larger 32-core `head_cores` set is the containing sub-core compute
domain, while the actual Q input uses eight cores. This satisfies concat's
one-core-per-user validation and gives it at least eight cores for output.

## Existing-Test and Model Evidence

`tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads_decode.py`:

- Lines 35-92 construct height-sharded input with one core per batch user.
- Lines 50-67 derive or require a compute sub-core set with at least
  `n_local_heads` cores.
- Lines 88-92 explicitly pass `sub_core_grids`.
- Sub-core cases at lines 143-239 include non-origin and disjoint grids and
  pass a containing compute grid separately from the input grid.

`models/demos/llama3_70b_galaxy/tt/llama_attention.py` lines 817-825 obtains a
grid from the gathered input memory config and passes it as
`sub_core_grids`. This confirms that Galaxy callers are expected to carry the
grid into the primitive rather than rely on automatic output placement.

`tests/ttnn/distributed/test_multidevice_TG.py` lines 1030-1071 exercises the
same `(batch=8, local_heads=8, head_dim=128)` model geometry. Its omission of
`sub_core_grids` is valid only because that test uses a simple origin-anchored
single-range input grid, which selects the non-subcore program.

## tt-buddy Relevance

Inspected `/tmp/tt-buddy-access-audit-20260819` at
`ba9021417442d59756aa8cdf154a25648c9a0de5`. Its CCL guidance emphasizes exact
topology/link configuration and source-derived parameters, but it contains no
operation-specific `nlp_concat_heads_decode` guidance. Since the observed
failure is an immediate host-side optional access after SDPA, no CCL retry,
reset, or link change is indicated.

## Read-only Commands Used

```bash
rg -n --hidden "nlp_concat_heads_decode|NLPConcatHeadsDecode|concat_heads_decode" \
  . /tmp/tt-buddy-access-audit-20260819

nl -ba ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp
nl -ba ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp
nl -ba ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/all_reduce_create_qkv_heads_device_operation.cpp

nl -ba tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads_decode.py
nl -ba tests/ttnn/distributed/test_multidevice_TG.py
nl -ba models/demos/llama3_70b_galaxy/tt/llama_attention.py
nl -ba models/common/modules/attention/attention_2d.py
nl -ba models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py

rg -n "subcore|optional|concat_heads|bad optional" \
  /tmp/tt-buddy-access-audit-20260819/knowledge \
  /tmp/tt-buddy-access-audit-20260819/skills

git rev-parse HEAD
git -C /tmp/tt-buddy-access-audit-20260819 rev-parse HEAD
```

No `pytest`, TT runtime, `tt-smi`, reset, or other hardware command was run.
