# Uniform CB allocation for full and tail cores

The multicore examples previously allocated the auxiliary CB separately for each
core group, using that group's auxiliary recipe length. An aligned native
reduction could therefore allocate one tile on a full core and three on a tail
core. Runtime tails do not require that difference in physical capacity.

Both multicore factories in
[test_reduce_helpers.py](../../../../tests/ttnn/unit_tests/kernel_lib/reduce/test_reduce_helpers.py)
now allocate a single auxiliary CB descriptor over the complete participating
core range. Its capacity is the largest aggregate recipe among all variants:

```python
auxiliary_tiles = max(len(plan.auxiliary.tiles) for plan in plans)
```

Full and tail cores have the same auxiliary capacity, page size and CB ID. The
input, output and optional accumulator capacities are also uniform. Each kernel
continues to prepare and consume only its own auxiliary recipe; spare capacity
on a full core is left unused. Only tail kernels receive runtime shape arguments.
No dummy mask generation or extra runtime arguments are needed on full cores.

## Allocation and L1 accounting

The changes apply to `test_reduce_local_blocks_on_multiple_cores` and
`test_reduce_runtime_tail_cores`. Their input and output tensors are resident,
so the factory subtracts their common per-core allocation sizes from the L1
budget before planning. The runtime-tail factory also reserves the common
accumulator allocation when it executes multiple calls. After planning, each
factory checks that the maximum aggregate auxiliary allocation fits the remaining
budget. All cores pay for that maximum capacity, including cores whose recipes
use fewer pages.

The planner still describes local work and local CB requirements; it does not
allocate CBs or own core ranges. These factories coordinate the allocations
across their local plans. The recipe length describes initialized pages, not a
requirement to allocate different-sized buffers on different cores.

For a future factory with streamed inputs, common physical allocation must also
respect each variant's fixed packet size. Independently choosing chunks and then
taking the maximum input capacity is insufficient if that capacity is not an
integer multiple of every packet size. Such a factory must plan compatible input
capacities and budget the common auxiliary reservation before assigning its
remaining L1 to input FIFOs. This change corrects the current resident multicore
examples; it does not add a planner that coordinates streaming variants.

## Allocation order

The runtime-tail example now declares the common auxiliary CB before the common
accumulator CB. It no longer depends on placing the accumulator ahead of
different-sized per-group auxiliary allocations. This supersedes the test-factory
workaround documented in
[the original runtime-tail report](reduce_runtime_tail_shapes_report_2026-09-11.md).
That report is retained as the record of the original change. Metal's allocator
has not been changed.

## Validation

On the attached Wormhole N300,
`PYTHONDONTWRITEBYTECODE=1 TT_METAL_HOME="$PWD" bash scripts/run_safe_pytest.sh --no-precompile tests/ttnn/unit_tests/kernel_lib/reduce/test_reduce_helpers.py -q --maxfail=1`
passed all **203 helper cases**, including the 36 cases using the two modified
multicore factories and all 20 streaming-tail cases. The 32 runtime multicore
cases each exercise three sets of per-core shapes. Coverage includes native and
additive reductions, reduced and non-reduced partial edges, both destination
precisions, and accumulated SUM/MAX calls with the auxiliary CB declared first.

This follow-up changes Python test factories and documentation; no C++ rebuild
is needed. Repository pre-commit checks passed for both changed files.

Logs: `/localdev/malimpic/reviews/pr56063-uniform-cbs-20260911-XaOeXd/`.
