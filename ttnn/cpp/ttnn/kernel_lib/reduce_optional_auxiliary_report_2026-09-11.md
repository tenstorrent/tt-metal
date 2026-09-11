# Optional reduction auxiliary buffers

The planner can now represent a reduction with no auxiliary tiles and no
auxiliary CB. The previous nonempty-recipe restriction maintained a uniform
buffer protocol by creating an unused tile for aligned additive reductions and
SFPU folds. That restriction was not needed by those computations.

## Planning and caller contract

Set `ReduceBlockSpec::allow_empty_auxiliary = true` in C++, or
`allow_empty_auxiliary=True` when constructing the Python block description.
The default remains false for existing factories that assume an auxiliary
requirement is always present.

With the option enabled, a call that needs no auxiliary operands returns an
empty `auxiliary_tiles` list and no `ReduceCbRole::Auxiliary` requirement.
`find_cb(Auxiliary)` returns null, and that buffer contributes no bytes to the
L1 budget. The caller may omit its allocation and bindings. A caller-owned
buffer may still exist for other work; the empty reduction recipe never uses it.

Native scalers, partial masks and runtime-tail masks remain mandatory when the
algorithm needs them. Tail planning still reserves both reduction-axis and
output-edge masks even when the corresponding full-core plan needs no tiles.
Factories maintaining a uniform allocation across full and tail cores must
take the maximum requirement across those plans; an empty full-core recipe
does not remove a buffer needed by a tail core. Existing multicore allocation
code is unchanged.

An additive sequence may initially have an empty call and later need a zero
tile for its selected reload. The sequence planner creates that requirement,
charges its actual L1 cost and validates the aggregate allocation on every
call. It compares auxiliary formats only among calls that use them. If the
caller passes `host::no_cb_id` (`planner.NO_CB_ID` in Python) for the sequence's
auxiliary CB, the planner skips the optional zero-pair optimization and retains
the correct ordinary pairs reload. A missing CB is still rejected when the
requested computation needs a scaler or mask.

Dense row-major planning similarly charges no auxiliary storage for a whole
additive block, then reserves a zero tile if the input must be staged in chunks.

## Serialization and device execution

Empty compute slices use count zero, offset zero and the existing reserved
no-CB value, 255. Empty dataflow recipes retain their one-word header, allowing
multiple planning units to retain their normal argument-offset calculation.
The host accepts an empty recipe whether the caller supplied a candidate CB ID
or the no-CB value, and serializes the same empty descriptor in both cases.

The dataflow helper's compile-time recipe walk performs no work for zero tiles.
The binding views preserve absent buffers instead of indexing a binding table
with 255. Required input, output and accumulator bindings retain their checks.

The compute helper accepts an absent auxiliary only for additive or SFPU paths.
It skips the auxiliary wait and validation when absent. Unused instantiated
branches receive the real input's metadata internally so they cannot index
metadata arrays with the no-CB sentinel; no auxiliary tile read is issued.
Host and device validation still reject missing masks and zero-pair operands.

## Updated callers

* Moreh height mean enables the option, conditionally declares the scaler CB
  and its producer/consumer bindings, and supplies a compiler definition for
  the optional binding. Its compute and reader kernels do not inspect the
  selected algorithm to decide whether an auxiliary exists. The final pop
  removed in the preceding change remains absent. Its current two-tile input
  cap selects native scaler plans for the tested BF16 heights, so those tests
  verify compatibility and relaunch behavior; the helper and example tests
  exercise actual CB omission. Input chunk selection is unchanged.
* The runnable `reduce_block` example enables the option and omits both the
  auxiliary CB and its producer kernel for an empty aggregate recipe.
* Focused helper tests exercise empty descriptors with an omitted CB, an
  allocated but unused CB, and a sequence whose second call introduces a zero.

## Validation

Validation is limited to the small migration suite and focused checks as
requested. The full helper suite was not run for this change.

* Native build passed:
  `CMAKE_BUILD_PARALLEL_LEVEL=8 PYTHONDONTWRITEBYTECODE=1 ./build_metal.sh --enable-ccache --build-ttnn-tests`.
* All 25 `ReduceHostPlanner.*` tests passed, including four new tests covering
  empty descriptors, omitted allocation, required scalers/tail masks, mixed
  sequences, the no-zero fallback, and dense staging budgets.
* All 35 selected helper tests passed:
  `bash scripts/run_safe_pytest.sh --no-precompile tests/ttnn/unit_tests/kernel_lib/reduce/test_reduce_helpers.py -k 'empty_auxiliary or mixed_auxiliary_format' -q --maxfail=1`.
  Fifteen cover optional auxiliaries; twenty retain the preceding mixed-format
  operand regressions. Empty-recipe cases use small integer-valued data and
  compare exactly, including all output padding. The dataflow test also
  exercises the bound empty descriptor with no allocated CB.
* Three Moreh height cases (128, 256, 273), each run twice, passed together with
  `test_reduce_block_correctness` and
  `test_reduce_block_accumulate_partial_zero_pair`: five pytest cases in total.
* All 63 migration sanity cases passed: 50 Python cases and 13 C++ cases,
  with no failed groups. The selected lanes were `common`, `wormhole` and
  `wormhole-n300` using the retained sanity manifest at
  `/localdev/malimpic/reviews/pr56063-local-spec-20260911-oer_2vst/sanity_test_suite.json`.
  Command:
  `python_env/bin/python scripts/run_reduce_migration_sanity.py --manifest /localdev/malimpic/reviews/pr56063-local-spec-20260911-oer_2vst/sanity_test_suite.json --lane common --lane wormhole --lane wormhole-n300 --output-dir /localdev/malimpic/reviews/pr56063-empty-auxiliary-20260911-fsViee/sanity-results`.
* Pre-commit passed on all 20 modified/new files; `git diff --check` also passed.

Device checks ran on Wormhole N300. Quasar was not tested; its planner continues
to use native reductions that require auxiliary scaler tiles. Dense optional
allocation and runtime-tail auxiliary budgeting have host regression coverage.

Logs are retained in
`/localdev/malimpic/reviews/pr56063-empty-auxiliary-20260911-fsViee/`.
