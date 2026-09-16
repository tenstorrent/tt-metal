# Matmul registry

The registry selects measured program and compute-kernel configurations for
calls that exactly match a checked-in entry.

## Runtime modes

Set `matmul_registry_mode` through `TTNN_CONFIG_OVERRIDES` or
`ttnn.CONFIG.matmul_registry_mode`:

- `off` uses the existing TTNN path.
- `shadow` performs lookup without changing the call.
- `on` applies an exact match and otherwise uses the existing TTNN path.

For coverage tests, combine `on` with the existing fallback guard:

```bash
TTNN_CONFIG_OVERRIDES='{"matmul_registry_mode":"on","throw_exception_on_fallback":true}' pytest ...
```

That command fails at the first matmul that the registry does not own. A
successful run therefore cannot silently pass through an explicit override,
an unsupported call, or a table miss.

## Current coverage

The dense table currently covers exact Blackhole, single-device, rank-2,
interleaved calls. Explicit program configs and core grids, sharded tensors,
mesh-wide tensors, and trace capture are not covered. The AGMM table currently
contains 32-chip entries; its lookup is used only when the caller omits both
the program config and compute-kernel config.

Do not describe an `on` model run as registry coverage unless it also enables
`throw_exception_on_fallback` and completes successfully.

## Updating the tables

The generator and banked sweep results live in `tt-matmul-codegen`. From that
repository, install a freshly generated dense table into a tt-metal checkout
with:

```bash
scripts/pipeline.sh registry ~/tt-metal
```

Then build the TTNN tests and run the focused registry tests before running
the same model command once with the registry `off` and once with strict
registry `on`. A model call site should drop a bespoke config only after its
runtime contract has an exact measured registry entry.
