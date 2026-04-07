# Module Registration Analysis: `normalization_nanobind.cpp`

## Overview

Primary reference: `ttnn/cpp/ttnn/operations/normalization/normalization_nanobind.cpp`

Supporting reference: `ttnn/cpp/ttnn/operations/normalization/normalization_nanobind.hpp`

This file is the module-level nanobind entrypoint for the normalization namespace. Its job is not to define operation signatures; it only wires per-op binders into the module.

Current implementation:

```cpp
void py_module(nb::module_& mod) { detail::bind_batch_norm_operation(mod); }
```

`normalization_nanobind.cpp:11-13`

## File Contract

### Header

`normalization_nanobind.hpp` declares:

- `namespace ttnn::operations::normalization`
- `namespace nb = nanobind`
- `void py_module(nb::module_& mod);` `normalization_nanobind.hpp:7-12`

### Implementation

`normalization_nanobind.cpp`:

- includes its own header,
- includes `<nanobind/nanobind.h>`,
- includes the per-op binder header for `batch_norm`,
- calls the per-op binder inside `py_module`. `normalization_nanobind.cpp:5-13`

This is the direct pattern layernorm should follow.

## Registration Pattern

The normalization module currently registers one operation:

- include `batch_norm/batch_norm_nanobind.hpp`
- call `detail::bind_batch_norm_operation(mod)` `normalization_nanobind.cpp:9-13`

Implication for `layernorm`:

- add `#include "layernorm/layernorm_nanobind.hpp"`
- add `detail::bind_layernorm_operation(mod);`

If both `batch_norm` and `layer_norm` live in the same normalization module, the final shape will likely be:

```cpp
void py_module(nb::module_& mod) {
    detail::bind_batch_norm_operation(mod);
    detail::bind_layernorm_operation(mod);
}
```

The precise ordering is usually not semantically important, but keeping the file in simple include/call form matches the existing style.

## Separation Of Concerns

This reference makes the intended layering explicit:

| Layer | Responsibility | Evidence |
| --- | --- | --- |
| Per-op nanobind file | define docstring and bind public C++ function | `batch_norm_nanobind.cpp:18-93` |
| Module registrar | aggregate per-op binders into one namespace module | `normalization_nanobind.cpp:9-13` |

That means layernorm registration should not duplicate docstrings or argument metadata in `normalization_nanobind.cpp`; all operation-specific binding logic belongs in `layernorm_nanobind.cpp`.

## Naming Expectations

The module registrar uses the `detail::bind_<op>_operation` naming convention. `normalization_nanobind.cpp:13`, `batch_norm_nanobind.hpp:12`

For consistency, the new binder should be named:

- `detail::bind_layernorm_operation(mod)` if the repo prefers the filesystem/op name `layernorm`, or
- `detail::bind_layer_norm_operation(mod)` if the team wants the binder name to mirror the public Python symbol `layer_norm`.

The important point is internal consistency across:

- the binder header declaration,
- the binder implementation,
- the include path in `normalization_nanobind.cpp`.

## Minimal Changes Required For Layernorm

Based on this reference alone, module registration for layernorm requires exactly two edits in the registrar:

1. include the per-op layernorm nanobind header
2. call the layernorm binder from `py_module`

Nothing in this file suggests additional registration tables, macros, or dispatch metadata are needed at the normalization-module level.

## Risks And Unknowns

- This reference shows only one registered op in the normalization module. If the missing local layernorm/rmsnorm sources used additional submodule structure, that is not visible here.
- The naming convention for the future layernorm binder is not dictated by this file alone; downstream implementation should pick one form and use it consistently.

## Assumptions

- Assumed `normalization_nanobind.cpp` remains the single aggregation point for normalization ops in this workspace.
- Assumed layernorm is intended to surface in the same Python namespace as `batch_norm`, so registration belongs in this file rather than in a separate module registrar.
