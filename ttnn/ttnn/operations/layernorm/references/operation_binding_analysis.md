# Operation Binding Analysis: `batch_norm_nanobind.cpp`

## Overview

Primary reference: `ttnn/cpp/ttnn/operations/normalization/batch_norm/batch_norm_nanobind.cpp`

Supporting references:

- `ttnn/cpp/ttnn/operations/normalization/batch_norm/batch_norm_nanobind.hpp`
- `ttnn/cpp/ttnn-nanobind/bind_function.hpp`

The per-op nanobind file for `batch_norm` is intentionally small. Its responsibilities are:

1. define a detailed Python-facing docstring,
2. bind the public C++ free function `ttnn::batch_norm`,
3. expose keyword names/defaults in Python order,
4. export a single helper `bind_batch_norm_operation(nb::module_&)` for the normalization module registrar to call. `batch_norm_nanobind.cpp:18-93`, `batch_norm_nanobind.hpp:9-12`

For a new `layernorm` op, this is the direct template for the per-op binding file.

## Binding Structure

### Namespace and entrypoint

The binding function lives in `ttnn::operations::normalization::detail`:

```cpp
void bind_batch_norm_operation(nb::module_& mod)
```

`batch_norm_nanobind.cpp:16-18`, `batch_norm_nanobind.hpp:9-12`

This implies the expected layout for layernorm:

- per-op binder declaration in `layernorm_nanobind.hpp`
- implementation in `layernorm_nanobind.cpp`
- same `detail` namespace pattern

### Bound symbol

The file binds the free function directly:

```cpp
ttnn::bind_function<"batch_norm">(..., &ttnn::batch_norm, ...)
```

`batch_norm_nanobind.cpp:75-92`

This is important because the Python-visible symbol is not bound as a method on an operation struct. The binder expects an exported C++ free function in the `ttnn` namespace.

For layernorm, the binding should target `&ttnn::layer_norm` (or the final chosen public symbol name) rather than a device-op primitive.

## `bind_function` Semantics

`ttnn::bind_function` does more than call `nb::def`:

- it creates a unique wrapper class per operation name,
- attaches `name` and `python_fully_qualified_name` properties,
- adds a `__ttnn_operation__` marker property,
- installs one or more `__call__` overloads,
- binds a static instance to `mod.attr(FuncName)`. `bind_function.hpp:108-133`

That means the per-op binder only needs to provide:

- the function name template parameter,
- the docstring,
- the overload signature with `nb::arg(...)` metadata.

## Python Signature Shape

The bound overload is:

```cpp
ttnn::overload_t(
    &ttnn::batch_norm,
    nb::arg("input"),
    nb::kw_only(),
    nb::arg("running_mean") = nb::none(),
    nb::arg("running_var") = nb::none(),
    nb::arg("training") = false,
    nb::arg("eps") = 1e-05,
    nb::arg("momentum") = 0.1,
    nb::arg("weight") = nb::none(),
    nb::arg("bias") = nb::none(),
    nb::arg("output") = nb::none(),
    nb::arg("memory_config") = nb::none(),
    nb::arg("compute_kernel_config") = nb::none())
```

`batch_norm_nanobind.cpp:79-92`

Key conventions to preserve:

| Convention | Evidence | Layernorm implication |
| --- | --- | --- |
| First tensor positional, rest keyword-only | `batch_norm_nanobind.cpp:81-83` | Good fit for `layer_norm(input, *, ...)`. |
| Optional tensors default to `None` | `batch_norm_nanobind.cpp:83-91` | Reuse for optional `weight`, `bias`, and optional `output`. |
| Scalar hyperparameters get explicit Python defaults | `batch_norm_nanobind.cpp:85-87` | Reuse for `eps` and any future scalar flags. |
| Memory/kernel config options exposed last | `batch_norm_nanobind.cpp:90-92` | Reuse directly. |

The include of `<nanobind/stl/optional.h>` is required for these `std::optional` arguments to bind naturally. `batch_norm_nanobind.cpp:9-10`

## Docstring Pattern

The file stores the full docstring in a raw string literal `const auto* doc = R"doc(... )doc";`. `batch_norm_nanobind.cpp:19-73`

The docstring is structured as:

1. one-sentence operation summary,
2. optional external reference,
3. math block,
4. `Args:` section,
5. `Keyword args:` section,
6. `Returns:` section,
7. `Note:` section with supported dtypes/layouts/ranks,
8. `Memory Support:` section,
9. `Limitations:` section. `batch_norm_nanobind.cpp:20-73`

For layernorm, this is the exact documentation scaffold to copy. Only the semantics change:

- math becomes layer normalization over the last logical dimension,
- tensor-shape descriptions should reflect same-shape output and optional affine tensors,
- rank/layout restrictions should match the implemented device op instead of inheriting batch norm’s rank-4 text.

## Naming And Surface Consistency

There is one notable naming mismatch in the docstring:

- the human docs use `input_tensor` in prose,
- the actual bound Python argument name is `input`. `batch_norm_nanobind.cpp:21`, `batch_norm_nanobind.cpp:33-46`, `batch_norm_nanobind.cpp:81`

This is not fatal, but for layernorm it would be better to keep the prose and bound argument names aligned to reduce downstream confusion.

## Minimal File Contract

From this reference, the minimal per-op binding contract is:

- header includes `ttnn-nanobind/nanobind_fwd.hpp`,
- declares `namespace nb = nanobind;`,
- declares one `bind_<op>_operation(nb::module_&)`,
- implementation includes nanobind optional support, the op header, and `ttnn-nanobind/bind_function.hpp`,
- implementation binds exactly the public op symbol. `batch_norm_nanobind.hpp:5-12`, `batch_norm_nanobind.cpp:7-15`, `batch_norm_nanobind.cpp:18-93`

## Layernorm-Relevant Binding Shape

The likely layernorm Python signature suggested by this reference is:

```cpp
ttnn::bind_function<"layer_norm">(
    mod,
    doc,
    ttnn::overload_t(
        &ttnn::layer_norm,
        nb::arg("input"),
        nb::kw_only(),
        nb::arg("eps") = ...,
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("output") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none()));
```

Any extra parameters such as `residual_input_tensor` should only be added if later phases confirm they belong in the target API.

## Risks And Unknowns

- This reference only shows a single-overload binding. If layernorm needs multiple overloads, `bind_function` supports them, but that pattern is not demonstrated here.
- The docstring hardcodes batch-norm-specific rank/layout limits. Layernorm should not copy those text blocks until the device op contract is finalized.
- The bound function name template (`"batch_norm"`) and the exported C++ symbol (`ttnn::batch_norm`) must stay aligned. If the layernorm public symbol ends up named differently from the Python API surface, the binder will need an intentional divergence.

## Assumptions

- Assumed target public Python name is `ttnn.layer_norm` and the corresponding public C++ free function will be `ttnn::layer_norm`.
- Assumed a single callable binding object is sufficient for the initial layernorm surface, matching the pattern used here.
