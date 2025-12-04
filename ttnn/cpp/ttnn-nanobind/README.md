# TTNN Nanobind Bindings

This directory contains the Python bindings for TTNN using [nanobind](https://github.com/wjakob/nanobind).

## Documentation

If you're porting from pybind11 to nanobind, start with these resources:

- **[Porting Guide](https://nanobind.readthedocs.io/en/latest/porting.html)** - Key differences from pybind11
- **[nanobind Documentation](https://nanobind.readthedocs.io/en/latest/)** - Full documentation
- **[API Reference](https://nanobind.readthedocs.io/en/latest/api_core.html)** - C++ API reference

### Quick Reference: Name Changes

| pybind11 | nanobind |
|----------|----------|
| `py::module&` | `nb::module_&` |
| `PYBIND11_MODULE` | `NB_MODULE` |
| `.def_readwrite()` | `.def_rw()` |
| `.def_readonly()` | `.def_ro()` |
| `.def_property()` | `.def_prop_rw()` |
| `.def_property_readonly()` | `.def_prop_ro()` |
| `py::arg(...)` | `nb::arg(...)` |
| `reinterpret_borrow<T>()` | `borrow<T>()` |
| `reinterpret_steal<T>()` | `steal<T>()` |
| `error_already_set` | `python_error` |

---

## Common Pitfalls (with examples)

The following issues were encountered while porting TTNN from pybind11 to nanobind. Each includes the problem, symptoms, and solution.

### 1. Missing Typecasters

**Problem:** Nanobind requires explicit includes for STL type casters. Unlike pybind11, they are not always automatically available.

**Symptoms:**
```
TypeError: __call__(): incompatible function arguments
```

**Solution:** Include the necessary typecaster headers in your binding file:

```cpp
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/set.h>
```

**Example fix (from `moreh_adam_nanobind.cpp`):**
```cpp
// Before: missing includes caused TypeError
#include <nanobind/nanobind.h>

// After: add vector and variant casters
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>
```

---

### 2. `.noconvert()` Breaks Overload Resolution

**Problem:** Using `.noconvert()` on arguments can prevent nanobind from finding valid overloads, especially with implicit type conversions.

**Symptoms:**
- Function calls that worked in pybind11 fail with argument type errors
- Overload resolution fails unexpectedly

**Solution:** Remove `.noconvert()` unless strictly necessary:

```cpp
// Before: noconvert prevents implicit conversion from list to set
.def("__init__",
    [](CoreRangeSet* t, const std::set<CoreRange>& core_ranges) {
        new (t) CoreRangeSet(core_ranges);
    },
    nb::arg("core_ranges").noconvert())  // ❌ Too restrictive

// After: allow implicit conversion
.def("__init__",
    [](CoreRangeSet* t, const std::set<CoreRange>& core_ranges) {
        new (t) CoreRangeSet(core_ranges);
    },
    nb::arg("core_ranges"))  // ✅ Works
```

---

### 3. `keep_alive` is Required for Object Lifetime

**Problem:** Nanobind does not automatically keep parent objects alive when returning references. Without explicit `keep_alive`, objects may be garbage collected prematurely.

**Symptoms:**
- Segmentation faults
- Use-after-free errors
- Objects become invalid unexpectedly

**Solution:** Add `nb::keep_alive<return_idx, arg_idx>()` annotations:

```cpp
// keep_alive<0, 1> means: keep argument 1 alive as long as return value (0) exists
mod.def("to_device", &to_device,
    nb::arg("tensor"),
    nb::arg("device"),
    nb::keep_alive<0, 2>());  // Keep device alive while tensor exists

// For methods, index 1 is 'self'
.def("cpu", &Tensor::cpu,
    nb::arg("blocking") = true,
    nb::keep_alive<0, 1>());  // Keep self alive while returned tensor exists
```

---

### 4. `None` Does Not Implicitly Cast to `bool`

**Problem:** In pybind11, checking `if optional_value:` worked when the value was `None`. Nanobind is stricter about this.

**Symptoms:**
- Code that uses Python's truthy checks on optional values fails
- Boolean checks on `None` behave unexpectedly

**Solution:** Use explicit `None` checks in Python code:

```python
# Before: relies on None being falsy
reshard_if_not_optimal = config.reshard_if_not_optimal if not auto else None  # ❌

# After: explicit handling
reshard_if_not_optimal = (
    config.reshard_if_not_optimal
    if not isinstance(config, AutoConfig)
    else False
)  # ✅
```

---

### 5. Variants Cannot Be Bound as Parent Classes

**Problem:** `std::variant<A, B, C>` types cannot be directly exposed as a class. Attempting to do so causes compilation or runtime errors.

**Symptoms:**
- Compilation errors with variant bindings
- `NB_MAKE_OPAQUE` doesn't work as expected for variants

**Solution:** Create a placeholder class for documentation purposes and include the variant typecaster:

```cpp
// Before: trying to bind variant directly
NB_MAKE_OPAQUE(ttnn::operations::matmul::MatmulProgramConfig);  // ❌ Doesn't work

// After: use placeholder class
struct MatmulProgramConfigPlaceholder {};

void py_module(nb::module_& mod) {
    // Placeholder for documentation - actual types are the variant members
    nb::class_<MatmulProgramConfigPlaceholder>(mod, "MatmulProgramConfig",
        R"doc(Variant defining matmul program config)doc");

    // Bind the actual variant member types
    nb::class_<MatmulMultiCoreReuseProgramConfig>(...);
    // Include variant typecaster
    #include <nanobind/stl/variant.h>  // ✅
}
```

---

### 6. Default Arguments with `optional<T>` Behave Differently

**Problem:** In pybind11, `nb::arg("x") = value` with an optional worked seamlessly. Nanobind requires `nb::none()` for optional defaults.

**Symptoms:**
- Optional arguments don't accept `None`
- Default values don't work as expected

**Solution:** Use `nb::none()` for optional defaults:

```cpp
// Before: pybind11 style
nb::arg("dim"),  // Required, but should be optional ❌

// After: nanobind style
nb::arg("dim") = nb::none(),  // Optional, defaults to None ✅
```

---

### 7. Setting Optional Fields to `None` After Construction

**Problem:** Setting an optional field to `None` after object construction may not work with `def_rw()`.

**Symptoms:**
- `config.shard_layout = None` raises TypeError
- Optional member assignment fails

**Solution:** Use `def_prop_rw()` with explicit getter/setter:

```cpp
// Before: def_rw doesn't handle None assignment properly
py_conv_config.def_rw("shard_layout", &Conv2dConfig::shard_layout);  // ❌

// After: explicit property with getter/setter
py_conv_config.def_prop_rw(
    "shard_layout",
    [](Conv2dConfig& self) { return self.shard_layout; },
    [](Conv2dConfig& self, std::optional<TensorMemoryLayout> val) {
        self.shard_layout = val;
    });  // ✅
```

**Alternative Python workaround:**
```python
# Instead of setting to None after construction:
# config.shard_layout = None  ❌

# Set conditionally during construction:
config = Conv2dConfig(
    shard_layout=shard_layout if not auto_shard else None  # ✅
)
```

---

### 8. `kw_only()` Placement Matters

**Problem:** `nb::kw_only()` must come before the keyword-only arguments, not at the beginning.

**Symptoms:**
- All arguments become keyword-only unexpectedly
- Positional arguments fail

**Solution:** Place `kw_only()` correctly:

```cpp
// Before: all args become keyword-only
mod.def("from_host_shards", &from_host_shards,
    nb::kw_only(),  // ❌ Wrong placement
    nb::arg("tensors"),
    nb::arg("mesh_shape"));

// After: only args after kw_only are keyword-only
mod.def("from_host_shards", &from_host_shards,
    nb::arg("tensors"),  // Positional
    nb::kw_only(),       // ✅ Correct placement
    nb::arg("mesh_shape"));  // Keyword-only
```

---

### 9. Overload Ordering and Specificity

**Problem:** Nanobind may resolve overloads differently than pybind11. More specific overloads should be registered first.

**Symptoms:**
- Wrong overload is called
- Type conversion happens when it shouldn't

**Solution:** Order overloads from most specific to least specific:

```cpp
// Register more specific overload first
.def("__init__",
    [](CoreRangeSet* t, const std::vector<CoreRange>& core_ranges) {
        new (t) CoreRangeSet(tt::stl::Span<const CoreRange>(core_ranges));
    },
    nb::arg("core_ranges"))  // ✅ More specific (vector)
.def("__init__",
    [](CoreRangeSet* t, const std::set<CoreRange>& core_ranges) {
        new (t) CoreRangeSet(core_ranges);
    },
    nb::arg("core_ranges").noconvert())  // Less specific (set)
```

---

### 10. Return Value Policies

**Problem:** Nanobind's default return value policy may differ from pybind11. Returning references without proper policies causes issues.

**Symptoms:**
- Returned objects are copies when references are expected
- Memory leaks or dangling references

**Solution:** Explicitly specify return value policies:

```cpp
// For properties returning references to internal data
.def_prop_ro("spec",
    [](const Tensor& self) { return self.tensor_spec(); },
    nb::rv_policy::reference_internal)  // Keep self alive

// For functions that transfer ownership
.def("to_torch", &to_torch,
    nb::rv_policy::move)  // Transfer ownership

// For functions returning new objects
.def("clone", &clone,
    nb::rv_policy::copy)  // Return a copy
```

---

### 11. `bfloat16` Not Supported by Default in ndarray

**Problem:** Nanobind's ndarray doesn't recognize `bfloat16` dtype code by default.

**Symptoms:**
- Tensor conversion to torch/numpy fails for bfloat16
- DLPack dtype errors

**Solution:** We had to patch nanobind to add bfloat16 support. See `third_party/nanobind-bfloat.patch`:

```cpp
// Added to nanobind's nb_ndarray.cpp
case (uint8_t) dlpack::dtype_code::Bfloat:
    prefix = "bfloat";
    break;
```

---

### 12. Iterator Types Need `begin()`/`end()` (not just `cbegin()`/`cend()`)

**Problem:** Nanobind's automatic container bindings may require non-const iterators.

**Symptoms:**
- Container iteration fails
- `to_list()` methods don't work

**Solution:** Expose both const and non-const iterators:

```cpp
// In shape.hpp, we had to add:
using ShapeBase::begin;  // Added
using ShapeBase::end;    // Added
using ShapeBase::cbegin;
using ShapeBase::cend;
```

---

## Additional Resources

- [ttnn-pybind](../ttnn-pybind/) - pybind11 implementation for comparison
- [operations/examples/example](../ttnn/operations/examples/example/) - Example operation with both pybind11 and nanobind bindings
- [decorators.hpp](./decorators.hpp) - Nanobind-specific decorator utilities for operation binding
