# TTNN IsIn

## Overview

The TTNN IsIn operation, `ttnn.experimental.isin`, checks, for every value of an `elements` tensor, whether the same value appears anywhere in a `test_elements` tensor.

The operation is a host-side wrapper around a single device operation. The wrapper flattens both inputs to 1-D row-major tensors. The device operation splits `elements` into chunks that fit in a core's L1 memory and assigns whole chunks to cores. Each core compares every value of its chunks with the values of `test_elements`, one pair at a time, until it finds a match. The comparisons run on one of the core's data-movement processors. Finally, the wrapper restores the shape and layout.

## Brief Functional Description

The operation returns a mask with one value for each value of `elements`: True if the value is in `test_elements`, False otherwise, or the opposite with `invert=True`. The mask is a `uint32` tensor (`0xFFFFFFFF` for True, `0` for False) with the shape, layout and memory configuration of `elements` (unless `out` is given). The shape of `test_elements` does not matter; all of its values are used.

```
elements      = [[1, 2, 3],
                 [4, 5, 6]]
test_elements = [2, 5, 9]

isin(elements, test_elements)              -> [[F, T, F],
                                               [F, T, F]]
isin(elements, test_elements, invert=True) -> [[T, F, T],
                                               [T, F, T]]
```

### Arguments

- **elements** (Tensor): The values to look up.
- **test_elements** (Tensor): The values to look for. It must have the same dtype as `elements`.
- **assume_unique** (bool, optional, keyword-only): Accepted, but it has no effect. Defaults to False.
- **invert** (bool, optional, keyword-only): If True, marks the values that are **not** in `test_elements`. Defaults to False.
- **out** (Tensor, optional, keyword-only): Preallocated `uint32` tensor for the result; see [Preallocated output](#preallocated-output-out). Defaults to None.

### Usage

#### TTNN

```python
import torch
import ttnn

device = ttnn.open_device(device_id=0)

elements = ttnn.from_torch(torch.tensor([[1, 2, 3], [4, 5, 6]]), dtype=ttnn.int32, device=device)
test_elements = ttnn.from_torch(torch.tensor([2, 5, 9]), dtype=ttnn.int32, device=device)

mask = ttnn.experimental.isin(elements, test_elements)
print(ttnn.to_torch(mask))
# tensor([[         0, 4294967295,          0],
#         [         0, 4294967295,          0]], dtype=torch.uint32)
print(ttnn.to_torch(mask) != 0)
# tensor([[False,  True, False],
#         [False,  True, False]])

inverted = ttnn.experimental.isin(elements, test_elements, invert=True)
print(ttnn.to_torch(inverted) != 0)
# tensor([[ True, False,  True],
#         [ True, False,  True]])

ttnn.close_device(device)
```

#### Metalium

```cpp
#include "ttnn/operations/experimental/isin/isin.hpp"

// Input tensors, on the device and with the same dtype
const ttnn::Tensor elements = ...;
const ttnn::Tensor test_elements = ...;

// Optional parameters
const bool assume_unique = false;  // has no effect
const bool invert = false;
const std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt;

ttnn::Tensor mask =
    ttnn::experimental::isin(elements, test_elements, assume_unique, invert, optional_output_tensor);
```

### Universal Input/Output Support

The table below lists the tensor formats isin accepts for its inputs and can produce for its output. Dtypes are listed separately in [Supported dtypes](#supported-dtypes).

| Format | isin | Notes |
| --- | --- | --- |
| Interleaved DRAM inputs | ✓ | |
| Interleaved L1 inputs | Partly | Small tensors only; see [Usage Limitations](#usage-limitations). |
| Sharded inputs (height, width, block or ND; L1 or DRAM) | ✗ | Rejected; see [Usage Limitations](#usage-limitations). |
| `ROW_MAJOR` and `TILE` inputs | ✓ | `TILE` inputs are converted to row-major internally. |
| Two inputs with different layouts or memories | ✓ | |
| Output memory configuration chosen by the user | Partly | There is no `memory_config` argument. The output follows `elements`, unless a preallocated `out` is given; the result then uses the memory of `out`. |
| Sharded output | ✗ | |
| Output layout different from `elements` | ✗ | The returned tensor always has the layout of `elements`. |
| Preallocated output (`out=`) | Partly | Only a `uint32`, row-major, 1-D, interleaved tensor; see [Preallocated output](#preallocated-output-out). |

#### Supported dtypes

Both inputs must have the same dtype. The output is always `uint32`.

| Dtype | Supported | Notes |
| --- | --- | --- |
| `int32`, `uint32` | ✓ | |
| `uint16` | Partly | An input with more than one row is flattened with `ttnn.reshape`, which for `uint16` needs Wormhole B0 or Blackhole and enough free L1 for its reshape program; see the `ttnn.reshape` documentation. |
| `uint8` | ✓ | |
| `bfloat16` | ✓ | Compared by bit pattern: `-0.0` does not match `0.0`, and `NaN` matches a `NaN` with the same bits. |
| `float32` | ✓ | Compared as floating-point values: `-0.0` matches `0.0`, and `NaN` never matches. |
| `bfloat8_b`, `bfloat4_b` | ✗ | |

#### Supported tensor shapes

- Any rank, including rank 0 (a single value). The two shapes are independent and need not be multiples of the tile size.
- `test_elements` may be empty. Every result is then False, or True with `invert=True`.
- `elements` must not be empty.

#### Preallocated output (`out`)

- `out` must be a `uint32`, row-major, 1-D, interleaved tensor with one value per value of `elements`, in DRAM or L1. It must not share storage with `test_elements`.
- Only the dtype is checked. With another shape or layout, `out` gets wrong values, and the write can overwrite memory past the end of `out`.
- If `elements` is 1-D and row-major, the returned tensor is `out` itself. Otherwise `out` holds the flat mask, and the returned tensor is `out` reshaped to the shape of `elements` (and converted to `TILE` if `elements` is tiled).

### Usage Limitations

| Input | What happens | What to do instead |
| --- | --- | --- |
| Inputs with different dtypes | `elements.dtype: ..., test_elements.dtype: ..., they must be equal` | Convert one input with `ttnn.typecast`, if its values fit in the other dtype. |
| A sharded input | `Elements tensor is sharded` or `Test elements tensor is sharded`. DRAM-sharded inputs fail earlier, with `Input tensor must be in L1`. | Convert it to interleaved, for example with `ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)`. |
| A `bfloat8_b` input | `Statically allocated circular buffers ... beyond max L1 size` | Convert both inputs to `bfloat16` with `ttnn.typecast`. |
| A `bfloat4_b` input | `Only TILE layout is supported for BFLOAT8_B dtype!` | Convert both inputs to `bfloat16` with `ttnn.typecast`. |
| An empty `elements` | `Failed to generate binaries for isin_reader` | Skip the call. |
| A host or deallocated `test_elements` | `Device Operations expect device tensors as inputs` or `Input Tensor is not allocated` | Move it to the device with `ttnn.to_device`, and keep it allocated. |
| A host or deallocated `elements` | The process crashes with a segmentation fault. | Move it to the device with `ttnn.to_device`, and keep it allocated. |
| A Python number, list or `torch` tensor as an input, or an optional argument passed by position | `TypeError: ... incompatible function arguments` | Use `ttnn.from_torch(..., device=device)`, and pass `assume_unique`, `invert` and `out` by keyword. |
| An `out` that is not `uint32` | `Preallocated output should be of uint32 dtype` | See [Preallocated output](#preallocated-output-out). |
| An `out` with another shape or layout | No error: wrong values, and memory past `out` can be overwritten. | Use a 1-D row-major `out`. |
| Inputs in L1 that are not small, or large tensors already in L1 | `Statically allocated circular buffers in program ... clash with L1 buffers` | Keep the inputs and other large tensors in DRAM. |
| A very large `elements` | `Out of Memory: ... DRAM buffer ... where each bank needs to store ...` | Call isin on smaller parts of `elements`. |
| `bfloat16` values `-0.0`, `0.0` or `NaN` | No error; they are compared by bit pattern, so `-0.0` does not match `0.0`, and `NaN` matches a `NaN` with the same bits. | If `-0.0` must match `0.0` and `NaN` must never match, create the tensors as `float32`. Converting `bfloat16` tensors with `ttnn.typecast` is not enough: on a Wormhole N150 it makes `-0.0` match `0.0`, but turns `NaN` into `inf`. |

Why the memory limits exist:
- **L1**: on every core it uses, isin reserves about 80% of L1 for its buffers, even for tiny inputs. All L1 tensors that exist during the call must fit in the rest: about 194 KB per L1 bank on a Wormhole N150. A 1-D input, a flattened input and the mask are single pages, and a single page takes its whole size in every bank. For example, a 1-D `int32` `elements` of 24,000 values in L1 works, and 25,000 values fail.
- **DRAM**: for the same reason, the flattened inputs and the mask together must fit in one DRAM bank (about 1 GB on a Wormhole N150).

## Tensor Transformations

The host-side wrapper (`preprocess_isin_inputs()` and `process_isin_output()` in `isin.cpp`) prepares the inputs before the device operation and restores the output afterwards. These steps are transparent to the user.

### Pre-IsIn Transformations

1. **Dtype check**: both inputs must have the same dtype.
2. **Chunk size**: computed from the L1 size of a core (see [Work Distribution](#work-distribution)).
3. **Layout normalization**: each `TILE` input is converted to `ROW_MAJOR`, with one untilize operation per tiled input.
4. **Flattening**: each input with rank > 1 is reshaped to 1-D. For an input with a single row (for example `[1, 1, N]`) only the metadata changes. With more rows, a reshape operation copies the data.

### Post-IsIn Transformations

1. **Shape restoration**: unless `elements` is 1-D, the flat mask is reshaped to the shape of `elements`.
2. **Layout restoration**: if `elements` is in `TILE` layout, the mask is converted back to `TILE` with one tilize operation.

So a call runs between 0 and 6 helper operations around the IsIn device operation.

### Example Transformation Flow

For `elements` of shape `[2, 95]` and `test_elements` of shape `[2, 3]`, both in `TILE` layout:

**Pre-IsIn:**
1. Untilize both inputs: `[2, 95]` and `[2, 3]` in `ROW_MAJOR`
2. Flatten both inputs: `[2, 95]` → `[190]` and `[2, 3]` → `[6]`

**IsIn device operation**: `[190]` and `[6]` → a `[190]` `uint32` mask, on one core (190 values fit in one chunk)

**Post-IsIn:**
1. Reshape the mask: `[190]` → `[2, 95]`
2. Tilize the mask: `ROW_MAJOR` → `TILE`

## Strategy Implementation Details

isin has a single program factory, `IsInProgramFactory`, so there is no strategy selection.

### Work Distribution

```
chunk  = floor(0.8 × L1_size_per_core / (elements_bytes + test_elements_bytes + 4)), rounded down to a multiple of 64
chunks = ceil(N / chunk)                                          where N = number of values in elements
```

- A core holds one chunk of `elements`, one chunk of `test_elements` and one chunk of the `uint32` output (4 bytes per value) in L1 at the same time. Together they take about 80% of L1. On a Wormhole N150, a chunk is 99,904 values for 4-byte dtypes, 149,888 for 2-byte dtypes and 199,872 for `uint8`.
- `split_work_to_cores` assigns whole chunks to cores. With fewer chunks than cores, each chunk gets its own core. Otherwise all cores are used, and the first `chunks % cores` cores take one chunk more than the others.
- Cores are filled column by column, starting at core (0, 0).
- The number of cores depends only on the size of `elements`. Every core reads all of `test_elements`, chunk by chunk.

### Core Utilization Pattern

```
N = 250,000 int32 values, chunk = 99,904 (Wormhole N150) -> 3 chunks -> 3 cores

├── Core (0,0): chunk 0 -> elements[0       : 99,904 ]
├── Core (0,1): chunk 1 -> elements[99,904  : 199,808]
└── Core (0,2): chunk 2 -> elements[199,808 : 250,000]   (50,192 values)
```

## Core IsIn Algorithm

### Compare Loop

The comparison is a double loop in the reader kernel (`isin_reader.cpp`, simplified):

```cpp
// The output chunk is prefilled with 0 (or with 0xFFFFFFFF when invert is set).
for (uint32_t i = 0; i < elements_chunk_size; ++i) {
    for (uint32_t j = 0; j < test_elements_chunk_size; ++j) {
        if (elements[i] == test_elements[j]) {
            output[i] = invert ? 0x00000000 : 0xFFFFFFFF;
            break;  // stop at the first match
        }
    }
}
```

- The search for a value stops at its first match. A value that is never found costs one comparison per test value.
- When `test_elements` is longer than one chunk, the loop runs once per test chunk, and the early stop applies within each test chunk.

### How Values Are Compared

The kernel compares values as the C++ type that matches the dtype (`df_to_std` in `device/kernels/dataflow/isin_common.hpp`):

| Dtype | C++ type | Effect |
| --- | --- | --- |
| `int32`, `uint32` | `uint32_t` | Exact 32-bit equality. |
| `uint16`, `bfloat16` | `uint16_t` | Exact 16-bit equality. For `bfloat16` this compares bit patterns. |
| `uint8` | `uint8_t` | Exact 8-bit equality. |
| `float32` | `float` | Floating-point equality. Each comparison compiles to a call to a software routine (`__eqsf2` / `__nesf2`). |

## Kernel Architecture

### Reader-Writer Split Architecture

Each core runs two data-movement kernels and no compute kernel:

```
DRAM or L1                                one core
elements (flat)       --- chunk --->   c_0: elements chunk        reader (NCRISC): reads,
test_elements (flat)  --- chunk --->   c_1: test_elements chunk   compares, fills c_2
output mask (flat)    <--- chunk ---   c_2: output chunk          writer (BRISC): writes c_2
```

#### Reader Kernel Responsibilities

`isin_reader.cpp` runs on RISCV_1 (NCRISC). For each chunk of `elements` assigned to the core, it reads the chunk into `c_0`, prefills the output chunk in `c_2`, reads `test_elements` chunk by chunk into `c_1` and runs the compare loop on each, then hands the output chunk to the writer.

#### Writer Kernel Responsibilities

`isin_writer.cpp` runs on RISCV_0 (BRISC). It waits for each output chunk in `c_2` and writes it to the output buffer at the matching offset.

### Circular Buffer Management

| CB | Holds | Size (one page) |
| --- | --- | --- |
| `c_0` | one chunk of `elements` | `chunk × elements_bytes` |
| `c_1` | one chunk of `test_elements` | `chunk × test_elements_bytes` |
| `c_2` | one chunk of the `uint32` output | `chunk × 4` |

Each CB holds one chunk, with no double buffering. The sizes follow the chunk length, not the size of the inputs, so they take about 80% of L1 on every core used, even for tiny inputs.

### Program Caching

A call with a new size, dtype, `invert` or `assume_unique` value, or memory configuration of `elements` adds a new entry to the program cache. The sizes of both inputs, the chunk length and `invert` are compile-time kernel arguments, so new values of these also compile new kernels. The buffer addresses are runtime arguments, so later calls with new data of the same shapes reuse the cached program.

## Performance Characteristics

- **The work on a core grows with its share of `elements` × `M`** (the number of values in `test_elements`), because values are compared one pair at a time.
- **The data matters.** A value that is found early in `test_elements` is cheap; a value that is not found costs `M` comparisons.
- **More cores only for more values.** Inputs smaller than one chunk run on a single core. Larger inputs are spread over cores in whole chunks, so the time follows the number of chunks on the busiest core.
- **`float32` is several times slower** than the other dtypes. Its comparisons call a software routine instead of a single integer compare.
- **Shape and layout affect transformation costs; memory placement affects transfers.** For the same flattened values, shape and layout do not change the comparison work, but tiled and multi-row inputs can require data-moving helper operations. DRAM versus L1 placement affects the kernel's input reads and output writes.
- **The first call with new sizes** compiles new kernels; see [Program Caching](#program-caching).

---

© Tenstorrent AI ULC 2026
