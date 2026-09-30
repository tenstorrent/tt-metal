# `ttnn::experimental::fft` device kernels

These are the Metal 2.0 kernels reachable from `ttnn.experimental.fft`
and `ttnn.experimental.ifft`. Each source is referenced by a
`KernelSpec` in the corresponding `device/*_factory.cpp`.

Kernels use named resources generated from their program specification:

- `dfb::...` for dataflow buffers
- `tensor::...` for tensor accessors
- `args::...` for compile-time and runtime arguments

## Active kernel groups

| Group | Sources | Used by |
| --- | --- | --- |
| Batched Stockham | `batch_fft_{reader,writer}.cpp`, `batch_fft_compute.cpp` | Power-of-two FFTs with `N <= 1024` |
| Radix pass | `radix_pass_{reader,writer}.cpp`, `batch_fft_compute.cpp` | Two- and three-pass FFTs |
| Table twiddles | `apply_twiddles_{reader,writer}.cpp`, `apply_twiddles_compute.cpp` | Two-pass between-pass multiply |
| XL twiddles | `apply_twiddles_xl_reader.cpp`, `apply_twiddles_writer.cpp`, `apply_twiddles_compute.cpp` | Three-pass large-modulus multiply |
| Complex multiply | `complex_mul_reader.cpp`, `apply_twiddles_writer.cpp`, `apply_twiddles_compute.cpp` | Bluestein pre/post chirp and spectrum multiply |
| Rebank | `rebank_rm_{reader,writer}.cpp` | Streaming row-major rebank |
| Rebank merge | `rebank_rm_merge_{reader,writer}.cpp` | Inverse row-major rebank |
| Transpose | `transpose_rm_{reader,writer}.cpp` | Precision-preserving inner-axis transpose |

The single-tile FFT factory delegates to the batched Stockham program, so it
does not have a separate kernel triple.

## Dispatch mapping

| Input | Composite path | Device primitives |
| --- | --- | --- |
| Power-of-two, `N <= 1024` | `ttnn::prim::fft` | Batched Stockham |
| Power-of-two, `1024 < N <= 2^20` | `fft_two_pass` | Transpose, radix pass, table twiddles |
| Power-of-two, `2^20 < N <= 2^30` | `fft_three_pass_auto` | Transpose, radix pass, table and XL twiddles |
| Non-power-of-two | `bluestein_dispatch` | Complex multiply plus the appropriate power-of-two path |

IFFT uses the same kernels with swapped real/imaginary inputs and applies
`1/N` scaling in the writer.

## Build and installation

The parent CMake configuration installs the kernel tree with the TTNN runtime.
Kernel sources are compiled on demand through their `KernelSpec`; adding or
renaming one requires updating the corresponding factory source path.
