# `ttnn::experimental::fft` device kernels

All on-device kernels reachable from `ttnn.experimental.fft` /
`ttnn.experimental.ifft`. Each kernel is `CreateKernel(...)`-referenced
by the corresponding `device/*_factory.cpp` ProgramDescriptor factory
via a canonical path
`ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/...`.

## Layout

```
device/kernels/
├── dataflow/                            (BRISC0 reader / BRISC1 writer)
│   ├── fft_reader.cpp                   ┐
│   ├── fft_writer.cpp                   │ single-tile Stockham FFT
│   ├── fft_common.h                     ┘  (sub_N ≤ 1024, real input) — fp32
│   ├── batch_fft_reader.cpp             ┐
│   ├── batch_fft_writer.cpp             │ batched single-tile Stockham,
│   ├── batch_fft_common.h               ┘  parallel sub-FFTs           — fp32
│   ├── radix_pass_reader.cpp            ┐
│   ├── radix_pass_writer.cpp            │ fused (batched-FFT + post-
│   ├── radix_pass_common.h              ┘  twiddle) primitive used by
│   │                                       two-pass and three-pass — fp32
│   ├── pass2_reader.cpp                 ┐
│   ├── pass2_writer.cpp                 │ pure between-pass twiddle
│   ├── pass2_common.h                   ┘  multiply (bf16 path)  — bf16
│   ├── packed_dft_reader.cpp            ┐
│   ├── packed_dft_writer.cpp            │ packed direct DFT for small
│   ├── packed_dft_common.h              ┘  radices (matmul-based) — fp32
│   ├── packed_dft_bf16_reader.cpp       ┐
│   ├── packed_dft_bf16_writer.cpp       │ packed direct DFT, bf16 FPU
│   ├── packed_dft_bf16_common.h         ┘  matmul reduction       — bf16
│   ├── apply_twiddles_reader.cpp        ┐
│   ├── apply_twiddles_writer.cpp        │ between-pass table-driven
│   ├── apply_twiddles_common.h          ┘  twiddle multiply (two-pass)
│   ├── apply_twiddles_xl_reader.cpp     large-modulus twiddle multiply
│   │                                    (three-pass, on-the-fly recurrence)
│   ├── complex_mul_reader.cpp           ┐
│   ├── complex_mul_writer.cpp           │ elementwise (a+bi)·(c+di)
│   ├── complex_mul_common.h             ┘  used by Bluestein pre/post
│   ├── rebank_rm_reader.cpp             ┐
│   ├── rebank_rm_writer.cpp             │ ROW_MAJOR (B, N) → (B·N/K, K)
│   │                                    │  streaming rebank (no CB blow-up)
│   ├── rebank_rm_merge_reader.cpp       ┐
│   ├── rebank_rm_merge_writer.cpp       ┘  inverse rebank: (B·N/K, K) → (B, N)
│   ├── transpose_rm_reader.cpp          ┐
│   ├── transpose_rm_writer.cpp          │ precision-preserving inner-axis
│   └── transpose_rm_common.h            ┘  ROW_MAJOR transpose
│
└── compute/                             (TRISC0/1/2 — FPU + SFPU)
    ├── fft_compute.cpp                  radix-2 butterfly via FPU matmul (fp32)
    ├── batch_fft_compute.cpp            same, batched per core           (fp32)
    ├── pass2_compute.cpp                per-element complex multiply     (fp32)
    ├── packed_dft_compute.cpp           packed direct DFT compute        (fp32)
    ├── packed_dft_bf16_compute.cpp      packed direct DFT compute        (bf16)
    ├── apply_twiddles_compute.cpp       between-pass twiddle multiply    (fp32)
    ├── complex_mul_compute.cpp          elementwise complex multiply     (fp32/bf16)
    ├── packed_dft_common.h              ┐  duplicated from dataflow/ —
    └── packed_dft_bf16_common.h         ┘  see "Why two copies" below.
```

41 files total: 32 dataflow (13 reader/writer pairs + 6 common headers)
and 9 compute (7 compute.cpp + 2 duplicated common.h headers).

### Why two copies of `packed_dft{,_bf16}_common.h`

The tt-metal kernel build resolves bare `#include "X_common.h"` only
against the kernel's own directory. The `packed_dft` and `packed_dft_bf16`
triples genuinely share state across both compute and dataflow, so each
common.h is duplicated into both `compute/` and `dataflow/`. Both copies
carry a sync-warning header. The other four common.h files (`fft`,
`batch_fft`, `pass2`, `radix_pass`, `apply_twiddles`, `complex_mul`,
`transpose_rm`) are only used by their reader/writer pair, so they live
in `dataflow/` only.

## Backend → kernel mapping

The public dispatch (see `../../fft.cpp`):

| `ttnn.experimental.fft` input           | Composite path            | Device primitives used                                                              |
|-----------------------------------------|---------------------------|-------------------------------------------------------------------------------------|
| pow-2, N ≤ 1024, B ≥ 1 real             | `ttnn::prim::fft` direct  | `single_tile_stockham` (`fft_*`) or `batched_stockham` (`batch_fft_*` + `pass2_*`)  |
| pow-2, 1024 < N ≤ 2^20                  | `fft_two_pass`            | `transpose_rm_*` + `radix_pass_*` + `apply_twiddles_*` + `ttnn::prim::fft`          |
| pow-2, 2^20 < N ≤ 2^30                  | `fft_three_pass_auto`     | as above + `apply_twiddles_xl_reader` (recurrence twiddles for large modulus)       |
| non-pow-2, M = next_pow2(2N-1) ≤ 2^30   | `bluestein_dispatch`      | `complex_mul_*` (pre/post chirp + H multiply) + one of the pow-2 paths above        |
| any pow-2 IFFT                          | `small_pow2_ifft` (≤1024) or the same two/three-pass composites (>1024) with `output_scale=1/N` folded into the writer                              |

`ttnn.experimental.ifft(re, im)` uses the swap-trick (real↔imag) and
folds the 1/N normalisation into the writer, so it reuses the same
compute kernels as forward FFT unchanged.

## Build / install

The parent `CMakeLists.txt` does
`file(GLOB_RECURSE kernels device/kernels/*)` and installs the whole
tree to
`${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/fft/`
under the `ttnn-runtime` component. Adding a new kernel file here needs
no CMake change — just rebuild `ninja -C build ttnn ttnncpp`.
