# `ttnn::experimental::fft` device kernels

All on-device kernels reachable from `ttnn.experimental.fft` /
`ttnn.experimental.ifft`. The host-side orchestrators (`device/*_host.hpp`)
`CreateKernel(...)` against the canonical paths under this directory.

## Layout

```
device/kernels/
├── dataflow/                            (BRISC0 reader / BRISC1 writer)
│   ├── fft_reader.cpp                   ┐
│   ├── fft_writer.cpp                   │ inner radix-2 single-tile FFT
│   ├── fft_common.h                     ┘  (sub_N <= 1024)  — fp32
│   ├── batch_fft_reader.cpp             ┐
│   ├── batch_fft_writer.cpp             │ batched single-tile FFT,
│   ├── batch_fft_common.h               ┘  parallel sub-FFTs   — fp32
│   ├── pass2_reader.cpp                 ┐
│   ├── pass2_writer.cpp                 │ Stockham pass-2: per-element
│   ├── pass2_common.h                   ┘  twiddle multiply   — fp32
│   ├── packed_dft_reader.cpp            ┐
│   ├── packed_dft_writer.cpp            │ packed direct DFT for small /
│   ├── packed_dft_common.h              ┘  composite radices  — fp32
│   ├── packed_dft_bf16_reader.cpp       ┐
│   ├── packed_dft_bf16_writer.cpp       │ packed direct DFT, bf16 FPU
│   └── packed_dft_bf16_common.h         ┘  matmul reduction   — bf16
└── compute/                             (TRISC0/1/2 — FPU + SFPU)
    ├── fft_compute.cpp                  radix-2 butterfly via FPU matmul (fp32)
    ├── batch_fft_compute.cpp            same as above, batched per core   (fp32)
    ├── pass2_compute.cpp                complex multiply via SFPU         (fp32)
    ├── packed_dft_compute.cpp           packed direct DFT compute         (fp32)
    ├── packed_dft_bf16_compute.cpp      packed direct DFT compute         (bf16)
    ├── packed_dft_common.h              ┐  duplicated from dataflow/ —
    └── packed_dft_bf16_common.h         ┘  see "Why two copies" below.
```

22 files: 15 dataflow (5 reader/writer pairs + 5 common headers),
7 compute (5 compute.cpp + 2 duplicated common.h headers).

### Why two copies of `packed_dft{,_bf16}_common.h`

The tt-metal kernel build resolves bare `#include "X_common.h"` only
against the kernel's own directory. The `packed_dft` and `packed_dft_bf16`
triples genuinely share state across both compute and dataflow, so each
common.h is duplicated into both `compute/` and `dataflow/`. Both copies
carry a sync-warning header. The other three common.h files (`fft`,
`batch_fft`, `pass2`) are only used by their reader/writer pair, so they
live in `dataflow/` only.

## Backend → kernel mapping

| `ttnn.experimental.fft` input         | Backend              | Kernels used                                                       |
|---------------------------------------|----------------------|--------------------------------------------------------------------|
| fp32, pow2, N ≤ 64K                   | `fft_stockham`       | `fft_*`                                                            |
| fp32, pow2, 64K < N ≤ 1M              | `fft_stockham`       | `batch_fft_*` + `pass2_*`                                          |
| fp32, pow2, 1M < N ≤ 16M              | `fft_universal_xl`   | (delegates to `fft_stockham` per sub-FFT)                          |
| fp32, non-pow2, `precision="precise"` | `fft_universal`      | recursive Stockham/Bluestein only (SFPU true-fp32)                 |
| fp32, non-pow2, `precision="fast"`    | `fft_universal`      | `packed_dft_*` + `fft_stockham` kernels for pow2 sub-FFTs          |
| bf16, any N                           | `fft_universal_bf16` | `packed_dft_bf16_*` + `fft_stockham` kernels                       |

`ttnn.experimental.ifft(re, im)` swaps real ↔ imag, calls the same forward
backend, then divides by N — so it reuses the table above unchanged.

## Build / install

The parent `CMakeLists.txt` does
`file(GLOB_RECURSE kernels device/kernels/*)` and installs the whole tree
to
`${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/fft/`
under the `ttnn-runtime` component. Adding new kernel files here needs
no CMake change — just rebuild `ninja -C build ttnn ttnncpp`.
