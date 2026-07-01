// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// FftRadixPassFactory implementation — fused FFT + optional post-twiddle.
//
// Layout mirrors BatchedStockhamFactory:
//   * fp32 path: 17 base CBs (0..16, same as BATCH_NUM_CBS) + 2 PT CBs (21, 22)
//   * bf16 I/O path: also CBs 17..20 (bf16 staging)
// Kernels:
//   reader  : radix_pass_reader.cpp (NEW — has APPLY_POST_TWIDDLE branch)
//   compute : batch_fft_compute.cpp (unchanged — sees no PT)
//   writer  : batch_fft_writer.cpp  (unchanged — STATE already post-mul'd)
//
// All host-side state (twiddle tables, zero-imag scratch) is per-(device,…)
// cached.  Hot path is a pure cache lookup → ProgramDescriptor assembly.

#include "fft_radix_pass_factory.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>

#include "stockham_host.hpp"          // fft_stockham::get_cached_batch_plan, buf_addr, ...
#include "apply_twiddles_host.hpp"    // apply_twiddles_host::get_or_create  (PT table)

namespace ttnn::experimental::prim {

namespace {

// `_rp` suffix avoids Unity-build ODR collision with the same-named
// anonymous-namespace symbols in single_tile_stockham_factory.cpp and
// batched_stockham_factory.cpp (Unity concatenates them into one TU).
constexpr uint32_t kTileHW_rp        = 32u;
constexpr uint32_t kTileElems_rp     = kTileHW_rp * kTileHW_rp;          // 1024
constexpr uint32_t kTileBytesFp32_rp = kTileElems_rp * sizeof(float);    // 4096
constexpr uint32_t kTileBytesBf16_rp = kTileElems_rp * sizeof(uint16_t); // 2048

constexpr uint32_t log2u_rp(uint32_t n) {
    uint32_t r = 0;
    while ((1u << r) < n) ++r;
    return r;
}

constexpr bool is_pow2_rp(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

// ── Zero-imag scratch (per (device, dtype, B)) ────────────────────────
// Identical purpose / layout to BatchedStockhamFactory's scratch; kept
// in its own cache to avoid taking a dependency on the other factory's
// anonymous namespace.
using MeshBufferPtr_rp = std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>;

struct ZeroScratch_rp {
    MeshBufferPtr_rp buf;
    uint32_t B = 0;
    // Weak reference to the owning device.  lock() returns nullptr once the
    // device is fully destroyed, correctly detecting stale entries even when
    // the heap allocator reuses the same raw MeshDevice* address.
    std::weak_ptr<tt::tt_metal::distributed::MeshDevice> device_weak;
};

inline std::unordered_map<uint64_t, std::shared_ptr<ZeroScratch_rp>>&
zero_scratch_cache_rp() {
    static std::unordered_map<uint64_t, std::shared_ptr<ZeroScratch_rp>> c;
    return c;
}

inline uint64_t zero_key_rp(
    tt::tt_metal::distributed::MeshDevice* md,
    tt::tt_metal::DataType                 dtype,
    uint32_t                               B)
{
    return reinterpret_cast<uint64_t>(md)
         ^ (static_cast<uint64_t>(dtype) * 0x9E3779B97F4A7C15ull)
         ^ (static_cast<uint64_t>(B)     * 0xBF58476D1CE4E5B9ull);
}

std::shared_ptr<ZeroScratch_rp> get_or_create_zero_scratch_rp(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md,
    tt::tt_metal::DataType                                 dtype,
    uint32_t                                               B)
{
    using namespace tt::tt_metal::distributed;
    const uint64_t key = zero_key_rp(md.get(), dtype, B);
    auto& cache = zero_scratch_cache_rp();
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (it->second->device_weak.lock()) return it->second;
        cache.erase(it);   // stale: device was destroyed (and ptr may be reused)
    }

    auto z = std::make_shared<ZeroScratch_rp>();
    z->B = B;
    z->device_weak = md;
    MeshCommandQueue& cq = md->mesh_command_queue();

    if (dtype == tt::tt_metal::DataType::BFLOAT16) {
        const uint32_t total = B * kTileBytesBf16_rp;
        z->buf = fft_example::make_mesh_buf(md, total, kTileBytesBf16_rp);
        std::vector<uint16_t> zeros(static_cast<size_t>(B) * kTileElems_rp, 0u);
        WriteShard(cq, z->buf, zeros, MeshCoordinate(0, 0), /*blocking=*/true);
    } else {
        const uint32_t total = B * kTileBytesFp32_rp;
        z->buf = fft_example::make_mesh_buf(md, total, kTileBytesFp32_rp);
        std::vector<float> zeros(static_cast<size_t>(B) * kTileElems_rp, 0.0f);
        WriteShard(cq, z->buf, zeros, MeshCoordinate(0, 0), /*blocking=*/true);
    }

    cache.emplace(key, z);
    return z;
}

}  // namespace

tt::tt_metal::ProgramDescriptor FftRadixPassFactory::create_descriptor(
    const FftRadixPassParams& operation_attributes,
    const FftRadixPassTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    const auto& in_real = tensor_args.input_real;
    const auto& shape   = in_real.padded_shape();
    const uint32_t N    = static_cast<uint32_t>(shape[-1]);

    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        B *= static_cast<uint32_t>(shape[d]);
    }

    TT_FATAL(N == operation_attributes.P,
        "FftRadixPassFactory: shape[-1]={} != params.P={}", N, operation_attributes.P);
    TT_FATAL(is_pow2_rp(N) && N >= 2u && N <= 1024u,
        "FftRadixPassFactory: P must be pow-2 in [2, 1024] (got {}).", N);
    TT_FATAL(is_pow2_rp(B) && B >= 1u,
        "FftRadixPassFactory: total batch (product of leading dims) must be "
        "pow-2 and >=1 (got {}).", B);

    const uint32_t log2N = log2u_rp(N);
    const bool apply_pt = operation_attributes.twiddle_N2 != 0u;
    const uint32_t pt_stride = apply_pt ? operation_attributes.stride : 1u;
    if (apply_pt) {
        const uint32_t twN2 = operation_attributes.twiddle_N2;
        TT_FATAL(is_pow2_rp(twN2) && twN2 >= 1u && twN2 <= 1024u,
            "FftRadixPassFactory: twiddle_N2 must be pow-2 in [1, 1024] (got {}).", twN2);
        TT_FATAL(is_pow2_rp(pt_stride) && pt_stride >= 1u && pt_stride <= B,
            "FftRadixPassFactory: stride must be pow-2 in [1, B={}] (got {}).",
            B, pt_stride);
        TT_FATAL((B % pt_stride) == 0u,
            "FftRadixPassFactory: total batch {} must be a multiple of stride {}.",
            B, pt_stride);
        TT_FATAL(((B / pt_stride) % twN2) == 0u,
            "FftRadixPassFactory: (B={} / stride={}) must be a multiple of "
            "twiddle_N2 {}.", B, pt_stride, twN2);
    }

    const auto dtype = in_real.dtype();
    TT_FATAL(dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT16,
        "FftRadixPassFactory: only fp32 / bf16 supported (got dtype {}).",
        static_cast<int>(dtype));
    const bool is_bf16 = (dtype == DataType::BFLOAT16);

    const auto& out_r_tensor = std::get<0>(tensor_return_value);
    const auto& out_i_tensor = std::get<1>(tensor_return_value);
    auto* const in_r_buf  = in_real.buffer();
    auto* const out_r_buf = out_r_tensor.buffer();
    auto* const out_i_buf = out_i_tensor.buffer();
    TT_FATAL(in_r_buf != nullptr && out_r_buf != nullptr && out_i_buf != nullptr,
        "FftRadixPassFactory: input/output tensors must be on device.");

    auto* device_raw = in_real.device();
    auto md = device_raw->get_mesh_device();

    // FFT internal twiddles — cached (device, N).
    auto plan = fft_stockham::get_cached_batch_plan(md, /*sub_N=*/N, /*batch=*/1u);

    // Imag input: caller-provided (complex input) or cached zero scratch.
    const bool have_imag = tensor_args.input_imag.has_value();
    auto* const in_i_buf = have_imag
            ? tensor_args.input_imag->buffer()
            : nullptr;
    std::shared_ptr<ZeroScratch_rp> zscratch;
    if (!have_imag) {
        zscratch = get_or_create_zero_scratch_rp(md, dtype, B);
    } else {
        TT_FATAL(in_i_buf != nullptr,
            "FftRadixPassFactory: input_imag must be on device.");
    }
    const uint32_t in_i_addr = have_imag
            ? in_i_buf->address()
            : fft_stockham::buf_addr(zscratch->buf);
    const uint32_t in_i_page_size_override = have_imag
            ? static_cast<uint32_t>(in_i_buf->aligned_page_size())
            : 0u;

    // Post-twiddle table — cached (device, N1=P, N2).  Only built if needed.
    std::shared_ptr<apply_twiddles_host::TwiddlePlan> pt_plan;
    uint32_t pt_r_addr = 0u;
    uint32_t pt_i_addr = 0u;
    const uint32_t pt_modulus = apply_pt ? operation_attributes.twiddle_N2 : 0u;
    if (apply_pt) {
        pt_plan = apply_twiddles_host::get_or_create(
            md, /*N1=*/N, /*N2=*/pt_modulus);
        pt_r_addr = fft_stockham::buf_addr(pt_plan->tw_r_buf);
        pt_i_addr = fft_stockham::buf_addr(pt_plan->tw_i_buf);
    }

    // ── Core grid: same logic as BatchedStockhamFactory ────────────────
    const auto dev_grid = md->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    const uint32_t num_cores = (B < max_cores) ? B : max_cores;
    TT_FATAL(B % num_cores == 0u,
        "FftRadixPassFactory: batch {} must divide num_cores {} cleanly.", B, num_cores);
    const uint32_t batch_per_core = B / num_cores;
    auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);

    ProgramDescriptor desc;

    const CoreCoord first{0, 0};
    const CoreCoord last{grid_cols - 1u, grid_rows - 1u};
    const CoreRange  cr(first, last);
    const CoreRangeSet crs({cr});

    // ── Circular Buffers ───────────────────────────────────────────────
    // CBs 0..16 = fp32 FFT base (identical to BatchedStockhamFactory).
    constexpr uint32_t kNumBaseCbs = 17;
    constexpr uint32_t kCbTiles[kNumBaseCbs] = {
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1,
        1, 1,
        1
    };
    for (uint32_t id = 0; id < kNumBaseCbs; ++id) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = kCbTiles[id] * kTileBytesFp32_rp,
            .core_ranges = crs,
            .format_descriptors = {
                CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(id),
                    .data_format  = tt::DataFormat::Float32,
                    .page_size    = kTileBytesFp32_rp,
                }
            },
        });
    }

    // bf16 I/O staging CBs 17..20 (only when input dtype is bf16).
    if (is_bf16) {
        constexpr uint32_t kBf16CbIds[4] = { 17u, 18u, 19u, 20u };
        for (uint32_t i = 0; i < 4; ++i) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = kTileBytesBf16_rp,
                .core_ranges = crs,
                .format_descriptors = {
                    CBFormatDescriptor{
                        .buffer_index = static_cast<uint8_t>(kBf16CbIds[i]),
                        .data_format  = tt::DataFormat::Float16_b,
                        .page_size    = kTileBytesBf16_rp,
                    }
                },
            });
        }
    }

    // Post-twiddle staging CBs 21..22.  Allocated ONLY when
    // APPLY_POST_TWIDDLE=1 so the pure-FFT path keeps a bit-identical
    // CB layout to BatchedStockhamFactory (avoids any L1-placement
    // sensitivity for that path).  Tile-sized fp32.
    if (apply_pt) {
        constexpr uint32_t kPtCbIds[2] = { 21u, 22u };
        for (uint32_t i = 0; i < 2; ++i) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = kTileBytesFp32_rp,
                .core_ranges = crs,
                .format_descriptors = {
                    CBFormatDescriptor{
                        .buffer_index = static_cast<uint8_t>(kPtCbIds[i]),
                        .data_format  = tt::DataFormat::Float32,
                        .page_size    = kTileBytesFp32_rp,
                    }
                },
            });
        }
    }

    // ── Kernels ────────────────────────────────────────────────────────
    const uint32_t input_bf16_flag  = is_bf16 ? 1u : 0u;
    const uint32_t output_bf16_flag = is_bf16 ? 1u : 0u;
    const uint32_t apply_pt_flag    = apply_pt ? 1u : 0u;
    // commit 6c: IFFT folds 1/N into the LAST radix_pass writer.
    // We compile the per-element scale loop into the kernel binary only
    // when the host actually asks for a non-unity scale (so the legacy
    // commit-5 path keeps its bit-identical writer kernel and program
    // cache entry).
    const bool     apply_scale      = (operation_attributes.output_scale != 1.0f);
    const uint32_t apply_scale_flag = apply_scale ? 1u : 0u;
    uint32_t output_scale_bits = 0u;
    {
        const float scale_value = operation_attributes.output_scale;
        std::memcpy(&output_scale_bits, &scale_value, sizeof(float));
    }

    KernelDescriptor reader{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/radix_pass_reader.cpp",
        .core_ranges = crs,
        // {SUB_N, LOG2_SUB_N, BIT_REVERSE_ON_LOAD=1, INPUT_BF16, APPLY_POST_TWIDDLE}
        .compile_time_args = {N, log2N, 1u, input_bf16_flag, apply_pt_flag},
        .runtime_args = {},
        .config = ReaderConfigDescriptor{},
    };

    KernelDescriptor writer{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/radix_pass_writer.cpp",
        .core_ranges = crs,
        // {OUTPUT_BF16, SUB_N, APPLY_POST_TWIDDLE, APPLY_SCALE}
        .compile_time_args = {output_bf16_flag, N, apply_pt_flag, apply_scale_flag},
        .runtime_args = {},
        .config = WriterConfigDescriptor{},
    };

    std::vector<UnpackToDestMode> u2d(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t id = 0; id < kNumBaseCbs; ++id) {
        u2d[id] = UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor compute{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/compute/batch_fft_compute.cpp",
        .core_ranges = crs,
        .compile_time_args = {log2N},
        .runtime_args = {},
        .config = ComputeConfigDescriptor{
            .math_fidelity       = MathFidelity::HiFi4,
            .fp32_dest_acc_en    = true,
            .unpack_to_dest_mode = u2d,
        },
    };

    // ── Per-core runtime args ──────────────────────────────────────────
    const uint32_t in_page_size_bytes  = static_cast<uint32_t>(in_r_buf->aligned_page_size());
    const uint32_t out_page_size_bytes = static_cast<uint32_t>(out_r_buf->aligned_page_size());

    reader.runtime_args.reserve(num_cores);
    writer.runtime_args.reserve(num_cores);
    compute.runtime_args.reserve(num_cores);

    for (uint32_t c = 0; c < num_cores; ++c) {
        const CoreCoord logical  = fft_stockham::batch_logical_core(c, grid_cols);
        const CoreCoord physical = md->worker_core_from_logical_core(logical);
        const uint32_t base = c * batch_per_core;

        reader.runtime_args.emplace_back(
            logical,
            KernelDescriptor::CoreRuntimeArgs{
                in_r_buf->address(),
                in_i_addr,
                fft_stockham::buf_addr(plan->tw_r_buf),
                fft_stockham::buf_addr(plan->tw_i_buf),
                base,
                batch_per_core,
                static_cast<uint32_t>(physical.x),
                static_cast<uint32_t>(physical.y),
                /*in_page_size_override=*/in_page_size_bytes,
                /*in_imag_page_size_override=*/in_i_page_size_override,
                /*pt_r_addr=*/pt_r_addr,
                /*pt_i_addr=*/pt_i_addr,
                /*pt_modulus=*/pt_modulus,
                /*pt_stride=*/pt_stride,
            });

        // commit 6c: arg 5 (output_scale_bits) is ONLY appended when
        // APPLY_SCALE=1.  When 0, the kernel never reads it and the
        // runtime arg vector stays at 5 entries (same as commit 5).
        if (apply_scale) {
            writer.runtime_args.emplace_back(
                logical,
                KernelDescriptor::CoreRuntimeArgs{
                    out_r_buf->address(),
                    out_i_buf->address(),
                    base,
                    batch_per_core,
                    /*out_page_size_override=*/out_page_size_bytes,
                    /*output_scale_bits=*/output_scale_bits,
                });
        } else {
            writer.runtime_args.emplace_back(
                logical,
                KernelDescriptor::CoreRuntimeArgs{
                    out_r_buf->address(),
                    out_i_buf->address(),
                    base,
                    batch_per_core,
                    /*out_page_size_override=*/out_page_size_bytes,
                });
        }

        compute.runtime_args.emplace_back(
            logical,
            KernelDescriptor::CoreRuntimeArgs{ batch_per_core });
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));

    return desc;
}

}  // namespace ttnn::experimental::prim
