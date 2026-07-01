// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BatchedStockhamFactory implementation — multi-core generalization of
// SingleTileStockhamFactory. Dispatches B independent length-N FFTs
// (N pow-2 in [2, 1024]) across up to max_cores Tensix cores.
//
// Per-core: handles `batch_per_core` consecutive sub-FFTs starting at
// `base_tile_idx`. The kernels' outer loop already handles batch_per_core > 1.
//
// All host-side setup (twiddle factors, zero-imag scratch) is one-time and
// cached — the per-call op path touches zero tensor data on host. Same
// fp32-internal / bf16-I/O design as SingleTileStockhamFactory.

#include "batched_stockham_factory.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>

#include "stockham_host.hpp"   // fft_stockham::get_cached_batch_plan, pick_batch_grid, buf_addr

namespace ttnn::experimental::prim {

namespace {

// All `_bs` suffix is to avoid Unity-build ODR collision with the same-named
// anonymous-namespace symbols in single_tile_stockham_factory.cpp (Unity
// concatenates them into one TU).
constexpr uint32_t kTileHW_bs        = 32u;
constexpr uint32_t kTileElems_bs     = kTileHW_bs * kTileHW_bs;            // 1024
constexpr uint32_t kTileBytesFp32_bs = kTileElems_bs * sizeof(float);      // 4096
constexpr uint32_t kTileBytesBf16_bs = kTileElems_bs * sizeof(uint16_t);   // 2048

constexpr uint32_t log2u_bs(uint32_t n) {
    uint32_t r = 0;
    while ((1u << r) < n) ++r;
    return r;
}

constexpr bool is_pow2_bs(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

// ── Zero-imag scratch buffer cache ─────────────────────────────────────
// Keyed by (device, dtype, B) — one tile per sub-FFT, all zeros. The
// reader treats this as the imag input for forward FFT of real signals.
// Re-allocated only when batch size grows (cache hits otherwise).
using MeshBufferPtr = std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>;

struct BatchedZeroScratch {
    MeshBufferPtr buf;
    uint32_t B = 0;
};

inline std::unordered_map<uint64_t, std::shared_ptr<BatchedZeroScratch>>&
batched_zero_scratch_cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<BatchedZeroScratch>> c;
    return c;
}

inline uint64_t batched_zero_key(
    tt::tt_metal::distributed::MeshDevice* md,
    tt::tt_metal::DataType                 dtype,
    uint32_t                               B)
{
    return reinterpret_cast<uint64_t>(md)
         ^ (static_cast<uint64_t>(dtype) * 0x9E3779B97F4A7C15ull)
         ^ (static_cast<uint64_t>(B)     * 0xBF58476D1CE4E5B9ull);
}

std::shared_ptr<BatchedZeroScratch> get_or_create_batched_zero_scratch(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md,
    tt::tt_metal::DataType                                 dtype,
    uint32_t                                               B)
{
    using namespace tt::tt_metal::distributed;
    const uint64_t key = batched_zero_key(md.get(), dtype, B);
    auto& cache = batched_zero_scratch_cache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    auto z = std::make_shared<BatchedZeroScratch>();
    z->B = B;
    MeshCommandQueue& cq = md->mesh_command_queue();

    if (dtype == tt::tt_metal::DataType::BFLOAT16) {
        const uint32_t total = B * kTileBytesBf16_bs;
        z->buf = fft_example::make_mesh_buf(md, total, kTileBytesBf16_bs);
        std::vector<uint16_t> zeros(static_cast<size_t>(B) * kTileElems_bs, 0u);
        WriteShard(cq, z->buf, zeros, MeshCoordinate(0, 0), /*blocking=*/true);
    } else {
        const uint32_t total = B * kTileBytesFp32_bs;
        z->buf = fft_example::make_mesh_buf(md, total, kTileBytesFp32_bs);
        std::vector<float> zeros(static_cast<size_t>(B) * kTileElems_bs, 0.0f);
        WriteShard(cq, z->buf, zeros, MeshCoordinate(0, 0), /*blocking=*/true);
    }

    cache.emplace(key, z);
    return z;
}

}  // namespace

tt::tt_metal::ProgramDescriptor BatchedStockhamFactory::create_descriptor(
    [[maybe_unused]] const FFTParams& operation_attributes,
    const FFTTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    // ── Resolve N (last dim) and B (product of leading dims) ───────────
    const auto& in_real = tensor_args.input_real;
    const auto& shape   = in_real.padded_shape();
    const uint32_t N    = static_cast<uint32_t>(shape[-1]);
    const uint32_t log2N = log2u_bs(N);

    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        B *= static_cast<uint32_t>(shape[d]);
    }

    TT_FATAL(is_pow2_bs(N) && N >= 2u && N <= kTileElems_bs,
        "BatchedStockhamFactory: requires pow-2 N in [2, 1024] (got N={}).", N);
    TT_FATAL(is_pow2_bs(B) && B >= 1u,
        "BatchedStockhamFactory: requires pow-2 batch (got B={}).", B);

    const DataType dtype = in_real.dtype();
    TT_FATAL(dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT16,
        "BatchedStockhamFactory: only fp32 / bf16 supported (got dtype {}).",
        static_cast<int>(dtype));
    const bool is_bf16 = (dtype == DataType::BFLOAT16);

    // ── Output tensor buffer addresses (already created by framework) ──
    const auto& out_r_tensor = std::get<0>(tensor_return_value);
    const auto& out_i_tensor = std::get<1>(tensor_return_value);
    auto* const in_r_buf  = in_real.buffer();
    auto* const out_r_buf = out_r_tensor.buffer();
    auto* const out_i_buf = out_i_tensor.buffer();
    TT_FATAL(in_r_buf != nullptr && out_r_buf != nullptr && out_i_buf != nullptr,
        "BatchedStockhamFactory: input/output tensors must be on device.");

    // ── Imag input (commit 3c): if the caller provided an imag tensor
    //    we use ITS buffer; otherwise we fall back to the cached B-tile
    //    zero scratch (real-only forward FFT, original commit-3a path).
    auto* device_raw = in_real.device();
    auto md = device_raw->get_mesh_device();

    // ── Cached fp32 twiddles (same per-N twiddle table, reused) ────────
    auto plan = fft_stockham::get_cached_batch_plan(md, /*sub_N=*/N, /*batch=*/1u);

    const bool have_imag = tensor_args.input_imag.has_value();
    auto* const in_i_buf = have_imag
            ? tensor_args.input_imag->buffer()
            : nullptr;
    std::shared_ptr<BatchedZeroScratch> zscratch;
    if (!have_imag) {
        zscratch = get_or_create_batched_zero_scratch(md, dtype, B);
    } else {
        TT_FATAL(in_i_buf != nullptr,
            "BatchedStockhamFactory: input_imag must be on device.");
    }
    const uint32_t in_i_addr = have_imag
            ? in_i_buf->address()
            : fft_stockham::buf_addr(zscratch->buf);
    // ROW_MAJOR imag (when provided by caller) has the same page_size as
    // the real input; zero scratch is tile-sized so override = 0 there.
    const uint32_t in_i_page_size_override = have_imag
            ? static_cast<uint32_t>(in_i_buf->aligned_page_size())
            : 0u;

    // ── Pick core grid: num_cores must divide B and fit grid ───────────
    const auto dev_grid = md->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    const uint32_t num_cores = (B < max_cores) ? B : max_cores;
    TT_FATAL(B % num_cores == 0u,
        "BatchedStockhamFactory: batch {} must divide num_cores {} cleanly.", B, num_cores);
    const uint32_t batch_per_core = B / num_cores;
    auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);

    ProgramDescriptor desc;

    const CoreCoord first{0, 0};
    const CoreCoord last{grid_cols - 1u, grid_rows - 1u};
    const CoreRange  cr(first, last);
    const CoreRangeSet crs({cr});

    // ── Circular Buffers (same layout as SingleTile) ───────────────────
    constexpr uint32_t kNumCbs = 17;
    constexpr uint32_t kCbTiles[kNumCbs] = {
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   // EVEN/ODD/TW/OUT real+imag
        1, 1, 1, 1,                     // TMP, TW_ODD
        1, 1,                           // STATE_R, STATE_I
        1                               // SYNC
    };
    for (uint32_t id = 0; id < kNumCbs; ++id) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = kCbTiles[id] * kTileBytesFp32_bs,
            .core_ranges = crs,
            .format_descriptors = {
                CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(id),
                    .data_format  = tt::DataFormat::Float32,
                    .page_size    = kTileBytesFp32_bs,
                }
            },
        });
    }

    if (is_bf16) {
        constexpr uint32_t kBf16CbIds[4] = { 17u, 18u, 19u, 20u };
        for (uint32_t i = 0; i < 4; ++i) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = kTileBytesBf16_bs,
                .core_ranges = crs,
                .format_descriptors = {
                    CBFormatDescriptor{
                        .buffer_index = static_cast<uint8_t>(kBf16CbIds[i]),
                        .data_format  = tt::DataFormat::Float16_b,
                        .page_size    = kTileBytesBf16_bs,
                    }
                },
            });
        }
    }

    // ── Kernels ────────────────────────────────────────────────────────
    const uint32_t input_bf16_flag  = is_bf16 ? 1u : 0u;
    const uint32_t output_bf16_flag = is_bf16 ? 1u : 0u;

    KernelDescriptor reader{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/batch_fft_reader.cpp",
        .core_ranges = crs,
        // {SUB_N, LOG2_SUB_N, BIT_REVERSE_ON_LOAD=1, INPUT_BF16}
        .compile_time_args = {N, log2N, 1u, input_bf16_flag},
        .runtime_args = {},   // filled per-core below
        .config = ReaderConfigDescriptor{},
    };

    KernelDescriptor writer{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/batch_fft_writer.cpp",
        .core_ranges = crs,
        // {OUTPUT_BF16, SUB_N}
        .compile_time_args = {output_bf16_flag, N},
        .runtime_args = {},
        .config = WriterConfigDescriptor{},
    };

    std::vector<UnpackToDestMode> u2d(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t id = 0; id < kNumCbs; ++id) {
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
    // Each core processes `batch_per_core` consecutive sub-FFTs starting at
    // base = core_idx * batch_per_core. The kernel's outer loop handles the
    // batch_per_core sequence; no host glue between sub-FFTs.
    //
    // ROW_MAJOR ttnn tensors use page_size = N*elem_size (NOT one full tile),
    // and the allocator places pages with `aligned_page_size` stride per bank
    // — that's the value InterleavedAddrGenFast must use for its within-bank
    // offset calculation.  page_size == aligned_page_size when the row is
    // already DRAM-aligned (32 B on Wormhole), but the safe call is
    // aligned_page_size() — matches what sdpa_decode / others use.
    //
    // The scratch / twiddle buffers we allocate ourselves are tile-sized,
    // so they use the kernel's default ts (override = 0).
    const uint32_t in_page_size_bytes  = static_cast<uint32_t>(in_r_buf->aligned_page_size());
    const uint32_t out_page_size_bytes = static_cast<uint32_t>(out_r_buf->aligned_page_size());
    reader.runtime_args.reserve(num_cores);
    writer.runtime_args.reserve(num_cores);
    compute.runtime_args.reserve(num_cores);

    for (uint32_t c = 0; c < num_cores; ++c) {
        const CoreCoord logical = fft_stockham::batch_logical_core(c, grid_cols);
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
            });

        writer.runtime_args.emplace_back(
            logical,
            KernelDescriptor::CoreRuntimeArgs{
                out_r_buf->address(),
                out_i_buf->address(),
                base,
                batch_per_core,
                /*out_page_size_override=*/out_page_size_bytes,
            });

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
