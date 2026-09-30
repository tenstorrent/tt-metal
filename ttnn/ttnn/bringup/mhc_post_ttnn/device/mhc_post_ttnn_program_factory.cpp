// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_post_ttnn_program_factory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::bringup::mhc_post_ttnn {

namespace {

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::DataType;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;

constexpr const char* KERNEL_DIR = "ttnn/ttnn/bringup/mhc_post_ttnn/kernels/";

constexpr uint32_t TILE_HW = 32;

// ---- CB slots (semantic names) ----
constexpr uint8_t CB_SUBLAYER_TILES = 0;  // F block tiles              reader (+ writer's read help) -> compute
constexpr uint8_t CB_RESIDUAL_TILES = 1;  // X block tiles (n streams)  reader  -> compute
constexpr uint8_t CB_COEF_RAW = 2;        // raw post / comb tiles of one token row, private scratch of the writer
constexpr uint8_t CB_COEF_BCAST = 3;     // n * P half-packed column-broadcast fp32 coefficient tiles, writer -> compute
constexpr uint8_t CB_OUTPUT_TILES = 16;  // X' block tiles (n streams) compute -> writer

// ---- Block-model knobs (the Python module's values; see mhc_post_program_descriptor.py for the measurements) ----
constexpr int64_t DEPTH_IN = 2;
constexpr int64_t DEPTH_OUT = 2;
constexpr int64_t COEF_DEPTH = 2;
constexpr int64_t L1_BUDGET_BYTES = 1 << 20;
constexpr int64_t MAX_BLOCK_COL_TILES = 8;
constexpr int64_t MIN_BLOCKS_PER_CORE = 3;
constexpr bool DST_FULL_SYNC = true;
constexpr int64_t HELP_MIN_BLOCKS = 6;
constexpr bool HELP_FP32_STREAMS = false;
constexpr uint32_t HELP_FROM_BLOCK = 1;
constexpr uint32_t SEM_RD_GO = 0;
constexpr uint32_t SEM_RD_DONE = 1;
constexpr double ROW_WEIGHT = 0.16;
constexpr double ROW_WEIGHT_MIXED_FP32_STREAMS = 0.0;
constexpr size_t NUM_CIRCULAR_BUFFERS = 64;

static_assert(COEF_DEPTH >= 2, "writer-side expansion needs COEF_DEPTH >= 2");

// Kernel order in the descriptor and the address slots of the data-movement runtime args, read by
// override_runtime_arguments.
constexpr uint32_t READER_KERNEL = 0;
constexpr uint32_t WRITER_KERNEL = 1;
constexpr size_t RT_F = 0, RT_X = 1, RT_O = 2, RT_P = 3, RT_M = 4, RT_ADDRESSES = 5;

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

int64_t tensor_token_tiles(const ttnn::Shape& shape) {
    int64_t lead = 1;
    for (size_t i = 0; i + 2 < shape.rank(); ++i) {
        lead *= shape[i];
    }
    return lead * ceil_div(shape[-2], TILE_HW);
}

struct Assignment {
    CoreCoord core;
    int64_t start = 0;
    int64_t count = 0;
};

// _work_assignment(): the cores split_work_to_cores(..., row_wise=True) selects, in split order (group 1 then
// group 2, each row-wise); core i gets 1 unit plus its largest-remainder share of the other total - num_cores units,
// weighted by its logical grid row. Same floating-point operations in the same order as the Python.
std::pair<CoreRangeSet, std::vector<Assignment>> work_assignment(
    CoreCoord grid_size, int64_t total_units, double row_weight) {
    auto [num_cores, all_cores, core_group_1, core_group_2, units_1, units_2] =
        tt::tt_metal::split_work_to_cores(grid_size, static_cast<uint32_t>(total_units), true);
    std::vector<CoreCoord> cores;
    for (const auto& group : {core_group_1, core_group_2}) {
        auto c = tt::tt_metal::corerange_to_cores(group, std::nullopt, true);
        cores.insert(cores.end(), c.begin(), c.end());
    }
    const double rows = static_cast<double>(grid_size.y);
    std::vector<double> weights;
    weights.reserve(cores.size());
    for (const auto& c : cores) {
        weights.push_back(1.0 + row_weight * (static_cast<double>(c.y) - (rows - 1) / 2.0) / std::max(1.0, rows - 1));
    }
    const int64_t extra = total_units - static_cast<int64_t>(cores.size());
    double weight_sum = 0.0;
    for (double w : weights) {
        weight_sum += w;  // Python's sum(): left to right
    }
    std::vector<double> exact;
    std::vector<int64_t> counts;
    for (double w : weights) {
        exact.push_back(static_cast<double>(extra) * w / weight_sum);
        counts.push_back(static_cast<int64_t>(std::floor(exact.back())));
    }
    std::vector<size_t> by_remainder(cores.size());
    std::iota(by_remainder.begin(), by_remainder.end(), 0);
    // sorted(..., key=remainder, reverse=True): stable, so equal remainders keep their order
    std::stable_sort(by_remainder.begin(), by_remainder.end(), [&](size_t a, size_t b) {
        return (exact[a] - counts[a]) > (exact[b] - counts[b]);
    });
    const int64_t assigned = std::accumulate(counts.begin(), counts.end(), int64_t{0});
    for (int64_t k = 0; k < extra - assigned; ++k) {
        counts[by_remainder[k]] += 1;
    }
    std::vector<Assignment> assignment;
    int64_t start = 0;
    for (size_t i = 0; i < cores.size(); ++i) {
        assignment.push_back({cores[i], start, counts[i] + 1});
        start += counts[i] + 1;
    }
    TT_FATAL(start == total_units, "mhc_post: work split covers {} of {} units", start, total_units);
    return {all_cores, assignment};
}

int64_t max_segment_col_tiles(const std::vector<Assignment>& assignment, int64_t col_tiles_per_row) {
    int64_t longest = 0;
    for (const auto& a : assignment) {
        int64_t u = a.start, left = a.count;
        while (left) {
            const int64_t seg = std::min(col_tiles_per_row - u % col_tiles_per_row, left);
            longest = std::max(longest, seg);
            u += seg;
            left -= seg;
        }
    }
    return longest;
}

int64_t core_blocks(int64_t start, int64_t count, int64_t col_tiles_per_row, int64_t block_col_tiles) {
    int64_t blocks = 0, u = start, left = count;
    while (left) {
        const int64_t seg = std::min(col_tiles_per_row - u % col_tiles_per_row, left);
        blocks += ceil_div(seg, block_col_tiles);
        u += seg;
        left -= seg;
    }
    return blocks;
}

int64_t block_col_tiles_fit(
    int64_t n,
    int64_t sublayer_tile_bytes,
    int64_t residual_tile_bytes,
    int64_t coef_tile_bytes,
    int64_t num_coef_tiles,
    int64_t num_raw_tiles) {
    const int64_t coef_bytes = COEF_DEPTH * num_coef_tiles * coef_tile_bytes + num_raw_tiles * coef_tile_bytes;
    const int64_t per_col =
        DEPTH_IN * (sublayer_tile_bytes + n * residual_tile_bytes) + DEPTH_OUT * n * residual_tile_bytes;
    const int64_t num = L1_BUDGET_BYTES - coef_bytes;
    // Python floor division
    return num >= 0 ? num / per_col : -ceil_div(-num, per_col);
}

CBDescriptor make_cb(uint8_t index, DataType dtype, int64_t page_bytes, int64_t num_pages, const CoreRangeSet& cores) {
    CBDescriptor cb;
    cb.total_size = static_cast<uint32_t>(num_pages * page_bytes);
    cb.core_ranges = cores;
    cb.format_descriptors.push_back(CBFormatDescriptor{
        .buffer_index = index,
        .data_format = tt::tt_metal::datatype_to_dataformat_converter(dtype),
        .page_size = static_cast<uint32_t>(page_bytes)});
    return cb;
}

KernelDescriptor make_kernel(
    const char* file,
    const CoreRangeSet& cores,
    std::vector<uint32_t> ct,
    KernelDescriptor::RuntimeArgs rt,
    KernelDescriptor::ConfigDescriptor config) {
    KernelDescriptor k;
    k.kernel_source = std::string(KERNEL_DIR) + file;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = std::move(ct);
    k.runtime_args = std::move(rt);
    k.config = std::move(config);
    return k;
}

uint32_t page_size(const Tensor& t) { return t.buffer()->page_size(); }

uint32_t address(const Tensor& t) { return t.buffer()->address(); }

}  // namespace

ProgramDescriptor create_program_descriptor(
    const Tensor& input,
    const Tensor& residual,
    const Tensor& post,
    const Tensor& comb,
    const Tensor& output,
    const ComputeConfigDescriptor& compute_config,
    bool comb_transposed) {
    auto* device = input.device();

    const int64_t n = post.logical_shape()[-1];
    const int64_t col_tiles_per_row = input.logical_shape()[-1] / TILE_HW;  // Ct
    const int64_t token_tiles = tensor_token_tiles(input.padded_shape());
    const int64_t total_units = token_tiles * col_tiles_per_row;
    const int64_t post_tiles_per_row = ceil_div(n, TILE_HW);
    const int64_t comb_tiles_per_row = ceil_div(n * n, TILE_HW);
    const int64_t num_raw_tiles = post_tiles_per_row + comb_tiles_per_row;
    const int64_t coef_tiles_per_stream = ceil_div(n + 1, 2);  // two coefficient terms per tile (mhc_post_common.hpp)
    const int64_t num_coef_tiles = n * coef_tiles_per_stream;

    const int64_t sublayer_page = page_size(input);
    const int64_t residual_page = page_size(residual);
    const int64_t output_page = page_size(output);
    const int64_t coef_page = page_size(post);
    TT_FATAL(page_size(comb) == coef_page, "mhc_post: comb and post page sizes differ");
    TT_FATAL(output_page == residual_page, "mhc_post: output and residual page sizes differ");

    // ---- work split + block size ----
    const CoreCoord grid_size = device->compute_with_storage_grid_size();
    double row_weight = ROW_WEIGHT;
    if (residual.dtype() == DataType::FLOAT32 && input.dtype() == DataType::BFLOAT16) {
        row_weight = ROW_WEIGHT_MIXED_FP32_STREAMS;  // carve-out (measured regression)
    }
    auto [all_cores, assignment] = work_assignment(grid_size, total_units, row_weight);
    const int64_t fit = block_col_tiles_fit(n, sublayer_page, residual_page, coef_page, num_coef_tiles, num_raw_tiles);
    TT_FATAL(fit >= 1, "mhc_post: coefficient set + one column block does not fit L1_BUDGET_BYTES");
    int64_t block_col_tiles = std::min(fit, max_segment_col_tiles(assignment, col_tiles_per_row));
    int64_t max_units_per_core = 0;
    for (const auto& a : assignment) {
        max_units_per_core = std::max(max_units_per_core, a.count);
    }
    block_col_tiles = std::min(block_col_tiles, ceil_div(max_units_per_core, MIN_BLOCKS_PER_CORE));
    block_col_tiles = std::min(block_col_tiles, MAX_BLOCK_COL_TILES);
    block_col_tiles = std::max<int64_t>(1, block_col_tiles);

    // ---- circular buffers ----
    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(CB_SUBLAYER_TILES, input.dtype(), sublayer_page, DEPTH_IN * block_col_tiles, all_cores));
    desc.cbs.push_back(
        make_cb(CB_RESIDUAL_TILES, residual.dtype(), residual_page, DEPTH_IN * n * block_col_tiles, all_cores));
    desc.cbs.push_back(make_cb(CB_COEF_RAW, post.dtype(), coef_page, num_raw_tiles, all_cores));
    desc.cbs.push_back(make_cb(CB_COEF_BCAST, post.dtype(), coef_page, COEF_DEPTH * num_coef_tiles, all_cores));
    desc.cbs.push_back(
        make_cb(CB_OUTPUT_TILES, output.dtype(), output_page, DEPTH_OUT * n * block_col_tiles, all_cores));

    // ---- read help (see HELP_MIN_BLOCKS) ----
    int64_t max_blocks_per_core = 0;
    for (const auto& a : assignment) {
        max_blocks_per_core =
            std::max(max_blocks_per_core, core_blocks(a.start, a.count, col_tiles_per_row, block_col_tiles));
    }
    bool read_help = max_blocks_per_core >= HELP_MIN_BLOCKS;
    if (residual.dtype() == DataType::FLOAT32 && !HELP_FP32_STREAMS) {
        read_help = false;  // carve-out: compute-bound datapath
    }

    // ---- data movement: one source (mhc_post_dm.cpp), role 0 = reader (NCRISC), role 1 = writer (BRISC) ----
    auto dm_ct = [&](uint32_t role) {
        std::vector<uint32_t> ct = {
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(col_tiles_per_row),
            static_cast<uint32_t>(block_col_tiles),
            role,
            static_cast<uint32_t>(read_help),
            HELP_FROM_BLOCK,
            static_cast<uint32_t>(DEPTH_IN),
            CB_SUBLAYER_TILES,
            CB_RESIDUAL_TILES,
            CB_OUTPUT_TILES,
            CB_COEF_RAW,
            CB_COEF_BCAST,
            static_cast<uint32_t>(post_tiles_per_row),
            static_cast<uint32_t>(comb_tiles_per_row),
            static_cast<uint32_t>(sublayer_page),
            static_cast<uint32_t>(residual_page),
            static_cast<uint32_t>(coef_page),
            static_cast<uint32_t>(coef_tiles_per_stream),
            SEM_RD_GO,
            SEM_RD_DONE,
            TILE_HW,
        };
        for (const Tensor* t : {&input, &residual, &output, &post, &comb}) {
            const auto a = tt::tt_metal::TensorAccessorArgs(*t->buffer()).get_compile_time_args();
            ct.insert(ct.end(), a.begin(), a.end());
        }
        return ct;
    };

    // ---- compute ----
    std::vector<uint32_t> compute_ct = {
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(col_tiles_per_row),
        static_cast<uint32_t>(block_col_tiles),
        CB_SUBLAYER_TILES,
        CB_RESIDUAL_TILES,
        CB_COEF_BCAST,
        CB_OUTPUT_TILES,
        static_cast<uint32_t>(coef_tiles_per_stream),
    };

    KernelDescriptor::RuntimeArgs dm_rt;
    KernelDescriptor::RuntimeArgs compute_rt;
    const uint32_t f_addr = address(input), x_addr = address(residual);
    const uint32_t p_addr = address(post), m_addr = address(comb);
    const uint32_t o_addr = address(output);
    for (const auto& a : assignment) {
        dm_rt.emplace_back(
            a.core,
            std::vector<uint32_t>{
                f_addr,
                x_addr,
                o_addr,
                p_addr,
                m_addr,
                static_cast<uint32_t>(a.start),
                static_cast<uint32_t>(a.count)});  // smuggled-rta-ok: patched in override_runtime_arguments
        compute_rt.emplace_back(
            a.core, std::vector<uint32_t>{static_cast<uint32_t>(a.start), static_cast<uint32_t>(a.count)});
    }

    // UnpackToDestFp32 on every Float32 CB compute reads with copy_tile (derived from the CB format).
    ComputeConfigDescriptor compute_cfg;
    compute_cfg.math_fidelity = compute_config.math_fidelity;
    compute_cfg.fp32_dest_acc_en = compute_config.fp32_dest_acc_en;
    compute_cfg.math_approx_mode = compute_config.math_approx_mode;
    compute_cfg.dst_full_sync_en = DST_FULL_SYNC;
    compute_cfg.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (auto [index, dtype] :
         {std::pair{CB_SUBLAYER_TILES, input.dtype()},
          std::pair{CB_RESIDUAL_TILES, residual.dtype()},
          std::pair{CB_COEF_BCAST, post.dtype()}}) {
        if (dtype == DataType::FLOAT32) {
            compute_cfg.unpack_to_dest_mode[index] = UnpackToDestMode::UnpackToDestFp32;
        }
    }

    const auto noc_mode = read_help ? tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC : tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
    desc.kernels.push_back(make_kernel(
        "mhc_post_dm.cpp",
        all_cores,
        dm_ct(0),
        dm_rt,
        tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .noc_mode = noc_mode}));
    desc.kernels.push_back(make_kernel(
        "mhc_post_dm.cpp",
        all_cores,
        dm_ct(1),
        dm_rt,
        tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .noc_mode = noc_mode}));
    desc.kernels.push_back(make_kernel("mhc_post_compute.cpp", all_cores, compute_ct, compute_rt, compute_cfg));
    if (!comb_transposed) {
        // Only the coefficient expansion (DM kernels) reads comb's layout; the default adds no define (parity).
        for (uint32_t k : {READER_KERNEL, WRITER_KERNEL}) {
            desc.kernels[k].defines.emplace_back("MHC_POST_COMB_DIRECT", "1");
        }
    }
    for (uint32_t sem : {SEM_RD_GO, SEM_RD_DONE}) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = sem, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    }
    return desc;
}

ProgramDescriptor MhcPostProgramFactory::create_descriptor(
    const MhcPostParams& operation_attributes, const MhcPostInputs& tensor_args, Tensor& output) {
    return create_program_descriptor(
        tensor_args.input,
        tensor_args.residual,
        tensor_args.post,
        tensor_args.comb,
        output,
        operation_attributes.compute_config,
        operation_attributes.comb_transposed);
}

void MhcPostProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MhcPostParams& /*operation_attributes*/,
    const MhcPostInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Everything the builder reads besides the buffer ADDRESSES is in the program hash, so a cache hit can only move
    // addresses: patch the five address slots of both data-movement kernels.
    const uint32_t f_addr = address(tensor_args.input), x_addr = address(tensor_args.residual);
    const uint32_t p_addr = address(tensor_args.post), m_addr = address(tensor_args.comb);
    const uint32_t o_addr = address(output);
    for (uint32_t kernel : {READER_KERNEL, WRITER_KERNEL}) {
        auto& all = tt::tt_metal::GetRuntimeArgs(program, kernel);
        for (auto& col : all) {
            for (auto& args : col) {
                if (args.size() < RT_ADDRESSES) {
                    continue;
                }
                args[RT_F] = f_addr;
                args[RT_X] = x_addr;
                args[RT_O] = o_addr;
                args[RT_P] = p_addr;
                args[RT_M] = m_addr;
            }
        }
    }
}

}  // namespace ttnn::operations::bringup::mhc_post_ttnn
