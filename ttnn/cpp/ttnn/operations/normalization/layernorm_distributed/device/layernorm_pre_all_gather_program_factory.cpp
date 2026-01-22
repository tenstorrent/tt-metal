// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_pre_all_gather_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/math.hpp"

#include <optional>
#include <string>
#include <variant>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

inline uint16_t bfloat16(float float_num) {
    uint32_t uint32_data;
    TT_FATAL(
        sizeof float_num == sizeof uint32_data,
        "Float size ({}) must equal uint32 size ({})",
        sizeof float_num,
        sizeof uint32_data);

    uint32_data = *reinterpret_cast<uint32_t*>(&float_num);
    // just move upper 16 to lower 16 (truncate)
    uint32_data = (uint32_data >> 16);

    // store lower 16 as 16-bit uint
    return (uint16_t)uint32_data;
}
inline uint32_t pack_two_bfloat16_into_uint32(std::pair<uint16_t, uint16_t> two_bfloats) {
    // first -> lower 16
    // second -> upper 16
    return (uint32_t)two_bfloats.first | ((uint32_t)two_bfloats.second << 16);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

// =============================================================================
// LayerNormPreAllGatherProgramFactory - Normal (non-Welford, non-2D) operation
// =============================================================================

LayerNormPreAllGatherProgramFactory::cached_program_t LayerNormPreAllGatherProgramFactory::create(
    const LayerNormPreAllGatherParams& operation_attributes, const Tensor& tensor_args, Tensor& output) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& a = tensor_args;
    const bool is_rmsnorm = operation_attributes.norm_type == LayerNormDistributedType::RMSNORM;
    const auto& shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;

    IDevice* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();

    uint32_t num_tile_rows = NC * Ht;

    log_debug(tt::LogOp, "is_rmsnorm: {}", is_rmsnorm);
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "H: {}", H);
    log_debug(tt::LogOp, "num_tile_rows: {}", num_tile_rows);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "Ht: {}", Ht);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    uint32_t block_size = 1;
    uint32_t writer_block_size = 1;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    log_debug(tt::LogOp, "in_data_format: {}", in_data_format);
    log_debug(tt::LogOp, "out_data_format: {}", out_data_format);

    auto a_addr = a.buffer()->address();
    auto dst_addr = output.buffer()->address();

    const uint32_t double_buffer_constant = 2;
    const uint32_t in0_tiles = Wt * double_buffer_constant;
    const uint32_t in1_tiles = 1;  // reduce scalar

    const uint32_t intermed0_tiles = Wt * double_buffer_constant;  // x^2
    uint32_t out0_tiles = 1;
    if (!is_rmsnorm) {
        out0_tiles = 2;
    }

    TT_FATAL(
        W <= TILE_WIDTH * in0_tiles,
        "W ({}) exceeds the maximum supported size of tile buffer ({} * {}, kernel limitation right now).",
        W,
        TILE_WIDTH,
        in0_tiles);
    TT_FATAL(
        in0_tiles % block_size == 0,
        "Size of buffer ({}) must be divisible by the size of block ({}) used by the reader and compute kernel.",
        in0_tiles,
        block_size);
    TT_FATAL(
        intermed0_tiles % block_size == 0,
        "Size of buffer ({}) must be divisible by the size of block ({}) used by the reader and compute kernel.",
        intermed0_tiles,
        block_size);

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(tt::LogOp, "grid_size: {}", grid_size);
    log_debug(tt::LogOp, "core_group_1: {}", core_group_1.str());
    log_debug(tt::LogOp, "num_tile_rows_per_core_group_1: {}", num_tile_rows_per_core_group_1);
    log_debug(tt::LogOp, "core_group_2: {}", core_group_2.str());
    log_debug(tt::LogOp, "num_tile_rows_per_core_group_2: {}", num_tile_rows_per_core_group_2);

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)block_size,
    };
    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)writer_block_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    std::map<std::string, std::string> compute_defines;

    auto reader_kernels_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernels_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_blocked.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Get program config
    ttnn::prim::LayerNormDefaultProgramConfig program_config;
    if (std::holds_alternative<ttnn::prim::LayerNormDefaultProgramConfig>(operation_attributes.program_config)) {
        program_config = std::get<ttnn::prim::LayerNormDefaultProgramConfig>(operation_attributes.program_config);
    }

    bool float32_reduction = fp32_dest_acc_en && !program_config.legacy_reduction;
    std::vector<uint32_t> compute_args = {Wt, block_size, float32_reduction ? 1u : 0u};

    const auto* compute_kernel_file =
        is_rmsnorm ? "ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/"
                     "rmsnorm_pre_allgather.cpp"
                   : "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
                     "layernorm_pre_allgather.cpp";
    auto compute_config = tt::tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_args,
        .defines = compute_defines};
    auto compute_kernels_id = tt::tt_metal::CreateKernel(program, compute_kernel_file, all_cores, compute_config);

    // Create circular buffers
    // c_in0 -> a
    auto cb_src0_config =
        tt::tt_metal::CircularBufferConfig(in0_tiles * in_single_tile_size, {{tt::CBIndex::c_0, in_data_format}})
            .set_page_size(tt::CBIndex::c_0, in_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);
    // c_in1 -> reduce scalar
    auto cb_reduce_config =
        tt::tt_metal::CircularBufferConfig(in1_tiles * bfloat16_tile_size, {{tt::CBIndex::c_1, cb_data_format}})
            .set_page_size(tt::CBIndex::c_1, bfloat16_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_reduce_config);

    // LN and RMS shared intermediates
    // c_intermed0 -> x^2
    auto cb_intermed0_config =
        tt::tt_metal::CircularBufferConfig(intermed0_tiles * single_tile_size, {{tt::CBIndex::c_6, cb_data_format}})
            .set_page_size(tt::CBIndex::c_6, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed0_config);

    auto cb_out0_config =
        tt::tt_metal::CircularBufferConfig(out0_tiles * out_single_tile_size, {{tt::CBIndex::c_14, out_data_format}})
            .set_page_size(tt::CBIndex::c_14, out_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out0_config);

    // Log all circular buffers
    for (const auto& cb : program.circular_buffers()) {
        for ([[maybe_unused]] const auto index : cb->buffer_indices()) {
            log_debug(tt::LogOp, "cb_id {}", index);
            log_debug(tt::LogOp, "page_size: {}", cb->page_size(index));
            log_debug(tt::LogOp, "num_pages: {}", cb->num_pages(index));
            log_debug(tt::LogOp, "data_format: {}", cb->data_format(index));
        }
    }

    uint32_t curr_row = 0;
    float winv = 1.0f;
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t in_tile_offset = curr_row * Wt;
        uint32_t out_tile_offset = curr_row * out0_tiles;

        tt::tt_metal::SetRuntimeArgs(
            program, reader_kernels_id, core, {a_addr, num_tile_rows_per_core, Wt, in_tile_offset, packed_winv_value});
        tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id, core, {num_tile_rows_per_core});
        tt::tt_metal::SetRuntimeArgs(
            program, writer_kernels_id, core, {dst_addr, num_tile_rows_per_core * out0_tiles, out_tile_offset});
        curr_row += num_tile_rows_per_core;
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernels_id,
         .writer_kernel_id = writer_kernels_id,
         .num_cores = num_cores,
         .grid_size = grid_size}};
}

void LayerNormPreAllGatherProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const LayerNormPreAllGatherParams& /*operation_attributes*/,
    const Tensor& tensor_args,
    Tensor& output) {
    auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    const auto input_addr = tensor_args.buffer()->address();
    const auto output_addr = output.buffer()->address();

    auto& reader_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_vars.reader_kernel_id);
    auto& writer_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_vars.writer_kernel_id);

    for (uint32_t i = 0; i < shared_vars.num_cores; ++i) {
        const CoreCoord core = {i % shared_vars.grid_size.x, i / shared_vars.grid_size.x};

        {
            auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);
            reader_args[0] = input_addr;
        }

        {
            auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
            writer_args[0] = output_addr;
        }
    }
}

// =============================================================================
// LayerNormPreAllGather2DProgramFactory - 2D core grid operation
// =============================================================================

LayerNormPreAllGather2DProgramFactory::cached_program_t LayerNormPreAllGather2DProgramFactory::create(
    const LayerNormPreAllGatherParams& operation_attributes, const Tensor& tensor_args, Tensor& output) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& a = tensor_args;
    const auto& shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;

    uint32_t num_tile_rows = NC * Ht;

    IDevice* device = a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    uint32_t block_size = 1;
    uint32_t writer_block_size = 1;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    auto a_addr = a.buffer()->address();
    auto dst_addr = output.buffer()->address();

    const uint32_t double_buffer_constant = 2;
    const uint32_t in0_tiles = Wt * double_buffer_constant;
    const uint32_t in1_tiles = 1;  // reduce scalar

    const uint32_t intermed0_tiles = Wt * double_buffer_constant;  // x^2
    uint32_t out0_tiles = 1;

    TT_FATAL(
        W <= TILE_WIDTH * in0_tiles,
        "W ({}) exceeds the maximum supported size of tile buffer ({} * {}, kernel limitation right now).",
        W,
        TILE_WIDTH,
        in0_tiles);
    TT_FATAL(
        in0_tiles % block_size == 0,
        "Size of buffer ({}) must be divisible by the size of block ({}) used by the reader and compute kernel.",
        in0_tiles,
        block_size);
    TT_FATAL(
        intermed0_tiles % block_size == 0,
        "Size of buffer ({}) must be divisible by the size of block ({}) used by the reader and compute kernel.",
        intermed0_tiles,
        block_size);

    auto grid_size = device->compute_with_storage_grid_size();

    uint32_t max_cores_y = grid_size.y;
    uint32_t cores_x = std::min(max_cores_y, num_tile_rows);
    while (num_tile_rows % cores_x != 0 && cores_x > 1) {
        cores_x--;
    }
    uint32_t tiles_per_core_x = num_tile_rows / cores_x;
    uint32_t cores_y = std::min(max_cores_y, Wt);
    while (Wt % cores_y != 0 && cores_y > 1) {
        cores_y--;
    }
    uint32_t tiles_per_core_y = Wt / cores_y;

    CoreRange all_cores_range({0, 0}, {cores_x - 1, cores_y - 1});
    CoreRangeSet all_cores = CoreRangeSet(std::vector{all_cores_range});
    auto cores = corerange_to_cores(all_cores, std::nullopt);

    std::vector<CoreRange> merge_core_ranges_vec;
    for (uint32_t x = 0; x < cores_x; ++x) {
        CoreCoord merge_core = {x, 0};
        merge_core_ranges_vec.push_back(CoreRange(merge_core, merge_core));
    }
    CoreRangeSet merge_cores(merge_core_ranges_vec);

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    auto reducer_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)block_size,
        (std::uint32_t)reducer_semaphore_id,
        (std::uint32_t)cores_y,
    };
    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)writer_block_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    std::map<std::string, std::string> compute_defines;

    auto reader_kernels_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_layernorm_preallgather_2d.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernels_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_blocked.cpp",
        merge_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Get program config
    ttnn::prim::LayerNormDefaultProgramConfig program_config;
    if (std::holds_alternative<ttnn::prim::LayerNormDefaultProgramConfig>(operation_attributes.program_config)) {
        program_config = std::get<ttnn::prim::LayerNormDefaultProgramConfig>(operation_attributes.program_config);
    }

    bool float32_reduction = fp32_dest_acc_en && !program_config.legacy_reduction;
    std::vector<uint32_t> compute_args = {
        tiles_per_core_x, tiles_per_core_y, block_size, cores_y, float32_reduction ? 1u : 0u};

    auto compute_kernels_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
        "layernorm_pre_allgather_2d.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_args,
            .defines = compute_defines});

    // Create circular buffers
    // c_in0 -> a
    auto cb_src0_config =
        tt::tt_metal::CircularBufferConfig(in0_tiles * in_single_tile_size, {{tt::CBIndex::c_0, in_data_format}})
            .set_page_size(tt::CBIndex::c_0, in_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);
    // c_in1 -> reduce scalar
    auto cb_reduce_config =
        tt::tt_metal::CircularBufferConfig(in1_tiles * bfloat16_tile_size, {{tt::CBIndex::c_1, cb_data_format}})
            .set_page_size(tt::CBIndex::c_1, bfloat16_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_reduce_config);

    // LN and RMS shared intermediates
    // c_intermed0 -> x^2
    auto cb_intermed0_config =
        tt::tt_metal::CircularBufferConfig(intermed0_tiles * single_tile_size, {{tt::CBIndex::c_6, cb_data_format}})
            .set_page_size(tt::CBIndex::c_6, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed0_config);

    auto cb_intermed1_config =
        tt::tt_metal::CircularBufferConfig(tiles_per_core_y * single_tile_size, {{tt::CBIndex::c_15, cb_data_format}})
            .set_page_size(tt::CBIndex::c_15, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed1_config);

    auto cb_out0_config =
        tt::tt_metal::CircularBufferConfig(out0_tiles * single_tile_size, {{tt::CBIndex::c_16, cb_data_format}})
            .set_page_size(tt::CBIndex::c_16, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out0_config);
    auto cb_zero_config =
        tt::tt_metal::CircularBufferConfig(out0_tiles * single_tile_size, {{tt::CBIndex::c_13, cb_data_format}})
            .set_page_size(tt::CBIndex::c_13, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_zero_config);

    auto cb_out_final_config =
        tt::tt_metal::CircularBufferConfig(out0_tiles * out_single_tile_size, {{tt::CBIndex::c_14, out_data_format}})
            .set_page_size(tt::CBIndex::c_14, out_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, merge_cores, cb_out_final_config);

    float winv = 1.0f;
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    for (uint32_t x = 0; x < cores_x; ++x) {
        for (uint32_t y = 0; y < cores_y; ++y) {
            CoreCoord core = {x, y};
            bool is_merge_core = y == 0;
            const auto merge_core = device->worker_core_from_logical_core({x, 0});

            uint32_t num_tile_rows_per_core = tiles_per_core_x;

            uint32_t in_tile_offset = (x * Wt) + (y * tiles_per_core_y);
            uint32_t out_tile_offset = x * out0_tiles;

            tt::tt_metal::SetRuntimeArgs(
                program,
                reader_kernels_id,
                core,
                {a_addr,
                 tiles_per_core_x,
                 tiles_per_core_y,
                 in_tile_offset,
                 is_merge_core,
                 merge_core.x,
                 merge_core.y,
                 y,
                 packed_winv_value});
            tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id, core, {is_merge_core});
            if (is_merge_core) {
                tt::tt_metal::SetRuntimeArgs(
                    program, writer_kernels_id, core, {dst_addr, num_tile_rows_per_core * out0_tiles, out_tile_offset});
            }
        }
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernels_id, .writer_kernel_id = writer_kernels_id, .cores = cores}};
}

void LayerNormPreAllGather2DProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const LayerNormPreAllGatherParams& /*operation_attributes*/,
    const Tensor& tensor_args,
    Tensor& output) {
    auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    const auto input_addr = tensor_args.buffer()->address();
    const auto output_addr = output.buffer()->address();

    auto& reader_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_vars.reader_kernel_id);
    auto& writer_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_vars.writer_kernel_id);

    for (const auto& core : shared_vars.cores) {
        {
            auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);
            reader_args[0] = input_addr;
        }

        if (core.y == 0) {
            auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
            writer_args[0] = output_addr;
        }
    }
}

}  // namespace ttnn::prim
