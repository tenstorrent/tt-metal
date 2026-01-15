// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_post_all_gather_program_factory.hpp"

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
// LayerNormPostAllGatherProgramFactory - Normal (non-Welford) operation
// =============================================================================

LayerNormPostAllGatherProgramFactory::cached_program_t LayerNormPostAllGatherProgramFactory::create(
    const LayerNormPostAllGatherParams& operation_attributes,
    const LayerNormPostAllGatherInputs& tensor_args,
    Tensor& output) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using tt::tt_metal::CBHandle;
    using tt::tt_metal::CircularBuffer;
    using tt::tt_metal::CircularBufferConfig;

    const auto& a = tensor_args.input;
    const auto& stats = tensor_args.stats;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    const bool is_rmsnorm = operation_attributes.norm_type == LayerNormDistributedType::RMSNORM;
    const auto& shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;
    const uint32_t stats_tiles_cols = stats.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t tile_cols_per_device = is_rmsnorm ? 1 : 2;
    const uint32_t num_devices = stats_tiles_cols / tile_cols_per_device;
    TT_FATAL(num_devices > 0, "Number of devices must be greater than 0");
    TT_FATAL(
        num_devices * tile_cols_per_device == stats_tiles_cols, "Number of devices must divide number of stats tiles");

    uint32_t num_tile_rows = NC * Ht;

    log_debug(tt::LogOp, "is_rmsnorm: {}", is_rmsnorm);
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "H: {}", H);
    log_debug(tt::LogOp, "num_tile_rows: {}", num_tile_rows);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "Ht: {}", Ht);
    log_debug(tt::LogOp, "stats_tiles_cols: {}", stats_tiles_cols);
    log_debug(tt::LogOp, "num_devices: {}", num_devices);

    IDevice* device = a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat stats_data_format = tt::tt_metal::datatype_to_dataformat_converter(stats.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat gamma_cb_data_format = gamma.has_value()
                                              ? tt::tt_metal::datatype_to_dataformat_converter(gamma.value().dtype())
                                              : tt::DataFormat::Float16_b;
    tt::DataFormat beta_cb_data_format = beta.has_value()
                                             ? tt::tt_metal::datatype_to_dataformat_converter(beta.value().dtype())
                                             : tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t stats_single_tile_size = tt::tile_size(stats_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    uint32_t gamma_single_tile_size = tt::tile_size(gamma_cb_data_format);
    uint32_t beta_single_tile_size = tt::tile_size(beta_cb_data_format);

    log_debug(tt::LogOp, "in_data_format: {}", in_data_format);
    log_debug(tt::LogOp, "out_data_format: {}", out_data_format);
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "gamma_cb_data_format: {}", gamma_cb_data_format);
    log_debug(tt::LogOp, "beta_cb_data_format: {}", beta_cb_data_format);
    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "math_approx_mode: {}", math_approx_mode);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);

    auto a_addr = a.buffer()->address();
    auto stats_addr = stats.buffer()->address();
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    auto dst_addr = output.buffer()->address();

    [[maybe_unused]] uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().physical_volume() / TILE_HW : 0;
    [[maybe_unused]] uint32_t num_beta_tiles = beta.has_value() ? beta.value().physical_volume() / TILE_HW : 0;

    // For bert, tensor is packed as RM with width 32
    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        num_gamma_tiles = gamma.has_value() ? gamma.value().physical_volume() / TILE_WIDTH : 0;
    }
    if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        num_beta_tiles = beta.has_value() ? beta.value().physical_volume() / TILE_WIDTH : 0;
    }

    log_debug(tt::LogOp, "num_gamma_tiles: {}", num_gamma_tiles);
    log_debug(tt::LogOp, "num_beta_tiles: {}", num_beta_tiles);

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t max_cores_y = grid_size.y;
    uint32_t tiles_per_core_y = Wt;

    // Declare all variables that will be used later
    uint32_t cores_x = 0;
    uint32_t cores_y = 0;
    uint32_t tiles_per_core_x = 0;
    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_tile_rows_per_core_group_1 = 0;
    uint32_t num_tile_rows_per_core_group_2 = 0;

    // Determine if we should use 2D kernel layout
    bool use_2d_kernel = false;
    if (operation_attributes.use_2d_core_grid.has_value()) {
        use_2d_kernel = *operation_attributes.use_2d_core_grid;
    }

    if (use_2d_kernel) {
        // 2D kernel layout: distribute work across cores in a 2D grid
        cores_x = std::min(max_cores_y, num_tile_rows);
        while (num_tile_rows % cores_x != 0 && cores_x > 1) {
            cores_x--;
        }
        tiles_per_core_x = num_tile_rows / cores_x;
        cores_y = std::min(max_cores_y, Wt);
        while (Wt % cores_y != 0 && cores_y > 1) {
            cores_y--;
        }
        tiles_per_core_y = Wt / cores_y;

        CoreRange all_cores_range({0, 0}, {cores_x - 1, cores_y - 1});
        all_cores = CoreRangeSet(std::vector{all_cores_range});
    } else {
        auto
            [num_cores_result,
             all_cores_result,
             core_group_1_result,
             core_group_2_result,
             num_tile_rows_per_core_group_1_result,
             num_tile_rows_per_core_group_2_result] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

        num_cores = num_cores_result;
        all_cores = all_cores_result;
        core_group_1 = core_group_1_result;
        core_group_2 = core_group_2_result;
        num_tile_rows_per_core_group_1 = num_tile_rows_per_core_group_1_result;
        num_tile_rows_per_core_group_2 = num_tile_rows_per_core_group_2_result;

        log_debug(tt::LogOp, "num_cores: {}", num_cores);
        log_debug(tt::LogOp, "grid_size: {}", grid_size);
        log_debug(tt::LogOp, "core_group_1: {}", core_group_1.str());
        log_debug(tt::LogOp, "num_tile_rows_per_core_group_1: {}", num_tile_rows_per_core_group_1);
        log_debug(tt::LogOp, "core_group_2: {}", core_group_2.str());
        log_debug(tt::LogOp, "num_tile_rows_per_core_group_2: {}", num_tile_rows_per_core_group_2);
    }
    uint32_t block_size = fp32_dest_acc_en ? tt::tt_metal::find_max_divisor(tiles_per_core_y, 4)
                                           : tt::tt_metal::find_max_divisor(tiles_per_core_y, 8);
    uint32_t cb_length = tiles_per_core_y;

    const double available_L1 =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    if ((!(operation_attributes.use_2d_core_grid.has_value() && *operation_attributes.use_2d_core_grid)) &&
        (cb_length * in_single_tile_size > available_L1 * 0.95)) {
        cb_length = ((available_L1 / in_single_tile_size) * 0.95) / 7;
    }
    const uint32_t in0_tiles = cb_length;
    const uint32_t in1_tiles = stats_tiles_cols;
    const uint32_t in2_tiles = cb_length;
    const uint32_t in3_tiles = cb_length;
    const uint32_t in4_tiles = 1;  // epsilon
    const uint32_t in5_tiles = 1;  // reduce scalar

    const uint32_t intermed0_tiles = tile_cols_per_device;
    const uint32_t intermed1_tiles = 1;
    const uint32_t intermed2_tiles = 1;
    const uint32_t intermed3_tiles = 1;
    const uint32_t intermed4_tiles = 1;
    const uint32_t intermed5_tiles = cb_length;
    const uint32_t intermed6_tiles = cb_length;
    const uint32_t intermed7_tiles = cb_length;
    const uint32_t out0_tiles = cb_length;

    auto cores = corerange_to_cores(all_cores, std::nullopt);

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)block_size,
        (std::uint32_t)stats_tiles_cols,
    };

    uint32_t gamma_stick_size = 0;
    uint32_t gamma_is_row_major = 0;
    uint32_t beta_is_row_major = 0;
    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        gamma_stick_size = gamma.value().padded_shape()[-1] * gamma.value().element_size();
        bool gamma_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(gamma_stick_size);
        TT_FATAL(gamma_stick_size_is_power_of_two, "Only power of 2 gammas are supported");
        gamma_is_row_major = 1;
    } else if (gamma.has_value() and gamma.value().layout() == Layout::TILE) {
        gamma_stick_size = gamma.value().element_size() * 1024;  // size of tile in bytes bf16
    }
    uint32_t beta_stick_size = 0;
    if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        beta_stick_size = beta.value().padded_shape()[-1] * beta.value().element_size();
        bool beta_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(beta_stick_size);
        TT_FATAL(beta_stick_size_is_power_of_two, "Only power of 2 betas are supported");
        beta_is_row_major = 1;
    } else if (beta.has_value() and beta.value().layout() == Layout::TILE) {
        beta_stick_size = beta.value().element_size() * 1024;  // size of tile in bytes bf16
    }
    reader_compile_time_args.push_back((std::uint32_t)gamma_stick_size);
    reader_compile_time_args.push_back((std::uint32_t)beta_stick_size);
    reader_compile_time_args.push_back((std::uint32_t)gamma_is_row_major);
    reader_compile_time_args.push_back((std::uint32_t)beta_is_row_major);
    reader_compile_time_args.push_back((std::uint32_t)cb_length);
    reader_compile_time_args.push_back((std::uint32_t)tiles_per_core_y);

    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(stats.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(gamma.has_value() ? gamma.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(beta.has_value() ? beta.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)block_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> compute_defines;
    if (gamma.has_value()) {
        reader_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_defines["FUSE_BETA"] = "1";
    }

    auto reader_kernels_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_unary_interleaved_ln_rm_gb_post_allgather.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

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
    std::vector<uint32_t> compute_args = {
        tiles_per_core_y,
        block_size,
        stats_tiles_cols,
        gamma.has_value(),
        beta.has_value(),
        fp32_dest_acc_en,
        float32_reduction ? 1 : 0,
        program_config.legacy_rsqrt ? 1 : 0,
        cb_length};

    const auto* compute_kernel_file =
        is_rmsnorm ? "ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/"
                     "rmsnorm_post_allgather.cpp"
                   : "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
                     "layernorm_post_allgather.cpp";
    auto compute_config = tt::tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_args,
        .defines = compute_defines};
    auto compute_kernels_id = tt::tt_metal::CreateKernel(program, compute_kernel_file, all_cores, compute_config);

    // Create circular buffers
    // c_in0 -> a
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(in0_tiles * in_single_tile_size, {{tt::CBIndex::c_0, in_data_format}})
            .set_page_size(tt::CBIndex::c_0, in_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);
    // c_in1 -> stats
    CircularBufferConfig cb_stats_config =
        CircularBufferConfig(in1_tiles * stats_single_tile_size, {{tt::CBIndex::c_1, stats_data_format}})
            .set_page_size(tt::CBIndex::c_1, stats_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_stats_config);
    // c_in2 -> gamma
    if (gamma.has_value()) {
        CircularBufferConfig cb_gamma_config =
            CircularBufferConfig(in2_tiles * gamma_single_tile_size, {{tt::CBIndex::c_2, gamma_cb_data_format}})
                .set_page_size(tt::CBIndex::c_2, gamma_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_gamma_config);
    }
    // c_in3 -> beta
    if (beta.has_value()) {
        CircularBufferConfig cb_beta_config =
            CircularBufferConfig(in3_tiles * beta_single_tile_size, {{tt::CBIndex::c_3, beta_cb_data_format}})
                .set_page_size(tt::CBIndex::c_3, beta_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_beta_config);
    }
    // c_in4 -> epsilon
    CircularBufferConfig cb_eps_config =
        CircularBufferConfig(in4_tiles * bfloat16_tile_size, {{tt::CBIndex::c_4, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_4, bfloat16_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_eps_config);
    // c_in5 -> reduce scalar
    CircularBufferConfig cb_reduce_config =
        CircularBufferConfig(in5_tiles * bfloat16_tile_size, {{tt::CBIndex::c_5, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_5, bfloat16_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_reduce_config);

    // LN and RMS shared intermediates
    // c_intermed0 -> [mean(x**2), mean(x)]
    CircularBufferConfig cb_intermed0_config =
        CircularBufferConfig(intermed0_tiles * single_tile_size, {{tt::CBIndex::c_6, cb_data_format}})
            .set_page_size(tt::CBIndex::c_6, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed0_config);
    // c_intermed2 -> var = mean(x**2) - mean(x)**2
    CircularBufferConfig cb_intermed2_config =
        CircularBufferConfig(intermed2_tiles * single_tile_size, {{tt::CBIndex::c_8, cb_data_format}})
            .set_page_size(tt::CBIndex::c_8, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed2_config);
    // c_intermed3 -> var + epsilon
    CircularBufferConfig cb_intermed3_config =
        CircularBufferConfig(intermed3_tiles * single_tile_size, {{tt::CBIndex::c_9, cb_data_format}})
            .set_page_size(tt::CBIndex::c_9, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed3_config);
    // c_intermed4 -> 1/sqrt(var + epsilon)
    CircularBufferConfig cb_intermed4_config =
        CircularBufferConfig(intermed4_tiles * single_tile_size, {{tt::CBIndex::c_10, cb_data_format}})
            .set_page_size(tt::CBIndex::c_10, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed4_config);
    // c_intermed6 -> (x - mean(x)) * 1/sqrt(var + epsilon)
    CircularBufferConfig cb_intermed6_config =
        CircularBufferConfig(intermed6_tiles * single_tile_size, {{tt::CBIndex::c_12, cb_data_format}})
            .set_page_size(tt::CBIndex::c_12, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed6_config);

    // LN-specific intermediates
    if (!is_rmsnorm) {
        // c_intermed1 -> mean(x)**2
        CircularBufferConfig cb_intermed1_config =
            CircularBufferConfig(intermed1_tiles * single_tile_size, {{tt::CBIndex::c_7, cb_data_format}})
                .set_page_size(tt::CBIndex::c_7, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed1_config);
        // c_intermed5 -> x - mean(x)
        CircularBufferConfig cb_intermed5_config =
            CircularBufferConfig(intermed5_tiles * single_tile_size, {{tt::CBIndex::c_11, cb_data_format}})
                .set_page_size(tt::CBIndex::c_11, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed5_config);
        if (beta.has_value()) {
            // c_intermed7 -> (x - mean(x)) * 1/sqrt(var + epsilon) * gamma
            CircularBufferConfig cb_intermed7_config =
                CircularBufferConfig(intermed7_tiles * single_tile_size, {{tt::CBIndex::c_13, cb_data_format}})
                    .set_page_size(tt::CBIndex::c_13, single_tile_size);
            tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed7_config);
        }
    }

    CircularBufferConfig cb_out0_config =
        CircularBufferConfig(out0_tiles * out_single_tile_size, {{tt::CBIndex::c_14, out_data_format}})
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
    float winv = 1.0f / (W * num_devices);  // bcast-w scaler
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    union {
        float f;
        uint32_t u;
    } e{};
    e.f = operation_attributes.eps;  // epsilon

    // Set runtime arguments based on kernel layout type
    if (use_2d_kernel) {
        for (uint32_t x = 0; x < cores_x; ++x) {
            for (uint32_t y = 0; y < cores_y; ++y) {
                CoreCoord core = {x, y};

                uint32_t tile_offset = (x * Wt) + (y * tiles_per_core_y);
                uint32_t stats_offset = x * stats_tiles_cols;

                log_debug(
                    tt::LogOp,
                    "Setting reader runtime args for core: {}, tile_offset: {}, tiles_per_core_y: {}",
                    core.x,
                    tile_offset,
                    tiles_per_core_y);
                tt::tt_metal::SetRuntimeArgs(
                    program,
                    reader_kernels_id,
                    core,
                    {a_addr,
                     tiles_per_core_x,
                     tiles_per_core_y,
                     tile_offset,
                     stats_offset,
                     packed_winv_value,
                     e.u,
                     gamma_dram_addr,
                     beta_dram_addr,
                     stats_addr,
                     y * tiles_per_core_y});
                tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id, core, {tiles_per_core_x});
                tt::tt_metal::SetRuntimeArgs(
                    program, writer_kernels_id, core, {dst_addr, tiles_per_core_x * tiles_per_core_y, tile_offset});
            }
        }
    } else {
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

            uint32_t tile_offset = curr_row * Wt;
            uint32_t stats_offset = curr_row * stats_tiles_cols;
            uint32_t y_offset = 0;

            tt::tt_metal::SetRuntimeArgs(
                program,
                reader_kernels_id,
                core,
                {a_addr,
                 num_tile_rows_per_core,
                 Wt,
                 tile_offset,
                 stats_offset,
                 packed_winv_value,
                 e.u,
                 gamma_dram_addr,
                 beta_dram_addr,
                 stats_addr,
                 y_offset});
            tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id, core, {num_tile_rows_per_core});
            tt::tt_metal::SetRuntimeArgs(
                program, writer_kernels_id, core, {dst_addr, num_tile_rows_per_core * Wt, tile_offset});
            curr_row += num_tile_rows_per_core;
        }
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernels_id, .writer_kernel_id = writer_kernels_id, .cores = cores}};
}

void LayerNormPostAllGatherProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const LayerNormPostAllGatherParams& /*operation_attributes*/,
    const LayerNormPostAllGatherInputs& tensor_args,
    Tensor& output) {
    auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    const auto input_addr = tensor_args.input.buffer()->address();
    const auto stats_addr = tensor_args.stats.buffer()->address();
    const bool has_gamma = tensor_args.gamma.has_value();
    const bool has_beta = tensor_args.beta.has_value();
    const auto gamma_addr = has_gamma ? tensor_args.gamma.value().buffer()->address() : 0;
    const auto beta_addr = has_beta ? tensor_args.beta.value().buffer()->address() : 0;
    const auto output_addr = output.buffer()->address();

    auto& reader_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_vars.reader_kernel_id);
    auto& writer_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_vars.writer_kernel_id);

    for (const auto& core : shared_vars.cores) {
        {
            auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);

            reader_args[0] = input_addr;
            reader_args[9] = stats_addr;
            if (has_gamma) {
                reader_args[7] = gamma_addr;
            }
            if (has_beta) {
                reader_args[8] = beta_addr;
            }
        }

        {
            auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
            writer_args[0] = output_addr;
        }
    }
}

}  // namespace ttnn::prim
