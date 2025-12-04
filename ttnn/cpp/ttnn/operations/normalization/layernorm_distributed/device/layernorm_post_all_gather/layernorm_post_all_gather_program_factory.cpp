// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_post_all_gather_program_factory.hpp"

#include "ttnn/operations/math.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;

namespace ttnn::operations::normalization::layernorm_post_all_gather::program {

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

LayerNormPostAllGatherProgramFactory::cached_program_t LayerNormPostAllGatherProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
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

    // Kernels are configured to support BFLOAT8_B, but bad pcc so we need mixed precision support in compute

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;
    const uint32_t stats_tiles_cols = stats.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t tile_cols_per_device = is_rmsnorm ? 1 : 2;
    const uint32_t num_devices = stats_tiles_cols / tile_cols_per_device;
    TT_FATAL(num_devices > 0, "Number of devices must be greater than 0");
    TT_FATAL(
        num_devices * tile_cols_per_device == stats_tiles_cols, "Number of devices must divide number of stats tiles");

    uint32_t num_tile_rows = NC * Ht;

    log_debug(
        tt::LogOp,
        "LayerNormPostAllGatherProgramFactory: is_rmsnorm={}, shape=[{},{}], num_tile_rows={}, num_devices={}",
        is_rmsnorm,
        W,
        H,
        num_tile_rows,
        num_devices);

    ////////////////////////////////////////////////////////////////////////////
    //                       Device Setup
    //////////////////////////////////////////////////////////////////////////
    IDevice* device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                Circular Buffer Data Format Setup
    //////////////////////////////////////////////////////////////////////////
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    uint32_t block_size =
        fp32_dest_acc_en ? tt::tt_metal::find_max_divisor(Wt, 4) : tt::tt_metal::find_max_divisor(Wt, 8);

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

    const bool gamma_is_row_major = gamma.has_value() && gamma.value().layout() == Layout::ROW_MAJOR;
    const bool beta_is_row_major = beta.has_value() && beta.value().layout() == Layout::ROW_MAJOR;

    auto a_addr = a.buffer()->address();
    auto stats_addr = stats.buffer()->address();
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    auto dst_addr = output.buffer()->address();

    [[maybe_unused]] uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().physical_volume() / TILE_HW : 0;
    [[maybe_unused]] uint32_t num_beta_tiles = beta.has_value() ? beta.value().physical_volume() / TILE_HW : 0;

    // For bert, tensor is packed as RM with width 32
    if (gamma_is_row_major) {
        num_gamma_tiles = gamma.value().physical_volume() / TILE_WIDTH;
    }
    if (beta_is_row_major) {
        num_beta_tiles = beta.value().physical_volume() / TILE_WIDTH;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    //////////////////////////////////////////////////////////////////////////
    /*
    in0_cb: a
    in1_cb: stats
    in2_cb: gamma
    in3_cb: beta
    in4_cb: epsilon
    in5_cb: 1/row_size (reduction scalar)

    intermediate CBs are packed such that in layernorm, first tile is for x**2 stats, second tile is for x stats
    in RMSNorm, only first tile has valid data.

    intermed0_cb: [mean(x**2), mean(x)] # reduce with reduce_scalar
    intermed1_cb: mean(x)**2 # LN only
    intermed2_cb: var = mean(x**2) - mean(x)**2 # for RMSNorm, this is just mean(x**2)
    intermed3_cb: var + epsilon # RMSNorm takes mean(x**2) instead of var
    intermed4_cb: 1/sqrt(var + epsilon)
    intermed5_cb: x - mean(x) # LN only
    intermed6_cb: (x - mean(x)) * 1/sqrt(var + epsilon) # RMSNorm takes x instead of (x - mean(x))
    intermed7_cb: (x - mean(x)) * 1/sqrt(var + epsilon) * gamma
    out0_cb: (x - mean(x)) * 1/sqrt(var + epsilon) * gamma + beta # RMSNorm doesn't include beta

    */

    const uint32_t in0_tiles = Wt;
    const uint32_t in1_tiles = stats_tiles_cols;
    const uint32_t in2_tiles = Wt;
    const uint32_t in3_tiles = Wt;
    const uint32_t in4_tiles = 1;  // epsilon
    const uint32_t in5_tiles = 1;  // reduce scalar

    const uint32_t intermed0_tiles = tile_cols_per_device;
    const uint32_t intermed1_tiles = 1;
    const uint32_t intermed2_tiles = 1;
    const uint32_t intermed3_tiles = 1;
    const uint32_t intermed4_tiles = 1;
    const uint32_t intermed5_tiles = Wt;
    const uint32_t intermed6_tiles = Wt;
    const uint32_t intermed7_tiles = Wt;
    const uint32_t out0_tiles = Wt;

    TT_FATAL(
        W <= TILE_WIDTH * in0_tiles,
        "W ({}) exceeds the maximum supported size of tile buffer ({} * {}, kernel limitation right now)",
        W,
        TILE_WIDTH,
        in0_tiles);
    TT_FATAL(
        in0_tiles % block_size == 0,
        "Buffer size in0_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        in0_tiles,
        block_size);
    TT_FATAL(
        in2_tiles % block_size == 0,
        "Buffer size in2_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        in2_tiles,
        block_size);
    TT_FATAL(
        in3_tiles % block_size == 0,
        "Buffer size in3_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        in3_tiles,
        block_size);
    TT_FATAL(
        out0_tiles % block_size == 0,
        "Buffer size out0_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        out0_tiles,
        block_size);
    TT_FATAL(
        intermed5_tiles % block_size == 0,
        "Buffer size im0_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        intermed5_tiles,
        block_size);
    TT_FATAL(
        intermed6_tiles % block_size == 0,
        "Buffer size im6_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        intermed6_tiles,
        block_size);
    TT_FATAL(
        intermed7_tiles % block_size == 0,
        "Buffer size im7_t ({}) must be divisible by block_size ({}) for proper reader and compute kernel operation",
        intermed7_tiles,
        block_size);

    auto grid_size = device->compute_with_storage_grid_size();

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    auto cores = corerange_to_cores(all_cores, std::nullopt);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = tt::tt_metal::CreateProgram();

    std::vector<uint32_t> reader_compile_time_args = {
        block_size,
        stats_tiles_cols,
    };

    uint32_t gamma_stick_size = 0;
    if (gamma_is_row_major) {
        gamma_stick_size = gamma.value().padded_shape()[-1] * gamma.value().element_size();
        bool gamma_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(gamma_stick_size);
        TT_FATAL(gamma_stick_size_is_power_of_two, "Only power of 2 gammas are supported");
    }
    reader_compile_time_args.push_back(gamma_stick_size);

    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(stats.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(gamma.has_value() ? gamma.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(beta.has_value() ? beta.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {block_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> compute_defines;
    if (gamma.has_value()) {
        reader_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_defines["FUSE_BETA"] = "1";
    }

    auto use_row_major_kernel = gamma_is_row_major || beta_is_row_major;
    TT_FATAL(
        use_row_major_kernel || (!gamma.has_value() && !beta.has_value()),
        "Only row major gamma and beta are supported");

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_unary_interleaved_ln_rm_gb_post_allgather.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_blocked.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    bool float32_reduction = fp32_dest_acc_en && !operation_attributes.program_config.legacy_reduction;
    std::vector<uint32_t> compute_args = {
        Wt,
        block_size,
        stats_tiles_cols,
        gamma.has_value(),
        beta.has_value(),
        fp32_dest_acc_en,
        float32_reduction ? 1 : 0,
        operation_attributes.program_config.legacy_rsqrt ? 1 : 0};

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
    auto compute_kernels_id = CreateKernel(program, compute_kernel_file, all_cores, compute_config);

    // Create circular buffers
    // c_in0 -> a
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(in0_tiles * in_single_tile_size, {{tt::CBIndex::c_0, in_data_format}})
            .set_page_size(tt::CBIndex::c_0, in_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_src0_config);
    // c_in1 -> stats
    CircularBufferConfig cb_stats_config =
        CircularBufferConfig(in1_tiles * stats_single_tile_size, {{tt::CBIndex::c_1, stats_data_format}})
            .set_page_size(tt::CBIndex::c_1, stats_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_stats_config);
    // c_in2 -> gamma
    if (gamma.has_value()) {
        CircularBufferConfig cb_gamma_config =
            CircularBufferConfig(in2_tiles * gamma_single_tile_size, {{tt::CBIndex::c_2, gamma_cb_data_format}})
                .set_page_size(tt::CBIndex::c_2, gamma_single_tile_size);
        CreateCircularBuffer(program, all_cores, cb_gamma_config);
    }
    // c_in3 -> beta
    if (beta.has_value()) {
        CircularBufferConfig cb_beta_config =
            CircularBufferConfig(in3_tiles * beta_single_tile_size, {{tt::CBIndex::c_3, beta_cb_data_format}})
                .set_page_size(tt::CBIndex::c_3, beta_single_tile_size);
        CreateCircularBuffer(program, all_cores, cb_beta_config);
    }
    // c_in4 -> epsilon
    CircularBufferConfig cb_eps_config =
        CircularBufferConfig(in4_tiles * bfloat16_tile_size, {{tt::CBIndex::c_4, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_4, bfloat16_tile_size);
    CreateCircularBuffer(program, all_cores, cb_eps_config);
    // c_in5 -> reduce scalar
    CircularBufferConfig cb_reduce_config =
        CircularBufferConfig(in5_tiles * bfloat16_tile_size, {{tt::CBIndex::c_5, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_5, bfloat16_tile_size);
    CreateCircularBuffer(program, all_cores, cb_reduce_config);

    // LN and RMS shared intermediates //
    // c_intermed0 -> [mean(x**2), mean(x)]
    CircularBufferConfig cb_intermed0_config =
        CircularBufferConfig(intermed0_tiles * single_tile_size, {{tt::CBIndex::c_6, cb_data_format}})
            .set_page_size(tt::CBIndex::c_6, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_intermed0_config);
    // c_intermed2 -> var = mean(x**2) - mean(x)**2
    CircularBufferConfig cb_intermed2_config =
        CircularBufferConfig(intermed2_tiles * single_tile_size, {{tt::CBIndex::c_8, cb_data_format}})
            .set_page_size(tt::CBIndex::c_8, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_intermed2_config);
    // c_intermed3 -> var + epsilon
    CircularBufferConfig cb_intermed3_config =
        CircularBufferConfig(intermed3_tiles * single_tile_size, {{tt::CBIndex::c_9, cb_data_format}})
            .set_page_size(tt::CBIndex::c_9, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_intermed3_config);
    // c_intermed4 -> 1/sqrt(var + epsilon)
    CircularBufferConfig cb_intermed4_config =
        CircularBufferConfig(intermed4_tiles * single_tile_size, {{tt::CBIndex::c_10, cb_data_format}})
            .set_page_size(tt::CBIndex::c_10, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_intermed4_config);
    // c_intermed6 -> (x - mean(x)) * 1/sqrt(var + epsilon)
    CircularBufferConfig cb_intermed6_config =
        CircularBufferConfig(intermed6_tiles * single_tile_size, {{tt::CBIndex::c_12, cb_data_format}})
            .set_page_size(tt::CBIndex::c_12, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_intermed6_config);

    // LN-specific intermediates
    if (!is_rmsnorm) {
        // c_intermed1 -> mean(x)**2
        CircularBufferConfig cb_intermed1_config =
            CircularBufferConfig(intermed1_tiles * single_tile_size, {{tt::CBIndex::c_7, cb_data_format}})
                .set_page_size(tt::CBIndex::c_7, single_tile_size);
        CreateCircularBuffer(program, all_cores, cb_intermed1_config);
        // c_intermed5 -> x - mean(x)
        CircularBufferConfig cb_intermed5_config =
            CircularBufferConfig(intermed5_tiles * single_tile_size, {{tt::CBIndex::c_11, cb_data_format}})
                .set_page_size(tt::CBIndex::c_11, single_tile_size);
        CreateCircularBuffer(program, all_cores, cb_intermed5_config);
        if (beta.has_value()) {
            // Layernorm has gamma and beta so we need an extra intermediate buffer
            // c_intermed7 -> (x - mean(x)) * 1/sqrt(var + epsilon) * gamma
            CircularBufferConfig cb_intermed7_config =
                CircularBufferConfig(intermed7_tiles * single_tile_size, {{tt::CBIndex::c_13, cb_data_format}})
                    .set_page_size(tt::CBIndex::c_13, single_tile_size);
            CreateCircularBuffer(program, all_cores, cb_intermed7_config);
        }
    }

    CircularBufferConfig cb_out0_config =
        CircularBufferConfig(out0_tiles * out_single_tile_size, {{tt::CBIndex::c_14, out_data_format}})
            .set_page_size(tt::CBIndex::c_14, out_single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_out0_config);

    uint32_t curr_row = 0;
    float winv = 1.0f / (W * num_devices);  // bcast-w scaler
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    union {
        float f;
        uint32_t u;
    } e{};
    e.f = operation_attributes.eps;  // epsilon

    // Set runtime arguments
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

        SetRuntimeArgs(
            program,
            reader_kernels_id,
            core,
            {a_addr,
             num_tile_rows_per_core,
             Wt,
             tile_offset,
             stats_offset,
             packed_winv_value,
             e.u,  // 0-5
             gamma_dram_addr,
             beta_dram_addr,
             stats_addr,
             y_offset}  // 6-8
        );
        SetRuntimeArgs(program, compute_kernels_id, core, {num_tile_rows_per_core});
        SetRuntimeArgs(program, writer_kernels_id, core, {dst_addr, num_tile_rows_per_core * Wt, tile_offset});
        curr_row += num_tile_rows_per_core;
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernels_id,
         .writer_kernel_id = writer_kernels_id,
         .compute_kernel_id = compute_kernels_id,
         .cores = std::move(cores),
         .num_cores = num_cores,
         .core_group_1 = core_group_1,
         .core_group_2 = core_group_2,
         .num_tile_rows_per_core_group_1 = num_tile_rows_per_core_group_1,
         .num_tile_rows_per_core_group_2 = num_tile_rows_per_core_group_2,
         .Wt = Wt,
         .stats_tiles_cols = stats_tiles_cols}};
}

void LayerNormPostAllGatherProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    const auto& input_tensor = tensor_args.input;
    const auto& stats_tensor = tensor_args.stats;
    const auto& gamma_tensor = tensor_args.gamma;
    const auto& beta_tensor = tensor_args.beta;

    const auto input_addr = input_tensor.buffer()->address();
    const auto stats_addr = stats_tensor.buffer()->address();
    const bool has_gamma = gamma_tensor.has_value();
    const bool has_beta = beta_tensor.has_value();
    const auto gamma_addr = has_gamma ? gamma_tensor.value().buffer()->address() : 0;
    const auto beta_addr = has_beta ? beta_tensor.value().buffer()->address() : 0;
    const auto output_addr = output.buffer()->address();

    auto& reader_runtime_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernel_id);
    auto& writer_runtime_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernel_id);

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

}  // namespace ttnn::operations::normalization::layernorm_post_all_gather::program
