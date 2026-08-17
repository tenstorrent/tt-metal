// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_post_all_gather_device_operation.hpp"
#include "layernorm_distributed_metal2_helpers.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <bit>
#include <string>
#include <variant>

using uint32_t = std::uint32_t;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
namespace m2 = tt::tt_metal::experimental;
using namespace ttnn::prim::layernorm_distributed_metal2;

const m2::KernelSpecName POST_READER{"post_reader"};
const m2::KernelSpecName POST_WRITER{"post_writer"};
const m2::KernelSpecName POST_COMPUTE{"post_compute"};

const m2::DFBSpecName POST_INPUT{"post_input"};
const m2::DFBSpecName POST_STATS{"post_stats"};
const m2::DFBSpecName POST_GAMMA{"post_gamma"};
const m2::DFBSpecName POST_BETA{"post_beta"};
const m2::DFBSpecName POST_EPS{"post_eps"};
const m2::DFBSpecName POST_REDUCE{"post_reduce"};
const m2::DFBSpecName POST_STATS_REDUCED{"post_stats_reduced"};
const m2::DFBSpecName POST_MEAN_SQUARED{"post_mean_squared"};
const m2::DFBSpecName POST_VAR{"post_var"};
const m2::DFBSpecName POST_RECIP_SQRT_VAR{"post_recip_sqrt_var"};
const m2::DFBSpecName POST_X_MINUS_MEAN{"post_x_minus_mean"};
const m2::DFBSpecName POST_X_NORMED{"post_x_normed"};
const m2::DFBSpecName POST_TIMES_GAMMA_OUT{"post_times_gamma_out"};
const m2::DFBSpecName POST_OUT{"post_out"};

const m2::TensorParamName POST_INPUT_T{"post_input_t"};
const m2::TensorParamName POST_STATS_T{"post_stats_t"};
const m2::TensorParamName POST_GAMMA_T{"post_gamma_t"};
const m2::TensorParamName POST_BETA_T{"post_beta_t"};
const m2::TensorParamName POST_OUTPUT_T{"post_output_t"};

constexpr const char* POST_READER_KERNEL =
    "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
    "reader_unary_interleaved_ln_rm_gb_post_allgather.cpp";
constexpr const char* POST_WRITER_KERNEL =
    "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
    "writer_unary_interleaved_start_id_blocked.cpp";

}  // namespace

// =============================================================================
// LayerNormPostAllGatherProgramFactory - Normal (non-Welford) operation
// =============================================================================

ttnn::device_operation::ProgramArtifacts LayerNormPostAllGatherProgramFactory::create_program_artifacts(
    const LayerNormPostAllGatherParams& operation_attributes,
    const LayerNormPostAllGatherInputs& tensor_args,
    Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& stats = tensor_args.stats;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

    const bool is_rmsnorm = operation_attributes.norm_type == LayerNormDistributedType::RMSNORM;
    const auto& shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;
    // Logical (un-padded) width is used for the normalization scaler so that
    // non-tile-aligned widths normalise by the true N, not the tile-padded N.
    const uint32_t logical_W = a.logical_shape()[-1];

    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;
    const uint32_t stats_tiles_cols = stats.padded_shape()[-1] / tile_width;
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

    const auto& input_mesh = a.mesh_tensor();
    const auto& stats_mesh = stats.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    IDevice* device = a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat stats_data_format = tt::tt_metal::datatype_to_dataformat_converter(stats.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    // FP32 input/stats require fp32_dest_acc_en: the intermediate buffers (cb_data_format) only
    // become Float32 when it is set, otherwise a Float32 input/stats buffer feeds Float16_b
    // intermediates and precision is silently lost. Reject the unsupported combination up front.
    TT_FATAL(
        !((in_data_format == tt::DataFormat::Float32 || stats_data_format == tt::DataFormat::Float32) &&
          !fp32_dest_acc_en),
        "FLOAT32 input/stats require fp32_dest_acc_en=true in the compute kernel config.");
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

    [[maybe_unused]] uint32_t num_gamma_tiles = gamma.has_value() ? gamma.value().physical_volume() / tile_hw : 0;
    [[maybe_unused]] uint32_t num_beta_tiles = beta.has_value() ? beta.value().physical_volume() / tile_hw : 0;

    // For bert, tensor is packed as RM with width 32
    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        num_gamma_tiles = gamma.has_value() ? gamma.value().physical_volume() / tile_width : 0;
    }
    if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        num_beta_tiles = beta.has_value() ? beta.value().physical_volume() / tile_width : 0;
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
    const uint32_t intermed4_tiles = 1;
    const uint32_t intermed5_tiles = cb_length;
    const uint32_t intermed6_tiles = cb_length;
    const uint32_t intermed7_tiles = cb_length;
    const uint32_t out0_tiles = cb_length;

    uint32_t gamma_stick_size = 0;
    uint32_t gamma_is_row_major = 0;
    uint32_t beta_is_row_major = 0;
    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        gamma_stick_size = gamma.value().padded_shape()[-1] * gamma.value().element_size();
        bool gamma_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(gamma_stick_size);
        TT_FATAL(gamma_stick_size_is_power_of_two, "Only power of 2 gammas are supported");
        gamma_is_row_major = 1;
    }
    uint32_t beta_stick_size = 0;
    if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        beta_stick_size = beta.value().padded_shape()[-1] * beta.value().element_size();
        bool beta_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(beta_stick_size);
        TT_FATAL(beta_stick_size_is_power_of_two, "Only power of 2 betas are supported");
        beta_is_row_major = 1;
    }
    // Reader uses this compile-time reduction width to generate the AVG scaler tile.
    const uint32_t reduce_factor = logical_W * num_devices;

    // Get program config
    ttnn::prim::LayerNormDefaultProgramConfig program_config;
    if (std::holds_alternative<ttnn::prim::LayerNormDefaultProgramConfig>(operation_attributes.program_config)) {
        program_config = std::get<ttnn::prim::LayerNormDefaultProgramConfig>(operation_attributes.program_config);
    }

    bool float32_reduction = fp32_dest_acc_en && !program_config.legacy_reduction;

    const auto* compute_kernel_file =
        is_rmsnorm ? "ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/"
                     "rmsnorm_post_allgather_metal2.cpp"
                   : "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
                     "layernorm_post_allgather.cpp";

    uint32_t eps_u = std::bit_cast<uint32_t>(operation_attributes.eps);  // epsilon

    // gamma and beta are optional, and the kernel-side handle for a buffer exists only where the host
    // binds that buffer: the build emits `dfb::gamma` into a kernel's generated bindings only if this
    // factory gave that kernel a gamma binding. So when gamma is absent, a kernel's source must not
    // contain the text `dfb::gamma` at all. Gating the use with `if constexpr` does not achieve that,
    // the gate has to be `#ifdef`, which removes the text before the compiler sees it.
    // These two defines are what the kernels gate on.
    m2::KernelSpec::CompilerOptions::Defines gb_defines;
    if (gamma.has_value()) {
        gb_defines.emplace("FUSE_GAMMA", "1");
    }
    if (beta.has_value()) {
        gb_defines.emplace("FUSE_BETA", "1");
    }

    // The normalized result is staged in its own buffer only while gamma or beta still has to be
    // applied; otherwise it is packed straight into the output. RMSNorm applies beta only in the
    // company of gamma, so a beta without a gamma leaves it staged nowhere.
    const bool uses_x_normed = is_rmsnorm ? (gamma.has_value()) : (gamma.has_value() || beta.has_value());
    // The gamma product needs a buffer of its own only when beta still has to be added to it.
    const bool uses_times_gamma_out = gamma.has_value() && beta.has_value();

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers
    ////////////////////////////////////////////////////////////////////////////
    const tt::DataFormat scaler_data_format =
        in_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t scaler_tile_size = tt::tile_size(scaler_data_format);

    m2::Group<m2::DataflowBufferSpec> dfbs;
    dfbs.push_back(make_dfb(POST_INPUT, in0_tiles, in_single_tile_size, in_data_format));
    dfbs.push_back(make_dfb(POST_STATS, in1_tiles, stats_single_tile_size, stats_data_format));
    if (gamma.has_value()) {
        dfbs.push_back(make_dfb(POST_GAMMA, in2_tiles, gamma_single_tile_size, gamma_cb_data_format));
    }
    if (beta.has_value()) {
        dfbs.push_back(make_dfb(POST_BETA, in3_tiles, beta_single_tile_size, beta_cb_data_format));
    }
    dfbs.push_back(make_dfb(POST_EPS, in4_tiles, bfloat16_tile_size, tt::DataFormat::Float16_b));
    dfbs.push_back(make_dfb(POST_REDUCE, in5_tiles, scaler_tile_size, scaler_data_format));
    // [mean(x**2), mean(x)], layernorm only. RMSNorm reduces the stats straight into the variance
    // buffer and never touches this one.
    if (!is_rmsnorm) {
        dfbs.push_back(make_dfb(POST_STATS_REDUCED, intermed0_tiles, single_tile_size, cb_data_format));
        // mean(x)**2
        dfbs.push_back(make_dfb(POST_MEAN_SQUARED, intermed1_tiles, single_tile_size, cb_data_format));
        // x - mean(x)
        dfbs.push_back(make_dfb(POST_X_MINUS_MEAN, intermed5_tiles, single_tile_size, cb_data_format));
    }
    // var = mean(x**2) - mean(x)**2, or mean(x**2) for RMSNorm
    dfbs.push_back(make_dfb(POST_VAR, intermed2_tiles, single_tile_size, cb_data_format));
    // 1/sqrt(var + epsilon)
    dfbs.push_back(make_dfb(POST_RECIP_SQRT_VAR, intermed4_tiles, single_tile_size, cb_data_format));
    if (uses_x_normed) {
        // (x - mean(x)) * 1/sqrt(var + epsilon)
        dfbs.push_back(make_dfb(POST_X_NORMED, intermed6_tiles, single_tile_size, cb_data_format));
    }
    if (uses_times_gamma_out) {
        // (x - mean(x)) * 1/sqrt(var + epsilon) * gamma
        dfbs.push_back(make_dfb(POST_TIMES_GAMMA_OUT, intermed7_tiles, single_tile_size, cb_data_format));
    }
    dfbs.push_back(make_dfb(POST_OUT, out0_tiles, out_single_tile_size, out_data_format));

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec reader{
        .unique_id = POST_READER,
        .source = POST_READER_KERNEL,
        .compiler_options = {.defines = gb_defines},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = POST_INPUT,
                    .accessor_name = "inp",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = POST_STATS,
                    .accessor_name = "stats",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = POST_EPS, .accessor_name = "eps", .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = POST_REDUCE,
                    .accessor_name = "reduce",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = POST_INPUT_T, .accessor_name = "src"},
                m2::TensorBinding{.tensor_parameter_name = POST_STATS_T, .accessor_name = "stats_src"},
            },
        .compile_time_args =
            {{"blk", block_size},
             {"stats_tiles_cols", stats_tiles_cols},
             {"gamma_is_row_major", gamma_is_row_major},
             {"beta_is_row_major", beta_is_row_major},
             {"dfb_length", cb_length},
             {"Wt", tiles_per_core_y},
             {"reduce_factor", reduce_factor}},
        .runtime_arg_schema = {.runtime_arg_names = {"NCHt", "tile_offset", "stats_tile_offset", "eps", "y_offset"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };
    if (gamma.has_value()) {
        reader.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = POST_GAMMA, .accessor_name = "gamma", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        reader.tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = POST_GAMMA_T, .accessor_name = "gamma_src"});
    }
    if (beta.has_value()) {
        reader.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = POST_BETA, .accessor_name = "beta", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        reader.tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = POST_BETA_T, .accessor_name = "beta_src"});
    }

    m2::KernelSpec writer{
        .unique_id = POST_WRITER,
        .source = POST_WRITER_KERNEL,
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = POST_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = POST_OUTPUT_T, .accessor_name = "dst"}},
        .compile_time_args = {{"blk", block_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "tile_offset"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    m2::KernelSpec compute{
        .unique_id = POST_COMPUTE,
        .source = compute_kernel_file,
        .compiler_options = {.defines = gb_defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = POST_INPUT,
                    .accessor_name = "inp",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = POST_STATS,
                    .accessor_name = "stats",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = POST_EPS, .accessor_name = "eps", .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = POST_REDUCE,
                    .accessor_name = "reduce",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = POST_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {{"Wt", tiles_per_core_y},
             {"blk", block_size},
             {"stats_tiles_cols", stats_tiles_cols},
             {"fp32_dtype", static_cast<uint32_t>(fp32_dest_acc_en)},
             {"float32_reduction", static_cast<uint32_t>(float32_reduction)},
             {"legacy_rsqrt", static_cast<uint32_t>(program_config.legacy_rsqrt)},
             {"dfb_length", cb_length}},
        .runtime_arg_schema = {.runtime_arg_names = {"NCHt"}},
        .hw_config = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config),
    };
    // Every intermediate below is private to the compute kernel: it packs into the buffer and
    // unpacks it back, so it is that buffer's only endpoint on both sides.
    if (!is_rmsnorm) {
        bind_self_loop(compute, POST_STATS_REDUCED, "stats_reduced");
        bind_self_loop(compute, POST_MEAN_SQUARED, "mean_squared");
        bind_self_loop(compute, POST_X_MINUS_MEAN, "x_minus_mean");
    }
    bind_self_loop(compute, POST_VAR, "var");
    bind_self_loop(compute, POST_RECIP_SQRT_VAR, "recip_sqrt_var");
    if (uses_x_normed) {
        bind_self_loop(compute, POST_X_NORMED, "x_normed");
    }
    if (uses_times_gamma_out) {
        bind_self_loop(compute, POST_TIMES_GAMMA_OUT, "times_gamma_out");
    }
    if (gamma.has_value()) {
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = POST_GAMMA, .accessor_name = "gamma", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }
    if (beta.has_value()) {
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = POST_BETA, .accessor_name = "beta", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }
    auto& compute_gen1 = gen1_compute_config(std::get<m2::ComputeHardwareConfig>(compute.hw_config));
    // With the 32-bit Dest register enabled, every Float32 buffer the compute kernel consumes needs an
    // explicit unpack mode. In this kernel every consumed buffer is read by an FPU op (the stats row
    // reduce, mul_tiles / sub_tiles / add_tiles for the variance, and the broadcast multiplies and adds
    // of the gamma / beta chain), and the FPU takes its operands from SrcA/SrcB, so SrcA/B is the mode
    // for all of them. They are listed one by one on purpose: any "catch all" clever method that set
    // unpack_via_src on all of them could  accidentally include a future DFB where unpack_via_src would
    // not be appropriate.
    if (compute_gen1.enable_32_bit_dest) {
        // The intermediates all carry cb_data_format, which is Float32 exactly when the Dest register is.
        unpack_via_src(compute_gen1, POST_VAR);
        unpack_via_src(compute_gen1, POST_RECIP_SQRT_VAR);
        if (!is_rmsnorm) {
            unpack_via_src(compute_gen1, POST_STATS_REDUCED);
            unpack_via_src(compute_gen1, POST_MEAN_SQUARED);
            unpack_via_src(compute_gen1, POST_X_MINUS_MEAN);
        }
        if (uses_x_normed) {
            unpack_via_src(compute_gen1, POST_X_NORMED);
        }
        if (uses_times_gamma_out) {
            unpack_via_src(compute_gen1, POST_TIMES_GAMMA_OUT);
        }
        // The inputs carry their own tensor's dtype. The epsilon buffer is always Float16_b.
        if (in_data_format == tt::DataFormat::Float32) {
            unpack_via_src(compute_gen1, POST_INPUT);
            unpack_via_src(compute_gen1, POST_REDUCE);  // the scaler tile mirrors the input's dtype
        }
        if (stats_data_format == tt::DataFormat::Float32) {
            unpack_via_src(compute_gen1, POST_STATS);
        }
        if (gamma.has_value() && gamma_cb_data_format == tt::DataFormat::Float32) {
            unpack_via_src(compute_gen1, POST_GAMMA);
        }
        if (beta.has_value() && beta_cb_data_format == tt::DataFormat::Float32) {
            unpack_via_src(compute_gen1, POST_BETA);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::TensorParameter> tensor_parameters;
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = POST_INPUT_T, .spec = input_mesh.tensor_spec()});
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = POST_STATS_T, .spec = stats_mesh.tensor_spec()});
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = POST_OUTPUT_T, .spec = output_mesh.tensor_spec()});
    if (gamma.has_value()) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = POST_GAMMA_T, .spec = gamma.value().mesh_tensor().tensor_spec()});
    }
    if (beta.has_value()) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = POST_BETA_T, .spec = beta.value().mesh_tensor().tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime arguments
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelRunArgs reader_run{.kernel = POST_READER};
    m2::KernelRunArgs writer_run{.kernel = POST_WRITER};
    m2::KernelRunArgs compute_run{.kernel = POST_COMPUTE};

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
                m2::AddRuntimeArgsForNode(
                    reader_run.runtime_arg_values,
                    core,
                    {{"NCHt", tiles_per_core_x},
                     {"tile_offset", tile_offset},
                     {"stats_tile_offset", stats_offset},
                     {"eps", eps_u},
                     {"y_offset", y * tiles_per_core_y}});
                m2::AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"NCHt", tiles_per_core_x}});
                m2::AddRuntimeArgsForNode(
                    writer_run.runtime_arg_values,
                    core,
                    {{"num_tiles", tiles_per_core_x * tiles_per_core_y}, {"tile_offset", tile_offset}});
            }
        }
    } else {
        uint32_t curr_row = 0;
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

            m2::AddRuntimeArgsForNode(
                reader_run.runtime_arg_values,
                core,
                {{"NCHt", num_tile_rows_per_core},
                 {"tile_offset", tile_offset},
                 {"stats_tile_offset", stats_offset},
                 {"eps", eps_u},
                 {"y_offset", y_offset}});
            m2::AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"NCHt", num_tile_rows_per_core}});
            m2::AddRuntimeArgsForNode(
                writer_run.runtime_arg_values,
                core,
                {{"num_tiles", num_tile_rows_per_core * Wt}, {"tile_offset", tile_offset}});
            curr_row += num_tile_rows_per_core;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec{
        .name = "layernorm_post_all_gather",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {m2::WorkUnitSpec{
            .name = "main", .kernels = {POST_READER, POST_WRITER, POST_COMPUTE}, .target_nodes = all_cores}},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run), std::move(compute_run)};
    run_args.tensor_args.emplace(POST_INPUT_T, input_mesh);
    run_args.tensor_args.emplace(POST_STATS_T, stats_mesh);
    run_args.tensor_args.emplace(POST_OUTPUT_T, output_mesh);
    if (gamma.has_value()) {
        run_args.tensor_args.emplace(POST_GAMMA_T, gamma.value().mesh_tensor());
    }
    if (beta.has_value()) {
        run_args.tensor_args.emplace(POST_BETA_T, beta.value().mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
