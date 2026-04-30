// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "welford_reduce_device_operation.hpp"

namespace ttnn::prim {
namespace welford_reduce_metal2 {

namespace metal2 = tt::tt_metal::experimental::metal2_host_api;

constexpr const char* kProgramName = "welford_reduce";
constexpr const char* kReader = "reader";
constexpr const char* kWriter = "writer";
constexpr const char* kComputeGroup1 = "compute_group_1";
constexpr const char* kComputeGroup2 = "compute_group_2";

constexpr const char* kInputDfb = "input_dfb_c0";
constexpr const char* kUnusedDfb = "unused_dfb_c1";
constexpr const char* kScalarDfb = "scalar_dfb_c2";
constexpr const char* kOutputDfb = "output_dfb";
constexpr const char* kScratchDfb = "scratch_dfb";
constexpr const char* kScaledDfb = "scaled_dfb";
constexpr const char* kPartialDfb = "partial_dfb";
constexpr const char* kCombinedDfb = "combined_dfb";

constexpr uint32_t kInputCb = 0;
constexpr uint32_t kUnusedCb = 1;
constexpr uint32_t kScalarCb = 2;
constexpr uint32_t kOutputCb = 3;
constexpr uint32_t kScratchCb = 4;
constexpr uint32_t kScaledCb = 5;
constexpr uint32_t kPartialCb = 4;
constexpr uint32_t kCombinedCb = 5;

std::vector<std::pair<std::string, std::string>> to_define_vector(const std::map<std::string, std::string>& defines) {
    return {defines.begin(), defines.end()};
}

std::vector<std::pair<std::string, uint32_t>> make_tensor_accessor_named_args(
    const std::string& prefix, const tt::tt_metal::Buffer& buffer) {
    auto args = tt::tt_metal::TensorAccessorArgs(buffer).get_compile_time_args();
    TT_FATAL(args.size() == 2, "Welford reduction Metal 2.0 path only supports interleaved tensor accessors");
    return {{prefix + "_args_config", args[0]}, {prefix + "_page_size", args[1]}};
}

void append_named_args(
    std::vector<std::pair<std::string, uint32_t>>& dst, std::vector<std::pair<std::string, uint32_t>> src) {
    dst.insert(dst.end(), std::make_move_iterator(src.begin()), std::make_move_iterator(src.end()));
}

metal2::DataMovementConfiguration make_dm_config(tt::tt_metal::DataMovementProcessor processor, tt::tt_metal::NOC noc) {
    return metal2::DataMovementConfiguration{
        .gen1_data_movement_config =
            metal2::DataMovementConfiguration::Gen1DataMovementConfig{.processor = processor, .noc = noc},
        .gen2_data_movement_config = metal2::DataMovementConfiguration::Gen2DataMovementConfig{},
    };
}

metal2::ComputeConfiguration make_compute_config(tt::tt_metal::MathFidelity math_fidelity, bool fp32_dest_acc_en) {
    return metal2::ComputeConfiguration{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };
}

metal2::KernelSpec::DFBBinding dfb_binding(
    const std::string& dfb_name, const std::string& accessor_name, metal2::KernelSpec::DFBEndpointType endpoint_type) {
    return {
        .dfb_spec_name = dfb_name,
        .local_accessor_name = accessor_name,
        .endpoint_type = endpoint_type,
        .access_pattern = metal2::DFBAccessPattern::STRIDED,
    };
}

metal2::DataflowBufferSpec make_dfb(
    const std::string& name,
    const metal2::NodeRangeSet& nodes,
    uint32_t entry_size,
    uint32_t num_entries,
    tt::DataFormat data_format) {
    return {
        .unique_id = name,
        .target_nodes = nodes,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = data_format,
        .disable_implicit_sync = true,
    };
}

struct WelfordProgramSpecAndParams {
    metal2::ProgramSpec spec;
    metal2::ProgramRunParams params;
};

WelfordProgramSpecAndParams build_welford_program_spec_and_params(
    const WelfordReduceDeviceOperation::operation_attributes_t& operation_attributes,
    const WelfordReduceDeviceOperation::tensor_args_t& tensor_arg,
    WelfordReduceDeviceOperation::tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Shape& padded_shape = tensor_arg.padded_shape();
    const Shape& logical_shape = tensor_arg.logical_shape();

    const uint32_t W = logical_shape[-1];
    const uint32_t H = logical_shape[-2];
    const uint32_t W_padded = padded_shape[-1];
    const uint32_t H_padded = padded_shape[-2];
    TT_FATAL(
        H_padded > 0 && W_padded > 0,
        "Padded H and W dimensions must be non-zero, got H_padded={}, W_padded={}",
        H_padded,
        W_padded);

    const uint32_t NC = tensor_arg.physical_volume() / (H_padded * W_padded);
    const uint32_t tile_height = tensor_arg.tensor_spec().tile().get_height();
    const uint32_t tile_width = tensor_arg.tensor_spec().tile().get_width();

    const uint32_t Wt = W_padded / tile_width;
    const uint32_t Ht = H_padded / tile_height;
    const uint32_t HtWt = Ht * Wt;

    const bool reduce_w = (operation_attributes.reduce_dim == ReduceOpDim::W);
    const bool reduce_h = (operation_attributes.reduce_dim == ReduceOpDim::H);
    const bool reduce_hw = (operation_attributes.reduce_dim == ReduceOpDim::HW);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tensor_arg.device()->arch(), operation_attributes.compute_kernel_config);
    (void)math_approx_mode;
    (void)packer_l1_acc;

    const tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(tensor_arg.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    const tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    const uint32_t scalar_single_tile_size = tt::tile_size(scalar_cb_data_format);
    const tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const uint32_t reduce_batch_size = operation_attributes.reduce_batch_size;
    tt_metal::IDevice* device = tensor_arg.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const auto num_work_units = reduce_w ? (NC * Ht) : (reduce_hw ? (NC / reduce_batch_size) : (NC * Wt));

    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_work_units_per_core_group_1 = 0;
    uint32_t num_work_units_per_core_group_2 = 0;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_work_units_per_core_group_1,
            num_work_units_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_work_units);
    } else {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_work_units_per_core_group_1,
            num_work_units_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_work_units);
    }

    const bool do_scale = (operation_attributes.scalar != 1.0f);
    const bool is_std = (operation_attributes.math_op == ReduceOpMath::STD);
    if (!reduce_hw && operation_attributes.correction) {
        const uint32_t reduce_size = reduce_w ? W : H;
        TT_FATAL(
            reduce_size >= 2,
            "Bessel's correction requires at least 2 elements along the reduction dimension, got {}",
            reduce_size);
    }
    if (reduce_hw && operation_attributes.correction) {
        TT_FATAL(
            H * W * reduce_batch_size >= 2,
            "Bessel's correction requires at least 2 elements across all reduction dimensions, got {}",
            H * W * reduce_batch_size);
    }

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, operation_attributes.reduce_dim);
    reduce_defines["REDUCE_METAL2_NAMED_ARGS"] = "1";
    reduce_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
    reduce_defines["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
    const auto define_vector = to_define_vector(reduce_defines);

    const uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scalar);
    std::vector<std::pair<std::string, uint32_t>> reader_ct_args;
    std::string reader_kernel;
    size_t reader_num_varargs = 0;
    if (reduce_h || reduce_hw) {
        reader_ct_args = {{"Ht", Ht}, {"Wt", Wt}, {"HtWt", HtWt}, {"scaler_bits", scaler_bits}, {"use_welford", 1}};
        append_named_args(reader_ct_args, make_tensor_accessor_named_args("src", *tensor_arg.buffer()));
        reader_kernel =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp";
        reader_num_varargs = 4;
    } else {
        reader_ct_args = {{"scaler_bits", scaler_bits}};
        append_named_args(reader_ct_args, make_tensor_accessor_named_args("src", *tensor_arg.buffer()));
        reader_kernel =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_reduce_universal_start_id.cpp";
        reader_num_varargs = 3;
    }

    metal2::KernelSpec reader{
        .unique_id = kReader,
        .source = metal2::KernelSpec::SourceFilePath{reader_kernel},
        .target_nodes = all_cores,
        .compiler_options = {.defines = define_vector},
        .dfb_bindings =
            {
                dfb_binding(kInputDfb, "input_dfb", metal2::KernelSpec::DFBEndpointType::PRODUCER),
                dfb_binding(kUnusedDfb, "unused_dfb", metal2::KernelSpec::DFBEndpointType::PRODUCER),
                dfb_binding(kScalarDfb, "scalar_dfb", metal2::KernelSpec::DFBEndpointType::PRODUCER),
            },
        .compile_time_arg_bindings = reader_ct_args,
        .runtime_arguments_schema = {.num_runtime_varargs = reader_num_varargs},
        .config_spec = make_dm_config(
            DataMovementProcessor::RISCV_1,
            tt::tt_metal::detail::preferred_noc_for_dram_read(tensor_arg.device()->arch())),
    };

    std::vector<std::pair<std::string, uint32_t>> writer_ct_args;
    std::string writer_kernel;
    if (reduce_hw) {
        writer_ct_args = {
            {"Wt", Wt},
            {"W", W},
            {"tile_width", tile_width},
            {"H", H},
            {"correction", static_cast<uint32_t>(operation_attributes.correction)},
            {"reduce_batch_size", reduce_batch_size},
            {"cb_partial", kPartialCb},
            {"cb_combined", kCombinedCb},
            {"cb_out", kOutputCb},
        };
        append_named_args(writer_ct_args, make_tensor_accessor_named_args("dst", *tensor_return_value.buffer()));
        writer_kernel =
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "writer_welford_hw.cpp";
    } else {
        writer_ct_args = {{"output_cb_index", kOutputCb}};
        append_named_args(writer_ct_args, make_tensor_accessor_named_args("dst", *tensor_return_value.buffer()));
        writer_kernel =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id.cpp";
    }

    std::vector<metal2::KernelSpec::DFBBinding> writer_dfbs = {
        dfb_binding(kOutputDfb, "output_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER),
    };
    if (reduce_hw) {
        writer_dfbs.push_back(dfb_binding(kPartialDfb, "partial_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER));
        writer_dfbs.push_back(dfb_binding(kCombinedDfb, "combined_dfb", metal2::KernelSpec::DFBEndpointType::PRODUCER));
    }

    metal2::KernelSpec writer{
        .unique_id = kWriter,
        .source = metal2::KernelSpec::SourceFilePath{writer_kernel},
        .target_nodes = all_cores,
        .compiler_options = {.defines = define_vector},
        .dfb_bindings = writer_dfbs,
        .compile_time_arg_bindings = writer_ct_args,
        .runtime_arguments_schema = {.num_runtime_varargs = 3},
        .config_spec = make_dm_config(
            DataMovementProcessor::RISCV_0,
            tt::tt_metal::detail::preferred_noc_for_dram_write(tensor_arg.device()->arch())),
    };

    std::vector<std::pair<std::string, uint32_t>> compute_ct_args;
    std::string compute_kernel;
    if (reduce_hw) {
        compute_ct_args = {
            {"Ht", Ht},
            {"H", H},
            {"tile_height", tile_height},
            {"Wt", Wt},
            {"do_scale", static_cast<uint32_t>(do_scale)},
            {"reduce_batch_size", reduce_batch_size},
            {"is_std", static_cast<uint32_t>(is_std)},
            {"cb_in", kInputCb},
            {"cb_scalar", kScalarCb},
            {"cb_out", kOutputCb},
            {"cb_partial", kPartialCb},
            {"cb_combined", kCombinedCb},
        };
        compute_kernel = "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_hw.cpp";
    } else if (reduce_w) {
        compute_ct_args = {
            {"Wt", Wt},
            {"W", W},
            {"tile_width", tile_width},
            {"do_scale", static_cast<uint32_t>(do_scale)},
            {"correction", static_cast<uint32_t>(operation_attributes.correction)},
            {"is_std", static_cast<uint32_t>(is_std)},
            {"cb_in", kInputCb},
            {"cb_scalar", kScalarCb},
            {"cb_out", kOutputCb},
            {"cb_var", kScratchCb},
            {"cb_scaled", kScaledCb},
        };
        compute_kernel = "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_w.cpp";
    } else {
        compute_ct_args = {
            {"Ht", Ht},
            {"H", H},
            {"tile_height", tile_height},
            {"do_scale", static_cast<uint32_t>(do_scale)},
            {"correction", static_cast<uint32_t>(operation_attributes.correction)},
            {"is_std", static_cast<uint32_t>(is_std)},
            {"cb_in", kInputCb},
            {"cb_scalar", kScalarCb},
            {"cb_out", kOutputCb},
        };
        compute_kernel = "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/welford_reduce_h.cpp";
    }

    auto make_compute = [&](const char* name, const CoreRangeSet& cores) {
        std::vector<metal2::KernelSpec::DFBBinding> dfb_bindings = {
            dfb_binding(kInputDfb, "input_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER),
            dfb_binding(kUnusedDfb, "unused_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER),
            dfb_binding(kScalarDfb, "scalar_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER),
            dfb_binding(kOutputDfb, "output_dfb", metal2::KernelSpec::DFBEndpointType::PRODUCER),
        };
        if (reduce_w) {
            dfb_bindings.push_back(
                dfb_binding(kScratchDfb, "scratch_in", metal2::KernelSpec::DFBEndpointType::CONSUMER));
            dfb_bindings.push_back(
                dfb_binding(kScratchDfb, "scratch_out", metal2::KernelSpec::DFBEndpointType::PRODUCER));
            dfb_bindings.push_back(dfb_binding(kScaledDfb, "scaled_in", metal2::KernelSpec::DFBEndpointType::CONSUMER));
            dfb_bindings.push_back(
                dfb_binding(kScaledDfb, "scaled_out", metal2::KernelSpec::DFBEndpointType::PRODUCER));
        }
        if (reduce_hw) {
            dfb_bindings.push_back(
                dfb_binding(kPartialDfb, "partial_dfb", metal2::KernelSpec::DFBEndpointType::PRODUCER));
            dfb_bindings.push_back(
                dfb_binding(kCombinedDfb, "combined_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER));
        }
        return metal2::KernelSpec{
            .unique_id = name,
            .source = metal2::KernelSpec::SourceFilePath{compute_kernel},
            .target_nodes = cores,
            .compiler_options = {.defines = define_vector},
            .dfb_bindings = dfb_bindings,
            .compile_time_arg_bindings = compute_ct_args,
            .runtime_arguments_schema = {.num_runtime_varargs = 1},
            .config_spec = make_compute_config(math_fidelity, fp32_dest_acc_en),
        };
    };

    metal2::KernelSpec compute_group_1 = make_compute(kComputeGroup1, core_group_1);
    std::optional<metal2::KernelSpec> compute_group_2;
    if (!core_group_2.ranges().empty()) {
        compute_group_2 = make_compute(kComputeGroup2, core_group_2);
        compute_group_2->dfb_bindings.clear();
    }

    metal2::ProgramSpec spec{
        .program_id = kProgramName,
        .kernels = {reader, writer, compute_group_1},
        .dataflow_buffers =
            {
                make_dfb(kInputDfb, all_cores, input_single_tile_size, 2, input_cb_data_format),
                make_dfb(kUnusedDfb, all_cores, input_single_tile_size, 1, input_cb_data_format),
                make_dfb(kScalarDfb, all_cores, scalar_single_tile_size, 1, scalar_cb_data_format),
                make_dfb(kOutputDfb, all_cores, dst_single_tile_size, 2, dst_cb_data_format),
            },
        .workers =
            std::vector<metal2::WorkerSpec>{
                {.unique_id = "worker_group_1",
                 .kernels = {kReader, kWriter, kComputeGroup1},
                 .dataflow_buffers = {kInputDfb, kUnusedDfb, kScalarDfb, kOutputDfb},
                 .target_nodes = core_group_1},
            },
    };

    if (reduce_w) {
        const tt::DataFormat scratch_cb_data_format =
            fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        spec.dataflow_buffers.push_back(
            make_dfb(kScratchDfb, all_cores, tt::tile_size(scratch_cb_data_format), 1, scratch_cb_data_format));
        spec.dataflow_buffers.push_back(
            make_dfb(kScaledDfb, all_cores, input_single_tile_size, 1, input_cb_data_format));
        spec.workers->front().dataflow_buffers.push_back(kScratchDfb);
        spec.workers->front().dataflow_buffers.push_back(kScaledDfb);
    }
    if (reduce_hw) {
        spec.dataflow_buffers.push_back(
            make_dfb(kPartialDfb, all_cores, tt::tile_size(tt::DataFormat::Float32), 4, tt::DataFormat::Float32));
        spec.dataflow_buffers.push_back(
            make_dfb(kCombinedDfb, all_cores, tt::tile_size(tt::DataFormat::Float32), 1, tt::DataFormat::Float32));
        spec.workers->front().dataflow_buffers.push_back(kPartialDfb);
        spec.workers->front().dataflow_buffers.push_back(kCombinedDfb);
    }

    if (compute_group_2.has_value()) {
        spec.kernels.push_back(*compute_group_2);
        std::vector<metal2::DFBSpecName> dfbs = spec.workers->front().dataflow_buffers;
        spec.workers->push_back(
            {.unique_id = "worker_group_2",
             .kernels = {kReader, kWriter, kComputeGroup2},
             .dataflow_buffers = dfbs,
             .target_nodes = core_group_2});
    }

    std::vector<CoreCoord> cores;
    if (operation_attributes.sub_core_grids.has_value()) {
        for (const auto& range : all_cores.ranges()) {
            for (int y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (int x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    cores.emplace_back(x, y);
                }
            }
        }
    } else {
        cores = grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    }

    metal2::ProgramRunParams::KernelRunParams reader_params{.kernel_spec_name = kReader};
    metal2::ProgramRunParams::KernelRunParams writer_params{.kernel_spec_name = kWriter};
    metal2::ProgramRunParams::KernelRunParams compute_g1_params{.kernel_spec_name = kComputeGroup1};
    metal2::ProgramRunParams::KernelRunParams compute_g2_params{.kernel_spec_name = kComputeGroup2};

    if (reduce_w) {
        uint32_t input_tiles_offset = 0;
        uint32_t output_tiles_offset = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = cores[i];
            const bool in_g1 = core_group_1.contains(core);
            const uint32_t num_work_units_per_core =
                in_g1 ? num_work_units_per_core_group_1 : num_work_units_per_core_group_2;
            const uint32_t num_input_tiles_per_core = num_work_units_per_core * Wt;
            const uint32_t num_output_tiles_per_core = num_work_units_per_core;
            reader_params.runtime_varargs.push_back(
                {core, {tensor_arg.buffer()->address(), num_input_tiles_per_core, input_tiles_offset}});
            (in_g1 ? compute_g1_params : compute_g2_params)
                .runtime_varargs.push_back({core, {num_work_units_per_core}});
            writer_params.runtime_varargs.push_back(
                {core, {tensor_return_value.buffer()->address(), num_output_tiles_per_core, output_tiles_offset}});
            input_tiles_offset += num_input_tiles_per_core;
            output_tiles_offset += num_output_tiles_per_core;
        }
    } else if (reduce_hw) {
        TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
        TT_FATAL(
            NC % reduce_batch_size == 0, "NC ({}) must be divisible by reduce_batch_size ({})", NC, reduce_batch_size);
        uint32_t nc_slice_offset = 0;
        uint32_t output_offset = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = cores[i];
            const bool in_g1 = core_group_1.contains(core);
            const uint32_t num_outputs_per_core =
                in_g1 ? num_work_units_per_core_group_1 : num_work_units_per_core_group_2;
            const uint32_t nc_slices_per_core = num_outputs_per_core * reduce_batch_size;
            const uint32_t num_cols = Wt * nc_slices_per_core;
            const uint32_t col_start_tile_id = nc_slice_offset * HtWt;
            reader_params.runtime_varargs.push_back(
                {core, {tensor_arg.buffer()->address(), col_start_tile_id, 0u, num_cols}});
            (in_g1 ? compute_g1_params : compute_g2_params).runtime_varargs.push_back({core, {nc_slices_per_core}});
            writer_params.runtime_varargs.push_back(
                {core, {tensor_return_value.buffer()->address(), nc_slices_per_core, output_offset}});
            nc_slice_offset += nc_slices_per_core;
            output_offset += num_outputs_per_core;
        }
    } else {
        TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
        uint32_t num_cols_read = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord& core = cores[i];
            const bool in_g1 = core_group_1.contains(core);
            const uint32_t num_cols_per_core =
                in_g1 ? num_work_units_per_core_group_1 : num_work_units_per_core_group_2;
            reader_params.runtime_varargs.push_back(
                {core,
                 {tensor_arg.buffer()->address(),
                  (num_cols_read / Wt * HtWt) + (num_cols_read % Wt),
                  num_cols_read % Wt,
                  num_cols_per_core}});
            (in_g1 ? compute_g1_params : compute_g2_params).runtime_varargs.push_back({core, {num_cols_per_core}});
            writer_params.runtime_varargs.push_back(
                {core, {tensor_return_value.buffer()->address(), num_cols_per_core, num_cols_read}});
            num_cols_read += num_cols_per_core;
        }
    }

    metal2::ProgramRunParams params{.kernel_run_params = {reader_params, writer_params, compute_g1_params}};
    if (compute_group_2.has_value()) {
        params.kernel_run_params.push_back(compute_g2_params);
    }

    return {.spec = std::move(spec), .params = std::move(params)};
}

}  // namespace welford_reduce_metal2

WelfordReduceDeviceOperation::WelfordReduceProgramFactory::cached_program_t
WelfordReduceDeviceOperation::WelfordReduceProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto spec_and_params = welford_reduce_metal2::build_welford_program_spec_and_params(
        operation_attributes, tensor_args, tensor_return_value);
    auto program = tt::tt_metal::experimental::metal2_host_api::MakeProgramFromSpec(spec_and_params.spec);
    tt::tt_metal::experimental::metal2_host_api::SetProgramRunParameters(program, spec_and_params.params);
    return cached_program_t{std::move(program), shared_variables_t{}};
}

void WelfordReduceDeviceOperation::WelfordReduceProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto spec_and_params = welford_reduce_metal2::build_welford_program_spec_and_params(
        operation_attributes, tensor_args, tensor_return_value);
    tt::tt_metal::experimental::metal2_host_api::SetProgramRunParameters(
        cached_program.program, spec_and_params.params);
}

}  // namespace ttnn::prim
