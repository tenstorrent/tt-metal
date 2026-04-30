// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <bit>
#include <cmath>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ttnn::prim {
namespace reduce_single_core_hw_metal2 {

namespace metal2 = tt::tt_metal::experimental::metal2_host_api;

constexpr const char* kProgramName = "reduce_single_core_hw";
constexpr const char* kReader = "reader";
constexpr const char* kWriter = "writer";
constexpr const char* kCompute = "compute";

constexpr const char* kInputDfb = "input_dfb_c0";
constexpr const char* kUnusedDfb = "unused_dfb_c1";
constexpr const char* kScalerDfb = "scaler_dfb_c2";
constexpr const char* kOutputDfb = "output_dfb_c3";
constexpr const char* kAccDfb = "acc_dfb_c4";
constexpr const char* kINegDfb = "ineg_dfb_c5";

std::vector<std::pair<std::string, std::string>> to_define_vector(const std::map<std::string, std::string>& defines) {
    return {defines.begin(), defines.end()};
}

std::vector<std::pair<std::string, uint32_t>> make_tensor_accessor_named_args(
    const std::string& prefix, const tt::tt_metal::Buffer& buffer) {
    auto args = tt::tt_metal::TensorAccessorArgs(buffer).get_compile_time_args();
    TT_FATAL(args.size() == 2, "Reduce single-core HW Metal 2.0 path only supports interleaved tensor accessors");
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

metal2::ComputeConfiguration make_compute_config(
    tt::tt_metal::MathFidelity math_fidelity, bool fp32_dest_acc_en, bool dst_full_sync_en) {
    return metal2::ComputeConfiguration{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
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

struct ReduceHwProgramSpecAndParams {
    metal2::ProgramSpec spec;
    metal2::ProgramRunParams params;
};

ReduceHwProgramSpecAndParams build_reduce_hw_program_spec_and_params(
    const ReduceDeviceOperation::operation_attributes_t& operation_attributes,
    const ReduceDeviceOperation::tensor_args_t& tensor_args,
    ReduceDeviceOperation::tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args;
    auto& output = tensor_return_value;
    const auto& shape = input.padded_shape();
    const uint32_t W = shape[3];
    const uint32_t H = shape[2];
    const uint32_t NC = shape[1] * shape[0];
    const uint32_t tile_height = input.tensor_spec().tile().get_height();
    const uint32_t tile_width = input.tensor_spec().tile().get_width();
    const uint32_t tile_hw = input.tensor_spec().tile().get_tile_hw();

    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;
    TT_FATAL(operation_attributes.scaler >= 0, "Scalar must be non-negative");
    const float scaler = std::sqrt(operation_attributes.scaler);
    const uint32_t num_tensor_tiles = NC * H * W / tile_hw;

    CoreCoord selected_core_coord = {0, 0};
    if (operation_attributes.sub_core_grids.has_value() && !operation_attributes.sub_core_grids->ranges().empty()) {
        selected_core_coord = operation_attributes.sub_core_grids->ranges().front().start_coord;
    }
    const CoreRangeSet core_set(CoreRange(selected_core_coord, selected_core_coord));

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), operation_attributes.compute_kernel_config);
    (void)math_approx_mode;
    (void)packer_l1_acc;

    const tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    const tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    const tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    const uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, tt::tt_metal::ReduceOpDim::HW);
    reduce_defines["REDUCE_METAL2_NAMED_ARGS"] = "1";
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }
    const auto define_vector = to_define_vector(reduce_defines);

    std::vector<std::pair<std::string, uint32_t>> reader_ct_args = {
        {"scaler_bits", std::bit_cast<uint32_t>(scaler)},
    };
    append_named_args(reader_ct_args, make_tensor_accessor_named_args("src", *input.buffer()));

    metal2::KernelSpec reader{
        .unique_id = kReader,
        .source =
            metal2::KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
                                               "reader_unary_reduce_universal_start_id.cpp"},
        .target_nodes = core_set,
        .compiler_options = {.defines = define_vector},
        .dfb_bindings =
            {
                dfb_binding(kInputDfb, "input_dfb", metal2::KernelSpec::DFBEndpointType::PRODUCER),
                dfb_binding(kUnusedDfb, "unused_dfb", metal2::KernelSpec::DFBEndpointType::PRODUCER),
                dfb_binding(kScalerDfb, "scaler_dfb", metal2::KernelSpec::DFBEndpointType::PRODUCER),
            },
        .compile_time_arg_bindings = reader_ct_args,
        .runtime_arguments_schema = {.num_runtime_varargs = 3},
        .config_spec = make_dm_config(
            DataMovementProcessor::RISCV_1, tt::tt_metal::detail::preferred_noc_for_dram_read(input.device()->arch())),
    };

    std::vector<std::pair<std::string, uint32_t>> writer_ct_args = {{"output_cb_index", tt::CBIndex::c_3}};
    append_named_args(writer_ct_args, make_tensor_accessor_named_args("dst", *output.buffer()));

    metal2::KernelSpec writer{
        .unique_id = kWriter,
        .source = metal2::KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                                     "writer_unary_interleaved_start_id.cpp"},
        .target_nodes = core_set,
        .compiler_options = {.defines = define_vector},
        .dfb_bindings = {dfb_binding(kOutputDfb, "output_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER)},
        .compile_time_arg_bindings = writer_ct_args,
        .runtime_arguments_schema = {.num_runtime_varargs = 3},
        .config_spec = make_dm_config(
            DataMovementProcessor::RISCV_0, tt::tt_metal::detail::preferred_noc_for_dram_write(input.device()->arch())),
    };

    const std::string compute_kernel =
        std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce") +
        (operation_attributes.negate ? "_hw_neg" : "") + ".cpp";

    std::vector<metal2::KernelSpec::DFBBinding> compute_dfbs = {
        dfb_binding(kInputDfb, "input_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER),
        dfb_binding(kUnusedDfb, "unused_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER),
        dfb_binding(kScalerDfb, "scaler_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER),
        dfb_binding(kOutputDfb, "output_dfb", metal2::KernelSpec::DFBEndpointType::PRODUCER),
    };
    if (operation_attributes.negate) {
        compute_dfbs.push_back(dfb_binding(kAccDfb, "acc_in", metal2::KernelSpec::DFBEndpointType::CONSUMER));
        compute_dfbs.push_back(dfb_binding(kAccDfb, "acc_out", metal2::KernelSpec::DFBEndpointType::PRODUCER));
        compute_dfbs.push_back(dfb_binding(kINegDfb, "ineg_in", metal2::KernelSpec::DFBEndpointType::CONSUMER));
        compute_dfbs.push_back(dfb_binding(kINegDfb, "ineg_out", metal2::KernelSpec::DFBEndpointType::PRODUCER));
    }

    metal2::KernelSpec compute{
        .unique_id = kCompute,
        .source = metal2::KernelSpec::SourceFilePath{compute_kernel},
        .target_nodes = core_set,
        .compiler_options = {.defines = define_vector},
        .dfb_bindings = compute_dfbs,
        .compile_time_arg_bindings =
            {
                {"Ht", Ht},
                {"Wt", Wt},
                {"NC", NC},
                {"post_mul_scaler_bits", post_mul_scaler_bits},
            },
        .config_spec = make_compute_config(math_fidelity, fp32_dest_acc_en, dst_full_sync_en),
    };

    metal2::ProgramSpec spec{
        .program_id = kProgramName,
        .kernels = {reader, writer, compute},
        .dataflow_buffers =
            {
                make_dfb(kInputDfb, core_set, src0_single_tile_size, 2, src0_cb_data_format),
                make_dfb(kUnusedDfb, core_set, src0_single_tile_size, 1, src0_cb_data_format),
                make_dfb(kScalerDfb, core_set, scaler_single_tile_size, 1, scaler_cb_data_format),
                make_dfb(kOutputDfb, core_set, dst_single_tile_size, 2, dst_cb_data_format),
            },
        .workers =
            std::vector<metal2::WorkerSpec>{
                {.unique_id = "worker",
                 .kernels = {kReader, kWriter, kCompute},
                 .dataflow_buffers = {kInputDfb, kUnusedDfb, kScalerDfb, kOutputDfb},
                 .target_nodes = core_set},
            },
    };

    if (operation_attributes.negate) {
        spec.dataflow_buffers.push_back(make_dfb(kAccDfb, core_set, dst_single_tile_size, 1, dst_cb_data_format));
        spec.dataflow_buffers.push_back(make_dfb(kINegDfb, core_set, dst_single_tile_size, 1, dst_cb_data_format));
        spec.workers->front().dataflow_buffers.push_back(kAccDfb);
        spec.workers->front().dataflow_buffers.push_back(kINegDfb);
    }

    TT_FATAL(Ht != 0 && Wt != 0, "Height and width in tiles must be non-zero (Ht={}, Wt={}, H={}, W={})", Ht, Wt, H, W);
    const uint32_t out_dim_divider = Ht * Wt;

    metal2::ProgramRunParams params{
        .kernel_run_params =
            {
                {.kernel_spec_name = kReader,
                 .runtime_varargs = {{selected_core_coord, {input.buffer()->address(), num_tensor_tiles, 0}}}},
                {.kernel_spec_name = kWriter,
                 .runtime_varargs =
                     {{selected_core_coord, {output.buffer()->address(), num_tensor_tiles / out_dim_divider, 0}}}},
                {.kernel_spec_name = kCompute},
            },
    };

    return {.spec = std::move(spec), .params = std::move(params)};
}

}  // namespace reduce_single_core_hw_metal2

ReduceDeviceOperation::ReduceSingleCoreHwProgramFactory::cached_program_t
ReduceDeviceOperation::ReduceSingleCoreHwProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto spec_and_params = reduce_single_core_hw_metal2::build_reduce_hw_program_spec_and_params(
        operation_attributes, tensor_args, tensor_return_value);
    auto program = tt::tt_metal::experimental::metal2_host_api::MakeProgramFromSpec(spec_and_params.spec);
    tt::tt_metal::experimental::metal2_host_api::SetProgramRunParameters(program, spec_and_params.params);
    return cached_program_t{std::move(program), shared_variables_t{}};
}

void ReduceDeviceOperation::ReduceSingleCoreHwProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto spec_and_params = reduce_single_core_hw_metal2::build_reduce_hw_program_spec_and_params(
        operation_attributes, tensor_args, tensor_return_value);
    tt::tt_metal::experimental::metal2_host_api::SetProgramRunParameters(
        cached_program.program, spec_and_params.params);
}

}  // namespace ttnn::prim
