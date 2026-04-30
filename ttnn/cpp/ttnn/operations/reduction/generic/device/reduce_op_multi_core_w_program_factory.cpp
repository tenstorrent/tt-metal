// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <bit>
#include <cmath>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ttnn::prim {
namespace reduce_multi_core_w_metal2 {

namespace metal2 = tt::tt_metal::experimental::metal2_host_api;

constexpr const char* kProgramName = "reduce_multi_core_w";
constexpr const char* kReader = "reader";
constexpr const char* kWriter = "writer";
constexpr const char* kComputeGroup1 = "compute_group_1";
constexpr const char* kComputeGroup2 = "compute_group_2";

// Keep DFB creation order aligned with the legacy CB indices used by the existing kernels.
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
    TT_FATAL(args.size() == 2, "Reduce multi-core W Metal 2.0 POC only supports interleaved tensor accessors");
    return {
        {prefix + "_args_config", args[0]},
        {prefix + "_page_size", args[1]},
    };
}

void append_named_args(
    std::vector<std::pair<std::string, uint32_t>>& dst, std::vector<std::pair<std::string, uint32_t>> src) {
    dst.insert(dst.end(), std::make_move_iterator(src.begin()), std::make_move_iterator(src.end()));
}

metal2::DataMovementConfiguration make_dm_config(tt::tt_metal::DataMovementProcessor processor, tt::tt_metal::NOC noc) {
    return metal2::DataMovementConfiguration{
        .gen1_data_movement_config =
            metal2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = processor,
                .noc = noc,
            },
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
    return metal2::KernelSpec::DFBBinding{
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
    return metal2::DataflowBufferSpec{
        .unique_id = name,
        .target_nodes = nodes,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = data_format,
        .disable_implicit_sync = true,
    };
}

struct ReduceWProgramSpecAndParams {
    metal2::ProgramSpec spec;
    metal2::ProgramRunParams params;
};

ReduceWProgramSpecAndParams build_reduce_w_program_spec_and_params(
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

    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;

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

    tt_metal::IDevice* device = input.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const auto num_rows = NC * Ht;

    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_rows_per_core_group_1 = 0;
    uint32_t num_rows_per_core_group_2 = 0;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_rows);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);
    }

    const uint32_t output_cb_index = tt::CBIndex::c_3;
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    const uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::W);
    reduce_defines["REDUCE_METAL2_NAMED_ARGS"] = "1";
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }

    const auto define_vector = to_define_vector(reduce_defines);

    std::vector<std::pair<std::string, uint32_t>> reader_ct_args = {
        {"scaler_bits", std::bit_cast<uint32_t>(operation_attributes.scaler)},
    };
    append_named_args(reader_ct_args, make_tensor_accessor_named_args("src", *input.buffer()));

    metal2::KernelSpec reader{
        .unique_id = kReader,
        .source =
            metal2::KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
                                               "reader_unary_reduce_universal_start_id.cpp"},
        .target_nodes = all_cores,
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

    std::vector<std::pair<std::string, uint32_t>> writer_ct_args = {
        {"output_cb_index", output_cb_index},
    };
    append_named_args(writer_ct_args, make_tensor_accessor_named_args("dst", *output.buffer()));

    metal2::KernelSpec writer{
        .unique_id = kWriter,
        .source = metal2::KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                                     "writer_unary_interleaved_start_id.cpp"},
        .target_nodes = all_cores,
        .compiler_options = {.defines = define_vector},
        .dfb_bindings = {dfb_binding(kOutputDfb, "output_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER)},
        .compile_time_arg_bindings = writer_ct_args,
        .runtime_arguments_schema = {.num_runtime_varargs = 3},
        .config_spec = make_dm_config(
            DataMovementProcessor::RISCV_0, tt::tt_metal::detail::preferred_noc_for_dram_write(input.device()->arch())),
    };

    const std::string compute_kernel =
        std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce") +
        (operation_attributes.negate ? "_w_neg" : "") + ".cpp";

    auto make_compute = [&](const char* name, const CoreRangeSet& cores, uint32_t rows_per_core) {
        std::vector<metal2::KernelSpec::DFBBinding> dfb_bindings = {
            dfb_binding(kInputDfb, "input_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER),
            dfb_binding(kUnusedDfb, "unused_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER),
            dfb_binding(kScalerDfb, "scaler_dfb", metal2::KernelSpec::DFBEndpointType::CONSUMER),
            dfb_binding(kOutputDfb, "output_dfb", metal2::KernelSpec::DFBEndpointType::PRODUCER),
        };
        if (operation_attributes.negate) {
            dfb_bindings.push_back(dfb_binding(kAccDfb, "acc_in", metal2::KernelSpec::DFBEndpointType::CONSUMER));
            dfb_bindings.push_back(dfb_binding(kAccDfb, "acc_out", metal2::KernelSpec::DFBEndpointType::PRODUCER));
            dfb_bindings.push_back(dfb_binding(kINegDfb, "ineg_in", metal2::KernelSpec::DFBEndpointType::CONSUMER));
            dfb_bindings.push_back(dfb_binding(kINegDfb, "ineg_out", metal2::KernelSpec::DFBEndpointType::PRODUCER));
        }

        return metal2::KernelSpec{
            .unique_id = name,
            .source = metal2::KernelSpec::SourceFilePath{compute_kernel},
            .target_nodes = cores,
            .compiler_options = {.defines = define_vector},
            .dfb_bindings = dfb_bindings,
            .compile_time_arg_bindings =
                {
                    {"Ht", rows_per_core},
                    {"Wt", Wt},
                    {"NC", 1},
                    {"post_mul_scaler_bits", post_mul_scaler_bits},
                },
            .config_spec = make_compute_config(math_fidelity, fp32_dest_acc_en, dst_full_sync_en),
        };
    };

    metal2::KernelSpec compute_group_1 = make_compute(kComputeGroup1, core_group_1, num_rows_per_core_group_1);
    std::optional<metal2::KernelSpec> compute_group_2;
    if (!core_group_2.ranges().empty()) {
        // The existing kernels address CB/DFB resources by numeric id. The DFB endpoint bindings above establish the
        // shared DFB configs once; group 2 intentionally shares those logical DFB ids.
        compute_group_2 = make_compute(kComputeGroup2, core_group_2, num_rows_per_core_group_2);
        compute_group_2->dfb_bindings.clear();
    }

    metal2::ProgramSpec spec{
        .program_id = kProgramName,
        .kernels = {reader, writer, compute_group_1},
        .dataflow_buffers =
            {
                make_dfb(kInputDfb, all_cores, src0_single_tile_size, 2, src0_cb_data_format),
                make_dfb(kUnusedDfb, all_cores, src0_single_tile_size, 1, src0_cb_data_format),
                make_dfb(kScalerDfb, all_cores, scaler_single_tile_size, 1, scaler_cb_data_format),
                make_dfb(kOutputDfb, all_cores, dst_single_tile_size, 2, dst_cb_data_format),
            },
        .workers =
            std::vector<metal2::WorkerSpec>{
                metal2::WorkerSpec{
                    .unique_id = "worker_group_1",
                    .kernels = {kReader, kWriter, kComputeGroup1},
                    .dataflow_buffers = {kInputDfb, kUnusedDfb, kScalerDfb, kOutputDfb},
                    .target_nodes = core_group_1,
                },
            },
    };

    if (operation_attributes.negate) {
        spec.dataflow_buffers.push_back(make_dfb(kAccDfb, all_cores, dst_single_tile_size, 1, dst_cb_data_format));
        spec.dataflow_buffers.push_back(make_dfb(kINegDfb, all_cores, dst_single_tile_size, 1, dst_cb_data_format));
        spec.workers->front().dataflow_buffers.push_back(kAccDfb);
        spec.workers->front().dataflow_buffers.push_back(kINegDfb);
    }

    if (compute_group_2.has_value()) {
        spec.kernels.push_back(*compute_group_2);
        std::vector<metal2::DFBSpecName> dfbs = {kInputDfb, kUnusedDfb, kScalerDfb, kOutputDfb};
        if (operation_attributes.negate) {
            dfbs.push_back(kAccDfb);
            dfbs.push_back(kINegDfb);
        }
        spec.workers->push_back(metal2::WorkerSpec{
            .unique_id = "worker_group_2",
            .kernels = {kReader, kWriter, kComputeGroup2},
            .dataflow_buffers = dfbs,
            .target_nodes = core_group_2,
        });
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

    TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
    const uint32_t out_dim_divider = Wt;

    metal2::ProgramRunParams::KernelRunParams reader_params{.kernel_spec_name = kReader};
    metal2::ProgramRunParams::KernelRunParams writer_params{.kernel_spec_name = kWriter};

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
        const CoreCoord& core = cores[i];
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        const uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;
        reader_params.runtime_varargs.push_back(
            {core, {input.buffer()->address(), num_tensor_tiles_per_core, num_tiles_read}});
        writer_params.runtime_varargs.push_back(
            {core,
             {
                 output.buffer()->address(),
                 num_tensor_tiles_per_core / out_dim_divider,
                 num_tiles_read / out_dim_divider,
             }});
        num_tiles_read += num_tensor_tiles_per_core;
    }

    metal2::ProgramRunParams params{
        .kernel_run_params =
            {
                reader_params,
                writer_params,
                metal2::ProgramRunParams::KernelRunParams{.kernel_spec_name = kComputeGroup1},
            },
    };
    if (compute_group_2.has_value()) {
        params.kernel_run_params.push_back(
            metal2::ProgramRunParams::KernelRunParams{.kernel_spec_name = kComputeGroup2});
    }

    return {.spec = std::move(spec), .params = std::move(params)};
}

}  // namespace reduce_multi_core_w_metal2

ReduceDeviceOperation::ReduceMultiCoreWProgramFactory::cached_program_t
ReduceDeviceOperation::ReduceMultiCoreWProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto spec_and_params = reduce_multi_core_w_metal2::build_reduce_w_program_spec_and_params(
        operation_attributes, tensor_args, tensor_return_value);
    auto program = tt::tt_metal::experimental::metal2_host_api::MakeProgramFromSpec(spec_and_params.spec);
    tt::tt_metal::experimental::metal2_host_api::SetProgramRunParameters(program, spec_and_params.params);
    return cached_program_t{std::move(program), shared_variables_t{}};
}

void ReduceDeviceOperation::ReduceMultiCoreWProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto spec_and_params = reduce_multi_core_w_metal2::build_reduce_w_program_spec_and_params(
        operation_attributes, tensor_args, tensor_return_value);
    tt::tt_metal::experimental::metal2_host_api::SetProgramRunParameters(
        cached_program.program, spec_and_params.params);
}

}  // namespace ttnn::prim
