// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-core HW reduction program factory, Metal 2.0 host-API port.

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/metal2_artifacts.hpp"

#include <bit>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental::metal2_host_api;

namespace {

// Unity-build hygiene: `HW_` prefix.
constexpr const char* HW_READER = "reader";
constexpr const char* HW_WRITER = "writer";
constexpr const char* HW_COMPUTE = "compute";

constexpr const char* HW_WU_MAIN = "wu_main";

constexpr const char* HW_INPUT_DFB = "input";
constexpr const char* HW_SCALER_DFB = "scaler";
constexpr const char* HW_OUTPUT_DFB = "output";
constexpr const char* HW_ACC_DFB = "acc";
constexpr const char* HW_INEG_DFB = "ineg";

constexpr const char* HW_INPUT_TENSOR = "input";
constexpr const char* HW_OUTPUT_TENSOR = "output";

m2::KernelSpec::CompilerOptions::Defines hw_defines_from_map(const std::map<std::string, std::string>& src) {
    m2::KernelSpec::CompilerOptions::Defines out;
    out.reserve(src.size());
    for (const auto& [k, v] : src) {
        out.emplace_back(k, v);
    }
    return out;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts ReduceDeviceOperation::ReduceSingleCoreHwProgramFactory::create_program_spec(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& a = tensor_args;
    auto& output = tensor_return_value;
    const auto& shape = a.padded_shape();
    const uint32_t W = shape[3];
    const uint32_t H = shape[2];
    const uint32_t NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;
    TT_FATAL(Ht != 0 && Wt != 0, "Height and width in tiles must be non-zero (Ht={}, Wt={}, H={}, W={})", Ht, Wt, H, W);
    TT_FATAL(
        operation_attributes.dim == ReduceOpDim::HW,
        "ReduceSingleCoreHwProgramFactory supports HW dim only, got dim enum value {}",
        static_cast<int>(operation_attributes.dim));

    // The single-core HW path uses REDUCE_SCALAR mode, which applies the
    // scaler twice internally (once per dimension). Here we compensate with
    // sqrt(scaler). However, sqrt of a negative number is NaN, so negative scalers
    // must not reach this code path.
    TT_FATAL(operation_attributes.scaler >= 0, "Scalar must be non-negative");
    const float scaler = std::sqrt(operation_attributes.scaler);

    TT_FATAL(
        H % tile_height == 0 && W % tile_width == 0, "Reduce HW expects tile-aligned padded shape H={}, W={}", H, W);
    const uint32_t num_tensor_tiles = NC * H * W / tile_hw;
    const uint32_t num_tensor_tiles_ht_wt = NC * Ht * Wt;
    TT_FATAL(
        num_tensor_tiles == num_tensor_tiles_ht_wt,
        "Reduce HW tile count mismatch: tile_hw path={} vs Ht*Wt path={}",
        num_tensor_tiles,
        num_tensor_tiles_ht_wt);

    CoreCoord selected_core_coord = {0, 0};
    if (operation_attributes.sub_core_grids.has_value() && !operation_attributes.sub_core_grids->ranges().empty()) {
        const auto& r = operation_attributes.sub_core_grids->ranges().front();
        selected_core_coord = r.start_coord;
        TT_FATAL(
            operation_attributes.sub_core_grids->contains(selected_core_coord),
            "Selected core {} must be contained in provided sub_core_grids {}",
            selected_core_coord,
            *operation_attributes.sub_core_grids);
    }
    const CoreRange core_range(selected_core_coord, selected_core_coord);
    const CoreRangeSet core_set(core_range);

    const tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    const tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    const tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    const uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);
    const uint32_t scaler_bits = std::bit_cast<uint32_t>(scaler);

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::HW);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }

    // ----- DataflowBufferSpecs -----

    constexpr uint32_t kNumInputEntries = 2;
    constexpr uint32_t kNumScalerEntries = 1;
    constexpr uint32_t kNumOutputEntries = 2;
    constexpr uint32_t kNumNegateEntries = 1;

    std::vector<m2::DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = HW_INPUT_DFB,
        .entry_size = src0_single_tile_size,
        .num_entries = kNumInputEntries,
        .data_format_metadata = src0_cb_data_format,
        .tile_format_metadata = a.tensor_spec().tile(),
    });
    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = HW_SCALER_DFB,
        .entry_size = scaler_single_tile_size,
        .num_entries = kNumScalerEntries,
        .data_format_metadata = scaler_cb_data_format,
    });
    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = HW_OUTPUT_DFB,
        .entry_size = dst_single_tile_size,
        .num_entries = kNumOutputEntries,
        .data_format_metadata = dst_cb_data_format,
        .tile_format_metadata = output.tensor_spec().tile(),
    });
    if (operation_attributes.negate) {
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = HW_ACC_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = kNumNegateEntries,
            .data_format_metadata = dst_cb_data_format,
        });
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = HW_INEG_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = kNumNegateEntries,
            .data_format_metadata = dst_cb_data_format,
        });
    }

    // ----- KernelSpecs -----

    m2::KernelSpec reader{
        .unique_id = HW_READER,
        .source = m2::KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
                                                 "reader_unary_reduce_universal_start_id.cpp"},
        .compiler_options =
            {
                .defines = hw_defines_from_map(reduce_defines),
            },
        .dfb_bindings =
            {
                {.dfb_spec_name = HW_INPUT_DFB,
                 .local_accessor_name = "input",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
                {.dfb_spec_name = HW_SCALER_DFB,
                 .local_accessor_name = "scaler",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                {.tensor_parameter_name = HW_INPUT_TENSOR, .accessor_name = "input"},
            },
        .compile_time_arg_bindings = {{"scaler_bits", scaler_bits}},
        .runtime_arguments_schema = {.named_runtime_args = {"num_tiles", "start_id"}},
        .config_spec =
            m2::DataMovementConfiguration{
                .gen1_data_movement_config =
                    m2::DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                    },
            },
    };

    m2::KernelSpec writer{
        .unique_id = HW_WRITER,
        .source = m2::KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                                 "writer_unary_interleaved_start_id_metal2.cpp"},
        .compiler_options =
            {
                .defines = hw_defines_from_map(reduce_defines),
            },
        .dfb_bindings =
            {
                {.dfb_spec_name = HW_OUTPUT_DFB,
                 .local_accessor_name = "output",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                {.tensor_parameter_name = HW_OUTPUT_TENSOR, .accessor_name = "output"},
            },
        .runtime_arguments_schema = {.named_runtime_args = {"num_pages", "start_id"}},
        .config_spec =
            m2::DataMovementConfiguration{
                .gen1_data_movement_config =
                    m2::DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                    },
            },
    };

    const std::string compute_kernel_source =
        std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/") +
        (operation_attributes.negate ? "reduce_hw_neg.cpp" : "reduce.cpp");

    m2::KernelSpec compute{
        .unique_id = HW_COMPUTE,
        .source = m2::KernelSpec::SourceFilePath{compute_kernel_source},
        .compiler_options =
            {
                .defines = hw_defines_from_map(reduce_defines),
            },
        .dfb_bindings =
            {
                {.dfb_spec_name = HW_INPUT_DFB,
                 .local_accessor_name = "input",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
                {.dfb_spec_name = HW_SCALER_DFB,
                 .local_accessor_name = "scaler",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
                {.dfb_spec_name = HW_OUTPUT_DFB,
                 .local_accessor_name = "output",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
            },
        .compile_time_arg_bindings = {{"Ht", Ht}, {"Wt", Wt}, {"NC", NC}},
        .config_spec =
            m2::ComputeConfiguration{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
            },
    };
    if (use_post_mul) {
        compute.compile_time_arg_bindings.push_back({"post_mul_scaler_bits", post_mul_scaler_bits});
    }
    if (operation_attributes.negate) {
        compute.dfb_bindings.push_back(
            {.dfb_spec_name = HW_ACC_DFB,
             .local_accessor_name = "acc",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER});
        compute.dfb_bindings.push_back(
            {.dfb_spec_name = HW_ACC_DFB,
             .local_accessor_name = "acc",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER});
        compute.dfb_bindings.push_back(
            {.dfb_spec_name = HW_INEG_DFB,
             .local_accessor_name = "ineg",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER});
        compute.dfb_bindings.push_back(
            {.dfb_spec_name = HW_INEG_DFB,
             .local_accessor_name = "ineg",
             .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER});
    }

    // ----- WorkUnitSpecs -----

    std::vector<m2::WorkUnitSpec> work_units;
    work_units.push_back(m2::WorkUnitSpec{
        .unique_id = HW_WU_MAIN,
        .kernels = {HW_READER, HW_WRITER, HW_COMPUTE},
        .target_nodes = core_set,
    });

    // ----- ProgramSpec assembly -----

    std::vector<m2::KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    kernels.push_back(std::move(compute));

    m2::ProgramSpec spec{
        .program_id = "reduce_single_core_hw",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                {.unique_id = HW_INPUT_TENSOR, .spec = a.tensor_spec()},
                {.unique_id = HW_OUTPUT_TENSOR, .spec = output.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    // ----- ProgramRunParams -----

    m2::ProgramRunParams run_params;
    m2::ProgramRunParams::KernelRunParams reader_rp{.kernel_spec_name = HW_READER};
    m2::ProgramRunParams::KernelRunParams writer_rp{.kernel_spec_name = HW_WRITER};

    const uint32_t out_dim_divider = Ht * Wt;
    TT_FATAL(
        num_tensor_tiles % out_dim_divider == 0,
        "Reduce HW per-core input tiles {} must be divisible by Ht*Wt={}",
        num_tensor_tiles,
        out_dim_divider);

    reader_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
        .node = selected_core_coord,
        .args = {{"num_tiles", num_tensor_tiles}, {"start_id", 0u}},
    });
    writer_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
        .node = selected_core_coord,
        .args = {{"num_pages", num_tensor_tiles / out_dim_divider}, {"start_id", 0u}},
    });

    run_params.kernel_run_params = {std::move(reader_rp), std::move(writer_rp)};

    run_params.tensor_args = {
        m2::ProgramRunParams::TensorArg{.tensor_parameter_name = HW_INPUT_TENSOR, .tensor = std::cref(a.mesh_tensor())},
        m2::ProgramRunParams::TensorArg{
            .tensor_parameter_name = HW_OUTPUT_TENSOR, .tensor = std::cref(output.mesh_tensor())},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
