// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include "moreh_mean_backward_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::tt_metal;
using ttnn::Tensor;
using namespace tt::tt_metal::experimental;

void get_tensor_dim(ttsl::SmallVector<uint32_t>& dim, const ttnn::Shape& shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;

        // last 2-dim
        if (idx == rank - 1 || idx == rank - 2) {
            dim[i] = (shape[idx] + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
        } else {
            dim[i] = shape[idx];
        }
    }
}

ttnn::Shape get_output_grad_shape(
    const Tensor& output_grad, const Tensor& input_grad, const ttsl::SmallVector<int64_t>& dims, const bool& keepdim) {
    if (keepdim) {
        return output_grad.logical_shape();
    }

    auto shape = input_grad.logical_shape();
    auto rank = shape.rank();
    for (auto dim : dims) {
        TT_FATAL(dim < rank, "dim {} < rank {}", dim, rank);
        shape[dim] = 1;
    }

    return shape;
}

namespace ttnn::operations::moreh::moreh_mean_backward {

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_mean_backward/device/kernels/reader_moreh_mean_backward.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_mean_backward/device/kernels/writer_moreh_mean_backward.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_mean_backward/device/kernels/moreh_mean_backward.cpp";

namespace {
// Metal 2.0 named resource handles for the moreh_mean_backward ProgramSpec.
const DFBSpecName IN_DFB{"in"};              // c_0  input (output_grad tile)
const DFBSpecName ZERO_DFB{"zero"};          // c_1  zero tile
const DFBSpecName SCALAR_DFB{"scalar"};      // c_2  1/num_dim bcast-scalar operand
const DFBSpecName INTERMED_DFB{"intermed"};  // c_24 compute self-loop scratch
const DFBSpecName OUT_DFB{"out"};            // c_16 output (input_grad tile)

const TensorParamName OUTPUT_GRAD_PARAM{"output_grad"};  // input tensor
const TensorParamName INPUT_GRAD_PARAM{"input_grad"};    // output tensor

const KernelSpecName READER_KERNEL{"reader"};
const KernelSpecName WRITER_KERNEL{"writer"};
const KernelSpecName COMPUTE_KERNEL_G1{"compute_g1"};
const KernelSpecName COMPUTE_KERNEL_G2{"compute_g2"};
}  // namespace

ttnn::device_operation::ProgramArtifacts
MorehMeanBackwardOperation::MorehMeanBackwardProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input_grad = output;
    const auto& keepdim = operation_attributes.keepdim;
    const auto& dims = operation_attributes.dims;
    auto compute_kernel_config =
        init_device_compute_kernel_config(output_grad.device()->arch(), operation_attributes.compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = output_grad.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    const uint32_t cb_tile_size = tile_size(cb_data_format);

    const auto& input_grad_shape = input_grad.logical_shape();
    const uint32_t input_grad_rank = input_grad_shape.rank();

    ttsl::SmallVector<uint32_t> input_grad_dim(input_grad_rank, 1);
    get_tensor_dim(input_grad_dim, input_grad_shape);
    const auto& output_grad_shape = get_output_grad_shape(output_grad, input_grad, dims, keepdim);

    ttsl::SmallVector<uint32_t> output_grad_dim(input_grad_rank, 1);
    get_tensor_dim(output_grad_dim, output_grad_shape);

    ttsl::SmallVector<uint32_t> need_bcast_dim(input_grad_rank, 0);
    for (auto i = 0; i < input_grad_rank; ++i) {
        auto idx = input_grad_rank - 1 - i;
        need_bcast_dim[i] = (output_grad_shape[idx] != input_grad_shape[idx]);
    }
    const auto num_input_grad_tiles = input_grad.physical_volume() / tt::constants::TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(output_grad.device()->arch(), compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = split_work_to_cores(grid, num_input_grad_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    // One DataflowBuffer per legacy buffer. entry_size / num_entries fixed at spec construction.
    ////////////////////////////////////////////////////////////////////////////
    DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = cb_tile_size,
        .num_entries = 2,
        .data_format_metadata = cb_data_format,
    };  // input
    DataflowBufferSpec zero_dfb_spec{
        .unique_id = ZERO_DFB,
        .entry_size = cb_tile_size,
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    };  // zero
    DataflowBufferSpec scalar_dfb_spec{
        .unique_id = SCALAR_DFB,
        .entry_size = cb_tile_size,
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    };  // scalar
    DataflowBufferSpec intermed_dfb_spec{
        .unique_id = INTERMED_DFB,
        .entry_size = cb_tile_size,
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    };  // intermediate (compute self-loop)
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = cb_tile_size,
        .num_entries = 2,
        .data_format_metadata = cb_data_format,
    };  // output

    ////////////////////////////////////////////////////////////////////////////
    //                         Tensor parameters (both Case 1)
    // output_grad delivered by the framework binding (supersedes the interim
    // Buffer* RTA the legacy factory used for AdamW-style cache-hit re-patching).
    ////////////////////////////////////////////////////////////////////////////
    TensorParameter output_grad_param{.unique_id = OUTPUT_GRAD_PARAM, .spec = output_grad.tensor_spec()};
    TensorParameter input_grad_param{.unique_id = INPUT_GRAD_PARAM, .spec = input_grad.tensor_spec()};

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_KERNEL_PATH},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = ZERO_DFB, .accessor_name = "zero", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = SCALAR_DFB, .accessor_name = "scalar", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_GRAD_PARAM, .accessor_name = "output_grad"}},
        .compile_time_args = {{"input_grad_rank", input_grad_rank}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_output_tiles", "start_id", "num_dim"}},
        .hw_config = create_reader_datamovement_config(),
        .advanced_options = {.num_runtime_varargs = 3 * input_grad_rank},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_KERNEL_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_GRAD_PARAM, .accessor_name = "input_grad"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = create_writer_datamovement_config(),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    // One KernelSpec per legacy compute KernelDescriptor (per core group),
    // preserving the work-split multiplicity. INTERMED is a self-loop binding
    // (PRODUCER + CONSUMER on the same compute KernelSpec).
    ////////////////////////////////////////////////////////////////////////////
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines.insert({"FP32_DEST_ACC_EN", "1"});
    }
    const auto compute_hw_config = to_compute_hardware_config(compute_kernel_config);

    auto make_compute_spec = [&](const KernelSpecName& unique_id, uint32_t num_output_tiles) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = std::filesystem::path{COMPUTE_KERNEL_PATH},
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = ZERO_DFB, .accessor_name = "zero", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = SCALAR_DFB,
                     .accessor_name = "scalar",
                     .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = INTERMED_DFB,
                     .accessor_name = "intermed",
                     .endpoint_type = DFBEndpointType::PRODUCER},
                 DFBBinding{
                     .dfb_spec_name = INTERMED_DFB,
                     .accessor_name = "intermed",
                     .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"num_output_tiles", num_output_tiles},
                 {"wt_need_bcast", need_bcast_dim[0]},
                 {"ht_need_bcast", need_bcast_dim[1]}},
            .hw_config = compute_hw_config,
        };
    };

    KernelSpec compute_spec_g1 = make_compute_spec(COMPUTE_KERNEL_G1, num_cols_per_core_group_1);
    const bool has_core_group_2 = !core_group_2.ranges().empty();
    KernelSpec compute_spec_g2 =
        has_core_group_2 ? make_compute_spec(COMPUTE_KERNEL_G2, num_cols_per_core_group_2) : KernelSpec{};

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    uint32_t num_dim = 1;
    for (auto dim : dims) {
        num_dim *= input_grad_shape[dim];
    }

    // The reader's three per-dimension blocks (output_grad_dim / input_grad_dim /
    // need_bcast_dim) are a CTA-bounded, per-instantiation-varying count → ported
    // as runtime varargs (concatenated in that order), not named RTAs.
    std::vector<uint32_t> reader_varargs;
    reader_varargs.reserve(3 * input_grad_rank);
    for (uint32_t v : output_grad_dim) {
        reader_varargs.push_back(v);
    }
    for (uint32_t v : input_grad_dim) {
        reader_varargs.push_back(v);
    }
    for (uint32_t v : need_bcast_dim) {
        reader_varargs.push_back(v);
    }

    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        NodeCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"num_output_tiles", num_tiles_per_core}, {"start_id", tile_offset}, {"num_dim", num_dim}});
        reader_run.advanced_options.runtime_varargs[core] = reader_varargs;

        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values, core, {{"num_tiles", num_tiles_per_core}, {"start_id", tile_offset}});

        tile_offset += num_tiles_per_core;
    }

    // Compute kernels carry no runtime args (CTAs only), so they get NO KernelRunArgs
    // entry — matching the proven moreh_group_norm port (a kernel with an empty runtime
    // schema is omitted from kernel_run_args entirely).

    ////////////////////////////////////////////////////////////////////////////
    //                      Work units: one per (non-empty) core group.
    // Reader/writer are members of both WUs (single KernelSpec each); their
    // effective node set is core_group_1 ∪ core_group_2 = all_cores.
    ////////////////////////////////////////////////////////////////////////////
    WorkUnitSpec wu_g1{
        .name = "moreh_mean_backward_g1",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G1},
        .target_nodes = core_group_1,
    };

    ProgramSpec spec{
        .name = "moreh_mean_backward",
        .kernels = {reader_spec, writer_spec, compute_spec_g1},
        .dataflow_buffers = {in_dfb_spec, zero_dfb_spec, scalar_dfb_spec, intermed_dfb_spec, out_dfb_spec},
        .tensor_parameters = {output_grad_param, input_grad_param},
        .work_units = {wu_g1},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args.insert({OUTPUT_GRAD_PARAM, output_grad.mesh_tensor()});
    run_args.tensor_args.insert({INPUT_GRAD_PARAM, input_grad.mesh_tensor()});

    if (has_core_group_2) {
        spec.kernels.push_back(compute_spec_g2);
        spec.work_units.push_back(WorkUnitSpec{
            .name = "moreh_mean_backward_g2",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G2},
            .target_nodes = core_group_2,
        });
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_mean_backward
