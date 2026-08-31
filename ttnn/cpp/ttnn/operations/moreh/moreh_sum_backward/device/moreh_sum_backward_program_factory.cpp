// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "moreh_sum_backward_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_sum_backward {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

void get_tensor_dim(ttsl::SmallVector<uint32_t>& dim, const ttnn::Shape& padded_shape) {
    const auto rank = padded_shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;

        // last 2-dim
        if (idx == rank - 1 || idx == rank - 2) {
            dim[i] = padded_shape[idx] / tt::constants::TILE_HEIGHT;
        } else {
            dim[i] = padded_shape[idx];
        }
    }

    log_debug(tt::LogOp, "rank {}", rank);
    for (auto i = 0; i < rank; ++i) {
        log_debug(tt::LogOp, "dim[{}] = {}", i, dim[i]);
    }
}

std::pair<ttnn::Shape, ttnn::Shape> get_output_grad_shape(
    const Tensor& output_grad, const Tensor& input_grad, const ttsl::SmallVector<int64_t>& dims, const bool& keepdim) {
    if (keepdim) {
        return {output_grad.logical_shape(), output_grad.padded_shape()};
    }

    auto logical_shape = input_grad.logical_shape();
    auto padded_shape = input_grad.padded_shape();
    auto rank = logical_shape.rank();
    for (auto dim : dims) {
        TT_FATAL(dim < rank, "dim {} < rank {}", dim, rank);
        bool is_tile_dim = (dim == rank - 1 || dim == rank - 2);
        logical_shape[dim] = 1;
        if (is_tile_dim) {
            padded_shape[dim] = tt::constants::TILE_HEIGHT;
        } else {
            padded_shape[dim] = 1;
        }
    }

    return {logical_shape, padded_shape};
}

namespace {

// Resource names for the Metal 2.0 spec.
const DFBSpecName C0_IN{"c0_in"};      // legacy c_0 (input)
const DFBSpecName C1_ZERO{"c1_zero"};  // legacy c_1 (zero tile)
const DFBSpecName C16_OUT{"c16_out"};  // legacy c_16 (output)
const TensorParamName OUTPUT_GRAD{"output_grad"};
const TensorParamName INPUT_GRAD{"input_grad"};
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE_1{"compute_group_1"};
const KernelSpecName COMPUTE_2{"compute_group_2"};

constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_sum_backward/device/kernels/reader_moreh_sum_backward.cpp";
constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_sum_backward/device/kernels/writer_moreh_sum_backward.cpp";
constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_sum_backward/device/kernels/moreh_sum_backward.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts MorehSumBackwardOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input_grad = output_tensor;

    const auto& dims = operation_attributes.dims;
    auto keepdim = operation_attributes.keepdim;
    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = output_grad.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    const uint32_t cb_tile_size = tile_size(cb_data_format);

    const auto& input_grad_shape = input_grad.padded_shape();
    const auto& input_grad_shape_wo_padding = input_grad.logical_shape();
    const uint32_t input_grad_rank = input_grad_shape.rank();

    ttsl::SmallVector<uint32_t> input_grad_dim(input_grad_rank, 1);
    log_debug(tt::LogOp, "input_grad");
    get_tensor_dim(input_grad_dim, input_grad_shape);
    const auto [output_grad_shape_wo_padding, output_grad_shape] =
        get_output_grad_shape(output_grad, input_grad, dims, keepdim);

    ttsl::SmallVector<uint32_t> output_grad_dim(input_grad_rank, 1);
    log_debug(tt::LogOp, "output_grad");
    get_tensor_dim(output_grad_dim, output_grad_shape);

    ttsl::SmallVector<uint32_t> need_bcast_dim(input_grad_rank, 0);
    for (auto i = 0; i < input_grad_rank; ++i) {
        auto idx = input_grad_rank - 1 - i;
        bool is_tile_dim = (idx == input_grad_rank - 1 || idx == input_grad_rank - 2);

        if (is_tile_dim) {
            need_bcast_dim[i] = (output_grad_shape_wo_padding[idx] != input_grad_shape_wo_padding[idx]);
        } else {
            need_bcast_dim[i] = (output_grad_shape[idx] != input_grad_shape[idx]);
        }
    }
    const auto num_input_grad_tiles = input_grad.physical_volume() / tt::constants::TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(output_grad.device()->arch(), compute_kernel_config);

    for (auto i = 0; i < input_grad_rank; ++i) {
        log_debug(tt::LogOp, "need_bcast_dim [{}] = {}", i, need_bcast_dim[i]);
    }
    log_debug(tt::LogOp, "num_input_grad_tiles {}", num_input_grad_tiles);
    log_debug(
        tt::LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            split_work_to_cores(grid, num_input_grad_tiles);
    bool has_core_group_2 = !core_group_2.ranges().empty();

    ////////////////////////////////////////////////////////////////////////////
    //                         ProgramSpec
    ////////////////////////////////////////////////////////////////////////////
    ProgramSpec spec;
    spec.name = "moreh_sum_backward";

    spec.tensor_parameters = {
        TensorParameter{.unique_id = OUTPUT_GRAD, .spec = output_grad.tensor_spec()},
        TensorParameter{.unique_id = INPUT_GRAD, .spec = input_grad.tensor_spec()},
    };

    // DataflowBuffers (formerly the c_0 / c_1 / c_16 buffers). All three are compute-bound, so each
    // carries its data-format metadata. Sizes taken verbatim from the legacy buffer total sizes.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = C0_IN,
        .entry_size = cb_tile_size,
        .num_entries = 2,
        .data_format_metadata = cb_data_format,
    });  // input
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = C1_ZERO,
        .entry_size = cb_tile_size,
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    });  // zero
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = C16_OUT,
        .entry_size = cb_tile_size,
        .num_entries = 2,
        .data_format_metadata = cb_data_format,
    });  // output

    ////////////////////////////////////////////////////////////////////////////
    //                      Compute hardware config
    ////////////////////////////////////////////////////////////////////////////
    // Style B: the legacy factory built a Metal ComputeConfigDescriptor directly from the
    // resolved compute-kernel-config scalars. Reproduce those exact values on ComputeGen1Config,
    // minding the two non-1:1 transforms (math_approx_mode bool->Precision; dst_full_sync_en
    // inverted into double_buffer_dest).
    ComputeHardwareConfig compute_cfg{
        .fpu_math_fidelity = math_fidelity,
        .sfpu_precision_mode = math_approx_mode ? Precision::Approximate : Precision::Precise,
        .enable_32_bit_dest = fp32_dest_acc_en,
        .double_buffer_dest = !dst_full_sync_en,
    };
    // Legacy set no unpack_to_dest_mode (all CBs defaulted to UnpackToSrc). The Metal 2.0 validator
    // requires an explicit unpack_modes entry whenever a compute kernel consumes a Float32 DFB with
    // enable_32_bit_dest = true. The compute kernel consumes c0_in and c1_zero; add explicit
    // UnpackToSrc entries in that case, faithful to the legacy default.
    if (cb_data_format == tt::DataFormat::Float32 && fp32_dest_acc_en) {
        compute_cfg.unpack_modes.emplace(C0_IN, UnpackMode::UnpackToSrc);
        compute_cfg.unpack_modes.emplace(C1_ZERO, UnpackMode::UnpackToSrc);
    }

    KernelSpec::CompilerOptions::Defines compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      KernelSpecs
    ////////////////////////////////////////////////////////////////////////////
    KernelSpec reader{
        .unique_id = READER,
        .source = READER_KERNEL_PATH,
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = C0_IN, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = C1_ZERO, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_GRAD, .accessor_name = "output_grad"}},
        .compile_time_args = {{"input_grad_rank", input_grad_rank}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_output_tiles", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    // The three variable-length per-dim RTA blocks (output_grad_dim, input_grad_dim, need_bcast_dim,
    // each of length input_grad_rank) are passed as positional runtime varargs (read kernel-side via
    // get_vararg). Count is fixed per program instantiation.
    reader.advanced_options.num_runtime_varargs = 3 * input_grad_rank;

    KernelSpec writer{
        .unique_id = WRITER,
        .source = WRITER_KERNEL_PATH,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = C16_OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_GRAD, .accessor_name = "input_grad"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    auto make_compute = [&](const KernelSpecName& id, uint32_t num_output_tiles) {
        return KernelSpec{
            .unique_id = id,
            .source = COMPUTE_KERNEL_PATH,
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = C0_IN, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = C1_ZERO, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = C16_OUT, .accessor_name = "out0", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"num_output_tiles", num_output_tiles},
                 {"wt_need_bcast", need_bcast_dim[0]},
                 {"ht_need_bcast", need_bcast_dim[1]}},
            .hw_config = compute_cfg,
        };
    };
    KernelSpec compute_1 = make_compute(COMPUTE_1, num_cols_per_core_group_1);

    spec.kernels = {reader, writer, compute_1};
    if (has_core_group_2) {
        spec.kernels.push_back(make_compute(COMPUTE_2, num_cols_per_core_group_2));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      WorkUnitSpecs
    ////////////////////////////////////////////////////////////////////////////
    // Each work unit co-locates the reader, writer, and that group's compute kernel on its node set,
    // so every DFB's producer and consumer land together per node (matches the proven moreh_group_norm
    // layout). reader/writer are members of both work units; their effective placement is the union
    // core_group_1 union core_group_2 == all_cores.
    spec.work_units.push_back(
        WorkUnitSpec{.name = "group1", .kernels = {READER, WRITER, COMPUTE_1}, .target_nodes = core_group_1});
    if (has_core_group_2) {
        spec.work_units.push_back(
            WorkUnitSpec{.name = "group2", .kernels = {READER, WRITER, COMPUTE_2}, .target_nodes = core_group_2});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      ProgramRunArgs
    ////////////////////////////////////////////////////////////////////////////
    // The per-dim vararg block is identical across cores (does not depend on the core), built once.
    std::vector<uint32_t> reader_varargs;
    reader_varargs.reserve(3 * input_grad_rank);
    reader_varargs.insert(reader_varargs.end(), output_grad_dim.begin(), output_grad_dim.end());
    reader_varargs.insert(reader_varargs.end(), input_grad_dim.begin(), input_grad_dim.end());
    reader_varargs.insert(reader_varargs.end(), need_bcast_dim.begin(), need_bcast_dim.end());

    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values, core, {{"num_output_tiles", num_tiles_per_core}, {"start_id", tile_offset}});
        reader_run.advanced_options.runtime_varargs.insert({core, reader_varargs});

        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values, core, {{"num_tiles", num_tiles_per_core}, {"start_id", tile_offset}});

        tile_offset += num_tiles_per_core;
    }

    // The compute kernels take only compile-time args (no RTAs/CRTAs), so they get no KernelRunArgs
    // entry — matching the proven moreh_group_norm factory, which likewise omits run-args for its
    // arg-less compute kernels.
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args.emplace(OUTPUT_GRAD, TensorArgument{output_grad.mesh_tensor()});
    run_args.tensor_args.emplace(INPUT_GRAD, TensorArgument{input_grad.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_sum_backward
