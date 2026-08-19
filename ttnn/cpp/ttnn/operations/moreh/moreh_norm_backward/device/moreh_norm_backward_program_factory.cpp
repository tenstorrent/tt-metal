// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "moreh_norm_backward_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_norm_backward {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;
using ttnn::device_operation::ProgramArtifacts;

std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p) {
    auto floored_p = std::floor(p);
    auto decimal = p - floored_p;
    bool p_is_negative = floored_p < 0.0f;
    if (p_is_negative) {
        floored_p = -floored_p;
    }
    return std::make_tuple(static_cast<uint32_t>(floored_p), decimal, p_is_negative);
}

void get_tensor_dim(ttsl::SmallVector<uint32_t>& dim, const ttnn::Shape& shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;
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

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward/device/kernels/reader_moreh_norm_backward.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward/device/kernels/writer_moreh_norm_backward.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward/device/kernels/moreh_norm_backward_kernel.cpp";

ProgramArtifacts MorehNormBackwardOperation::MorehNormBackwardProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& input_grad) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;
    const auto& output_grad = tensor_args.output_grad;
    const auto p = operation_attributes.p;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = output_grad.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto& input_grad_shape = input_grad.logical_shape();
    const auto input_grad_rank = input_grad_shape.rank();

    ttsl::SmallVector<uint32_t> input_grad_dim(input_grad_rank, 1);
    get_tensor_dim(input_grad_dim, input_grad_shape);
    auto output_grad_shape =
        get_output_grad_shape(output_grad, input_grad, operation_attributes.dims, operation_attributes.keepdim);

    ttsl::SmallVector<uint32_t> output_grad_dim(input_grad_rank, 1);
    get_tensor_dim(output_grad_dim, output_grad_shape);

    ttsl::SmallVector<uint32_t> need_bcast_dim(input_grad_rank, 0);
    for (auto i = 0; i < input_grad_rank; ++i) {
        auto idx = input_grad_rank - 1 - i;
        need_bcast_dim[i] = (output_grad_shape[idx] != input_grad_shape[idx]);
    }

    const auto num_input_grad_tiles = input_grad.physical_volume() / tt::constants::TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(output_grad.device()->arch(), operation_attributes.compute_kernel_config);

    auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);
    auto [floored_p_minus_one, decimal_minus_one, p_minus_one_is_negative] =
        get_floored_p_and_decimal_and_p_is_negative(p - 1.0f);

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

    const bool has_core_group_2 = !core_group_2.ranges().empty();

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;
    const uint32_t cb_tile_size = tile_size(cb_data_format);
    const uint32_t intermed_tile_size = tile_size(intermed_data_format);

    // Resource names (typed constants, referenced at each use site).
    const m2::TensorParamName T_INPUT{"input"};
    const m2::TensorParamName T_OUTPUT{"output"};
    const m2::TensorParamName T_OUTPUT_GRAD{"output_grad"};
    const m2::TensorParamName T_INPUT_GRAD{"input_grad"};

    const m2::DFBSpecName INPUT{"mnb_input"};                // CBIndex::c_0  input(==x)
    const m2::DFBSpecName OUTPUT{"mnb_output"};              // CBIndex::c_1  output(==y)
    const m2::DFBSpecName OUTPUT_GRAD{"mnb_output_grad"};    // CBIndex::c_2  output_grad(==dy)
    const m2::DFBSpecName DECIMAL{"mnb_decimal"};            // CBIndex::c_3  decimal
    const m2::DFBSpecName DX{"mnb_dx"};                      // CBIndex::c_16 input_grad(==dx)
    const m2::DFBSpecName XPOW{"mnb_xpow"};                  // CBIndex::c_24
    const m2::DFBSpecName LOGX{"mnb_logx"};                  // CBIndex::c_25
    const m2::DFBSpecName EXP_LXMD{"mnb_exp_lxmd"};          // CBIndex::c_26
    const m2::DFBSpecName CORRECT_XPOW{"mnb_correct_xpow"};  // CBIndex::c_27
    const m2::DFBSpecName TMP4{"mnb_tmp4"};                  // CBIndex::c_28
    const m2::DFBSpecName TMP5{"mnb_tmp5"};                  // CBIndex::c_29
    const m2::DFBSpecName RECIP_YPOW{"mnb_recip_ypow"};      // CBIndex::c_30
    const m2::DFBSpecName SIGN{"mnb_sign"};                  // CBIndex::c_31

    const m2::KernelSpecName READER{"mnb_reader"};
    const m2::KernelSpecName WRITER{"mnb_writer"};
    const m2::KernelSpecName COMPUTE_1{"mnb_compute_1"};
    const m2::KernelSpecName COMPUTE_2{"mnb_compute_2"};

    // input(==x), output(==y), output_grad(==dy), decimal — reader-produced, compute-consumed.
    auto io_dfb = [&](const m2::DFBSpecName& name) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = cb_tile_size,
            .num_entries = 1,
            .data_format_metadata = cb_data_format,
        };
    };
    // compute-only intermediate (self-loop).
    auto intermed_dfb = [&](const m2::DFBSpecName& name) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = intermed_tile_size,
            .num_entries = 1,
            .data_format_metadata = intermed_data_format,
        };
    };

    std::vector<m2::DataflowBufferSpec> dfbs = {
        io_dfb(INPUT),
        io_dfb(OUTPUT),
        io_dfb(OUTPUT_GRAD),
        io_dfb(DECIMAL),
        io_dfb(DX),
        intermed_dfb(XPOW),
        intermed_dfb(LOGX),
        intermed_dfb(EXP_LXMD),
        intermed_dfb(CORRECT_XPOW),
        intermed_dfb(TMP4),
        intermed_dfb(TMP5),
        intermed_dfb(RECIP_YPOW),
        intermed_dfb(SIGN),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path(READER_KERNEL_PATH),
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = INPUT, .accessor_name = "input", .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = OUTPUT, .accessor_name = "output", .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = OUTPUT_GRAD,
                 .accessor_name = "output_grad",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = DECIMAL, .accessor_name = "decimal", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {m2::TensorBinding{.tensor_parameter_name = T_INPUT, .accessor_name = "input"},
             m2::TensorBinding{.tensor_parameter_name = T_OUTPUT, .accessor_name = "output"},
             m2::TensorBinding{.tensor_parameter_name = T_OUTPUT_GRAD, .accessor_name = "output_grad"}},
        .compile_time_args = {{"input_grad_rank", static_cast<uint32_t>(input_grad_rank)}},
        .runtime_arg_schema = {.runtime_arg_names = {"decimal", "num_output_tiles", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };
    // The three per-dimension blocks (output_grad_dim, input_grad_dim, need_bcast_dim) are read as
    // runtime varargs (count = input_grad_rank each).
    reader_spec.advanced_options.num_runtime_varargs = 3u * static_cast<uint32_t>(input_grad_rank);

    m2::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path(WRITER_KERNEL_PATH),
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = DX, .accessor_name = "input_grad", .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = T_INPUT_GRAD, .accessor_name = "input_grad"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_input_tiles_per_core", "tile_offset"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec::CompilerOptions::Defines compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    // Style B: the legacy factory builds a Metal ComputeConfigDescriptor directly, setting only
    // math_fidelity / fp32_dest_acc_en / math_approx_mode; dst_full_sync_en is left at the Metal
    // default (false), so double_buffer_dest stays at its matching Gen1 default (true). We build a
    // ComputeGen1Config directly to preserve that (resolved dst_full_sync_en / packer_l1_acc are
    // unused, matching legacy).
    // Plain copies (structured-binding names captured by value to keep the lambda portable).
    const MathFidelity cfg_math_fidelity = math_fidelity;
    const bool cfg_math_approx_mode = math_approx_mode;
    const bool cfg_fp32_dest_acc_en = fp32_dest_acc_en;
    auto make_compute_hw = [&]() {
        m2::ComputeGen1Config cfg{
            .fpu_math_fidelity = cfg_math_fidelity,
            .sfpu_precision_mode = cfg_math_approx_mode ? Precision::Approximate : Precision::Precise,
            .enable_32_bit_dest = cfg_fp32_dest_acc_en,
        };
        // With enable_32_bit_dest, every Float32 DFB the compute kernel consumes needs an explicit
        // unpack_modes entry (legacy silently defaulted). Legacy Default -> UnpackToSrc.
        if (cfg_fp32_dest_acc_en) {
            // Intermediates are Float32 when fp32_dest_acc_en; compute self-loops (consumes) each.
            for (const auto& n : {XPOW, LOGX, EXP_LXMD, CORRECT_XPOW, TMP4, TMP5, RECIP_YPOW, SIGN}) {
                cfg.unpack_modes.emplace(n, UnpackMode::UnpackToSrc);
            }
            // I/O DFBs consumed by compute are Float32 only when cb_data_format is Float32.
            if (cb_data_format == tt::DataFormat::Float32) {
                for (const auto& n : {INPUT, OUTPUT, OUTPUT_GRAD, DECIMAL}) {
                    cfg.unpack_modes.emplace(n, UnpackMode::UnpackToSrc);
                }
            }
        }
        return m2::ComputeHardwareConfig{cfg};
    };

    // Compute DFB bindings: consume the four reader-produced inputs, produce dx, self-loop the 8
    // intermediates (one PRODUCER + one CONSUMER binding each).
    auto compute_dfb_bindings = [&]() {
        m2::Group<m2::DFBBinding> b;
        b.push_back({.dfb_spec_name = INPUT, .accessor_name = "x", .endpoint_type = m2::DFBEndpointType::CONSUMER});
        b.push_back({.dfb_spec_name = OUTPUT, .accessor_name = "y", .endpoint_type = m2::DFBEndpointType::CONSUMER});
        b.push_back(
            {.dfb_spec_name = OUTPUT_GRAD, .accessor_name = "dy", .endpoint_type = m2::DFBEndpointType::CONSUMER});
        b.push_back(
            {.dfb_spec_name = DECIMAL, .accessor_name = "decimal", .endpoint_type = m2::DFBEndpointType::CONSUMER});
        b.push_back({.dfb_spec_name = DX, .accessor_name = "dx", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        // Self-loop intermediates.
        const std::pair<m2::DFBSpecName, const char*> intermeds[] = {
            {XPOW, "xpow"},
            {LOGX, "logx"},
            {EXP_LXMD, "exp_lxmd"},
            {CORRECT_XPOW, "correct_xpow"},
            {TMP4, "tmp4"},
            {TMP5, "tmp5"},
            {RECIP_YPOW, "recip_ypow"},
            {SIGN, "sign"}};
        for (const auto& [name, accessor] : intermeds) {
            b.push_back(
                {.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = m2::DFBEndpointType::PRODUCER});
            b.push_back(
                {.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        return b;
    };

    auto make_compute_spec = [&](const m2::KernelSpecName& id, uint32_t num_cols_per_core_group) {
        return m2::KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path(COMPUTE_KERNEL_PATH),
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings = compute_dfb_bindings(),
            .compile_time_args =
                {{"num_output_tiles", num_cols_per_core_group},
                 {"wt_need_bcast", need_bcast_dim[0]},
                 {"ht_need_bcast", need_bcast_dim[1]}},
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"num_input_tiles_per_core", "p", "p_is_negative", "p_minus_one", "p_minus_one_is_negative"}},
            .hw_config = make_compute_hw(),
        };
    };

    m2::KernelSpec compute_spec_1 = make_compute_spec(COMPUTE_1, num_cols_per_core_group_1);
    m2::KernelSpec compute_spec_2;
    if (has_core_group_2) {
        compute_spec_2 = make_compute_spec(COMPUTE_2, num_cols_per_core_group_2);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelRunArgs::RuntimeArgValues reader_args, writer_args, compute1_args, compute2_args;
    m2::KernelRunArgs reader_kra{.kernel = READER};
    m2::KernelRunArgs writer_kra{.kernel = WRITER};

    // The three dim blocks are identical on every node; build the vararg payload once.
    std::vector<uint32_t> reader_varargs;
    reader_varargs.reserve(3u * static_cast<uint32_t>(input_grad_rank));
    for (auto i = 0; i < input_grad_rank; ++i) {
        reader_varargs.push_back(output_grad_dim[i]);
    }
    for (auto i = 0; i < input_grad_rank; ++i) {
        reader_varargs.push_back(input_grad_dim[i]);
    }
    for (auto i = 0; i < input_grad_rank; ++i) {
        reader_varargs.push_back(need_bcast_dim[i]);
    }

    const uint32_t decimal_bits = *reinterpret_cast<uint32_t*>(&decimal);

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const m2::NodeCoord node{core.x, core.y};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader — base addresses are injected by the TensorBindings; only scalars + varargs here.
        reader_args["decimal"][node] = decimal_bits;
        reader_args["num_output_tiles"][node] = num_tiles_per_core;
        reader_args["start_id"][node] = tile_offset;
        reader_kra.advanced_options.runtime_varargs[node] = reader_varargs;

        // writer
        writer_args["num_input_tiles_per_core"][node] = num_tiles_per_core;
        writer_args["tile_offset"][node] = tile_offset;

        // compute — runtime args go to the correct kernel's table
        auto& compute_args = core_group_1.contains(core) ? compute1_args : compute2_args;
        compute_args["num_input_tiles_per_core"][node] = num_tiles_per_core;
        compute_args["p"][node] = floored_p;
        compute_args["p_is_negative"][node] = static_cast<uint32_t>(p_is_negative);
        compute_args["p_minus_one"][node] = floored_p_minus_one;
        compute_args["p_minus_one_is_negative"][node] = static_cast<uint32_t>(p_minus_one_is_negative);

        tile_offset += num_tiles_per_core;
    }

    reader_kra.runtime_arg_values = std::move(reader_args);
    writer_kra.runtime_arg_values = std::move(writer_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble the spec + run-args
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::KernelSpec> kernels = {reader_spec, writer_spec, compute_spec_1};
    m2::Group<m2::WorkUnitSpec> work_units = {
        m2::WorkUnitSpec{.name = "mnb_group_1", .kernels = {READER, WRITER, COMPUTE_1}, .target_nodes = core_group_1}};

    m2::Group<m2::KernelRunArgs> kernel_run_args = {
        reader_kra, writer_kra, m2::KernelRunArgs{.kernel = COMPUTE_1, .runtime_arg_values = std::move(compute1_args)}};

    if (has_core_group_2) {
        kernels.push_back(compute_spec_2);
        work_units.push_back(m2::WorkUnitSpec{
            .name = "mnb_group_2", .kernels = {READER, WRITER, COMPUTE_2}, .target_nodes = core_group_2});
        kernel_run_args.push_back(
            m2::KernelRunArgs{.kernel = COMPUTE_2, .runtime_arg_values = std::move(compute2_args)});
    }

    m2::ProgramSpec spec{
        .name = "moreh_norm_backward",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {{.unique_id = T_INPUT, .spec = input.tensor_spec()},
             {.unique_id = T_OUTPUT, .spec = output.tensor_spec()},
             {.unique_id = T_OUTPUT_GRAD, .spec = output_grad.tensor_spec()},
             {.unique_id = T_INPUT_GRAD, .spec = input_grad.tensor_spec()}},
        .work_units = std::move(work_units),
    };

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args = std::move(kernel_run_args);
    run_params.tensor_args.emplace(T_INPUT, m2::TensorArgument{input.mesh_tensor()});
    run_params.tensor_args.emplace(T_OUTPUT, m2::TensorArgument{output.mesh_tensor()});
    run_params.tensor_args.emplace(T_OUTPUT_GRAD, m2::TensorArgument{output_grad.mesh_tensor()});
    run_params.tensor_args.emplace(T_INPUT_GRAD, m2::TensorArgument{input_grad.mesh_tensor()});

    return ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::operations::moreh::moreh_norm_backward
