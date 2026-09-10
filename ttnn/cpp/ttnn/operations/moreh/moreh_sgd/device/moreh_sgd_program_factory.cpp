// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <vector>

#include "moreh_sgd_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_sgd {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/reader_moreh_sgd.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/writer_moreh_sgd.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_sgd/device/kernels/moreh_sgd.cpp";

namespace {

// Spec resource names (prefixed to stay distinct under unity builds). The comment records the
// legacy magic CB index / buffer-address RTA each one replaces.
const m2::DFBSpecName PARAM_IN_DFB{"sgd_param_in"};          // c_0
const m2::DFBSpecName GRAD_DFB{"sgd_grad"};                  // c_1
const m2::DFBSpecName MOMENTUM_IN_DFB{"sgd_momentum_in"};    // c_2 (optional)
const m2::DFBSpecName PARAM_OUT_DFB{"sgd_param_out"};        // c_16
const m2::DFBSpecName MOMENTUM_OUT_DFB{"sgd_momentum_out"};  // c_17 (optional)
const m2::DFBSpecName SCALAR_ARGS_DFB{"sgd_scalar_args"};    // c_24 (lr, momentum, dampening, weight_decay, one)
const m2::DFBSpecName TMP1_DFB{"sgd_tmp1"};                  // c_25
const m2::DFBSpecName TMP2_DFB{"sgd_tmp2"};                  // c_26
const m2::DFBSpecName TMP3_DFB{"sgd_tmp3"};                  // c_27
const m2::DFBSpecName TMP4_DFB{"sgd_tmp4"};                  // c_28

const m2::TensorParamName PARAM_IN_TENSOR{"sgd_t_param_in"};
const m2::TensorParamName GRAD_TENSOR{"sgd_t_grad"};
const m2::TensorParamName MOMENTUM_IN_TENSOR{"sgd_t_momentum_in"};  // optional
const m2::TensorParamName PARAM_OUT_TENSOR{"sgd_t_param_out"};
const m2::TensorParamName MOMENTUM_OUT_TENSOR{"sgd_t_momentum_out"};  // optional

const m2::KernelSpecName READER{"sgd_reader"};
const m2::KernelSpecName WRITER{"sgd_writer"};
const m2::KernelSpecName COMPUTE1{"sgd_compute1"};
const m2::KernelSpecName COMPUTE2{"sgd_compute2"};

}  // namespace

ttnn::device_operation::ProgramArtifacts MorehSgdOperation::MorehSgdProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& param_in = tensor_args.param_in;
    const auto& grad = tensor_args.grad;
    const std::optional<Tensor>& momentum_buffer_in = tensor_args.momentum_buffer_in;

    auto& output_tensors = output_tensor;
    auto& param_out = output_tensors.at(0).value();
    auto& momentum_buffer_out = output_tensors.at(1);

    auto lr = operation_attributes.lr;
    auto momentum = operation_attributes.momentum;
    auto dampening = operation_attributes.dampening;
    auto weight_decay = operation_attributes.weight_decay;
    auto nesterov = operation_attributes.nesterov;
    auto momentum_initialized = operation_attributes.momentum_initialized;

    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    auto shape = param_in.logical_shape();
    auto H = shape[-2];
    auto W = shape[-1];
    auto num = param_in.physical_volume() / H / W;
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    // Optional-resource conditions. These mirror the compile-defines exactly: MOMENTUM is emitted
    // iff momentum != 0, MOMENTUM_INITIALIZED iff momentum_initialized. The reader/compute access
    // momentum_buffer_in under (MOMENTUM && MOMENTUM_INITIALIZED); the writer/compute access
    // momentum_buffer_out under MOMENTUM. The corresponding output tensor exists exactly when
    // momentum != 0, and the caller supplies momentum_buffer_in exactly under the pair condition.
    const bool bind_momentum_in = (momentum != 0) && momentum_initialized;
    const bool bind_momentum_out = (momentum != 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = param_in.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t units_to_divide = num * Ht * Wt;
    uint32_t core_h = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    bool has_core_group_2 = !core_group_2.ranges().empty();

    auto arch = param_in.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    auto data_format = datatype_to_dataformat_converter(param_in.dtype());
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;
    const uint32_t data_tile_size = tile_size(data_format);
    const uint32_t intermed_tile_size = tile_size(intermed_cb_format);

    auto make_dfb = [](const m2::DFBSpecName& id, uint32_t entry_size, uint32_t num_entries, tt::DataFormat fmt) {
        return m2::DataflowBufferSpec{
            .unique_id = id, .entry_size = entry_size, .num_entries = num_entries, .data_format_metadata = fmt};
    };

    m2::Group<m2::DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(make_dfb(PARAM_IN_DFB, data_tile_size, 2, data_format));   // param_in
    dataflow_buffers.push_back(make_dfb(GRAD_DFB, data_tile_size, 2, data_format));       // grad
    dataflow_buffers.push_back(make_dfb(PARAM_OUT_DFB, data_tile_size, 2, data_format));  // param_out
    // cb_scalar_args holds lr, momentum, dampening, weight_decay, one (5 entries).
    dataflow_buffers.push_back(make_dfb(SCALAR_ARGS_DFB, intermed_tile_size, 5, intermed_cb_format));
    dataflow_buffers.push_back(make_dfb(TMP1_DFB, intermed_tile_size, 1, intermed_cb_format));
    dataflow_buffers.push_back(make_dfb(TMP2_DFB, intermed_tile_size, 1, intermed_cb_format));
    dataflow_buffers.push_back(make_dfb(TMP3_DFB, intermed_tile_size, 1, intermed_cb_format));
    dataflow_buffers.push_back(make_dfb(TMP4_DFB, intermed_tile_size, 1, intermed_cb_format));
    if (bind_momentum_in) {
        dataflow_buffers.push_back(make_dfb(MOMENTUM_IN_DFB, data_tile_size, 2, data_format));  // momentum_in
    }
    if (bind_momentum_out) {
        dataflow_buffers.push_back(make_dfb(MOMENTUM_OUT_DFB, data_tile_size, 2, data_format));  // momentum_out
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernel defines
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec::CompilerOptions::Defines reader_defines;
    m2::KernelSpec::CompilerOptions::Defines writer_defines;
    m2::KernelSpec::CompilerOptions::Defines compute_defines;

    if (weight_decay != 0) {
        reader_defines.emplace("WEIGHT_DECAY", "1");
        compute_defines.emplace("WEIGHT_DECAY", "1");
    }
    if (momentum != 0) {
        reader_defines.emplace("MOMENTUM", "1");
        compute_defines.emplace("MOMENTUM", "1");
        writer_defines.emplace("MOMENTUM", "1");
    }
    if (momentum_initialized) {
        reader_defines.emplace("MOMENTUM_INITIALIZED", "1");
        compute_defines.emplace("MOMENTUM_INITIALIZED", "1");
    }
    if (nesterov) {
        reader_defines.emplace("NESTEROV", "1");
        compute_defines.emplace("NESTEROV", "1");
    }
    if (fp32_dest_acc_en) {
        reader_defines.emplace("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::DFBBinding> reader_dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = PARAM_IN_DFB, .accessor_name = "param_in", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = GRAD_DFB, .accessor_name = "grad", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = SCALAR_ARGS_DFB,
            .accessor_name = "scalar_args",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
    };
    m2::Group<m2::TensorBinding> reader_tensor_bindings = {
        m2::TensorBinding{.tensor_parameter_name = PARAM_IN_TENSOR, .accessor_name = "param_in"},
        m2::TensorBinding{.tensor_parameter_name = GRAD_TENSOR, .accessor_name = "grad"},
    };
    if (bind_momentum_in) {
        reader_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = MOMENTUM_IN_DFB,
            .accessor_name = "momentum_in",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        reader_tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = MOMENTUM_IN_TENSOR, .accessor_name = "momentum_in"});
    }

    m2::KernelSpec reader{
        .unique_id = READER,
        .source = READER_KERNEL_PATH,
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .runtime_arg_schema =
            {.runtime_arg_names = {"num_tiles", "tile_offset", "lr", "momentum", "dampening", "weight_decay", "one"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    m2::Group<m2::DFBBinding> writer_dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = PARAM_OUT_DFB,
            .accessor_name = "param_out",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
    };
    m2::Group<m2::TensorBinding> writer_tensor_bindings = {
        m2::TensorBinding{.tensor_parameter_name = PARAM_OUT_TENSOR, .accessor_name = "param_out"},
    };
    if (bind_momentum_out) {
        writer_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = MOMENTUM_OUT_DFB,
            .accessor_name = "momentum_out",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
        writer_tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = MOMENTUM_OUT_TENSOR, .accessor_name = "momentum_out"});
    }

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = WRITER_KERNEL_PATH,
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .tensor_bindings = std::move(writer_tensor_bindings),
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "tile_offset"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Faithful reproduction of the legacy ComputeConfigDescriptor, which set math_fidelity,
    // fp32_dest_acc_en, dst_full_sync_en and math_approx_mode. double_buffer_dest = !dst_full_sync_en
    // (the field was inverted on rename). packer_l1_acc is resolved but the legacy op never applied it.
    m2::ComputeGen1Config compute_gen1{
        .fpu_math_fidelity = math_fidelity,
        .sfpu_precision_mode = math_approx_mode ? Precision::Approximate : Precision::Precise,
        .enable_32_bit_dest = fp32_dest_acc_en,
        .double_buffer_dest = !dst_full_sync_en,
    };
    // Under fp32_dest_acc_en the intermediate DFBs (scalar_args + tmp1..4) are Float32 and are
    // consumed by the compute kernel with enable_32_bit_dest = true, so the Metal 2.0 validator
    // requires an explicit unpack_modes entry. Legacy set no unpack_to_dest_mode (all Default) →
    // UnpackMode::UnpackToSrc.
    if (fp32_dest_acc_en) {
        compute_gen1.unpack_modes = {
            {SCALAR_ARGS_DFB, UnpackMode::UnpackToSrc},
            {TMP1_DFB, UnpackMode::UnpackToSrc},
            {TMP2_DFB, UnpackMode::UnpackToSrc},
            {TMP3_DFB, UnpackMode::UnpackToSrc},
            {TMP4_DFB, UnpackMode::UnpackToSrc},
        };
    }
    m2::ComputeHardwareConfig compute_hw = compute_gen1;

    // Compute DFB bindings. The four tmp DFBs are compute-only self-loops (bound both PRODUCER and
    // CONSUMER on the one compute kernel). momentum_in/out cross to reader/writer and are bound only
    // under their optional conditions. Identical for both per-core-group compute KernelSpecs.
    auto build_compute_dfb_bindings = [&]() {
        m2::Group<m2::DFBBinding> b = {
            m2::DFBBinding{
                .dfb_spec_name = PARAM_IN_DFB,
                .accessor_name = "param_in",
                .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = GRAD_DFB, .accessor_name = "grad", .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = SCALAR_ARGS_DFB,
                .accessor_name = "scalar_args",
                .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = PARAM_OUT_DFB,
                .accessor_name = "param_out",
                .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = TMP1_DFB, .accessor_name = "tmp1", .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = TMP1_DFB, .accessor_name = "tmp1", .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = TMP2_DFB, .accessor_name = "tmp2", .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = TMP2_DFB, .accessor_name = "tmp2", .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = TMP3_DFB, .accessor_name = "tmp3", .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = TMP3_DFB, .accessor_name = "tmp3", .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = TMP4_DFB, .accessor_name = "tmp4", .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = TMP4_DFB, .accessor_name = "tmp4", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        };
        if (bind_momentum_in) {
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = MOMENTUM_IN_DFB,
                .accessor_name = "momentum_in",
                .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        if (bind_momentum_out) {
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = MOMENTUM_OUT_DFB,
                .accessor_name = "momentum_out",
                .endpoint_type = m2::DFBEndpointType::PRODUCER});
        }
        return b;
    };

    // Preserved work-split multiplicity: two compute KernelSpecs over disjoint core groups, each with
    // its own live per-group CTA num_tiles (the compute kernel reads it as its loop count).
    m2::KernelSpec compute1{
        .unique_id = COMPUTE1,
        .source = COMPUTE_KERNEL_PATH,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings = build_compute_dfb_bindings(),
        .compile_time_args = {{"num_tiles", num_tiles_per_core_group_1}},
        .hw_config = compute_hw,
    };
    m2::KernelSpec compute2{
        .unique_id = COMPUTE2,
        .source = COMPUTE_KERNEL_PATH,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings = build_compute_dfb_bindings(),
        .compile_time_args = {{"num_tiles", num_tiles_per_core_group_2}},
        .hw_config = compute_hw,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble spec
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::KernelSpec> kernels = {reader, writer, compute1};
    if (has_core_group_2) {
        kernels.push_back(compute2);
    }

    m2::Group<m2::TensorParameter> tensor_parameters = {
        m2::TensorParameter{.unique_id = PARAM_IN_TENSOR, .spec = param_in.tensor_spec()},
        m2::TensorParameter{.unique_id = GRAD_TENSOR, .spec = grad.tensor_spec()},
        m2::TensorParameter{.unique_id = PARAM_OUT_TENSOR, .spec = param_out.tensor_spec()},
    };
    if (bind_momentum_in) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = MOMENTUM_IN_TENSOR, .spec = momentum_buffer_in->tensor_spec()});
    }
    if (bind_momentum_out) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = MOMENTUM_OUT_TENSOR, .spec = momentum_buffer_out->tensor_spec()});
    }

    m2::Group<m2::WorkUnitSpec> work_units;
    work_units.push_back(
        m2::WorkUnitSpec{.name = "group_1", .kernels = {READER, WRITER, COMPUTE1}, .target_nodes = core_group_1});
    if (has_core_group_2) {
        work_units.push_back(
            m2::WorkUnitSpec{.name = "group_2", .kernels = {READER, WRITER, COMPUTE2}, .target_nodes = core_group_2});
    }

    m2::ProgramSpec spec;
    spec.name = "moreh_sgd";
    spec.kernels = std::move(kernels);
    spec.dataflow_buffers = std::move(dataflow_buffers);
    spec.tensor_parameters = std::move(tensor_parameters);
    spec.work_units = std::move(work_units);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t u_lr = std::bit_cast<uint32_t>(lr);
    const uint32_t u_momentum = std::bit_cast<uint32_t>(momentum);
    const uint32_t u_dampening = std::bit_cast<uint32_t>(dampening);
    const uint32_t u_weight_decay = std::bit_cast<uint32_t>(weight_decay);
    const uint32_t u_one = std::bit_cast<uint32_t>(1.0f);

    m2::ProgramRunArgs run_args;
    m2::KernelRunArgs reader_run{.kernel = READER};
    m2::KernelRunArgs writer_run{.kernel = WRITER};

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        m2::NodeCoord core = {i / core_h, i % core_h};
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        m2::AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"num_tiles", num_tiles_per_core},
             {"tile_offset", tile_offset},
             {"lr", u_lr},
             {"momentum", u_momentum},
             {"dampening", u_dampening},
             {"weight_decay", u_weight_decay},
             {"one", u_one}});

        m2::AddRuntimeArgsForNode(
            writer_run.runtime_arg_values, core, {{"num_tiles", num_tiles_per_core}, {"tile_offset", tile_offset}});

        tile_offset += num_tiles_per_core;
    }

    // Compute kernels carry only a compile-time num_tiles (no RTAs); their KernelRunArgs are empty
    // entries so every kernel in the spec has a run-args record.
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run), m2::KernelRunArgs{.kernel = COMPUTE1}};
    if (has_core_group_2) {
        run_args.kernel_run_args.push_back(m2::KernelRunArgs{.kernel = COMPUTE2});
    }

    run_args.tensor_args.emplace(PARAM_IN_TENSOR, m2::TensorArgument{param_in.mesh_tensor()});
    run_args.tensor_args.emplace(GRAD_TENSOR, m2::TensorArgument{grad.mesh_tensor()});
    run_args.tensor_args.emplace(PARAM_OUT_TENSOR, m2::TensorArgument{param_out.mesh_tensor()});
    if (bind_momentum_in) {
        run_args.tensor_args.emplace(MOMENTUM_IN_TENSOR, m2::TensorArgument{momentum_buffer_in->mesh_tensor()});
    }
    if (bind_momentum_out) {
        run_args.tensor_args.emplace(MOMENTUM_OUT_TENSOR, m2::TensorArgument{momentum_buffer_out->mesh_tensor()});
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_sgd
