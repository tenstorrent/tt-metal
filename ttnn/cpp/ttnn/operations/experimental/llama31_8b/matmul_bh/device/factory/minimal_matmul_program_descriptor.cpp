// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/factory/minimal.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/matmul_bh_shapes.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/matmul_bh_l1.hpp"
#include "ttnn/kernel_lib/mcast/host/mcast.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include <algorithm>
#include <bit>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::experimental::prim::llama31_8b_matmul_bh {

using tt::tt_metal::experimental::AddRuntimeArgsForNode;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DataMovementHardwareConfig;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBEndpointType;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::Group;
using tt::tt_metal::experimental::KernelAdvancedOptions;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::SemaphoreBinding;
using tt::tt_metal::experimental::SemaphoreSpec;
using tt::tt_metal::experimental::SemaphoreSpecName;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::TensorSpecRelaxations;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace {

KernelSpec::CompilerOptions::Defines to_defines(const std::map<std::string, std::string>& defines) {
    return KernelSpec::CompilerOptions::Defines(defines);
}

const DFBSpecName DFB_IN0{"in0"};
const DFBSpecName DFB_IN1{"in1"};
const DFBSpecName DFB_OUT{"out"};
const DFBSpecName DFB_INTERMEDIATE{"intermediate"};
const DFBSpecName DFB_IN2{"in2"};
const DFBSpecName DFB_TERNARY_A{"ternary_a"};
const DFBSpecName DFB_TERNARY_B{"ternary_b"};

const TensorParamName TP_IN0{"in0"};
const TensorParamName TP_IN1{"in1"};
const TensorParamName TP_IN2{"in2"};
const TensorParamName TP_IN3{"in3"};
const TensorParamName TP_TERNARY_A{"ternary_a"};
const TensorParamName TP_TERNARY_B{"ternary_b"};

const KernelSpecName K_IN0_SENDER{"in0_sender"};
const KernelSpecName K_IN0_RECEIVER{"in0_receiver"};
const KernelSpecName K_IN1_SENDER{"in1_sender"};
const KernelSpecName K_IN1_RECEIVER{"in1_receiver"};
const KernelSpecName K_COMPUTE{"compute"};

std::string output_param_name(uint32_t chunk) { return "out" + std::to_string(chunk); }

DFBBinding producer_of(const DFBSpecName& dfb) {
    return DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = *dfb,
        .endpoint_type = DFBEndpointType::PRODUCER,
    };
}

DFBBinding consumer_of(const DFBSpecName& dfb) {
    return DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = *dfb,
        .endpoint_type = DFBEndpointType::CONSUMER,
    };
}

void bind_tensor_args(
    ProgramRunArgs& run_args, const MinimalMatmulInputs& tensor_args, const std::vector<Tensor>& output_tensors) {
    run_args.tensor_args.insert({TP_IN0, tensor_args.input_tensor.mesh_tensor()});
    run_args.tensor_args.insert({TP_IN1, tensor_args.weight_tensor.mesh_tensor()});
    if (tensor_args.bias_tensor.has_value()) {
        run_args.tensor_args.insert({TP_IN2, tensor_args.bias_tensor->mesh_tensor()});
    }
    if (tensor_args.optional_input_tensor.has_value()) {
        run_args.tensor_args.insert({TP_IN3, tensor_args.optional_input_tensor->mesh_tensor()});
    }
    if (tensor_args.fused_ternary_input_a.has_value() && tensor_args.fused_ternary_input_b.has_value()) {
        run_args.tensor_args.insert({TP_TERNARY_A, tensor_args.fused_ternary_input_a->mesh_tensor()});
        run_args.tensor_args.insert({TP_TERNARY_B, tensor_args.fused_ternary_input_b->mesh_tensor()});
    }
    for (uint32_t chunk = 0; chunk < output_tensors.size(); ++chunk) {
        run_args.tensor_args.insert({TensorParamName{output_param_name(chunk)}, output_tensors[chunk].mesh_tensor()});
    }
}

const MinimalPlan& plan_for(const MinimalMatmulParams& operation_attributes, const MinimalMatmulInputs& tensor_args) {
    const MinimalPlan* plan = ttnn::prim::llama31_8b_matmul_bh::minimal_plan(operation_attributes, tensor_args);
    TT_FATAL(plan != nullptr, "llama31_8b.minimal_matmul_bh: this call has no table row");
    return *plan;
}

tt::tt_metal::NOC noc_of(uint32_t noc) { return noc == 0 ? tt::tt_metal::NOC::NOC_0 : tt::tt_metal::NOC::NOC_1; }

tt::tt_metal::DataMovementProcessor risc_of(uint32_t noc) {
    return noc == 0 ? tt::tt_metal::DataMovementProcessor::RISCV_0 : tt::tt_metal::DataMovementProcessor::RISCV_1;
}

uint32_t m_tiles_of(const Tensor& input_tensor) {
    return input_tensor.physical_volume() / input_tensor.padded_shape()[-1] / tt::constants::TILE_HEIGHT;
}

}  // namespace

ProgramRunArgs MinimalProgramFactory::override_runtime_arguments(
    const MinimalMatmulParams& operation_attributes,
    const MinimalMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>&) {
    const MinimalPlan& plan = plan_for(operation_attributes, tensor_args);
    const KernelRunArgs::CommonRuntimeArgValues length_word{{"M_tiles", m_tiles_of(tensor_args.input_tensor)}};
    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(KernelRunArgs{.kernel = K_IN0_SENDER, .common_runtime_arg_values = length_word});
    if (plan.in1_cores > 1) {
        run_args.kernel_run_args.push_back(
            KernelRunArgs{.kernel = K_IN0_RECEIVER, .common_runtime_arg_values = length_word});
    }
    run_args.kernel_run_args.push_back(KernelRunArgs{.kernel = K_IN1_SENDER, .common_runtime_arg_values = length_word});
    if (plan.in0_cores > 1) {
        run_args.kernel_run_args.push_back(
            KernelRunArgs{.kernel = K_IN1_RECEIVER, .common_runtime_arg_values = length_word});
    }
    run_args.kernel_run_args.push_back(KernelRunArgs{.kernel = K_COMPUTE, .common_runtime_arg_values = length_word});
    bind_tensor_args(run_args, tensor_args, tensor_return_value);
    return run_args;
}

ttnn::device_operation::ProgramArtifacts create_minimal_program_artifacts(
    const MinimalMatmulParams& operation_attributes,
    const MinimalMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value,
    const std::optional<HeadsLayout>& heads) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& weight_tensor = tensor_args.weight_tensor;
    const std::vector<Tensor>& output_tensors = tensor_return_value;
    const MinimalPlan& plan = plan_for(operation_attributes, tensor_args);
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    const uint32_t N_chunks = static_cast<uint32_t>(operation_attributes.chunks);

    TT_FATAL(
        !tensor_args.bias_tensor.has_value() && !tensor_args.optional_input_tensor.has_value() &&
            !tensor_args.fused_ternary_input_a.has_value() && !tensor_args.fused_ternary_input_b.has_value() &&
            !operation_attributes.fuse_swiglu && !operation_attributes.fused_activation.has_value(),
        "llama31_8b.minimal_matmul_bh: the table rows take no bias, concat, ternary, SwiGLU or fused activation");
    TT_FATAL(
        N_chunks < 4 && plan.chunk_tiles[N_chunks] != 0,
        "llama31_8b.minimal_matmul_bh: chunk count not in the table row");
    TT_FATAL(
        !heads.has_value() || (heads->num_heads == plan.num_heads && heads->num_kv_heads == plan.num_kv_heads &&
                               heads->head_dim == plan.head_dim && plan.head_dim != 0),
        "llama31_8b.minimal_matmul_bh: head layout not in the table row");

    auto* device = input_tensor.device();
    const auto core_grid = CoreRange({0, 0}, {plan.grid_x - 1, plan.grid_y - 1});
    const CoreCoord core_0_0{0, 0};
    const CoreCoord core_endx_endy{plan.grid_x - 1, plan.grid_y - 1};

    auto in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto in0_tile_size = tt::tile_size(in0_data_format);
    auto in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(weight_tensor.dtype());
    auto in1_tile_size = tt::tile_size(in1_data_format);
    auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensors[0].dtype());
    auto out_tile_size = tt::tile_size(output_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    auto intermediate_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    auto intermediate_tile_size = tt::tile_size(intermediate_data_format);

    const uint32_t M_tiles = m_tiles_of(input_tensor);
    const auto in1_noc = noc_of(plan.in1_noc);
    const bool has_in0_receivers = plan.in1_cores > 1;
    const bool has_in1_receivers = plan.in0_cores > 1;

    ProgramSpec spec;
    spec.name = "minimal_matmul";

    const SemaphoreSpecName SEM_IN0_SENDER{"in0_sender"};
    const SemaphoreSpecName SEM_IN0_RECEIVER{"in0_receiver"};
    const SemaphoreSpecName SEM_IN0_VALID{"in0_valid"};
    const SemaphoreSpecName SEM_IN1_SENDER{"in1_sender"};
    const SemaphoreSpecName SEM_IN1_RECEIVER{"in1_receiver"};
    const SemaphoreSpecName SEM_IN1_VALID{"in1_valid"};

    for (const auto& [name, initial_value] : std::initializer_list<std::pair<SemaphoreSpecName, uint32_t>>{
             {SEM_IN0_SENDER, INVALID},
             {SEM_IN0_RECEIVER, INVALID},
             {SEM_IN0_VALID, VALID},
             {SEM_IN1_SENDER, INVALID},
             {SEM_IN1_RECEIVER, INVALID},
             {SEM_IN1_VALID, VALID},
         }) {
        SemaphoreSpec sem{
            .unique_id = name,
            .target_nodes = CoreRangeSet(core_grid),
        };
        sem.advanced_options.initial_value = initial_value;
        spec.semaphores.push_back(std::move(sem));
    }

    auto push_dfb = [&spec](const DFBSpecName& name, uint32_t entry_size, uint32_t num_entries, tt::DataFormat format) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = name,
            .entry_size = entry_size,
            .num_entries = num_entries,
            .data_format_metadata = format,
        });
    };
    push_dfb(DFB_IN0, in0_tile_size, plan.in0_cb_tiles, in0_data_format);
    push_dfb(DFB_IN1, in1_tile_size, plan.in1_cb_tiles, in1_data_format);
    push_dfb(DFB_OUT, out_tile_size, plan.out_cb_tiles, output_data_format);
    push_dfb(DFB_INTERMEDIATE, intermediate_tile_size, plan.interm_cb_tiles, intermediate_data_format);
    TT_FATAL(
        ttnn::prim::llama31_8b_matmul_bh::dfb_l1_bytes(spec.dataflow_buffers) ==
            ttnn::prim::llama31_8b_matmul_bh::l1_bytes_minimal(
                plan, input_tensor.dtype(), weight_tensor.dtype(), output_tensors[0].dtype(), fp32_dest_acc_en),
        "llama31_8b.minimal_matmul_bh: buffer plan and L1 check disagree");

    const TensorSpecRelaxations rows_vary{.dynamic_tensor_shape = true};
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = TP_IN0, .spec = input_tensor.tensor_spec(), .relaxations = rows_vary});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TP_IN1, .spec = weight_tensor.tensor_spec()});
    std::vector<std::string> output_accessor_names;
    output_accessor_names.reserve(output_tensors.size());
    for (uint32_t chunk = 0; chunk < output_tensors.size(); ++chunk) {
        const std::string name = output_param_name(chunk);
        spec.tensor_parameters.push_back(TensorParameter{
            .unique_id = TensorParamName{name}, .spec = output_tensors[chunk].tensor_spec(), .relaxations = rows_vary});
        output_accessor_names.push_back(name);
    }

    std::map<std::string, std::string> defines;
    for (uint32_t i = 0; i < plan.num_defines; i++) {
        defines[plan.defines[i].key] = plan.defines[i].value;
    }
    if (heads.has_value()) {
        defines["HEADS_OUT"] = "1";
        defines["HEADS_DT"] = std::to_string(plan.heads_dt);
        defines["HEADS_QT"] = std::to_string(plan.heads_qt);
        defines["HEADS_KT"] = std::to_string(plan.heads_kt);
    }

    const bool in0_is_output_writer = !plan.transpose;
    const bool in1_is_output_writer = plan.transpose;

    auto dm_defines = [&](bool is_output_writer) {
        std::map<std::string, std::string> d = defines;
        if (is_output_writer) {
            d["IS_OUTPUT_WRITER"] = "1";
        }
        return to_defines(d);
    };

    auto dm_compile_time_args =
        [&](const std::string& input_tile_size_name, uint32_t input_tile_size, bool is_injector_core) {
            return KernelSpec::CompileTimeArgs{
                {"M_cores", plan.in0_cores},
                {"K_tiles", plan.K_tiles},
                {"padded_K_tiles", plan.padded_K_tiles},
                {"N_tiles", plan.N_tiles},
                {"padded_N_tiles", plan.padded_N_tiles},
                {"M_block_tiles", plan.M_block_tiles},
                {"K_block_tiles", plan.K_block_tiles},
                {"N_block_tiles", plan.N_block_tiles},
                {"N_blocks_per_core", plan.N_blocks_per_core},
                {"out_tile_size", out_tile_size},
                {"in2_tile_size", in1_tile_size},
                {"is_injector_core", static_cast<uint32_t>(is_injector_core)},
                {"N_chunks", N_chunks},
                {"N_tiles_per_chunk", plan.chunk_tiles[N_chunks]},
                {input_tile_size_name, input_tile_size},
            };
        };

    auto dm_runtime_arg_names = [&](const std::string& prefix) {
        KernelSpec::RuntimeArgSchema schema;
        schema.runtime_arg_names = {
            "is_sink_core",
            prefix + "_dest_noc_x",
            prefix + "_dest_noc_y",
            prefix + "_sender_noc_x",
            prefix + "_sender_noc_y",
            "in0_idx",
            "N_start_tile",
            "N_end_tile",
            "defer_write_k_block",
            "max_defer_write_k_block",
        };
        if (prefix == "in1") {
            schema.runtime_arg_names.push_back("bulk_segn");
            for (uint32_t q = 0; q < kIn1SegmentSlots; q++) {
                schema.runtime_arg_names.push_back("bulk_s" + std::to_string(q));
            }
        }
        schema.common_runtime_arg_names = {"M_tiles"};
        return schema;
    };

    auto dm_tensor_bindings = [&](const TensorParamName& own_input) {
        Group<TensorBinding> bindings;
        bindings.push_back(TensorBinding{.tensor_parameter_name = own_input, .accessor_name = *own_input});
        for (const auto& name : output_accessor_names) {
            bindings.push_back(TensorBinding{.tensor_parameter_name = TensorParamName{name}, .accessor_name = name});
        }
        return bindings;
    };

    auto dm_dfb_bindings = [&](const DFBSpecName& own_input, bool is_output_writer) {
        Group<DFBBinding> bindings;
        bindings.push_back(producer_of(own_input));
        if (is_output_writer) {
            bindings.push_back(consumer_of(DFB_OUT));
        }
        return bindings;
    };

    auto dm_semaphore_bindings =
        [](const SemaphoreSpecName& sender, const SemaphoreSpecName& receiver, const SemaphoreSpecName& valid) {
            Group<SemaphoreBinding> bindings;
            bindings.push_back(SemaphoreBinding{.semaphore_spec_name = sender, .accessor_name = *sender});
            bindings.push_back(SemaphoreBinding{.semaphore_spec_name = receiver, .accessor_name = *receiver});
            bindings.push_back(SemaphoreBinding{.semaphore_spec_name = valid, .accessor_name = *valid});
            return bindings;
        };

    KernelAdvancedOptions dm_advanced_options;
    dm_advanced_options.tensor_binding_sequences.push_back(
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "outputs", .members = output_accessor_names});

    auto make_dm_kernel = [&](const KernelSpecName& name,
                              const char* source,
                              uint32_t input_tile_size,
                              bool is_injector_core,
                              bool is_output_writer,
                              const DFBSpecName& own_input,
                              const TensorParamName& own_input_tensor,
                              const std::string& rt_prefix,
                              uint32_t noc,
                              const SemaphoreSpecName& sem_sender,
                              const SemaphoreSpecName& sem_receiver,
                              const SemaphoreSpecName& sem_valid) {
        return KernelSpec{
            .unique_id = name,
            .source = std::filesystem::path(source),
            .compiler_options = {.defines = dm_defines(is_output_writer)},
            .dfb_bindings = dm_dfb_bindings(own_input, is_output_writer),
            .semaphore_bindings = dm_semaphore_bindings(sem_sender, sem_receiver, sem_valid),
            .tensor_bindings = dm_tensor_bindings(own_input_tensor),
            .compile_time_args = dm_compile_time_args(*own_input + "_tile_size", input_tile_size, is_injector_core),
            .runtime_arg_schema = dm_runtime_arg_names(rt_prefix),
            .hw_config =
                DataMovementHardwareConfig{
                    .config_1xx =
                        DataMovementHardwareConfig::DataMovement1XXConfig{
                            .processor = risc_of(noc),
                            .noc = noc_of(noc),
                        },
                },
            .advanced_options = dm_advanced_options,
        };
    };

    constexpr const char* kIn0Source =
        "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/minimal/dm_in0_sender_metal2.cpp";
    constexpr const char* kIn1Source =
        "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/minimal/"
        "dm_in1_sender_out_metal2.cpp";

    spec.kernels.push_back(make_dm_kernel(
        K_IN0_SENDER,
        kIn0Source,
        in0_tile_size,
        true,
        in0_is_output_writer,
        DFB_IN0,
        TP_IN0,
        "in0",
        plan.in0_noc,
        SEM_IN0_SENDER,
        SEM_IN0_RECEIVER,
        SEM_IN0_VALID));
    if (has_in0_receivers) {
        spec.kernels.push_back(make_dm_kernel(
            K_IN0_RECEIVER,
            kIn0Source,
            in0_tile_size,
            false,
            in0_is_output_writer,
            DFB_IN0,
            TP_IN0,
            "in0",
            plan.in0_noc,
            SEM_IN0_SENDER,
            SEM_IN0_RECEIVER,
            SEM_IN0_VALID));
    }
    spec.kernels.push_back(make_dm_kernel(
        K_IN1_SENDER,
        kIn1Source,
        in1_tile_size,
        true,
        in1_is_output_writer,
        DFB_IN1,
        TP_IN1,
        "in1",
        plan.in1_noc,
        SEM_IN1_SENDER,
        SEM_IN1_RECEIVER,
        SEM_IN1_VALID));
    if (has_in1_receivers) {
        spec.kernels.push_back(make_dm_kernel(
            K_IN1_RECEIVER,
            kIn1Source,
            in1_tile_size,
            false,
            in1_is_output_writer,
            DFB_IN1,
            TP_IN1,
            "in1",
            plan.in1_noc,
            SEM_IN1_SENDER,
            SEM_IN1_RECEIVER,
            SEM_IN1_VALID));
    }

    auto compute_defines = defines;
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), plan.num_cores, compute_defines, ttnn::get_throttle_level(compute_kernel_config));

    auto compute_hw = to_compute_hardware_config(compute_kernel_config);
    compute_hw.double_buffer_dest = true;
    if (fp32_dest_acc_en) {
        for (const auto& [dfb, format] : std::initializer_list<std::pair<DFBSpecName, tt::DataFormat>>{
                 {DFB_IN0, in0_data_format},
                 {DFB_IN1, in1_data_format},
                 {DFB_INTERMEDIATE, intermediate_data_format}}) {
            if (format == tt::DataFormat::Float32) {
                compute_hw.unpack_modes.emplace(dfb, tt::tt_metal::UnpackMode::UnpackToSrc);
            }
        }
    }

    Group<DFBBinding> compute_dfb_bindings;
    compute_dfb_bindings.push_back(consumer_of(DFB_IN0));
    compute_dfb_bindings.push_back(consumer_of(DFB_IN1));
    compute_dfb_bindings.push_back(producer_of(DFB_OUT));
    compute_dfb_bindings.push_back(producer_of(DFB_INTERMEDIATE));
    compute_dfb_bindings.push_back(consumer_of(DFB_INTERMEDIATE));

    KernelSpec::RuntimeArgSchema compute_schema;
    compute_schema.runtime_arg_names = {"in0_idx", "N_start_tile", "N_end_tile"};
    compute_schema.common_runtime_arg_names = {"M_tiles"};

    spec.kernels.push_back(KernelSpec{
        .unique_id = K_COMPUTE,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/minimal/compute_metal2.cpp"),
        .compiler_options =
            {.defines = to_defines(compute_defines), .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args =
            {
                {"K_num_blocks", plan.K_blocks},
                {"M_block_tiles", plan.M_block_tiles},
                {"K_block_tiles", plan.K_block_tiles},
                {"N_block_tiles", plan.N_block_tiles},
                {"M_cores", plan.in0_cores},
                {"N_blocks_per_core", plan.N_blocks_per_core},
                {"subblock_h", plan.subblock_h},
                {"subblock_w", plan.subblock_w},
                {"swiglu_in_k_loop", 0u},
                {"swiglu_block_float_output", plan.swiglu_block_float_output},
            },
        .runtime_arg_schema = compute_schema,
        .hw_config = compute_hw,
    });

    for (uint32_t i = 0; i < plan.num_regions; i++) {
        const MinimalPlanRegion& r = plan.regions[i];
        Group<KernelSpecName> kernels;
        kernels.push_back(r.in0_sender ? K_IN0_SENDER : K_IN0_RECEIVER);
        kernels.push_back(r.in1_sender ? K_IN1_SENDER : K_IN1_RECEIVER);
        kernels.push_back(K_COMPUTE);
        spec.work_units.push_back(WorkUnitSpec{
            .name = r.name,
            .kernels = std::move(kernels),
            .target_nodes = CoreRangeSet(CoreRange(CoreCoord{r.x0, r.y0}, CoreCoord{r.x1, r.y1}))});
    }

    ProgramRunArgs run_args;
    const KernelRunArgs::CommonRuntimeArgValues length_word{{"M_tiles", M_tiles}};
    KernelRunArgs in0_sender_args{.kernel = K_IN0_SENDER, .common_runtime_arg_values = length_word};
    KernelRunArgs in0_receiver_args{.kernel = K_IN0_RECEIVER, .common_runtime_arg_values = length_word};
    KernelRunArgs in1_sender_args{.kernel = K_IN1_SENDER, .common_runtime_arg_values = length_word};
    KernelRunArgs in1_receiver_args{.kernel = K_IN1_RECEIVER, .common_runtime_arg_values = length_word};
    KernelRunArgs compute_args{.kernel = K_COMPUTE, .common_runtime_arg_values = length_word};

    for (uint32_t i = 0; i < plan.num_cores; i++) {
        const MinimalPlanCore& c = plan.cores[i];
        const CoreCoord core{c.x, c.y};
        const auto in0_prev = device->worker_core_from_logical_core(CoreCoord{c.in0_prev_x, c.in0_prev_y});
        const auto in0_next = device->worker_core_from_logical_core(CoreCoord{c.in0_next_x, c.in0_next_y});
        const auto in1_prev = device->worker_core_from_logical_core(CoreCoord{c.in1_prev_x, c.in1_prev_y});
        const auto in1_next = device->worker_core_from_logical_core(CoreCoord{c.in1_next_x, c.in1_next_y});
        const uint32_t N_start_tile = c.N_start_tile;
        const uint32_t N_end_tile = c.N_end_tile;
        const uint32_t defer_write_k_block = c.defer_write_k_block;
        const uint32_t in0_idx = c.in0_idx;

        AddRuntimeArgsForNode(
            (c.in0_sender ? in0_sender_args : in0_receiver_args).runtime_arg_values,
            core,
            {{"is_sink_core", static_cast<uint32_t>(c.in0_sink)},
             {"in0_dest_noc_x", (std::uint32_t)in0_next.x},
             {"in0_dest_noc_y", (std::uint32_t)in0_next.y},
             {"in0_sender_noc_x", (std::uint32_t)in0_prev.x},
             {"in0_sender_noc_y", (std::uint32_t)in0_prev.y},
             {"in0_idx", in0_idx},
             {"N_start_tile", N_start_tile},
             {"N_end_tile", N_end_tile},
             {"defer_write_k_block", defer_write_k_block},
             {"max_defer_write_k_block", plan.max_defer_write_k_block}});

        auto& in1_rt = (c.in1_sender ? in1_sender_args : in1_receiver_args).runtime_arg_values;
        AddRuntimeArgsForNode(
            in1_rt,
            core,
            {{"is_sink_core", static_cast<uint32_t>(c.in1_sink)},
             {"in1_dest_noc_x", (std::uint32_t)in1_next.x},
             {"in1_dest_noc_y", (std::uint32_t)in1_next.y},
             {"in1_sender_noc_x", (std::uint32_t)in1_prev.x},
             {"in1_sender_noc_y", (std::uint32_t)in1_prev.y},
             {"in0_idx", in0_idx},
             {"N_start_tile", N_start_tile},
             {"N_end_tile", N_end_tile},
             {"defer_write_k_block", defer_write_k_block},
             {"max_defer_write_k_block", plan.max_defer_write_k_block}});
        AddRuntimeArgsForNode(in1_rt, core, {{"bulk_segn", static_cast<uint32_t>(plan.bulk_counts[c.in1_idx])}});
        for (uint32_t q = 0; q < kIn1SegmentSlots; q++) {
            AddRuntimeArgsForNode(in1_rt, core, {{"bulk_s" + std::to_string(q), plan.bulk_segments[c.in1_idx][q]}});
        }

        AddRuntimeArgsForNode(
            compute_args.runtime_arg_values,
            core,
            {{"in0_idx", in0_idx}, {"N_start_tile", N_start_tile}, {"N_end_tile", N_end_tile}});
    }

    run_args.kernel_run_args.push_back(std::move(in0_sender_args));
    if (has_in0_receivers) {
        run_args.kernel_run_args.push_back(std::move(in0_receiver_args));
    }
    run_args.kernel_run_args.push_back(std::move(in1_sender_args));
    if (has_in1_receivers) {
        run_args.kernel_run_args.push_back(std::move(in1_receiver_args));
    }
    run_args.kernel_run_args.push_back(std::move(compute_args));

    if (plan.in1_wmc) {
        ttnn::kernel_lib::host::Mcast in1_mcast(
            *device,
            ttnn::kernel_lib::host::McastConfig{.noc = in1_noc},
            CoreRangeSet(CoreRange(core_0_0, core_endx_endy)),
            plan.in0_cores,
            ttnn::kernel_lib::host::McastFixedSenderConfig{},
            plan.transpose ? ttnn::kernel_lib::host::McastCoreOrder::RowMajor
                           : ttnn::kernel_lib::host::McastCoreOrder::ColumnMajor);
        const std::array<KernelSpecName, 2> in1_mcast_kernels{K_IN1_SENDER, K_IN1_RECEIVER};
        in1_mcast.attach(spec, run_args, "in1_wmc", in1_mcast_kernels);
    }

    bind_tensor_args(run_args, tensor_args, output_tensors);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts MinimalProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    return create_minimal_program_artifacts(operation_attributes, tensor_args, tensor_return_value, std::nullopt);
}

}  // namespace ttnn::experimental::prim::llama31_8b_matmul_bh
