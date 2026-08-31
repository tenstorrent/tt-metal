// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_abs_pow_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/work_split.hpp>

#include <bit>

namespace ttnn::operations::moreh::moreh_abs_pow {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_abs_pow/device/kernels/reader_moreh_abs_pow.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_abs_pow/device/kernels/writer_moreh_abs_pow.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_abs_pow/device/kernels/moreh_abs_pow_kernel.cpp";

ttnn::device_operation::ProgramArtifacts MorehAbsPowOperation::MorehAbsPowProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto p = operation_attributes.p;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = input.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.padded_shape();
    const auto input_rank = input_shape.rank();
    auto logical_shape = input.logical_shape();
    if (logical_shape.rank() < 2) {
        logical_shape = logical_shape.to_rank(2);
    }

    const auto H = input_shape[-2];
    const auto W = input_shape[-1];

    const auto Ht = H / tt::constants::TILE_HEIGHT;
    const auto Wt = W / tt::constants::TILE_WIDTH;

    const auto num_units = input.physical_volume() / H / W * Ht;

    const auto origin_w = logical_shape[input_rank - 1];

    auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, operation_attributes.compute_kernel_config);

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_units_per_core_group_1,
         num_units_per_core_group_2] = split_work_to_cores(grid, num_units);

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;
    const uint32_t cb_tile_size = tile_size(cb_data_format);
    const uint32_t intermed_tile_size = tile_size(intermed_data_format);

    // Program-scope resource / parameter names (was: magic CB indices c_0..c_27, buffer-address RTAs).
    const m2::DFBSpecName INPUT_DFB{"input_dfb"};        // c_0: input (x)
    const m2::DFBSpecName ONE_DFB{"one_dfb"};            // c_1: one
    const m2::DFBSpecName DECIMAL_DFB{"decimal_dfb"};    // c_2: recip_p_decimal
    const m2::DFBSpecName MASK_W_DFB{"mask_w_dfb"};      // c_3: mask_w
    const m2::DFBSpecName OUTPUT_DFB{"output_dfb"};      // c_16: output (y)
    const m2::DFBSpecName XABS_DFB{"xabs_dfb"};          // c_24: |x|
    const m2::DFBSpecName XPOW_DFB{"xpow_dfb"};          // c_25: |x|^p
    const m2::DFBSpecName LOGX_DFB{"logx_dfb"};          // c_26: log(|x|)
    const m2::DFBSpecName EXP_LXMD_DFB{"exp_lxmd_dfb"};  // c_27: exp(log(|x|) * decimal)

    const m2::TensorParamName INPUT_TENSOR{"input"};
    const m2::TensorParamName OUTPUT_TENSOR{"output"};

    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    // One tile per DFB (legacy total_size = 1 * tile_size). data_format_metadata is required for
    // DFBs bound to the compute kernel; tile_format_metadata is left unset (legacy CBFormatDescriptor
    // did not set .tile).
    auto make_dfb = [](const m2::DFBSpecName& id, uint32_t entry_size, tt::DataFormat fmt) {
        return m2::DataflowBufferSpec{
            .unique_id = id,
            .entry_size = entry_size,
            .num_entries = 1,
            .data_format_metadata = fmt,
        };
    };

    m2::Group<m2::DataflowBufferSpec> dataflow_buffers = {
        make_dfb(INPUT_DFB, cb_tile_size, cb_data_format),
        make_dfb(ONE_DFB, cb_tile_size, cb_data_format),
        make_dfb(DECIMAL_DFB, cb_tile_size, cb_data_format),
        make_dfb(MASK_W_DFB, cb_tile_size, cb_data_format),
        make_dfb(OUTPUT_DFB, cb_tile_size, cb_data_format),
        make_dfb(XABS_DFB, intermed_tile_size, intermed_data_format),
        make_dfb(XPOW_DFB, intermed_tile_size, intermed_data_format),
        make_dfb(LOGX_DFB, intermed_tile_size, intermed_data_format),
        make_dfb(EXP_LXMD_DFB, intermed_tile_size, intermed_data_format),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // The reader's / writer's only legacy CTA was the TensorAccessorArgs plumbing; that is now
    // carried by the TensorBinding, so compile_time_args are empty. The buffer-address RTA arg 0
    // is likewise replaced by the TensorBinding.
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = READER_KERNEL_PATH,
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = INPUT_DFB, .accessor_name = "in", .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = ONE_DFB, .accessor_name = "one", .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = DECIMAL_DFB,
                    .accessor_name = "decimal",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = MASK_W_DFB,
                    .accessor_name = "mask_w",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"input_is_dram", "decimal", "num_rows_per_core", "Wt", "tile_offset", "origin_w"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = WRITER_KERNEL_PATH,
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = OUTPUT_DFB, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"output_is_dram", "num_rows_per_core", "Wt", "tile_offset"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Faithful reproduction of the legacy ComputeConfigDescriptor, which set only math_fidelity,
    // fp32_dest_acc_en, and math_approx_mode — leaving dst_full_sync_en at the descriptor default
    // (false). double_buffer_dest defaults to true (== !dst_full_sync_en), matching legacy. The TTNN
    // to_compute_hardware_config helper is deliberately NOT used: it would forward the resolved
    // dst_full_sync_en, which the legacy op never applied.
    m2::ComputeHardwareConfig compute_hw{
        .fpu_math_fidelity = math_fidelity,
        .sfpu_precision_mode =
            math_approx_mode ? tt::tt_metal::Precision::Approximate : tt::tt_metal::Precision::Precise,
        .enable_32_bit_dest = fp32_dest_acc_en,
    };
    // When fp32_dest_acc_en, the intermediate DFBs are Float32 and are consumed by the compute
    // kernel (self-loop), so the Metal 2.0 validator requires an explicit unpack_modes entry. Legacy
    // unpack_to_dest_mode was empty (all Default) → UnpackMode::UnpackToSrc.
    if (fp32_dest_acc_en) {
        compute_hw.unpack_modes = {
            {XABS_DFB, tt::tt_metal::UnpackMode::UnpackToSrc},
            {XPOW_DFB, tt::tt_metal::UnpackMode::UnpackToSrc},
            {LOGX_DFB, tt::tt_metal::UnpackMode::UnpackToSrc},
            {EXP_LXMD_DFB, tt::tt_metal::UnpackMode::UnpackToSrc},
        };
    }

    // Compute kernel. The two legacy per-core-group compute descriptors collapse to a single
    // KernelSpec: they differed only by the CTA {num_units_per_core_group_*}, which the compute
    // kernel never reads (its loop count is the RTA num_rows_per_core). See METAL2_PORT_PLAN.md.
    // The compute-only intermediate DFBs (xabs/xpow/logx/exp_lxmd) are self-looped (bound both
    // PRODUCER and CONSUMER on this one kernel).
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = COMPUTE_KERNEL_PATH,
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = INPUT_DFB, .accessor_name = "x", .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = ONE_DFB, .accessor_name = "one", .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = DECIMAL_DFB,
                    .accessor_name = "decimal",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = MASK_W_DFB,
                    .accessor_name = "mask_w",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = OUTPUT_DFB, .accessor_name = "y", .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = XABS_DFB, .accessor_name = "xabs", .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = XABS_DFB, .accessor_name = "xabs", .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = XPOW_DFB, .accessor_name = "xpow", .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = XPOW_DFB, .accessor_name = "xpow", .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = LOGX_DFB, .accessor_name = "logx", .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = LOGX_DFB, .accessor_name = "logx", .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = EXP_LXMD_DFB,
                    .accessor_name = "exp_lxmd",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = EXP_LXMD_DFB,
                    .accessor_name = "exp_lxmd",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_rows_per_core", "Wt", "origin_w", "p", "p_is_negative"}},
        .hw_config = compute_hw,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelRunArgs reader_run{.kernel = READER};
    m2::KernelRunArgs writer_run{.kernel = WRITER};
    m2::KernelRunArgs compute_run{.kernel = COMPUTE};

    const uint32_t input_is_dram = static_cast<uint32_t>(is_dram(input));
    const uint32_t output_is_dram = static_cast<uint32_t>(is_dram(output));
    const uint32_t decimal_bits = std::bit_cast<uint32_t>(decimal);
    const uint32_t Wt_u = static_cast<uint32_t>(Wt);
    const uint32_t origin_w_u = static_cast<uint32_t>(origin_w);
    const uint32_t p_is_negative_u = static_cast<uint32_t>(p_is_negative);

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        m2::NodeCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_units_per_core;
        if (core_group_1.contains(core)) {
            num_units_per_core = num_units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_units_per_core = num_units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        m2::AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"input_is_dram", input_is_dram},
             {"decimal", decimal_bits},
             {"num_rows_per_core", num_units_per_core},
             {"Wt", Wt_u},
             {"tile_offset", tile_offset},
             {"origin_w", origin_w_u}});

        // writer
        m2::AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"output_is_dram", output_is_dram},
             {"num_rows_per_core", num_units_per_core},
             {"Wt", Wt_u},
             {"tile_offset", tile_offset}});

        // compute
        m2::AddRuntimeArgsForNode(
            compute_run.runtime_arg_values,
            core,
            {{"num_rows_per_core", num_units_per_core},
             {"Wt", Wt_u},
             {"origin_w", origin_w_u},
             {"p", floored_p},
             {"p_is_negative", p_is_negative_u}});

        tile_offset += num_units_per_core * Wt;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble spec + run-args
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec{
        .name = "moreh_abs_pow",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = INPUT_TENSOR, .spec = input.tensor_spec()},
                m2::TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()},
            },
        .work_units = {m2::WorkUnitSpec{
            .name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run), std::move(compute_run)};
    run_args.tensor_args = {
        {INPUT_TENSOR, m2::TensorArgument{input.mesh_tensor()}},
        {OUTPUT_TENSOR, m2::TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::operations::moreh::moreh_abs_pow
