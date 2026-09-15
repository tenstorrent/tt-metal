// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 program factory for integral_image.
//
// Faithful port of the legacy single-descriptor create_descriptor (see METAL2_PORT_PLAN.md). The op has one
// config: interleaved, a fixed 2x4 core grid, every node running reader+compute+writer. The port is a
// syntax swap only:
//   - CBDescriptor            -> DataflowBufferSpec (9 DFBs, one per legacy CB index).
//   - magic CB-index CTAs      -> DFBBindings (endpoint census in the plan; 5 self-loops on compute).
//   - TensorAccessorArgs + buffer-address RTAs -> TensorParameter/TensorBinding (input on reader, output on writer).
//   - positional scalar CTAs   -> named compile_time_args (the same 9 on all three kernels).
// Numerics, placement, and hardware config are unchanged.

#include "intimg_device_operation.hpp"

#include <array>
#include <variant>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/types.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;
using ttnn::device_operation::ProgramArtifacts;

constexpr std::array<const char*, 3> KERNEL_PATHS{
    "ttnn/cpp/ttnn/operations/experimental/reduction/integral_image/device/kernels/"
    "intimg_reader.cpp",
    "ttnn/cpp/ttnn/operations/experimental/reduction/integral_image/device/kernels/intimg_compute.cpp",
    "ttnn/cpp/ttnn/operations/experimental/reduction/integral_image/device/kernels/"
    "intimg_writer.cpp"};

}  // namespace

namespace ttnn::experimental::prim {

// it is expected that this operator is used primarily on BOS' custom chips, which are 4 rows and 5 columns, however the
// expected parallelisation of the maximal input shape is calculated to be 4 rows and 2 columns
constexpr uint32_t CORES_X = 2;
constexpr uint32_t CORES_Y = 4;

ttnn::device_operation::ProgramArtifacts IntImgDeviceOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor{tensor_args};
    auto& output_tensor{tensor_return_value};
    const auto& input_shape{input_tensor.padded_shape()};

    constexpr uint32_t BLOCK_DEPTH = 48;

    const auto cb_data_format{datatype_to_dataformat_converter(input_tensor.dtype())};
    const uint32_t single_tile_size{tt::tile_size(cb_data_format)};
    const bool fp32_dest_acc_en{
        (cb_data_format == DataFormat::Float32) || (cb_data_format == DataFormat::Int32) ||
        (cb_data_format == DataFormat::UInt32)};

    const auto tile_spec = input_tensor.tensor_spec().tile();

    constexpr uint32_t tiles_num_per_full_block_depth_cb = BLOCK_DEPTH;
    constexpr uint32_t tiles_num_per_small_cb = 2;

    // ---- DataflowBuffer names (one per legacy CB index) ----
    const m2::DFBSpecName START{"start"};
    const m2::DFBSpecName INPUT{"input"};
    const m2::DFBSpecName ACC{"acc"};
    const m2::DFBSpecName CUMSUM_STAGE_0{"cumsum_stage_0"};
    const m2::DFBSpecName CUMSUM_STAGE_1{"cumsum_stage_1"};
    const m2::DFBSpecName CUMSUM_STAGE_2{"cumsum_stage_2"};
    const m2::DFBSpecName OUTPUT{"output"};
    const m2::DFBSpecName AXIS_2_BUFFER{"axis_2_buffer"};  // memoizing last tile (for the "deeper" block) for
                                                           // propagation along axis 2
    const m2::DFBSpecName AXIS_3_BUFFER{"axis_3_buffer"};  // memoizing upper 32 tiles for propagation along axis 3

    // ---- Tensor parameter names ----
    const m2::TensorParamName T_INPUT{"input"};
    const m2::TensorParamName T_OUTPUT{"output"};

    // ---- Kernel names ----
    const m2::KernelSpecName READER{"intimg_reader"};
    const m2::KernelSpecName COMPUTE{"intimg_compute"};
    const m2::KernelSpecName WRITER{"intimg_writer"};

    // ---- DataflowBufferSpecs (were CBDescriptors; entry_size/data_format copied verbatim, tile default 32x32) ----
    auto make_dfb = [&](const m2::DFBSpecName& name, uint32_t num_entries) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = single_tile_size,
            .num_entries = num_entries,
            .data_format_metadata = cb_data_format,
        };
    };
    m2::Group<m2::DataflowBufferSpec> dfbs{
        make_dfb(START, tiles_num_per_small_cb),
        make_dfb(INPUT, tiles_num_per_full_block_depth_cb),
        make_dfb(ACC, tiles_num_per_small_cb),
        make_dfb(CUMSUM_STAGE_0, tiles_num_per_full_block_depth_cb),
        make_dfb(CUMSUM_STAGE_1, tiles_num_per_full_block_depth_cb),
        make_dfb(CUMSUM_STAGE_2, tiles_num_per_full_block_depth_cb),
        make_dfb(OUTPUT, tiles_num_per_full_block_depth_cb),
        make_dfb(AXIS_2_BUFFER, tiles_num_per_small_cb),
        make_dfb(AXIS_3_BUFFER, tiles_num_per_full_block_depth_cb),
    };

    // ---- Named compile-time args (the 9 legacy scalar CTAs, slots 9-17). Emitted on all three kernels; each
    //      kernel's get_ctas() reads all nine by name. ----
    const m2::KernelSpec::CompileTimeArgs scalar_ctas{
        {"tile_height", tile_spec.get_height()},
        {"tile_width", tile_spec.get_width()},
        {"block_depth", BLOCK_DEPTH},
        {"num_channels", static_cast<uint32_t>(input_shape[3])},
        {"input_height", static_cast<uint32_t>(input_shape[2])},
        {"input_depth", static_cast<uint32_t>(input_shape[1])},
        {"num_batches", static_cast<uint32_t>(input_shape[0])},
        {"cores_x", CORES_X},
        {"cores_y", CORES_Y},
    };

    const auto arch = input_tensor.device()->arch();

    // ---- Reader (DM). Produces START (zero-fill) and INPUT (DRAM load). Reads the input tensor via binding. ----
    m2::KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path(KERNEL_PATHS[0]),
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = START, .accessor_name = "start", .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = INPUT, .accessor_name = "input", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = T_INPUT, .accessor_name = "input"}},
        .compile_time_args = scalar_ctas,
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    // ---- Compute. Consumes START/INPUT/AXIS_3_BUFFER, produces OUTPUT, self-loops ACC/CUMSUM_STAGE_0/1/2/
    //      AXIS_2_BUFFER (both PRODUCER and CONSUMER on the compute kernel). ----
    m2::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(KERNEL_PATHS[1]),
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = START, .accessor_name = "start", .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = INPUT, .accessor_name = "input", .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = ACC, .accessor_name = "acc", .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = ACC, .accessor_name = "acc", .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = CUMSUM_STAGE_0,
                 .accessor_name = "cumsum_stage_0",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = CUMSUM_STAGE_0,
                 .accessor_name = "cumsum_stage_0",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = CUMSUM_STAGE_1,
                 .accessor_name = "cumsum_stage_1",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = CUMSUM_STAGE_1,
                 .accessor_name = "cumsum_stage_1",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = CUMSUM_STAGE_2,
                 .accessor_name = "cumsum_stage_2",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = CUMSUM_STAGE_2,
                 .accessor_name = "cumsum_stage_2",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = OUTPUT, .accessor_name = "output", .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = AXIS_2_BUFFER,
                 .accessor_name = "axis_2_buffer",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = AXIS_2_BUFFER,
                 .accessor_name = "axis_2_buffer",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = AXIS_3_BUFFER,
                 .accessor_name = "axis_3_buffer",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .compile_time_args = scalar_ctas,
    };
    // Compute hw_config — Style B (legacy set a Metal ComputeConfig directly). Build ComputeGen1Config; carry the
    // resolved legacy values (HiFi4, math_approx_mode=false -> Precise, fp32_dest_acc_en -> enable_32_bit_dest).
    m2::ComputeGen1Config compute_cfg{
        .fpu_math_fidelity = MathFidelity::HiFi4,
        .sfpu_precision_mode = Precision::Precise,
        .enable_32_bit_dest = fp32_dest_acc_en,
    };
    // Validator: with enable_32_bit_dest, every consumed Float32 DFB needs an explicit unpack_mode. Legacy left the
    // mode default (=UnpackToSrc); replicate that for the 8 DFBs the compute kernel consumes (OUTPUT is producer-only).
    // For bf16 input enable_32_bit_dest is false and no entries are required.
    if (fp32_dest_acc_en) {
        compute_cfg.unpack_modes = m2::ComputeUnpackModes{
            {START, UnpackMode::UnpackToSrc},
            {INPUT, UnpackMode::UnpackToSrc},
            {ACC, UnpackMode::UnpackToSrc},
            {CUMSUM_STAGE_0, UnpackMode::UnpackToSrc},
            {CUMSUM_STAGE_1, UnpackMode::UnpackToSrc},
            {CUMSUM_STAGE_2, UnpackMode::UnpackToSrc},
            {AXIS_2_BUFFER, UnpackMode::UnpackToSrc},
            {AXIS_3_BUFFER, UnpackMode::UnpackToSrc},
        };
    }
    compute_spec.hw_config = m2::ComputeHardwareConfig{compute_cfg};

    // ---- Writer (DM). Consumes OUTPUT (DRAM write), produces AXIS_3_BUFFER (cross-row readback). One output
    //      TensorBinding covers both the write and the readback (both go through the same TensorAccessor). ----
    m2::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path(KERNEL_PATHS[2]),
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = OUTPUT, .accessor_name = "output", .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = AXIS_3_BUFFER,
                 .accessor_name = "axis_3_buffer",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = T_OUTPUT, .accessor_name = "output"}},
        .compile_time_args = scalar_ctas,
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    // ---- WorkUnit: all three kernels on the fixed 2x4 grid (placement derives DFB residency). ----
    m2::WorkUnitSpec wu{
        .name = "intimg",
        .kernels = {READER, COMPUTE, WRITER},
        .target_nodes = m2::NodeRange{{0, 0}, {CORES_X - 1, CORES_Y - 1}},
    };

    m2::ProgramSpec spec{
        .name = "intimg",
        .kernels = {std::move(reader_spec), std::move(compute_spec), std::move(writer_spec)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {m2::TensorParameter{.unique_id = T_INPUT, .spec = input_tensor.tensor_spec()},
             m2::TensorParameter{.unique_id = T_OUTPUT, .spec = output_tensor.tensor_spec()}},
        .work_units = {std::move(wu)},
    };

    // ---- Run-args: no runtime args on any kernel (both address RTAs became TensorBindings); only tensor args. ----
    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{.kernel = READER},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = WRITER},
    };
    run_params.tensor_args.emplace(T_INPUT, m2::ProgramRunArgs::TensorArgument{input_tensor.mesh_tensor()});
    run_params.tensor_args.emplace(T_OUTPUT, m2::ProgramRunArgs::TensorArgument{output_tensor.mesh_tensor()});

    return ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
        .op_owned_tensors = {},
    };
}

}  // namespace ttnn::experimental::prim
