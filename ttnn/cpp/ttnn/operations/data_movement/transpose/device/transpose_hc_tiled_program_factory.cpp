// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

const DFBSpecName HCT_SRC0_DFB{"hct_src0"};
const DFBSpecName HCT_SCRATCH_DFB{"hct_scratch"};
const TensorParamName HCT_INPUT{"hct_input"};
const TensorParamName HCT_OUTPUT{"hct_output"};
const KernelSpecName HCT_READER{"hct_reader"};
const KernelSpecName HCT_WRITER{"hct_writer"};

void emit_runtime_args_hc_tiled(
    KernelRunArgs& reader_run,
    KernelRunArgs& writer_run,
    const Tensor& input_tensor,
    const CoreRangeSet& all_cores,
    const CoreRangeSet& core_group_1,
    uint32_t num_tiles_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_tiles_per_core_group_2) {
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1];
    uint32_t HW = H * W;
    uint32_t HW_bytes = HW * input_tensor.element_size();
    uint32_t CHW_bytes = C * HW * input_tensor.element_size();

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ct = C / TILE_HEIGHT;
    uint32_t CtHWt = Ct * H * Wt;
    uint32_t CtWt = Ct * Wt;

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t num_tiles_read = 0;
    for (const auto& core : cores) {
        uint32_t num_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t h = num_tiles_read / CtWt % H;
        uint32_t ct = num_tiles_read / Wt % Ct;

        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"WT", Wt},
             {"H", H},
             {"CT", Ct},
             {"HW_bytes", HW_bytes},
             {"CHW_bytes", CHW_bytes},
             {"start_id", num_tiles_read},
             {"num_tiles", num_tiles_per_core},
             {"batch_addr", num_tiles_read / CtHWt * CHW_bytes},
             {"h", h},
             {"htWT", h / TILE_HEIGHT * Wt},
             {"ct", ct},
             {"ctoffs", ct * TILE_HEIGHT * HW_bytes},
             {"wt", num_tiles_read % Wt}});

        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values, core, {{"num_pages", num_tiles_per_core}, {"start_id", num_tiles_read}});

        num_tiles_read += num_tiles_per_core;
    }
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeHCTiledProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    uint32_t sub_tile_line_bytes = 16 * input_tensor.element_size();
    uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "transpose_hc_tiled");
    log_debug(tt::LogOp, "sub_tile_line_bytes: {}", sub_tile_line_bytes);
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);

    IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // The kernel reads several 16 element face lines (32B for BFLOAT16) from different input tiles to form a single
    // output tile. Each face line is 32 bytes, so if our minimum read alignment is greater than that (64B for
    // Blackhole) we need a scratch buffer to stage from the nearest aligned address.
    uint32_t alignment = dst_buffer->alignment();
    bool misaligned = alignment > sub_tile_line_bytes;

    // ---- ProgramSpec ----
    ProgramSpec spec;
    spec.name = "transpose_hc_tiled";

    spec.tensor_parameters = {
        TensorParameter{.unique_id = HCT_INPUT, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = HCT_OUTPUT, .spec = output_tensor.tensor_spec()},
    };

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = HCT_SRC0_DFB,
        .entry_size = single_tile_size,
        .num_entries = 2,
        .data_format_metadata = cb_data_format,
    });
    if (misaligned) {
        // Scratch staging buffer: touched only by the reader (base-pointer access via
        // get_write_ptr, no cross-kernel FIFO) → self-loop DFB, bound only when misaligned.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = HCT_SCRATCH_DFB,
            .entry_size = alignment,
            .num_entries = 1,
            .data_format_metadata = cb_data_format,
        });
    }

    KernelSpec::CompilerOptions::Defines scratch_defines;
    if (misaligned) {
        scratch_defines.emplace("TRANSPOSE_HC_SCRATCH", "1");
    }

    Group<DFBBinding> reader_dfb = {
        DFBBinding{.dfb_spec_name = HCT_SRC0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER}};
    if (misaligned) {
        reader_dfb.push_back(DFBBinding{
            .dfb_spec_name = HCT_SCRATCH_DFB, .accessor_name = "scratch", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_dfb.push_back(DFBBinding{
            .dfb_spec_name = HCT_SCRATCH_DFB, .accessor_name = "scratch", .endpoint_type = DFBEndpointType::CONSUMER});
    }

    KernelSpec reader{
        .unique_id = HCT_READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "reader_unary_transpose_hc_interleaved_partitioned.cpp",
        .compiler_options = {.defines = scratch_defines},
        .dfb_bindings = reader_dfb,
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = HCT_INPUT, .accessor_name = "src"}},
        .compile_time_args =
            {{"SUBTILE_LINE_BYTES", sub_tile_line_bytes},
             {"FLOAT32_DTYPE", cb_data_format == tt::DataFormat::Float32 ? 1u : 0u},
             {"ALIGNMENT", alignment}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"WT",
                  "H",
                  "CT",
                  "HW_bytes",
                  "CHW_bytes",
                  "start_id",
                  "num_tiles",
                  "batch_addr",
                  "h",
                  "htWT",
                  "ct",
                  "ctoffs",
                  "wt"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // Borrowed writer: bound from the Metal 2.0 fork that lives beside its legacy original under
    // eltwise/unary (the legacy source still serves ~45 unmigrated binders). The fork's binding
    // names are this factory's constraint, not the reverse.
    KernelSpec writer{
        .unique_id = HCT_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = HCT_SRC0_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = HCT_OUTPUT, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    spec.kernels = {reader, writer};
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {HCT_READER, HCT_WRITER}, .target_nodes = all_cores}};

    // ---- ProgramRunArgs ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = HCT_READER};
    KernelRunArgs writer_run{.kernel = HCT_WRITER};
    emit_runtime_args_hc_tiled(
        reader_run,
        writer_run,
        input_tensor,
        all_cores,
        core_group_1,
        num_tiles_per_core_group_1,
        core_group_2,
        num_tiles_per_core_group_2);

    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args.emplace(HCT_INPUT, TensorArgument{input_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(HCT_OUTPUT, TensorArgument{output_tensor.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
