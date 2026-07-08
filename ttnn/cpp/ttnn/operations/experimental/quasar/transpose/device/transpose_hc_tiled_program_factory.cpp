// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <vector>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts TransposeHCTiledProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName SRC_CB{"src_cb"};          // legacy c_0: reader produces tiles, writer consumes them
    const DFBSpecName SCRATCH_CB{"scratch_cb"};  // legacy c_1: misaligned-read scratch (reader-only)

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_interleaved_partitioned.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id.cpp";

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
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // check if we need to allocate a scratch buffer
    // The kernel reads several 16 element face lines (32B for BFLOAT16) from different input tiles to form a single
    // output tile, one output tile at a time Each face line is 32 bytes, so if our minimum read alignment is greater
    // than that (64B for Blackhole) then we will have reads from unaligned face-lines into differently aligned
    // destination face-lines
    // TODO: noc_async_write only require 16B alignment for both DRAM and L1 for Blackhole, so instead of reading in
    // face-lines from C tiles to form a single tile, we can load a single tile and then write out its face-lines to C
    // tiles
    uint32_t alignment = dst_buffer->alignment();
    bool misaligned = alignment > sub_tile_line_bytes;

    // ------------------------------------------------------------------------
    // DataflowBufferSpecs. src_cb (c_0) always present; the scratch CB (c_1) only when a misaligned
    // read forces a staged copy through nearest-aligned scratch L1.
    // ------------------------------------------------------------------------
    uint32_t num_input_tiles = 2;
    std::vector<DataflowBufferSpec> dfbs;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = SRC_CB,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    });
    // need some scratch memory here - if we need data from a misaligned address then we need to read from the
    // nearest aligned address and then copy the data to the correct location
    if (misaligned) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = SCRATCH_CB,
            .entry_size = alignment,
            .num_entries = 1,
            .data_format_metadata = cb_data_format,
        });
    }

    // ------------------------------------------------------------------------
    // Tensor parameters. The legacy accessors carried RuntimeTensorShape → dynamic_tensor_shape.
    // ------------------------------------------------------------------------
    TensorParameter input_param{
        .unique_id = INPUT_TENSOR,
        .spec = input_tensor.tensor_spec(),
        .advanced_options = {.dynamic_tensor_shape = true},
    };
    TensorParameter output_param{
        .unique_id = OUTPUT_TENSOR,
        .spec = output_tensor.tensor_spec(),
        .advanced_options = {.dynamic_tensor_shape = true},
    };

    // ------------------------------------------------------------------------
    // Reader: reverse-maps each output tile to source face-lines into src_cb (c_0). The scratch CB
    // (c_1) and the MISALIGNED_SCRATCH define are only added on the misaligned path.
    // ------------------------------------------------------------------------
    Group<DFBBinding> reader_dfbs = {
        DFBBinding{.dfb_spec_name = SRC_CB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    KernelSpec::CompilerOptions::Defines reader_defines;
    if (misaligned) {
        reader_dfbs.push_back(DFBBinding{
            .dfb_spec_name = SCRATCH_CB, .accessor_name = "scratch", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_defines.emplace("MISALIGNED_SCRATCH", "1");
    }

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_PATH},
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfbs),
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
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

    // ------------------------------------------------------------------------
    // Writer: consumes src_cb (c_0) and streams the transposed tiles out (interleaved start_id).
    // ------------------------------------------------------------------------
    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC_CB, .accessor_name = "cb_out0", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
        .compile_time_args = {{"page_size", single_tile_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ------------------------------------------------------------------------
    // Per-core runtime args (identical traversal/derivations to the legacy factory).
    // ------------------------------------------------------------------------
    auto input_shape = input_tensor.padded_shape();
    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1];
    uint32_t HW = H * W;
    uint32_t HW_bytes = HW * input_tensor.element_size();
    uint32_t CHW_bytes = C * HW * input_tensor.element_size();

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ct = C / TILE_HEIGHT;
    uint32_t CtHWt = Ct * H * Wt;
    uint32_t CtWt = Ct * Wt;

    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    reader_run.runtime_arg_values.reserve(num_cores_total);
    writer_run.runtime_arg_values.reserve(num_cores_total);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            num_tiles_per_core = 0;
        }

        uint32_t h = num_tiles_read / CtWt % H;
        uint32_t ct = num_tiles_read / Wt % Ct;

        const NodeCoord node = core;
        reader_run.runtime_arg_values.push_back(
            {node,
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
              {"wt", num_tiles_read % Wt}}});
        writer_run.runtime_arg_values.push_back(
            {node, {{"num_pages", num_tiles_per_core}, {"start_id", num_tiles_read}}});

        num_tiles_read += num_tiles_per_core;
    }

    WorkUnitSpec wu{
        .name = "transpose_hc_tiled",
        .kernels = {READER_KERNEL, WRITER_KERNEL},
        .target_nodes = total_cores,
    };

    ProgramSpec spec{
        .name = "transpose_hc_tiled",
        .kernels = {std::move(reader_spec), std::move(writer_spec)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output_tensor.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
