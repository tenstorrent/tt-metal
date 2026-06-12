// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "nlp_concat_heads_decode_program_factory.hpp"
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <string>
#include <vector>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts NLPConcatHeadsDecodeProgramFactory::create_program_spec(
    const NlpConcatHeadsDecodeParams& /*operation_attributes*/,
    const NlpConcatHeadsDecodeInputs& tensor_args,
    Tensor& output) {
    // Metal 2.0 named resource handles for the nlp_concat_heads_decode ProgramSpec.
    const DFBSpecName OUT_DFB{"q_out"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    constexpr const char* KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_concat_heads_decode.cpp";

    const auto& input_tensor = tensor_args.input;

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t head_dim = input_shape[-1];
    const uint32_t batch = input_shape[1];

    tt_metal::IDevice* device = input_tensor.device();

    const tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);

    const uint32_t head_tiles = head_dim / TILE_WIDTH;
    const uint32_t head_size = head_tiles * single_tile_size;

    const uint32_t element_size = input_tensor.element_size();
    const uint32_t sub_tile_line_bytes = 16 * element_size;
    const auto q_shard_spec = output.shard_spec().value();
    const auto q_cores = q_shard_spec.grid;
    const auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;
    const auto in_shard_spec = input_tensor.shard_spec().value();
    const auto in_cores = in_shard_spec.grid;

    // cores to read and write to output
    const uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    const auto core_grid = q_cores.bounding_box();
    const uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    // cores for input
    const auto in_core_grid = in_cores.bounding_box();
    const uint32_t in_num_cores_x = in_core_grid.end_coord.x + 1, in_num_cores_y = in_core_grid.end_coord.y + 1;

    // NoC coordinates of the input shard cores. Identical for every output core, so they are
    // passed as Metal 2.0 *common* runtime varargs (broadcast), laid out [x0..x_{nx-1}, y0..y_{ny-1}].
    std::vector<uint32_t> noc_coords;
    noc_coords.reserve(in_num_cores_x + in_num_cores_y);
    for (uint32_t x = 0; x < in_num_cores_x; ++x) {
        noc_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    for (uint32_t y = 0; y < in_num_cores_y; ++y) {
        noc_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    // ----------------------------------------------------------------------------
    // Tensor parameters. INPUT supplies the input shard base address (Case 2 bridge,
    // recovered kernel-side via get_bank_base_address). OUTPUT backs the borrowed DFB.
    // ----------------------------------------------------------------------------
    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()};

    // ----------------------------------------------------------------------------
    // Output DFB: borrowed memory on output.buffer() (legacy CB c_16). Write-only
    // address source (fake CB): bound reader=PRODUCER / writer=CONSUMER on the same
    // nodes to satisfy the validator's producer-and-consumer rule. See PORT_REPORT.
    // ----------------------------------------------------------------------------
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = q_num_tiles,
        .data_format_metadata = cb_data_format,
        .borrowed_from = OUTPUT_TENSOR,
    };

    // Named CTAs shared by reader and writer (only PHASES_TO_READ differs).
    auto make_cta = [&](uint32_t phases_to_read) {
        return KernelSpec::CompileTimeArgs{
            {"element_size", element_size},
            {"sub_tile_line_bytes", sub_tile_line_bytes},
            {"head_size", head_size},
            {"batch", batch},
            {"head_size_num_tiles", head_tiles},
            {"phases_to_read", phases_to_read},
            {"num_x", in_num_cores_x},
            {"num_y", in_num_cores_y}};
    };

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{KERNEL_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "q_out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .compile_time_args = make_cta(1),  // read first phase
        .runtime_arg_schema = {.runtime_arg_names = {"in_tile_offset_by_head"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };
    reader_spec.advanced_options.num_common_runtime_varargs = noc_coords.size();

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{KERNEL_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "q_out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .compile_time_args = make_cta(2),  // read second phase
        .runtime_arg_schema = {.runtime_arg_names = {"in_tile_offset_by_head"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };
    writer_spec.advanced_options.num_common_runtime_varargs = noc_coords.size();

    // ----------------------------------------------------------------------------
    // Per-node runtime args: in_tile_offset_by_head (per output core), plus the
    // broadcast NoC-coordinate varargs.
    // ----------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    reader_run.advanced_options.common_runtime_varargs = noc_coords;
    writer_run.advanced_options.common_runtime_varargs = noc_coords;

    for (uint32_t i = 0; i < num_cores; ++i) {
        // Each output core i corresponds to head index i. Within the input shard, that head lives in
        // head-tile (i / 32) at row (i % 32). The two cases below pick the row's byte offset within
        // a single 32x32 tile (face 0 for rows < 16, face 2 for rows >= 16); add the head-tile skip
        // to land in the right tile when padded_heads > 32.
        uint32_t head_tile_idx = i / 32;
        uint32_t head_in_tile = i % 32;
        uint32_t in_tile_offset_by_batch =
            (head_in_tile < 16 ? head_in_tile * sub_tile_line_bytes
                               : (head_in_tile - 16) * sub_tile_line_bytes + 512 * element_size) +
            head_tile_idx * head_size;

        const NodeCoord core = cores[i];
        const KernelRunArgs::RuntimeArgValues args{{"in_tile_offset_by_head", in_tile_offset_by_batch}};
        reader_run.runtime_arg_values.push_back({core, args});
        writer_run.runtime_arg_values.push_back({core, args});
    }

    WorkUnitSpec wu{
        .name = "nlp_concat_heads_decode",
        .kernels = {READER_KERNEL, WRITER_KERNEL},
        .target_nodes = q_cores,
    };

    ProgramSpec spec{
        .name = "nlp_concat_heads_decode",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {out_dfb_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
