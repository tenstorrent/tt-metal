// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/metal2_artifacts.hpp"
#include "nlp_concat_heads_decode_program_factory.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {
constexpr const char* KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
    "reader_tm_tile_layout_nlp_concat_heads_decode.cpp";
}  // namespace

ttnn::device_operation::ProgramArtifacts NLPConcatHeadsDecodeProgramFactory::create_program_artifacts(
    const NlpConcatHeadsDecodeParams& /*operation_attributes*/,
    const NlpConcatHeadsDecodeInputs& tensor_args,
    Tensor& output) {
    const auto& input_tensor = tensor_args.input;

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t head_dim = input_shape[-1];
    const uint32_t batch = input_shape[1];

    tt_metal::IDevice* device = input_tensor.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    uint32_t element_size = input_tensor.element_size();
    uint32_t sub_tile_line_bytes = 16 * element_size;
    auto q_shard_spec = output.shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;
    auto in_shard_spec = input_tensor.shard_spec().value();
    auto in_cores = in_shard_spec.grid;

    // Output CB (formerly c_16) is borrowed from the output tensor (sharded WIDTH_SHARDED buffer).
    m2::DataflowBufferSpec q_output_dfb{
        .unique_id = m2::DFBSpecName{"q_output"},
        .entry_size = single_tile_size,
        .num_entries = q_num_tiles,
        .data_format_metadata = cb_data_format,
        .borrowed_from = m2::TensorParamName{"output"},
    };

    // cores to read and write to output
    uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    // cores for input
    auto in_core_grid = in_cores.bounding_box();
    uint32_t in_num_cores_x = in_core_grid.end_coord.x + 1, in_num_cores_y = in_core_grid.end_coord.y + 1;

    std::vector<uint32_t> noc_x_coords;
    noc_x_coords.reserve(in_num_cores_x);
    for (uint32_t x = 0; x < in_num_cores_x; ++x) {
        noc_x_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    std::vector<uint32_t> noc_y_coords;
    noc_y_coords.reserve(in_num_cores_y);
    for (uint32_t y = 0; y < in_num_cores_y; ++y) {
        noc_y_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    // We parallelize the reader on risc0 and risc1, where each risc reads a sub-tile of the input (phase1 and phase2 of
    // a tile respectively). Positional compile-time args become NAMED compile-time args; the old cb_id slot is gone
    // (the CB id now comes from the dfb::q_output binding token), and PHASES_TO_READ differs per kernel (reader=1,
    // writer=2). The noc coordinate arrays travel as positional runtime VARARGS after the single named RTA.
    m2::KernelSpec::CompileTimeArgs common_cta = {
        {"element_size", element_size},
        {"sub_tile_line_bytes", sub_tile_line_bytes},
        {"head_size", head_size},
        {"batch", batch},
        {"head_size_num_tiles", head_tiles},
        {"num_x", in_num_cores_x},
        {"num_y", in_num_cores_y}};

    auto reader_cta = common_cta;
    reader_cta.insert({"phases_to_read", 1u});  // read the first phase
    auto writer_cta = common_cta;
    writer_cta.insert({"phases_to_read", 2u});  // read the second phase

    const uint32_t num_varargs = in_num_cores_x + in_num_cores_y;

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{KERNEL_PATH},
        // Borrowed-output CB. Reader and writer each write a different phase of the same output tile via
        // get_write_ptr. Metal 2.0 requires a DFB to have exactly one producer per WorkUnitSpec (and at least
        // one consumer), so we label the reader the sole PRODUCER and the writer the sole CONSUMER. The role
        // label only shapes the dependency graph; both kernels resolve dfb::q_output to the same CB id and the
        // kernel body is unchanged.
        .dfb_bindings = {m2::ProducerOf(m2::DFBSpecName{"q_output"}, "q_output")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "input"}},
        .compile_time_args = reader_cta,
        .runtime_arg_schema = {.runtime_arg_names = {"in_tile_offset_by_head"}},
        // Reader on NCRISC (RISCV_1 / NOC1) so the two data-movement kernels don't collide on the same DM processor.
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
        .advanced_options = m2::KernelAdvancedOptions{.num_runtime_varargs = num_varargs},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{KERNEL_PATH},
        // Writer is the sole CONSUMER of the borrowed-output DFB (see reader's note); it also writes its phase
        // via get_write_ptr on the same dfb::q_output CB id.
        .dfb_bindings = {m2::ConsumerOf(m2::DFBSpecName{"q_output"}, "q_output")},
        // The writer binds both tensors: "input" supplies the read base address; "output" satisfies the
        // ProgramSpec referential-integrity check for the borrowed_from DFB (and supplies its backing
        // L1 address). The "output" accessor token is unused in the kernel body — the writer touches the
        // output only through the borrowed CB (dfb::q_output).
        .tensor_bindings =
            {m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "input"},
             m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "output"}},
        .compile_time_args = writer_cta,
        .runtime_arg_schema = {.runtime_arg_names = {"in_tile_offset_by_head"}},
        // Writer on BRISC (default data-movement config).
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
        .advanced_options = m2::KernelAdvancedOptions{.num_runtime_varargs = num_varargs},
    };

    m2::ProgramSpec spec;
    spec.name = "nlp_concat_heads_decode";
    spec.kernels = {std::move(reader), std::move(writer)};
    spec.dataflow_buffers = {std::move(q_output_dfb)};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "nlp_concat_heads_decode",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
        .target_nodes = q_cores}};

    // Run-args: the complete set (degenerate concept). Per-core named RTA (in_tile_offset_by_head) +
    // per-core varargs (the input NOC coordinate arrays — identical content on every core).
    std::vector<uint32_t> varargs;
    varargs.reserve(num_varargs);
    varargs.insert(varargs.end(), noc_x_coords.begin(), noc_x_coords.end());
    varargs.insert(varargs.end(), noc_y_coords.begin(), noc_y_coords.end());

    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};
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

        const auto& core = cores[i];
        reader_args.runtime_arg_values.push_back({core, {{"in_tile_offset_by_head", in_tile_offset_by_batch}}});
        writer_args.runtime_arg_values.push_back({core, {{"in_tile_offset_by_head", in_tile_offset_by_batch}}});
        reader_args.advanced_options.runtime_varargs.emplace(core, varargs);
        writer_args.advanced_options.runtime_varargs.emplace(core, varargs);
    }

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args.push_back(std::move(reader_args));
    run_params.kernel_run_args.push_back(std::move(writer_args));
    run_params.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(input_tensor.mesh_tensor())});
    run_params.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_params)};
}

}  // namespace ttnn::experimental::prim
