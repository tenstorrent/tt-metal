// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 port — see METAL2_PORT_PLAN.md / METAL2_PORT_REPORT.md alongside the
// tilize op directory. This factory satisfies ProgramSpecFactoryConcept.

#include "tilize_multi_core_sharded_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::metal2_host_api;

namespace ttnn::prim {

// Named namespace (not anonymous) to avoid redefinition in Unity builds where
// multiple factory cpps in this op family get concatenated into one TU.
namespace tilize_multi_core_sharded_factory {

constexpr const char* PROGRAM_ID = "tilize_multi_core_sharded";
constexpr const char* READER = "reader";
constexpr const char* WRITER = "writer";
constexpr const char* COMPUTE = "compute";
constexpr const char* SRC_DFB = "src_dfb";
constexpr const char* OUT_DFB = "out_dfb";
constexpr const char* INPUT = "input";
constexpr const char* OUTPUT = "output";
constexpr const char* MAIN_WU = "main";

NodeRangeSet to_node_range_set(const CoreRangeSet& crs) {
    std::vector<NodeRange> ranges;
    ranges.reserve(crs.ranges().size());
    for (const auto& cr : crs.ranges()) {
        ranges.emplace_back(NodeCoord{cr.start_coord.x, cr.start_coord.y}, NodeCoord{cr.end_coord.x, cr.end_coord.y});
    }
    return NodeRangeSet(std::move(ranges));
}

NodeCoord to_node_coord(const CoreCoord& c) { return NodeCoord{c.x, c.y}; }

}  // namespace tilize_multi_core_sharded_factory

ttnn::device_operation::ProgramArtifacts TilizeMultiCoreShardedProgramFactory::create_program_spec(
    const TilizeParams& /*operation_attributes*/, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tilize_multi_core_sharded_factory;
    const auto& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);
    bool fp32_llk_acc = input.dtype() == DataType::FLOAT32;

    auto shard_spec = input.shard_spec().value();
    uint32_t num_tiles_per_shard = shard_spec.shape[0] * shard_spec.shape[1] / TILE_HW;
    uint32_t num_tiles_per_row = shard_spec.shape[1] / TILE_WIDTH;
    const CoreRangeSet& all_cores = shard_spec.grid;
    NodeRangeSet all_nodes = to_node_range_set(all_cores);

    // ---- DFB specs ----
    DataflowBufferSpec src_dfb{
        .unique_id = SRC_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = INPUT,
    };

    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = OUTPUT,
    };

    // ---- Kernels ----
    KernelSpec reader{
        .unique_id = READER,
        .source =
            KernelSpec::SourceFilePath{
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp"},
        .dfb_bindings = {{
            .dfb_spec_name = SRC_DFB,
            .local_accessor_name = "shard",
            .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER,
        }},
        // Validator requires a kernel to claim each TensorParameter (program_spec.cpp:421).
        .tensor_bindings = {{
            .tensor_parameter_name = INPUT,
            .accessor_name = "input",
        }},
        .runtime_arguments_schema = {.named_runtime_args = {"num_tiles_per_core"}},
        .config_spec =
            DataMovementConfiguration{
                .gen1_data_movement_config =
                    DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                    },
            },
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                                             "writer_unary_sharded_metal2.cpp"},
        .dfb_bindings = {{
            .dfb_spec_name = OUT_DFB,
            .local_accessor_name = "out",
            .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {{
            .tensor_parameter_name = OUTPUT,
            .accessor_name = "output",
        }},
        .runtime_arguments_schema = {.named_runtime_args = {"num_units"}},
        .config_spec =
            DataMovementConfiguration{
                .gen1_data_movement_config =
                    DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                    },
            },
    };

    // unpack_to_dest_mode is keyed by DFB spec name in Metal 2.0 (vector<pair<name,mode>>).
    // Per kernel_spec.hpp: required only when the kernel is a CONSUMER of an FP32 DFB
    // with fp32_dest_acc_en=true. SRC_DFB is consumed by compute; that's the only one
    // that needs the FP32 entry.
    std::vector<ComputeConfiguration::UnpackToDestModeEntry> unpack_to_dest_mode;
    if (fp32_llk_acc) {
        unpack_to_dest_mode.emplace_back(SRC_DFB, UnpackToDestMode::UnpackToDestFp32);
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp"},
        .dfb_bindings =
            {
                {.dfb_spec_name = SRC_DFB,
                 .local_accessor_name = "src",
                 .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER},
                {.dfb_spec_name = OUT_DFB,
                 .local_accessor_name = "dst",
                 .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER},
            },
        .compile_time_arg_bindings =
            {
                {"per_core_block_tile_cnt", num_tiles_per_row},
            },
        .runtime_arguments_schema = {.named_runtime_args = {"per_core_block_cnt"}},
        .config_spec =
            ComputeConfiguration{
                .fp32_dest_acc_en = fp32_llk_acc,
                .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            },
    };

    WorkUnitSpec main_wu{
        .unique_id = MAIN_WU,
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_nodes,
    };

    ProgramSpec spec{
        .program_id = PROGRAM_ID,
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = {std::move(src_dfb), std::move(out_dfb)},
        .tensor_parameters =
            {
                {.unique_id = INPUT, .spec = input.tensor_spec()},
                {.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = {std::move(main_wu)},
    };

    // ---- Run params ----
    using KernelRunParams = ProgramRunParams::KernelRunParams;

    KernelRunParams reader_rp{.kernel_spec_name = READER};
    KernelRunParams writer_rp{.kernel_spec_name = WRITER};
    for (const auto& core_range : all_cores.ranges()) {
        for (const auto& core : core_range) {
            reader_rp.named_runtime_args.push_back({
                .node = to_node_coord(core),
                .args = {{"num_tiles_per_core", num_tiles_per_shard}},
            });
            writer_rp.named_runtime_args.push_back({
                .node = to_node_coord(core),
                .args = {{"num_units", num_tiles_per_shard}},
            });
        }
    }

    KernelRunParams compute_rp{.kernel_spec_name = COMPUTE};
    const uint32_t per_core_block_cnt = num_tiles_per_shard / num_tiles_per_row;
    for (const auto& core_range : all_cores.ranges()) {
        for (const auto& core : core_range) {
            compute_rp.named_runtime_args.push_back({
                .node = to_node_coord(core),
                .args = {{"per_core_block_cnt", per_core_block_cnt}},
            });
        }
    }

    ProgramRunParams run_params;
    run_params.kernel_run_params.push_back(std::move(reader_rp));
    run_params.kernel_run_params.push_back(std::move(writer_rp));
    run_params.kernel_run_params.push_back(std::move(compute_rp));
    // Bind to underlying MeshTensor (Tensor wraps MeshTensor) — see s2i port report.
    run_params.tensor_args = {
        {.tensor_parameter_name = INPUT, .tensor = std::cref(input.mesh_tensor())},
        {.tensor_parameter_name = OUTPUT, .tensor = std::cref(output.mesh_tensor())},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
