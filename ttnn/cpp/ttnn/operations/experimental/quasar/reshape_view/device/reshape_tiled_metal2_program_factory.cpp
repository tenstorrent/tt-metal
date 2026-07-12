// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal-2.0 (Quasar) tiled reshape factory. Mirrors reshape_tiled_program_factory.cpp's host-computed
// input->output page-mapping (via the shared detail::compute_reshape_mapping_host_tensor), but emits a
// ProgramSpec + QuasarDataMovementKernel (Quasar rejects the legacy DataMovementKernel the descriptor /
// workload-descriptor path builds). The mapping tensor is parked on ProgramArtifacts::op_owned_tensors so
// its device allocation outlives the cached program. The reader stages mapping pages + input tiles through
// two shared DFBs; the writer assembles each output tile in a private node-local scratchpad, then writes it
// out (a single DM kernel filling and draining a DFB would be an unsupported Gen2 self-loop).

#include "ttnn/operations/experimental/quasar/reshape_view/device/reshape_tiled_metal2_program_factory.hpp"

#include <functional>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/experimental/quasar/reshape_view/device/reshape_tiled_program_factory.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/device/hostdevcommon/common.hpp"

namespace ttnn::prim::qsr {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
const TensorParamName RT_INPUT{"reshape_tiled_input"};
const TensorParamName RT_OUTPUT{"reshape_tiled_output"};
const TensorParamName RT_MAP{"reshape_tiled_map"};
const DFBSpecName RT_MAP_DFB{"reshape_tiled_map_dfb"};
const DFBSpecName RT_INPUT_DFB{"reshape_tiled_input_dfb"};
const ScratchpadSpecName RT_WORKING{"reshape_tiled_working"};
const KernelSpecName RT_READER{"reshape_tiled_reader"};
const KernelSpecName RT_WRITER{"reshape_tiled_writer"};

// PCC fails when this is greater than 1 (matches the legacy factory's reader_cb_len). TODO figure out why.
constexpr uint32_t reader_cb_len = 1;
}  // namespace

ttnn::device_operation::ProgramArtifacts ReshapeViewTiledMetalV2ProgramFactory::create_program_artifacts(
    const ReshapeViewParams& operation_attributes, const ReshapeViewInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;

    const auto& input_shape = input.logical_shape();
    const auto& output_shape = output.logical_shape();

    TT_FATAL(input_shape.volume() == output_shape.volume(), "Requested shapes are not of equal volume");
    TT_ASSERT(input_shape.size() == 3 && output_shape.size() == 3, "Kernel designed for rank 3 tensors");

    const auto& tile_shape = input.tensor_spec().tile().get_tile_shape();
    const auto& face_shape = input.tensor_spec().tile().get_face_shape();

    distributed::MeshDevice* device = input.device();
    const auto grid = device->compute_with_storage_grid_size();

    const uint32_t num_input_pages = tt::div_up(input.physical_volume(), tile_shape[0] * tile_shape[1]);
    const uint32_t num_output_pages = tt::div_up(output.physical_volume(), tile_shape[0] * tile_shape[1]);

    // ---- Host-compute and upload the input->output page-mapping tensor (op-owned) ----
    // recreate_mapping_tensor is intentionally ignored (excluded from the program hash); the mapping
    // depends only on the hashed input/output shapes.
    Tensor mapping_tensor = detail::compute_reshape_mapping_host_tensor(
                                num_input_pages, num_output_pages, input_shape, output_shape, tile_shape, face_shape)
                                .to_device(device);

    const uint32_t mapping_page_size = mapping_tensor.logical_shape()[-1];
    const uint32_t mapping_page_size_bytes = mapping_page_size * mapping_tensor.element_size();
    const uint32_t max_map_entries = mapping_page_size / detail::SegmentMapData::size;

    // Release the sole-owner MeshTensor (the source Tensor is left deallocated; ~Tensor will not
    // force-free the device buffer that the cached program still references).
    tt::tt_metal::MeshTensor mapping_mesh_tensor = mapping_tensor.device_storage().release_mesh_tensor();

    // ---- Tile CB sizing / formats ----
    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_tile_size_bytes = tt::tile_size(input_cb_data_format);
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_tile_size_bytes = tt::tile_size(output_cb_data_format);
    const uint32_t element_sz_bytes = tt::datum_size(output_cb_data_format);

    // ---- Work split over output pages ----
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
            operation_attributes.sub_core_grid.has_value()
                ? split_work_to_cores(operation_attributes.sub_core_grid.value(), num_output_pages)
                : split_work_to_cores(grid, num_output_pages);
    TT_ASSERT(num_cores <= num_output_pages);

    // ---- ProgramSpec ----
    ProgramSpec spec;
    spec.name = "reshape_view_tiled";

    spec.tensor_parameters = {
        TensorParameter{.unique_id = RT_INPUT, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = RT_OUTPUT, .spec = output.tensor_spec()},
        TensorParameter{.unique_id = RT_MAP, .spec = mapping_mesh_tensor.tensor_spec()},
    };

    // Reader produces mapping pages + input tiles; writer consumes them. num_entries mirrors the legacy
    // reader_cb_len (=1): reader/writer ping-pong one entry at a time.
    spec.dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = RT_MAP_DFB,
            .entry_size = mapping_page_size_bytes,
            .num_entries = reader_cb_len,
            // The mapping DFB carries raw uint32 page-map rows. Every DFB needs a valid data_format_metadata
            // (finalize_single_dfb_config calls tile_size on it -> DataFormat::Invalid throws "Invalid data
            // format"). Use Int32 (bit-identical to the uint32 indices, 4B) — Quasar has Int32 on the device
            // side but NOT UInt32, so Int32 is the portable choice.
            .data_format_metadata = tt::DataFormat::Int32},
        DataflowBufferSpec{
            .unique_id = RT_INPUT_DFB,
            .entry_size = input_tile_size_bytes,
            .num_entries = reader_cb_len,
            .data_format_metadata = input_cb_data_format},
    };

    // Private node-local scratch page the writer assembles each output tile into (single-kernel fill+drain).
    spec.scratchpads = {
        ScratchpadSpec{.unique_id = RT_WORKING, .size_per_node = output_tile_size_bytes},
    };

    // ---- Reader kernel ----
    KernelSpec reader{
        .unique_id = RT_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/reshape_view/device/device/dataflow/"
            "reader_reshape_tiled_metal2.cpp",
        .dfb_bindings = {ProducerOf(RT_MAP_DFB, "mapping"), ProducerOf(RT_INPUT_DFB, "input")},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = RT_INPUT, .accessor_name = "input"},
             TensorBinding{.tensor_parameter_name = RT_MAP, .accessor_name = "map"}},
        .compile_time_args =
            {{"max_map_size_bytes", mapping_page_size_bytes}, {"tile_size_bytes", input_tile_size_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_output_page_idx", "end_output_page_idx"}},
        // Explicit CB credit ops (staged reads) -> disable implicit sync (Quasar tile-counter double-count).
        .hw_config =
            DataMovementHardwareConfig{
                .role = DataMovementRoleHint::READER,
                .gen2_config = DataMovementHardwareConfig::Gen2Config{.disable_dfb_implicit_sync_for_all = true}},
    };

    // ---- Writer kernel ----
    KernelSpec writer{
        .unique_id = RT_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/reshape_view/device/device/dataflow/"
            "writer_reshape_tiled_metal2.cpp",
        // Field order must match KernelSpec declaration order (dfb_bindings, scratchpad_bindings,
        // tensor_bindings) -- designated initializers require it (-Werror=reorder-init-list).
        .dfb_bindings = {ConsumerOf(RT_MAP_DFB, "mapping"), ConsumerOf(RT_INPUT_DFB, "input")},
        .scratchpad_bindings = {ScratchpadBinding{.scratchpad_spec_name = RT_WORKING, .accessor_name = "working"}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = RT_OUTPUT, .accessor_name = "dst"}},
        .compile_time_args =
            {{"tile_size_bytes", input_tile_size_bytes},
             {"max_map_entries", max_map_entries},
             {"element_sz_bytes", element_sz_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_output_page", "end_output_page"}},
        .hw_config =
            DataMovementHardwareConfig{
                .role = DataMovementRoleHint::WRITER,
                .gen2_config = DataMovementHardwareConfig::Gen2Config{.disable_dfb_implicit_sync_for_all = true}},
    };

    spec.kernels = {reader, writer};
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {RT_READER, RT_WRITER}, .target_nodes = all_cores}};

    // ---- Per-core runtime args ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = RT_READER};
    KernelRunArgs writer_run{.kernel = RT_WRITER};

    uint32_t page_idx_start = 0;
    for (const auto& core : corerange_to_cores(all_cores, std::nullopt)) {
        uint32_t increment = 0;
        if (core_group_1.contains(core)) {
            increment = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            increment = num_tiles_per_core_group_2;
        } else {
            continue;
        }
        const uint32_t page_idx_end = page_idx_start + increment;

        reader_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core, .args = {{"start_output_page_idx", page_idx_start}, {"end_output_page_idx", page_idx_end}}});
        writer_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core, .args = {{"start_output_page", page_idx_start}, {"end_output_page", page_idx_end}}});

        page_idx_start = page_idx_end;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run));
    run_args.kernel_run_args.push_back(std::move(writer_run));

    // ---- Op-owned tensors ----
    std::vector<tt::tt_metal::MeshTensor> op_owned_tensors;
    op_owned_tensors.reserve(1);
    op_owned_tensors.push_back(std::move(mapping_mesh_tensor));
    const tt::tt_metal::MeshTensor& mapping_owned = op_owned_tensors[0];

    // ---- Tensor args ----
    run_args.tensor_args.emplace(RT_INPUT, std::cref(input.mesh_tensor()));
    run_args.tensor_args.emplace(RT_OUTPUT, std::cref(output.mesh_tensor()));
    run_args.tensor_args.emplace(RT_MAP, std::cref(mapping_owned));

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
        .op_owned_tensors = std::move(op_owned_tensors),
    };
}

}  // namespace ttnn::prim::qsr
