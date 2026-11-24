// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <map>

#include "offset_cumsum_program_factory.hpp"
#include "offset_cumsum_device_operation_types.hpp"

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim::qsr {

namespace {

ttnn::device_operation::ProgramArtifacts create_program_artifacts_at(
    const OffsetCumsumParams& operation_attributes,
    const Tensor& input,
    OffsetCumsumProgramFactory::tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinate& coord);

}  // namespace

ttnn::device_operation::MeshWorkloadArtifacts OffsetCumsumProgramFactory::create_mesh_workload_artifacts(
    const OffsetCumsumParams& operation_attributes,
    const Tensor& input,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    // Preserve the exact populated coordinate set, including sparse tensors.
    // Merge only chips with the same reader row_idx; no other program argument
    // depends on the coordinate, and buffers have mesh-wide uniform addresses.
    std::map<uint32_t, ttnn::MeshCoordinateRangeSet> rows;
    for (const auto& coord : tensor_coords.coords()) {
        rows[coord[operation_attributes.cluster_axis]].merge(ttnn::MeshCoordinateRange(coord));
    }

    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    for (const auto& [row_idx, coordinates] : rows) {
        for (const auto& range : coordinates.ranges()) {
            auto program =
                create_program_artifacts_at(operation_attributes, input, tensor_return_value, range.start_coord());
            artifacts.programs.push_back({
                .range = range,
                .spec = std::move(program.spec),
                .run_params = std::move(program.run_params),
            });
        }
    }
    return artifacts;
}

namespace {

ttnn::device_operation::ProgramArtifacts create_program_artifacts_at(
    const OffsetCumsumParams& operation_attributes,
    const Tensor& input,
    OffsetCumsumProgramFactory::tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinate& coord) {
    using namespace tt::tt_metal;

    const KernelSpecName READER{"reader"};
    const ScratchpadSpecName IN0{"in0"};
    const ScratchpadSpecName OUT0{"out0"};
    const ScratchpadSpecName ROWS{"rows"};
    const TensorParamName SRC{"src"};
    const TensorParamName DST_OFFSETS{"dst_offsets"};
    const TensorParamName DST_TOTALS{"dst_totals"};
    const TensorParamName DST_EXPERT_REGION{"dst_expert_region"};
    const TensorParamName DST_ALL_OFFSETS{"dst_all_offsets"};

    const uint32_t row_idx = coord[operation_attributes.cluster_axis];
    const uint32_t experts_per_chip = operation_attributes.experts_per_chip;

    CoreCoord core = {0, 0};
    CoreRangeSet core_set = CoreRangeSet(std::vector{CoreRange(core, core)});

    const auto& logical_shape = input.logical_shape();
    uint32_t W = logical_shape[-1];
    uint32_t H = logical_shape[-2];

    TT_FATAL(
        row_idx < H,
        "offset_cumsum: mesh coordinate {} along cluster_axis {} is outside the {}-row gathered histogram, so "
        "this device's own offsets row does not exist. The input must be gathered along cluster_axis.",
        row_idx,
        operation_attributes.cluster_axis,
        H);

    auto* src_buffer = input.buffer();
    auto* dst_offsets_buffer = tensor_return_value.at(0).buffer();
    auto* dst_totals_buffer = tensor_return_value.at(1).buffer();
    auto* dst_expert_region_buffer = tensor_return_value.at(2).buffer();
    auto* dst_all_offsets_buffer = tensor_return_value.at(3).buffer();
    uint32_t input_page_size = src_buffer->aligned_page_size();
    uint32_t offsets_page_size = dst_offsets_buffer->aligned_page_size();
    uint32_t totals_page_size = dst_totals_buffer->aligned_page_size();
    uint32_t expert_region_page_size = dst_expert_region_buffer->aligned_page_size();
    uint32_t all_offsets_page_size = dst_all_offsets_buffer->aligned_page_size();

    // offsets reuses a row of the all_offsets staging buffer, which requires both outputs to have the same page size.
    TT_FATAL(
        all_offsets_page_size == offsets_page_size,
        "offset_cumsum: all_offsets page is {} B and offsets page is {} B; the staging buffer serves both",
        all_offsets_page_size,
        offsets_page_size);

    // The kernel stages everything in these and never uses FIFO operations, so they are scratchpads.
    Group<ScratchpadSpec> scratchpads = {
        // Holds all H input rows: the kernel reads the input once, behind a single barrier.
        ScratchpadSpec{.unique_id = IN0, .size_per_node = H * input_page_size},
        ScratchpadSpec{.unique_id = OUT0, .size_per_node = offsets_page_size},
        // Holds every all_offsets row, so all H row writes share one barrier. `offsets` is written from
        // its own row of this buffer.
        ScratchpadSpec{.unique_id = ROWS, .size_per_node = H * all_offsets_page_size},
    };

    // row_idx is invariant for a given coord, baked into the cached program for that coord.
    KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/offset_cumsum/device/kernels/"
            "reader_offset_cumsum_interleaved.cpp"),
        .scratchpad_bindings =
            {
                ScratchpadBinding{.scratchpad_spec_name = IN0, .accessor_name = "in0"},
                ScratchpadBinding{.scratchpad_spec_name = OUT0, .accessor_name = "out0"},
                ScratchpadBinding{.scratchpad_spec_name = ROWS, .accessor_name = "rows"},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = SRC, .accessor_name = "src"},
                TensorBinding{.tensor_parameter_name = DST_OFFSETS, .accessor_name = "dst_offsets"},
                TensorBinding{.tensor_parameter_name = DST_TOTALS, .accessor_name = "dst_totals"},
                TensorBinding{.tensor_parameter_name = DST_EXPERT_REGION, .accessor_name = "dst_expert_region"},
                TensorBinding{.tensor_parameter_name = DST_ALL_OFFSETS, .accessor_name = "dst_all_offsets"},
            },
        .compile_time_args =
            {
                {"input_page_size", input_page_size},
                {"offsets_page_size", offsets_page_size},
                {"totals_page_size", totals_page_size},
                {"expert_region_page_size", expert_region_page_size},
                {"all_offsets_page_size", all_offsets_page_size},
                {"W", W},
                {"H", H},
                {"experts_per_chip", experts_per_chip},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"row_idx"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelRunArgs reader_run_args{
        .kernel = READER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(core, {{"row_idx", row_idx}}),
    };

    const auto& dst_offsets = tensor_return_value.at(0);
    const auto& dst_totals = tensor_return_value.at(1);
    const auto& dst_expert_region = tensor_return_value.at(2);
    const auto& dst_all_offsets = tensor_return_value.at(3);

    ProgramSpec spec{
        .name = "offset_cumsum",
        .kernels = {std::move(reader_spec)},
        .scratchpads = std::move(scratchpads),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = SRC, .spec = input.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = DST_OFFSETS, .spec = dst_offsets.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = DST_TOTALS, .spec = dst_totals.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = DST_EXPERT_REGION, .spec = dst_expert_region.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = DST_ALL_OFFSETS, .spec = dst_all_offsets.mesh_tensor().tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER},
                    .target_nodes = core_set,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.tensor_args.emplace(SRC, input.mesh_tensor());
    run_args.tensor_args.emplace(DST_OFFSETS, dst_offsets.mesh_tensor());
    run_args.tensor_args.emplace(DST_TOTALS, dst_totals.mesh_tensor());
    run_args.tensor_args.emplace(DST_EXPERT_REGION, dst_expert_region.mesh_tensor());
    run_args.tensor_args.emplace(DST_ALL_OFFSETS, dst_all_offsets.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace

}  // namespace ttnn::experimental::prim::qsr
