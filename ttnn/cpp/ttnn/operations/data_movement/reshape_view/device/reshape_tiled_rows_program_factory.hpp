// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/workload/run_args/program_run_args.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/reshape_device_operation_types.hpp"

namespace ttnn::prim {

// Whether the rows path can reshape `input` from rank-3 `input_shape_3d` to rank-3 `output_shape_3d`,
// both with the default tile padding. Routing asks it with the shapes the reshape is reduced to, before
// the input is viewed as rank 3.
bool tiled_rows_reshape_applies(
    const Tensor& input,
    const ttnn::Shape& input_shape_3d,
    const ttnn::Shape& output_shape_3d,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool has_sub_core_grid);

// The tiled reshape of an interleaved tensor that keeps the last dim, [C_in, H_in, W] -> [C_out, H_out,
// W]. It needs no mapping tensor: the kernels derive which input rows make each output tile from the
// row counts, which are common runtime args, so one program, without per-shape device memory, serves
// every row count. The program hash leaves the row counts out.
struct ReshapeViewTiledRowsProgramFactory {
    // Whether the rows path can run this reshape.
    static bool applies(const ReshapeViewParams& operation_attributes, const ReshapeViewInputs& tensor_args);

    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value);

    static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
