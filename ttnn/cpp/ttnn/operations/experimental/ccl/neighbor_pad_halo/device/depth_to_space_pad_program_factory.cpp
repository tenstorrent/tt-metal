// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "depth_to_space_pad_program_factory.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <algorithm>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

// ============================================================================
// create_mesh_workload — local + identical on every device: same single program per mesh coord.
// ============================================================================
DepthToSpacePadMeshWorkloadFactory::cached_mesh_workload_t DepthToSpacePadMeshWorkloadFactory::create_mesh_workload(
    const DepthToSpacePadParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const DepthToSpacePadInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto cached_program = create_at(operation_attributes, mesh_coord, tensor_args, tensor_return_value);
            shared_variables[single_coord_range] = cached_program.shared_variables;
            mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        }
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

// ============================================================================
// create_at — one writer kernel that scatters each depth-to-space output stick (C elements, read as a
// sub-slice of the larger conv-output page) into the padded output interior.
// ============================================================================
DepthToSpacePadMeshWorkloadFactory::cached_program_t DepthToSpacePadMeshWorkloadFactory::create_at(
    const DepthToSpacePadParams& op,
    const ttnn::MeshCoordinate& /*mesh_coordinate*/,
    const DepthToSpacePadInputs& tensor_args,
    Tensor& tensor_return_value) {
    Program program{};

    Buffer* in = tensor_args.conv_out.buffer();
    Buffer* dst = tensor_return_value.buffer();

    const auto& xshape = tensor_args.conv_out.logical_shape();  // [B,T,H,W, p1*p2*p3*C]
    const uint32_t B = xshape[0];
    const uint32_t T = xshape[1];
    const uint32_t H = xshape[2];
    const uint32_t W = xshape[3];
    const uint32_t block = op.p1 * op.p2 * op.p3;
    const uint32_t C = xshape[4] / block;

    const uint32_t elem_size = tensor_args.conv_out.element_size();
    const uint32_t c_bytes = C * elem_size;
    const uint32_t in_page_size = in->aligned_page_size();
    const uint32_t dst_page_size = dst->aligned_page_size();

    const uint32_t Hd_out = H * op.p2;
    const uint32_t Wd_out = W * op.p3;
    const uint32_t T_out = T * op.p1 - op.drop_first;
    const uint32_t work_count = B * T_out * Hd_out * Wd_out;  // interior sticks

    tt::DataFormat df = datatype_to_dataformat_converter(tensor_args.conv_out.dtype());

    // Spread across the full compute grid: each core scatters a contiguous global-stick range.
    const CoreCoord grid = tensor_return_value.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min<uint32_t>(grid.x * grid.y, std::max<uint32_t>(work_count, 1u));
    const CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const uint32_t per_core = (work_count + num_cores - 1) / num_cores;  // ceil

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_pages = 8;  // sticks per barrier batch (in-flight reads/writes)
    CircularBufferConfig cb_cfg =
        CircularBufferConfig(cb_pages * dst_page_size, {{cb_id, df}}).set_page_size(cb_id, dst_page_size);
    CreateCircularBuffer(program, all_cores, cb_cfg);

    auto writer_cfg = WriterDataMovementConfig{};
    writer_cfg.compile_args = {
        c_bytes,
        in_page_size,
        dst_page_size,
        B,
        T,
        H,
        W,
        op.p1,
        op.p2,
        op.p3,
        op.np_padding_h,
        op.np_padding_w,
        op.drop_first,
        cb_id,
        cb_pages};
    TensorAccessorArgs(*in).append_to(writer_cfg.compile_args);
    TensorAccessorArgs(*dst).append_to(writer_cfg.compile_args);

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/depth_to_space_pad_writer.cpp",
        all_cores,
        writer_cfg);

    uint32_t assigned = 0;
    for (const auto& cr : all_cores.ranges()) {
        for (const auto& c : cr) {
            const uint32_t start = assigned;
            const uint32_t count = (start >= work_count) ? 0u : std::min(per_core, work_count - start);
            SetRuntimeArgs(program, writer_kernel_id, c, {in->address(), dst->address(), start, count});
            assigned += count;
        }
    }

    return cached_program_t{std::move(program), {DepthToSpacePadArtifacts{writer_kernel_id, all_cores}}};
}

// ============================================================================
// override_runtime_arguments — refresh the (in, dst) DRAM addresses.
// ============================================================================
void DepthToSpacePadMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const DepthToSpacePadParams& /*op*/,
    const DepthToSpacePadInputs& tensor_args,
    Tensor& tensor_return_value) {
    const uint32_t in_addr = tensor_args.conv_out.buffer()->address();
    const uint32_t dst_addr = tensor_return_value.buffer()->address();

    for (auto& [coordinate_range, shared_vars] : cached_workload.shared_variables) {
        auto& program = cached_workload.workload.get_programs().at(coordinate_range);
        auto& args_by_core = GetRuntimeArgs(program, shared_vars.artifacts.writer_kernel_id);
        for (const auto& cr : shared_vars.artifacts.cores.ranges()) {
            for (const auto& c : cr) {
                auto& args = args_by_core[c.x][c.y];
                args[0] = in_addr;
                args[1] = dst_addr;
            }
        }
    }
}

}  // namespace ttnn::experimental::prim
