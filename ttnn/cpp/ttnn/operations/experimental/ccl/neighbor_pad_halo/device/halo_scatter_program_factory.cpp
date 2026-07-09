// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "halo_scatter_program_factory.hpp"

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
// create_mesh_workload — the scatter is local + identical on every device, so
// each mesh coordinate gets the same single-core program.
// ============================================================================
NpHaloScatterMeshWorkloadFactory::cached_mesh_workload_t NpHaloScatterMeshWorkloadFactory::create_mesh_workload(
    const NpHaloScatterParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const NpHaloScatterInputs& tensor_args,
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
// create_at — one writer kernel on a single core that copies each compact halo
// stick to its fixed padded-buffer border page (in place; interior untouched).
// ============================================================================
NpHaloScatterMeshWorkloadFactory::cached_program_t NpHaloScatterMeshWorkloadFactory::create_at(
    const NpHaloScatterParams& op,
    const ttnn::MeshCoordinate& /*mesh_coordinate*/,
    const NpHaloScatterInputs& tensor_args,
    Tensor& tensor_return_value) {
    Program program{};

    Buffer* compact = tensor_args.compact_buffer.buffer();  // [total_sticks, C] (border source)
    Buffer* dst = tensor_return_value.buffer();          // padded output (allocated) or in-place (border_only)
    // border_only: no interior reads; x_src unused (point it at dst, harmless). repack: unpadded interior.
    Buffer* x_src = op.border_only ? dst : tensor_args.interior_src.buffer();

    const auto& xshape = tensor_args.interior_src.logical_shape();
    const uint32_t rank = xshape.size();
    const uint32_t pH = op.np_padding_h;
    const uint32_t pW = op.np_padding_w;
    // border_only: interior_src is padded, so subtract to get interior dims. repack: it's unpadded already.
    const uint32_t Wd = op.border_only ? xshape[rank - 2] - 2 * pW : xshape[rank - 2];
    const uint32_t Hd = op.border_only ? xshape[rank - 3] - 2 * pH : xshape[rank - 3];
    uint32_t outer = 1;  // product of all dims before H (B*T)
    for (uint32_t d = 0; d + 3 < rank; d++) {
        outer *= xshape[d];
    }
    const uint32_t Hp = Hd + 2 * pH;
    const uint32_t Wp = Wd + 2 * pW;

    const uint32_t page_size = compact->aligned_page_size();
    tt::DataFormat df = datatype_to_dataformat_converter(tensor_args.compact_buffer.dtype());

    // Every padded page is written once: interior (Hd*Wd) from x + border (2 H-sections pH*Wd + 2
    // W-sections Hp*pW) from compact, per frame.
    const uint32_t interior_sticks = outer * Hd * Wd;
    const uint32_t border_sticks = outer * (2 * pH * Wd + 2 * pW * Hp);
    const uint32_t total_sticks = interior_sticks + border_sticks;
    // border_only: process only the border sticks [interior_sticks, total_sticks); the kernel's global
    // index maps >= interior_sticks to border sections, so the interior is never touched.
    const uint32_t work_start = op.border_only ? interior_sticks : 0u;
    const uint32_t work_count = total_sticks - work_start;

    // Spread across the full compute grid: each core copies a contiguous global-stick range.
    const CoreCoord grid = tensor_return_value.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min<uint32_t>(grid.x * grid.y, std::max<uint32_t>(work_count, 1u));
    const CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const uint32_t per_core = (work_count + num_cores - 1) / num_cores;  // ceil

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_pages = 8;  // sticks per barrier batch (in-flight reads/writes)
    CircularBufferConfig cb_cfg =
        CircularBufferConfig(cb_pages * page_size, {{cb_id, df}}).set_page_size(cb_id, page_size);
    CreateCircularBuffer(program, all_cores, cb_cfg);

    auto writer_cfg = WriterDataMovementConfig{};
    writer_cfg.compile_args = {page_size, outer, Hp, Wp, Hd, Wd, pH, pW, cb_id, cb_pages};
    TensorAccessorArgs(*x_src).append_to(writer_cfg.compile_args);
    TensorAccessorArgs(*compact).append_to(writer_cfg.compile_args);
    TensorAccessorArgs(*dst).append_to(writer_cfg.compile_args);

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/halo_scatter_writer.cpp",
        all_cores,
        writer_cfg);

    // Per-core [x_addr, compact_addr, dst_addr, stick_start, stick_count]; contiguous ranges row-wise
    // over [work_start, total_sticks).
    uint32_t assigned = work_start;
    for (const auto& cr : all_cores.ranges()) {
        for (const auto& c : cr) {
            const uint32_t start = assigned;
            const uint32_t count = (start >= total_sticks) ? 0u : std::min(per_core, total_sticks - start);
            SetRuntimeArgs(
                program,
                writer_kernel_id,
                c,
                {x_src->address(), compact->address(), dst->address(), start, count});
            assigned += count;
        }
    }

    return cached_program_t{std::move(program), {NpHaloScatterArtifacts{writer_kernel_id, all_cores}}};
}

// ============================================================================
// override_runtime_arguments — refresh the (x, compact, padded) DRAM addresses.
// ============================================================================
void NpHaloScatterMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const NpHaloScatterParams& /*op*/,
    const NpHaloScatterInputs& tensor_args,
    Tensor& tensor_return_value) {
    const uint32_t x_addr = tensor_args.interior_src.buffer()->address();
    const uint32_t compact_addr = tensor_args.compact_buffer.buffer()->address();
    const uint32_t dst_addr = tensor_return_value.buffer()->address();

    for (auto& [coordinate_range, shared_vars] : cached_workload.shared_variables) {
        auto& program = cached_workload.workload.get_programs().at(coordinate_range);
        auto& args_by_core = GetRuntimeArgs(program, shared_vars.artifacts.writer_kernel_id);
        for (const auto& cr : shared_vars.artifacts.cores.ranges()) {
            for (const auto& c : cr) {
                auto& args = args_by_core[c.x][c.y];
                args[0] = x_addr;
                args[1] = compact_addr;
                args[2] = dst_addr;
            }
        }
    }
}

}  // namespace ttnn::experimental::prim
