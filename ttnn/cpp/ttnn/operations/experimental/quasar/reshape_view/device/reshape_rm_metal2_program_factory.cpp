// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal-2.0 (Quasar) row-major reshape factory. Mirrors reshape_rm_program_factory.cpp's page-mapping
// math, but emits a ProgramSpec + QuasarDataMovementKernel (Quasar rejects the legacy DataMovementKernel
// the descriptor path builds). Single reader kernel per core (the descriptor path's dual-kernel split is
// a WH/BH perf optimization not reproduced here). The two L1 staging buffers are node-local scratchpads
// (not DFBs) since one DM kernel both fills and drains them.

#include "ttnn/operations/experimental/quasar/reshape_view/device/reshape_row_major_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#define MASK_64 0xFFFFFFFFFFFFFFC0

namespace ttnn::prim::qsr {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
const TensorParamName RS_SRC{"reshape_rm_src"};
const TensorParamName RS_DST{"reshape_rm_dst"};
const ScratchpadSpecName RS_SCRATCH_SRC{"reshape_rm_src_stage"};
const ScratchpadSpecName RS_SCRATCH_DST{"reshape_rm_dst_stage"};
const KernelSpecName RS_READER{"reshape_rm_reader"};
}  // namespace

ttnn::device_operation::ProgramArtifacts ReshapeViewRMMetalV2ProgramFactory::create_program_artifacts(
    const ReshapeViewParams& operation_attributes, const ReshapeViewInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& sub_core_grid = operation_attributes.sub_core_grid;

    const uint32_t data_size = input.element_size();
    IDevice* device = input.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1});
    CoreRangeSet total_cores = sub_core_grid.has_value() ? sub_core_grid.value() : CoreRangeSet(default_cores);
    uint32_t num_cores_total = total_cores.num_cores();

    const auto input_log_shape = input.logical_shape();
    const auto output_log_shape = output.logical_shape();

    const uint32_t source_page_size_bytes = input_log_shape[-1] * data_size;
    const uint32_t dest_page_size_bytes = output_log_shape[-1] * data_size;
    const uint32_t source_read_size_bytes = ((source_page_size_bytes - 1) & MASK_64) + 128;

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    const bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    // Pages-per-core such that each core starts on a read AND write page boundary (logical volumes match,
    // so the last page is always aligned).
    uint32_t responsibility = ((input_log_shape[-2] - 1) / num_cores_total) + 1;
    while ((responsibility * source_page_size_bytes) % dest_page_size_bytes != 0) {
        responsibility++;
    }
    const uint32_t cb_size0 = source_read_size_bytes;
    const uint32_t cb_size1 = ((dest_page_size_bytes - 1) & MASK_64) + 80;

    // ---- ProgramSpec ----
    ProgramSpec spec;
    spec.name = "reshape_view_rm";

    spec.tensor_parameters = {
        TensorParameter{.unique_id = RS_SRC, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = RS_DST, .spec = output.tensor_spec()},
    };

    // Node-local L1 staging (not DFBs: the reader both fills and drains them -> would be a Gen2 self-loop).
    spec.scratchpads = {
        ScratchpadSpec{.unique_id = RS_SCRATCH_SRC, .size_per_node = cb_size0},
        ScratchpadSpec{.unique_id = RS_SCRATCH_DST, .size_per_node = cb_size1},
    };

    KernelSpec reader{
        .unique_id = RS_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/reshape_view/device/device/"
            "rm_reshape_interleaved_metal2.cpp",
        // NOTE: field order must match KernelSpec declaration order (scratchpad_bindings before
        // tensor_bindings) — designated initializers require it (-Werror=reorder-init-list).
        .scratchpad_bindings =
            {ScratchpadBinding{.scratchpad_spec_name = RS_SCRATCH_SRC, .accessor_name = "src_stage"},
             ScratchpadBinding{.scratchpad_spec_name = RS_SCRATCH_DST, .accessor_name = "dst_stage"}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = RS_SRC, .accessor_name = "src"},
             TensorBinding{.tensor_parameter_name = RS_DST, .accessor_name = "dst"}},
        .compile_time_args =
            {{"src_aligned_to_64", (source_page_size_bytes % 64 == 0) ? 1u : 0u},
             {"src_aligned_to_16", (source_page_size_bytes % 16 == 0) ? 1u : 0u},
             {"src_is_dram", src_is_dram ? 1u : 0u},
             {"source_page_size_bytes", source_page_size_bytes},
             {"dest_page_size_bytes", dest_page_size_bytes}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"source_read_size_bytes",
                  "read_start_page",
                  "read_end_page",
                  "write_start_page",
                  "write_start_offset",
                  "nop"}},
        // Repages with sub-tile writes (dest_page can be < source_page; tt_memmove chunks) -> per
        // ~/implicit_sync.md rule (A), revert to explicit credits.
        .hw_config =
            DataMovementHardwareConfig{
                .role = DataMovementRoleHint::READER,
                .gen2_config = DataMovementHardwareConfig::Gen2Config{.disable_dfb_implicit_sync_for_all = true}},
    };

    spec.kernels = {reader};
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {RS_READER}, .target_nodes = total_cores}};

    // ---- Per-core runtime args ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = RS_READER};

    uint32_t read_start_page = 0;
    uint32_t write_start_page = 0;
    bool done = false;
    for (const auto& core : corerange_to_cores(total_cores, std::nullopt)) {
        if (done) {
            // Idle core: nop=1, kernel short-circuits (buffers still bound but never dereferenced).
            reader_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                .node = core,
                .args = {
                    {"source_read_size_bytes", source_read_size_bytes},
                    {"read_start_page", 0u},
                    {"read_end_page", 0u},
                    {"write_start_page", 0u},
                    {"write_start_offset", 0u},
                    {"nop", 1u}}});
            continue;
        }
        const uint32_t start_of_read = read_start_page;
        uint32_t end_of_read = read_start_page + responsibility;
        end_of_read = end_of_read < static_cast<uint32_t>(input_log_shape[-2])
                          ? end_of_read
                          : static_cast<uint32_t>(input_log_shape[-2]);
        const uint32_t pages_for_this_core = end_of_read - start_of_read;
        const uint32_t write_jump = (pages_for_this_core * source_page_size_bytes) / dest_page_size_bytes;

        reader_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core,
            .args = {
                {"source_read_size_bytes", source_read_size_bytes},
                {"read_start_page", start_of_read},
                {"read_end_page", end_of_read},
                {"write_start_page", write_start_page},
                {"write_start_offset", 0u},
                {"nop", 0u}}});

        write_start_page += write_jump;
        read_start_page = end_of_read;
        done = (end_of_read == static_cast<uint32_t>(input_log_shape[-2]));
    }

    run_args.kernel_run_args.push_back(reader_run);
    run_args.tensor_args.emplace(RS_SRC, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(RS_DST, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
