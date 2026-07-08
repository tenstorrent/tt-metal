// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <optional>
#include <tuple>

namespace ttnn::operations::experimental::topk_large_indices::program {

namespace {

struct RuntimeShapeArgs {
    uint32_t num_rows = 0;
    uint32_t num_chunks = 0;
    uint32_t tail_elements = 0;
    uint32_t input_tail_chunk_bytes = 0;
    uint32_t input_row_bytes = 0;
};

enum class LlkTargetK : uint32_t {
    K512 = 512,
    K1024 = 1024,
    K2048 = 2048,
};

constexpr uint32_t to_uint32(LlkTargetK target_k) { return static_cast<uint32_t>(target_k); }

LlkTargetK snap_to_llk_target_k(uint32_t k) {
    if (k <= to_uint32(LlkTargetK::K512)) {
        return LlkTargetK::K512;
    }
    if (k <= to_uint32(LlkTargetK::K1024)) {
        return LlkTargetK::K1024;
    }
    return LlkTargetK::K2048;
}

RuntimeShapeArgs get_runtime_shape_args(
    const Tensor& input, LlkTargetK llk_target_k, std::optional<uint32_t> valid_length) {
    const uint32_t llk_k = to_uint32(llk_target_k);
    const auto& shape = input.logical_shape();
    const uint32_t n = shape[shape.rank() - 1];
    // Number of columns to actually read and scan per row. Defaults to the full physical width n; a
    // valid_length bounds it to the real prefix so the stale tail is never read or ranked. The row STRIDE
    // (input_row_bytes) stays n so per-row addressing is unchanged — only how much we pull from each row shrinks.
    const uint32_t search_len = valid_length.value_or(n);
    const uint32_t num_chunks = tt::div_up(search_len, llk_k);
    const uint32_t tail_elements = search_len - ((num_chunks - 1) * llk_k);
    return RuntimeShapeArgs{
        .num_rows = flattened_rows_excluding_last_dim(shape),
        .num_chunks = num_chunks,
        .tail_elements = tail_elements,
        .input_tail_chunk_bytes = tail_elements * input.element_size(),
        .input_row_bytes = n * input.element_size()};
}

tt::tt_metal::TensorAccessorArgs interleaved_accessor_args(const Tensor& tensor) {
    return tensor.buffer()->is_dram() ? tt::tt_metal::TensorAccessorArgs::create_dram_interleaved()
                                      : tt::tt_metal::TensorAccessorArgs::create_l1_interleaved();
}

uint32_t rows_for_core(
    const CoreCoord& core,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_rows_per_core_group_1,
    uint32_t num_rows_per_core_group_2) {
    if (core_group_1.contains(core)) {
        return num_rows_per_core_group_1;
    }
    if (core_group_2.contains(core)) {
        return num_rows_per_core_group_2;
    }
    return 0;
}

void set_runtime_args(
    tt::tt_metal::Program& program,
    const TopkLargeIndicesSharedVariables& shared,
    const Tensor& input,
    const Tensor& indices,
    LlkTargetK llk_target_k,
    std::optional<uint32_t> valid_length) {
    const auto runtime_args = get_runtime_shape_args(input, llk_target_k, valid_length);
    const auto work_split = tt::tt_metal::split_work_to_cores(
        input.device()->compute_with_storage_grid_size(), runtime_args.num_rows, true);
    const auto num_active_cores = std::get<0>(work_split);
    const auto& core_group_1 = std::get<2>(work_split);
    const auto& core_group_2 = std::get<3>(work_split);
    const auto num_rows_per_core_group_1 = std::get<4>(work_split);
    const auto num_rows_per_core_group_2 = std::get<5>(work_split);
    TT_FATAL(num_active_cores > 0, "topk_large_indices requires at least one row of work");

    uint32_t start_row = 0;
    for (const auto& core : shared.cores) {
        const uint32_t rows =
            rows_for_core(core, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2);
        TT_FATAL(
            rows <= num_rows_per_core_group_1,
            "topk_large_indices assigned {} rows to a core, expected at most {}",
            rows,
            num_rows_per_core_group_1);

        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.reader_kernel_id,
            core,
            {input.buffer()->address(),
             start_row,
             rows,
             runtime_args.num_chunks,
             runtime_args.input_tail_chunk_bytes,
             runtime_args.input_row_bytes});
        tt::tt_metal::SetRuntimeArgs(
            program, shared.compute_kernel_id, core, {rows, runtime_args.num_chunks, runtime_args.tail_elements});
        tt::tt_metal::SetRuntimeArgs(
            program, shared.writer_kernel_id, core, {indices.buffer()->address(), start_row, rows});

        start_row += rows;
    }
    TT_FATAL(
        start_row == runtime_args.num_rows,
        "topk_large_indices assigned {} rows, expected {}",
        start_row,
        runtime_args.num_rows);
}

}  // namespace

TopkLargeIndicesProgramFactory::cached_program_t TopkLargeIndicesProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto program = tt::tt_metal::CreateProgram();

    const auto& input = tensor_args.input_tensor;
    auto& indices = tensor_return_value;

    const uint32_t k = operation_attributes.k;
    const auto llk_target_k = snap_to_llk_target_k(k);
    const uint32_t llk_k = to_uint32(llk_target_k);
    const uint32_t tiles_per_sequence = (llk_k + tt::constants::TILE_HW - 1) / tt::constants::TILE_HW;

    const auto grid = input.device()->compute_with_storage_grid_size();
    const CoreRangeSet all_cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const auto cores = corerange_to_cores(all_cores, std::nullopt, true);
    // Runtime row counts are intentionally patched through runtime args instead of the program hash.
    // Create kernels/CBs across the full worker grid so cache hits can use a different active core subset.

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_indices = tt::CBIndex::c_1;
    constexpr uint32_t cb_indices_scratch = tt::CBIndex::c_2;

    const uint32_t input_chunk_bytes = llk_k * input.element_size();
    const uint32_t input_tile_bytes = tt::constants::TILE_HW * input.element_size();
    constexpr uint32_t row_slice_elements = tt::constants::FACE_WIDTH;
    const uint32_t source_slices_per_row = llk_k / row_slice_elements;
    const uint32_t output_slices_per_row = k / row_slice_elements;
    const uint32_t indices_slice_bytes = row_slice_elements * indices.element_size();
    const uint32_t indices_row_bytes = k * indices.element_size();
    const uint32_t indices_cb_row_bytes = llk_k * indices.element_size();

    const uint32_t cb_depth = 2;
    const auto input_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_depth * tiles_per_sequence * input_tile_bytes, {{cb_in, tt::DataFormat::Float16_b}})
            .set_page_size(cb_in, input_tile_bytes);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, input_cb_config);

    auto indices_cb_config =
        tt::tt_metal::CircularBufferConfig(cb_depth * indices_cb_row_bytes, {{cb_indices, tt::DataFormat::Float32}})
            .set_page_size(cb_indices, indices_cb_row_bytes);
    if (llk_target_k == LlkTargetK::K512) {
        indices_cb_config.set_unpack_face_geometry(cb_indices, tt::constants::FACE_HEIGHT, 2);
    }
    tt::tt_metal::CreateCircularBuffer(program, all_cores, indices_cb_config);

    if (llk_target_k != LlkTargetK::K512) {
        const auto indices_scratch_cb_config =
            tt::tt_metal::CircularBufferConfig(indices_row_bytes, {{cb_indices_scratch, tt::DataFormat::Float32}})
                .set_page_size(cb_indices_scratch, indices_row_bytes);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, indices_scratch_cb_config);
    }

    std::vector<uint32_t> reader_compile_args = {cb_in, input_chunk_bytes, input_tile_bytes, tiles_per_sequence};
    interleaved_accessor_args(input).append_to(reader_compile_args);

    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    std::vector<uint32_t> compute_compile_args = {cb_in, cb_indices, llk_k};
    auto compute_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            // TopK XL stores fused BF16-value/u16-index words and unfused UINT32 indices in 32-bit DEST lanes.
            .fp32_dest_acc_en = true,
            // K=2048 multi-chunk merge uses DEST slots 0..7; FP32 half-sync mode exposes only 4 tiles.
            .dst_full_sync_en = true,
            .compile_args = compute_compile_args});

    std::vector<uint32_t> writer_compile_args = {
        cb_indices,
        cb_indices_scratch,
        indices_row_bytes,
        source_slices_per_row,
        output_slices_per_row,
        indices_slice_bytes};
    interleaved_accessor_args(indices).append_to(writer_compile_args);

    auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/writer.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    TopkLargeIndicesSharedVariables shared{
        .reader_kernel_id = reader_kernel,
        .compute_kernel_id = compute_kernel,
        .writer_kernel_id = writer_kernel,
        .cores = cores};
    set_runtime_args(program, shared, input, indices, llk_target_k, operation_attributes.valid_length);

    return cached_program_t{std::move(program), std::move(shared)};
}

void TopkLargeIndicesProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    set_runtime_args(
        cached_program.program,
        cached_program.shared_variables,
        tensor_args.input_tensor,
        tensor_return_value,
        snap_to_llk_target_k(operation_attributes.k),
        operation_attributes.valid_length);
}

}  // namespace ttnn::operations::experimental::topk_large_indices::program
