// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <algorithm>
#include <cmath>
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

// ---------------------------------------------------------------------------
// Column-parallel (intra-row multi-core) path
// ---------------------------------------------------------------------------

ColumnSplitConfig compute_column_split_config(uint32_t k, uint32_t n, uint32_t num_rows, const CoreCoord& grid) {
    // Cap chosen so the gathered CBs (2 * num_slices * tiles_per_sequence
    // 4 KB tiles, allocated at one shared address on local + final cores) stay
    // ~1 MB at K=2048 — comfortably inside the 1.5 MB Blackhole L1 budget next
    // to the local-core CBs (~56 KB) and final-core output CBs (~24 KB).
    constexpr uint32_t max_slices = 64;

    ColumnSplitConfig config{};
    // Intra-row parallelism only pays off when rows cannot saturate the grid;
    // the final-core merge is serial per row, so restrict to the single-row
    // case this path is built for. Every other shape keeps the row-parallel
    // factory (and its shape-free program hash) unchanged.
    if (num_rows != 1) {
        return config;
    }
    if (grid.x < 2 || grid.y < 2) {
        return config;
    }

    const uint32_t llk_k = to_uint32(snap_to_llk_target_k(k));
    // Physical width only: valid_length must stay runtime-only, so the split
    // never depends on it (short valid prefixes empty out trailing slices at
    // runtime instead of changing the program).
    const uint32_t num_chunks = tt::div_up(n, llk_k);
    if (num_chunks < 2) {
        return config;
    }

    // Rough cost model in merge units: processing one chunk locally
    // (copy + lsb + fused sort + index split + merge + rebuild) ~ 2 units,
    // one final-core merge+rebuild ~ 1 unit. The serial final merge chain is
    // num_slices units, so the optimum is near sqrt(2 * num_chunks).
    const auto rect_capacity = static_cast<uint32_t>(grid.x * (grid.y - 1));
    uint32_t num_slices = static_cast<uint32_t>(std::ceil(std::sqrt(2.0 * num_chunks)));
    num_slices = std::min({num_slices, num_chunks, max_slices, rect_capacity});

    // Local cores must form a rectangle for the gather multicast.
    uint32_t local_grid_x = 0;
    uint32_t local_grid_y = 0;
    if (num_slices <= grid.x) {
        local_grid_x = num_slices;
        local_grid_y = 1;
    } else {
        local_grid_x = grid.x;
        local_grid_y = std::min<uint32_t>(num_slices / grid.x, grid.y - 1);
        num_slices = local_grid_x * local_grid_y;
    }
    if (num_slices < 2) {
        return config;
    }

    const uint32_t cost_column = 2 * tt::div_up(num_chunks, num_slices) + num_slices;
    const uint32_t cost_row = 2 * num_chunks;  // single row -> single core on the row-parallel path
    if (cost_column >= cost_row) {
        return config;
    }

    config.enabled = true;
    config.num_slices = num_slices;
    config.local_grid_x = local_grid_x;
    config.local_grid_y = local_grid_y;
    return config;
}

namespace {

struct SliceRuntime {
    uint32_t start_chunk = 0;    // index of the slice's first chunk within the row
    uint32_t start_element = 0;  // start_chunk * llk_k
    uint32_t num_chunks = 0;     // active chunks after the valid_length cut (0 = empty slice)
    uint32_t tail_elements = 0;  // active elements in the slice's last chunk
};

// Splits the physical chunk range evenly over the slices, then bounds each
// slice by the runtime search width (valid_length). Slices wholly beyond the
// search width come back empty and are serviced by the writer's -inf fill.
std::vector<SliceRuntime> compute_slice_runtime(
    uint32_t n, uint32_t llk_k, uint32_t num_slices, std::optional<uint32_t> valid_length) {
    const uint32_t search_len = valid_length.value_or(n);
    const uint32_t num_chunks_phys = tt::div_up(n, llk_k);
    const uint32_t base_chunks = num_chunks_phys / num_slices;
    const uint32_t extra_chunks = num_chunks_phys % num_slices;

    std::vector<SliceRuntime> slices(num_slices);
    uint32_t start_chunk = 0;
    for (uint32_t s = 0; s < num_slices; ++s) {
        const uint32_t chunk_count = base_chunks + (s < extra_chunks ? 1 : 0);
        const uint32_t start_element = start_chunk * llk_k;
        const uint32_t end_element = std::min((start_chunk + chunk_count) * llk_k, n);
        const uint32_t active_end = std::min(end_element, search_len);

        SliceRuntime& slice = slices[s];
        slice.start_chunk = start_chunk;
        slice.start_element = start_element;
        if (active_end > start_element) {
            const uint32_t active_len = active_end - start_element;
            slice.num_chunks = tt::div_up(active_len, llk_k);
            slice.tail_elements = active_len - ((slice.num_chunks - 1) * llk_k);
        }
        start_chunk += chunk_count;
    }
    TT_FATAL(
        start_chunk == num_chunks_phys,
        "topk_large_indices column split assigned {} chunks, expected {}",
        start_chunk,
        num_chunks_phys);
    return slices;
}

void set_runtime_args_multi_core(
    tt::tt_metal::Program& program,
    const TopkLargeIndicesMultiCoreSharedVariables& shared,
    const Tensor& input,
    const Tensor& indices,
    LlkTargetK llk_target_k,
    std::optional<uint32_t> valid_length) {
    const uint32_t llk_k = to_uint32(llk_target_k);
    const auto& shape = input.logical_shape();
    const uint32_t n = shape[shape.rank() - 1];
    const uint32_t num_rows = flattened_rows_excluding_last_dim(shape);
    const uint32_t input_row_bytes = n * input.element_size();
    const uint32_t num_slices = static_cast<uint32_t>(shared.local_cores.size());

    const auto slices = compute_slice_runtime(n, llk_k, num_slices, valid_length);
    const CoreCoord final_core_physical = input.device()->worker_core_from_logical_core(shared.final_core);

    for (uint32_t s = 0; s < num_slices; ++s) {
        const auto& core = shared.local_cores[s];
        const auto& slice = slices[s];
        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.reader_local_kernel_id,
            core,
            {input.buffer()->address(),
             0 /* start_row */,
             num_rows,
             slice.num_chunks,
             slice.tail_elements * input.element_size(),
             input_row_bytes,
             slice.start_element * input.element_size()});
        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.compute_local_kernel_id,
            core,
            {num_rows,
             slice.num_chunks,
             slice.tail_elements,
             slice.start_chunk,
             s == 0 ? 0u : 1u /* output_ascending: slice 0 is merge operand 0 */});
        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.writer_local_kernel_id,
            core,
            {static_cast<uint32_t>(final_core_physical.x),
             static_cast<uint32_t>(final_core_physical.y),
             num_rows,
             s,
             slice.num_chunks == 0 ? 1u : 0u /* is_empty */});
    }

    tt::tt_metal::SetRuntimeArgs(program, shared.reader_final_kernel_id, shared.final_core, {num_rows});
    tt::tt_metal::SetRuntimeArgs(program, shared.compute_final_kernel_id, shared.final_core, {num_rows, num_slices});
    tt::tt_metal::SetRuntimeArgs(
        program, shared.writer_final_kernel_id, shared.final_core, {indices.buffer()->address(), 0, num_rows});
}

}  // namespace

TopkLargeIndicesMultiCoreProgramFactory::cached_program_t TopkLargeIndicesMultiCoreProgramFactory::create(
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

    const auto& shape = input.logical_shape();
    const uint32_t n = shape[shape.rank() - 1];
    const uint32_t num_rows = flattened_rows_excluding_last_dim(shape);
    const auto grid = input.device()->compute_with_storage_grid_size();
    const auto config = compute_column_split_config(k, n, num_rows, grid);
    TT_FATAL(config.enabled, "topk_large_indices multi-core factory selected for a shape it does not support");
    const uint32_t num_slices = config.num_slices;

    // Local cores: the rectangle (0,0)..(local_grid_x-1, local_grid_y-1).
    // Final core: (0, local_grid_y), just below the rectangle (outside the
    // gather multicast destination set).
    const CoreRange local_cores_range({0, 0}, {config.local_grid_x - 1, config.local_grid_y - 1});
    const CoreRangeSet local_cores_set(local_cores_range);
    const auto local_cores = corerange_to_cores(local_cores_set, std::nullopt, true);
    TT_FATAL(
        local_cores.size() == num_slices,
        "topk_large_indices column split expected {} local cores, got {}",
        num_slices,
        local_cores.size());
    const CoreCoord final_core(0, config.local_grid_y);
    const CoreRangeSet final_core_set(CoreRange(final_core, final_core));
    const CoreRangeSet all_cores = local_cores_set.merge(final_core_set);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_indices_out = tt::CBIndex::c_1;
    constexpr uint32_t cb_indices_scratch = tt::CBIndex::c_2;
    constexpr uint32_t cb_gathered_values = tt::CBIndex::c_3;
    constexpr uint32_t cb_gathered_indices = tt::CBIndex::c_4;
    constexpr uint32_t cb_local_values = tt::CBIndex::c_5;
    constexpr uint32_t cb_local_indices = tt::CBIndex::c_6;
    constexpr uint32_t cb_neginf_scratch = tt::CBIndex::c_7;

    const uint32_t input_chunk_bytes = llk_k * input.element_size();
    const uint32_t input_tile_bytes = tt::constants::TILE_HW * input.element_size();
    // The unfused sequences travel as opaque 32-bit words (FP32 values /
    // UINT32 indices packed raw from DST); one 32x32 tile is 4 KB.
    const uint32_t tile32_bytes = tt::constants::TILE_HW * sizeof(uint32_t);
    const uint32_t sequence_bytes = tiles_per_sequence * tile32_bytes;
    constexpr uint32_t row_slice_elements = tt::constants::FACE_WIDTH;
    const uint32_t source_slices_per_row = llk_k / row_slice_elements;
    const uint32_t output_slices_per_row = k / row_slice_elements;
    const uint32_t indices_slice_bytes = row_slice_elements * indices.element_size();
    const uint32_t indices_row_bytes = k * indices.element_size();
    const uint32_t indices_cb_row_bytes = llk_k * indices.element_size();
    constexpr uint32_t cb_depth = 2;

    // The gathered CBs span local + final cores so every core sees the same
    // L1 address (local writers derive the final core's destination from
    // their own copy). CB allocation assigns one address per CB across its
    // whole range, so multi-range CBs must be created FIRST to avoid gaps.
    const auto gathered_values_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_slices * tiles_per_sequence * tile32_bytes, {{cb_gathered_values, tt::DataFormat::UInt32}})
            .set_page_size(cb_gathered_values, tile32_bytes);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, gathered_values_cb_config);

    const auto gathered_indices_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_slices * tiles_per_sequence * tile32_bytes, {{cb_gathered_indices, tt::DataFormat::UInt32}})
            .set_page_size(cb_gathered_indices, tile32_bytes);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, gathered_indices_cb_config);

    // Local-core CBs.
    const auto input_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_depth * tiles_per_sequence * input_tile_bytes, {{cb_in, tt::DataFormat::Float16_b}})
            .set_page_size(cb_in, input_tile_bytes);
    tt::tt_metal::CreateCircularBuffer(program, local_cores_set, input_cb_config);

    const auto local_values_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_depth * tiles_per_sequence * tile32_bytes, {{cb_local_values, tt::DataFormat::UInt32}})
            .set_page_size(cb_local_values, tile32_bytes);
    tt::tt_metal::CreateCircularBuffer(program, local_cores_set, local_values_cb_config);

    const auto local_indices_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_depth * tiles_per_sequence * tile32_bytes, {{cb_local_indices, tt::DataFormat::UInt32}})
            .set_page_size(cb_local_indices, tile32_bytes);
    tt::tt_metal::CreateCircularBuffer(program, local_cores_set, local_indices_cb_config);

    // Writer-owned scratch for the empty-slice -inf sequence (values + indices).
    const auto neginf_scratch_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * sequence_bytes, {{cb_neginf_scratch, tt::DataFormat::UInt32}})
            .set_page_size(cb_neginf_scratch, tile32_bytes);
    tt::tt_metal::CreateCircularBuffer(program, local_cores_set, neginf_scratch_cb_config);

    // Final-core CBs: identical shape to the row-parallel factory's output CBs.
    auto indices_cb_config =
        tt::tt_metal::CircularBufferConfig(cb_depth * indices_cb_row_bytes, {{cb_indices_out, tt::DataFormat::Float32}})
            .set_page_size(cb_indices_out, indices_cb_row_bytes);
    if (llk_target_k == LlkTargetK::K512) {
        indices_cb_config.set_unpack_face_geometry(cb_indices_out, tt::constants::FACE_HEIGHT, 2);
    }
    tt::tt_metal::CreateCircularBuffer(program, final_core_set, indices_cb_config);

    if (llk_target_k != LlkTargetK::K512) {
        const auto indices_scratch_cb_config =
            tt::tt_metal::CircularBufferConfig(indices_row_bytes, {{cb_indices_scratch, tt::DataFormat::Float32}})
                .set_page_size(cb_indices_scratch, indices_row_bytes);
        tt::tt_metal::CreateCircularBuffer(program, final_core_set, indices_scratch_cb_config);
    }

    // Gather flow control (see writer_local.cpp / reader_final.cpp).
    const uint32_t receiver_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    const uint32_t sender_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);

    // Local reader: the row-parallel reader plus a per-core slice offset.
    // Separate source file so reader.cpp (and its JIT binary) stays
    // byte-identical for the row-parallel path.
    std::vector<uint32_t> reader_compile_args = {cb_in, input_chunk_bytes, input_tile_bytes, tiles_per_sequence};
    interleaved_accessor_args(input).append_to(reader_compile_args);
    auto reader_local_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/reader_local.cpp",
        local_cores_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    const std::vector<uint32_t> compute_local_compile_args = {cb_in, cb_local_values, cb_local_indices, llk_k};
    auto compute_local_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/compute_local.cpp",
        local_cores_set,
        tt::tt_metal::ComputeConfig{// Same DST configuration as the row-parallel compute kernel: the
                                    // unfused K=2048 merge occupies DEST slots 0..7 (FP32, full sync).
                                    .fp32_dest_acc_en = true,
                                    .dst_full_sync_en = true,
                                    .compile_args = compute_local_compile_args});

    const std::vector<uint32_t> writer_local_compile_args = {
        cb_local_values,
        cb_local_indices,
        cb_neginf_scratch,
        cb_gathered_values,
        cb_gathered_indices,
        receiver_sem_id,
        sender_sem_id,
        tiles_per_sequence,
        tile32_bytes};
    auto writer_local_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/writer_local.cpp",
        local_cores_set,
        tt::tt_metal::WriterDataMovementConfig(writer_local_compile_args));

    const auto* device = input.device();
    const CoreCoord mcast_start = device->worker_core_from_logical_core(local_cores_range.start_coord);
    const CoreCoord mcast_end = device->worker_core_from_logical_core(local_cores_range.end_coord);
    const std::vector<uint32_t> reader_final_compile_args = {
        receiver_sem_id,
        sender_sem_id,
        static_cast<uint32_t>(mcast_start.x),
        static_cast<uint32_t>(mcast_start.y),
        static_cast<uint32_t>(mcast_end.x),
        static_cast<uint32_t>(mcast_end.y),
        num_slices,
        tiles_per_sequence,
        cb_gathered_values,
        cb_gathered_indices};
    auto reader_final_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/reader_final.cpp",
        final_core_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_final_compile_args));

    const std::vector<uint32_t> compute_final_compile_args = {
        cb_gathered_values, cb_gathered_indices, cb_indices_out, llk_k};
    auto compute_final_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/compute_final.cpp",
        final_core_set,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = true, .dst_full_sync_en = true, .compile_args = compute_final_compile_args});

    std::vector<uint32_t> writer_final_compile_args = {
        cb_indices_out,
        cb_indices_scratch,
        indices_row_bytes,
        source_slices_per_row,
        output_slices_per_row,
        indices_slice_bytes};
    interleaved_accessor_args(indices).append_to(writer_final_compile_args);
    auto writer_final_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/writer.cpp",
        final_core_set,
        tt::tt_metal::WriterDataMovementConfig(writer_final_compile_args));

    TopkLargeIndicesMultiCoreSharedVariables shared{
        .reader_local_kernel_id = reader_local_kernel,
        .compute_local_kernel_id = compute_local_kernel,
        .writer_local_kernel_id = writer_local_kernel,
        .reader_final_kernel_id = reader_final_kernel,
        .compute_final_kernel_id = compute_final_kernel,
        .writer_final_kernel_id = writer_final_kernel,
        .local_cores = local_cores,
        .final_core = final_core};
    set_runtime_args_multi_core(program, shared, input, indices, llk_target_k, operation_attributes.valid_length);

    return cached_program_t{std::move(program), std::move(shared)};
}

void TopkLargeIndicesMultiCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    set_runtime_args_multi_core(
        cached_program.program,
        cached_program.shared_variables,
        tensor_args.input_tensor,
        tensor_return_value,
        snap_to_llk_target_k(operation_attributes.k),
        operation_attributes.valid_length);
}

}  // namespace ttnn::operations::experimental::topk_large_indices::program
