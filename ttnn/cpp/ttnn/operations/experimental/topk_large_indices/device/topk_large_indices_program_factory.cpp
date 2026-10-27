// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_program_factory.hpp"
#include "kernels/topk_large_indices_runtime_args.hpp"

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/math.hpp>

#include <optional>
#include <tuple>

namespace ttnn::operations::experimental::topk_large_indices::program {

namespace {

struct RuntimeShapeArgs {
    uint32_t num_rows = 0;
    uint32_t search_len = 0;
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

RuntimeShapeArgs get_runtime_shape_args(const Tensor& input, std::optional<uint32_t> valid_length) {
    const auto& shape = input.logical_shape();
    const uint32_t n = shape[shape.rank() - 1];
    // Number of columns to actually read and scan per row. Defaults to the full physical width n; a
    // valid_length bounds it to the real prefix so the stale tail is never read or ranked. The row STRIDE
    // (input_row_bytes) stays n so per-row addressing is unchanged — only how much we pull from each row shrinks.
    const uint32_t search_len = valid_length.value_or(n);
    return RuntimeShapeArgs{
        .num_rows = flattened_rows_excluding_last_dim(shape),
        .search_len = search_len,
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
    TopkLargeIndicesSharedVariables& shared,
    const Tensor& input,
    std::optional<uint32_t> valid_length) {
    const auto runtime_args = get_runtime_shape_args(input, valid_length);
    auto& reader_args = tt::tt_metal::GetCommonRuntimeArgs(program, shared.reader_kernel_id);
    reader_args[topk_common_args::search_length] = runtime_args.search_len;
    reader_args[topk_common_args::input_row_bytes] = runtime_args.input_row_bytes;
    tt::tt_metal::GetCommonRuntimeArgs(program, shared.compute_kernel_id)[topk_common_args::compute_search_length] =
        runtime_args.search_len;
    // The row assignment depends on the row count and, through the column split, on the searched chunk count.
    const uint32_t num_chunks = tt::div_up(runtime_args.search_len, shared.llk_k);
    if (shared.num_rows == runtime_args.num_rows && shared.num_chunks == num_chunks) {
        return;
    }
    const auto assignments =
        derive_core_row_assignments(shared.core_grid, runtime_args.num_rows, num_chunks, shared.row_split);
    TT_FATAL(
        assignments.size() == shared.cores.size(),
        "topk_large_indices runtime assignment core count {} differs from compiled core count {}",
        assignments.size(),
        shared.cores.size());
    const auto* device = input.device();
    for (uint32_t i = 0; i < assignments.size(); ++i) {
        const auto& assignment = assignments[i];
        TT_FATAL(
            assignment.core == shared.cores[i],
            "topk_large_indices runtime assignment core {} differs from compiled core {} at position {}",
            assignment.core,
            shared.cores[i],
            i);
        if (!shared.row_split) {
            tt::tt_metal::SetRuntimeArgs(
                program, shared.reader_kernel_id, assignment.core, {assignment.start_row, assignment.num_rows});
            tt::tt_metal::SetRuntimeArgs(program, shared.compute_kernel_id, assignment.core, {assignment.num_rows});
            tt::tt_metal::SetRuntimeArgs(
                program, shared.writer_kernel_id, assignment.core, {assignment.start_row, assignment.num_rows});
            continue;
        }
        const uint32_t has_split = assignment.has_split ? 1u : 0u;
        const uint32_t second_half = assignment.second_half ? 1u : 0u;
        const auto partner = assignment.has_split
                                 ? device->worker_core_from_logical_core(assignments[assignment.partner_index].core)
                                 : CoreCoord{0, 0};
        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.reader_kernel_id,
            assignment.core,
            {assignment.start_row, assignment.num_rows, assignment.split_row, has_split, second_half});
        tt::tt_metal::SetRuntimeArgs(
            program, shared.compute_kernel_id, assignment.core, {assignment.num_rows, has_split, second_half});
        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.writer_kernel_id,
            assignment.core,
            {assignment.start_row,
             assignment.num_rows,
             has_split,
             second_half,
             static_cast<uint32_t>(partner.x),
             static_cast<uint32_t>(partner.y),
             assignment.split_row});
    }
    shared.num_rows = runtime_args.num_rows;
    shared.num_chunks = num_chunks;
}

}  // namespace

ComputeBodyMode compute_body_mode(uint32_t k, uint32_t input_last_dim) {
    const uint32_t llk_k = to_uint32(snap_to_llk_target_k(k));

    // For an internal K >= 1024, segmented fusion handles every width with
    // one binary; rows of at most 32 chunks naturally execute as one segment.
    // Gate on the snapped LLK K so public k values in [528, 1008] get the same
    // fused body as k=1024 instead of silently falling back to classic.
    if (llk_k >= to_uint32(LlkTargetK::K1024)) {
        return ComputeBodyMode::FusedSegmented;
    }

    const uint32_t physical_chunks = tt::div_up(input_last_dim, llk_k);
    return physical_chunks <= max_fused_chunks ? ComputeBodyMode::FusedEndToEnd : ComputeBodyMode::Classic;
}

bool multi_chunk_rows(uint32_t k, uint32_t input_last_dim) {
    return tt::div_up(input_last_dim, to_uint32(snap_to_llk_target_k(k))) >= 2;
}

bool long_rows(uint32_t k, uint32_t input_last_dim) {
    return tt::div_up(input_last_dim, to_uint32(snap_to_llk_target_k(k))) > max_fused_chunks;
}

bool row_split_enabled(uint32_t k, uint32_t num_rows, uint32_t input_last_dim, uint32_t num_cores) {
    return compute_body_mode(k, input_last_dim) != ComputeBodyMode::Classic && multi_chunk_rows(k, input_last_dim) &&
           row_split_possible(num_rows, num_cores);
}

bool split_whole_rows(uint32_t k, uint32_t num_rows, uint32_t input_last_dim, uint32_t num_cores) {
    // The Classic body has no split form either (see row_split_enabled).
    if (compute_body_mode(k, input_last_dim) == ComputeBodyMode::Classic) {
        return false;
    }
    // Only K' 2048 has a split body for a row past one fused segment. Below that the split needs the whole row
    // to fit a segment, so a longer row could never take it, and compiling reduce_split_row for such a program
    // just spends kernel config buffer on a path it cannot reach.
    if (long_rows(k, input_last_dim) && to_uint32(snap_to_llk_target_k(k)) != to_uint32(LlkTargetK::K2048)) {
        return false;
    }
    return multi_chunk_rows(k, input_last_dim) && (!row_split_possible(num_rows, num_cores) || num_rows >= num_cores);
}

std::vector<CoreRowAssignment> derive_core_row_assignments(
    const CoreRangeSet& core_grid, uint32_t num_rows, uint32_t num_chunks, bool row_split) {
    TT_FATAL(num_rows > 0, "topk_large_indices requires at least one row of work");
    TT_FATAL(num_chunks > 0, "topk_large_indices requires at least one chunk per row");
    const auto all_cores = corerange_to_cores(core_grid, std::nullopt, true);
    const uint64_t num_cores = all_cores.size();

    // Every core takes the same number of whole rows; the rows left over are halved across core pairs when two
    // cores per row are free, so no core works a whole row more than the others.
    const uint32_t whole_rounds = static_cast<uint32_t>(num_rows / num_cores);
    const uint32_t split_rows = static_cast<uint32_t>(num_rows % num_cores);
    // Without the compiled transfer code the caller reads only start_row and num_rows, and the halved layout
    // does not describe a contiguous row range -- producing it anyway would drop rows. The geometry test is
    // row_split_possible's, threaded through rather than re-derived, so the two cannot disagree.
    const bool pair_split =
        row_split && row_split_possible(num_rows, static_cast<uint32_t>(num_cores)) && num_chunks >= 2;

    if (pair_split) {
        const uint64_t num_units = 2ull * split_rows;
        // Every unit below num_units reads assignments[unit ^ 1], so the pair range must fit the grid.
        // row_split_possible() already guarantees this; restated here because the loop below indexes on it.
        TT_FATAL(
            num_units <= num_cores,
            "topk_large_indices halved {} rows over {} cores, needing {} of them",
            split_rows,
            num_cores,
            num_units);
        std::vector<CoreRowAssignment> assignments;
        assignments.reserve(all_cores.size());
        for (uint64_t unit = 0; unit < all_cores.size(); ++unit) {
            CoreRowAssignment assignment{
                .core = all_cores[unit],
                .start_row = static_cast<uint32_t>(unit * whole_rounds),
                .num_rows = whole_rounds};
            if (unit < num_units) {
                // Halves of one row are adjacent, so the partner is the unit with the low bit flipped.
                assignment.num_rows += 1;
                assignment.partner_index = static_cast<uint32_t>(unit ^ 1ull);
                assignment.split_row = static_cast<uint32_t>(uint64_t{whole_rounds} * num_cores + unit / 2);
                assignment.has_split = true;
                assignment.second_half = (unit & 1ull) != 0;
            }
            assignments.push_back(assignment);
        }
        return assignments;
    }

    const auto work_split = tt::tt_metal::split_work_to_cores(core_grid, num_rows, true);
    const auto num_active_cores = std::get<0>(work_split);
    const auto& core_group_1 = std::get<2>(work_split);
    const auto& core_group_2 = std::get<3>(work_split);
    const auto num_rows_per_core_group_1 = std::get<4>(work_split);
    const auto num_rows_per_core_group_2 = std::get<5>(work_split);
    TT_FATAL(num_active_cores > 0, "topk_large_indices requires at least one row of work");

    std::vector<CoreRowAssignment> assignments;
    assignments.reserve(all_cores.size());
    uint32_t start_row = 0;
    for (const auto& core : all_cores) {
        const uint32_t rows =
            rows_for_core(core, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2);
        TT_FATAL(
            rows <= num_rows_per_core_group_1,
            "topk_large_indices assigned {} rows to a core, expected at most {}",
            rows,
            num_rows_per_core_group_1);
        assignments.push_back(CoreRowAssignment{.core = core, .start_row = start_row, .num_rows = rows});
        start_row += rows;
    }
    TT_FATAL(start_row == num_rows, "topk_large_indices assigned {} rows, expected {}", start_row, num_rows);
    return assignments;
}

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

    const auto& all_cores = operation_attributes.resolved_worker_core_grid;
    const auto cores = corerange_to_cores(all_cores, std::nullopt, true);
    // Runtime row counts are intentionally patched through runtime args instead of the program hash. The
    // caller-selected structural core grid is fixed in the hash, so cache hits can change shape without ever
    // creating kernels or CBs on cores owned by another subdevice.

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_indices = tt::CBIndex::c_1;
    constexpr uint32_t cb_indices_scratch = tt::CBIndex::c_2;
    // Reader-to-compute mailbox for the derived chunk count and tail length. It also receives the metadata read.
    constexpr uint32_t cb_meta = tt::CBIndex::c_3;
    // Row split: a partner's survivor lands in cb_landing (one slot, same address on every core).
    constexpr uint32_t cb_landing = tt::CBIndex::c_4;
    constexpr uint32_t cb_send = tt::CBIndex::c_5;
    constexpr uint32_t cb_acc = tt::CBIndex::c_6;

    const uint32_t input_chunk_bytes = llk_k * input.element_size();
    const uint32_t input_tile_bytes = tt::constants::TILE_HW * input.element_size();
    constexpr uint32_t row_slice_elements = tt::constants::FACE_WIDTH;
    const uint32_t source_slices_per_row = llk_k / row_slice_elements;
    const uint32_t output_slices_per_row = k / row_slice_elements;
    const uint32_t indices_slice_bytes = row_slice_elements * indices.element_size();
    const uint32_t indices_row_bytes = k * indices.element_size();
    const uint32_t indices_cb_row_bytes = llk_k * indices.element_size();
    constexpr uint32_t survivor_tile_bytes = tt::constants::TILE_HW * sizeof(uint32_t);
    const uint32_t survivor_tiles = 2 * tiles_per_sequence;
    const uint32_t survivor_bytes = survivor_tiles * survivor_tile_bytes;

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

    const bool has_meta = tensor_args.has_valid_length_metadata();
    if (has_meta) {
        auto meta_cb_config =
            tt::tt_metal::CircularBufferConfig(64, {{cb_meta, tt::DataFormat::UInt32}}).set_page_size(cb_meta, 64);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, meta_cb_config);
    }

    // A row of one chunk cannot be halved, so its program carries no transfer code (multi_chunk_rows and
    // row_split_possible are both in the program hash).
    const uint32_t row_split = row_split_enabled(
                                   k,
                                   flattened_rows_excluding_last_dim(input.logical_shape()),
                                   input.logical_shape()[-1],
                                   static_cast<uint32_t>(cores.size()))
                                   ? 1u
                                   : 0u;
    uint32_t credit_semaphore_id = 0;
    uint32_t data_semaphore_id = 0;
    if (row_split) {
        const auto create_survivor_cb = [&](uint8_t cb) {
            const auto survivor_cb_config =
                // Raw 32 bit words: the index tiles are integers and must not pass through the float packer.
                tt::tt_metal::CircularBufferConfig(survivor_bytes, {{cb, tt::DataFormat::UInt32}})
                    .set_page_size(cb, survivor_tile_bytes);
            tt::tt_metal::CreateCircularBuffer(program, all_cores, survivor_cb_config);
        };
        create_survivor_cb(cb_landing);
        create_survivor_cb(cb_send);
        credit_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
        data_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    }

    std::vector<uint32_t> reader_compile_args = {cb_in, input_chunk_bytes, input_tile_bytes, tiles_per_sequence};
    interleaved_accessor_args(input).append_to(reader_compile_args);
    // Keep this block fixed-width so the kernel can use direct compile-argument offsets. The scalar path's
    // placeholders and duplicate input accessor are compiled out.
    reader_compile_args.push_back(has_meta ? 1u : 0u);
    reader_compile_args.push_back(has_meta ? cb_meta : 0u);
    reader_compile_args.push_back(has_meta ? operation_attributes.valid_length_offset : 0u);
    interleaved_accessor_args(has_meta ? *tensor_args.valid_length_tensor : input).append_to(reader_compile_args);
    // Real-token-end block: accessor only (placeholder when absent). No presence flag -- metadata mode is
    // one flag now, and the kernel reads presence from the common arg's VALUE (0 = uncapped).
    const bool has_vend = tensor_args.has_valid_end_metadata();
    interleaved_accessor_args(has_vend ? *tensor_args.valid_end_tensor : input).append_to(reader_compile_args);

    // The reader streams the input rows from DRAM, the heavy traffic; the writer emits k indices per row and
    // the row-split survivor hand-off, so it takes whichever NoC the reader leaves free.
    const auto reader_noc = operation_attributes.dram_read_noc;
    const auto writer_noc =
        reader_noc == tt::tt_metal::NOC::NOC_0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;
    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/reader.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = reader_noc,
            .compile_args = reader_compile_args,
            .named_compile_args = {{"row_split", row_split}}});

    const auto body_mode = compute_body_mode(k, input.logical_shape()[-1]);
    const bool rows_are_multi_chunk = multi_chunk_rows(k, input.logical_shape()[-1]);
    const bool whole_rows_split = split_whole_rows(
        k,
        flattened_rows_excluding_last_dim(input.logical_shape()),
        input.logical_shape()[-1],
        static_cast<uint32_t>(cores.size()));
    // The K' 2048 segmented body holds its unfused accumulator in L1 while a later segment is split across the
    // threads: two sequences of raw 32-bit words, packed out of Dst and unpacked back. Whole rows and
    // block-aligned halves both take it, but only past one segment -- a shorter row never touches the CB.
    const bool has_acc =
        llk_k == to_uint32(LlkTargetK::K2048) && whole_rows_split && long_rows(k, input.logical_shape()[-1]);
    if (has_acc) {
        // Same shape as a survivor: two sequences of raw 32-bit words.
        const auto acc_cb_config =
            tt::tt_metal::CircularBufferConfig(survivor_bytes, {{cb_acc, tt::DataFormat::UInt32}})
                .set_page_size(cb_acc, survivor_tile_bytes);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, acc_cb_config);
    }

    std::vector<uint32_t> compute_compile_args = {cb_in, cb_indices, llk_k, static_cast<uint32_t>(body_mode)};
    compute_compile_args.push_back(has_meta ? 1u : 0u);
    compute_compile_args.push_back(has_meta ? cb_meta : 0u);
    compute_compile_args.push_back(cb_landing);
    compute_compile_args.push_back(cb_send);
    compute_compile_args.push_back(rows_are_multi_chunk ? 1u : 0u);
    compute_compile_args.push_back(cb_acc);
    // cb_in is CB 0, so the accumulator's presence cannot be inferred from its id being zero.
    compute_compile_args.push_back(has_acc ? 1u : 0u);
    auto compute_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            // TopK XL stores fused BF16-value/u16-index words and unfused UINT32 indices in 32-bit DEST lanes.
            .fp32_dest_acc_en = true,
            // K=2048 multi-chunk merge uses DEST slots 0..7; FP32 half-sync mode exposes only 4 tiles.
            .dst_full_sync_en = true,
            .compile_args = compute_compile_args,
            .named_compile_args = {
                {"row_split", row_split},
                {"split_whole_rows", whole_rows_split ? 1u : 0u},
                {"long_rows", long_rows(k, input.logical_shape()[-1]) ? 1u : 0u}}});

    std::vector<uint32_t> writer_compile_args = {
        cb_indices,
        cb_indices_scratch,
        indices_row_bytes,
        source_slices_per_row,
        output_slices_per_row,
        indices_slice_bytes};
    interleaved_accessor_args(indices).append_to(writer_compile_args);
    writer_compile_args.push_back(cb_landing);
    writer_compile_args.push_back(cb_send);
    writer_compile_args.push_back(credit_semaphore_id);
    writer_compile_args.push_back(data_semaphore_id);
    writer_compile_args.push_back(survivor_tiles);
    writer_compile_args.push_back(survivor_bytes);

    auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/writer.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_noc,
            .compile_args = writer_compile_args,
            .named_compile_args = {{"row_split", row_split}}});

    TopkLargeIndicesSharedVariables shared{
        .reader_kernel_id = reader_kernel,
        .compute_kernel_id = compute_kernel,
        .writer_kernel_id = writer_kernel,
        .core_grid = all_cores,
        .cores = cores,
        .input_shape = input.logical_shape(),
        .valid_length = operation_attributes.valid_length,
        .llk_k = llk_k,
        .row_split = row_split != 0};
    const uint32_t meta_addr =
        tensor_args.has_valid_length_metadata() ? tensor_args.valid_length_tensor->buffer()->address() : 0u;
    const uint32_t valid_end_addr =
        tensor_args.has_valid_end_metadata() ? tensor_args.valid_end_tensor->buffer()->address() : 0u;
    tt::tt_metal::SetCommonRuntimeArgs(
        program, reader_kernel, {input.buffer()->address(), meta_addr, 0, 0, valid_end_addr});
    tt::tt_metal::SetCommonRuntimeArgs(program, compute_kernel, {0});
    tt::tt_metal::SetCommonRuntimeArgs(program, writer_kernel, {indices.buffer()->address()});
    set_runtime_args(program, shared, input, operation_attributes.valid_length);

    return cached_program_t{std::move(program), std::move(shared)};
}

void TopkLargeIndicesProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    TT_FATAL(
        operation_attributes.resolved_worker_core_grid == cached_program.shared_variables.core_grid,
        "topk_large_indices cache hit resolved grid {} differs from compiled grid {}",
        operation_attributes.resolved_worker_core_grid,
        cached_program.shared_variables.core_grid);
    auto& shared = cached_program.shared_variables;
    const auto& input = tensor_args.input_tensor;
    auto& reader_args = tt::tt_metal::GetCommonRuntimeArgs(cached_program.program, shared.reader_kernel_id);
    auto& writer_args = tt::tt_metal::GetCommonRuntimeArgs(cached_program.program, shared.writer_kernel_id);
    reader_args[topk_common_args::input_address] = input.buffer()->address();
    reader_args[topk_common_args::metadata_address] =
        tensor_args.has_valid_length_metadata() ? tensor_args.valid_length_tensor->buffer()->address() : 0u;
    // Refreshed alongside metadata_address: the bound lives in a device tensor whose buffer can move
    // between dispatches, and a stale address here would silently clamp the search to garbage.
    reader_args[topk_common_args::valid_end_address] =
        tensor_args.has_valid_end_metadata() ? tensor_args.valid_end_tensor->buffer()->address() : 0u;
    writer_args[topk_common_args::output_address] = tensor_return_value.buffer()->address();

    // The cache key fixes k, dtype, grid and compute body mode. Shape and valid_length
    // are runtime controls: update their common arguments when either changes, and
    // rebuild the per-core row assignment only when the flattened row count changes.
    if (shared.input_shape == input.logical_shape() && shared.valid_length == operation_attributes.valid_length) {
        return;
    }

    set_runtime_args(cached_program.program, shared, input, operation_attributes.valid_length);
    shared.input_shape = input.logical_shape();
    shared.valid_length = operation_attributes.valid_length;
}

}  // namespace ttnn::operations::experimental::topk_large_indices::program
