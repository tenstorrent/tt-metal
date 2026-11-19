// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_program_factory.hpp"
#include "kernels/topk_large_indices_runtime_args.hpp"
#include "kernels/topk_large_indices_two_level.hpp"

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/math.hpp>

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
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
    // The row assignment depends on the row count and, through the column split, on the searched chunk count. A
    // two-level program picks its mode on device, so both modes share an assignment cut from the physical width.
    const uint32_t assignment_len =
        shared.two_level ? runtime_args.input_row_bytes / input.element_size() : runtime_args.search_len;
    const uint32_t num_chunks = tt::div_up(assignment_len, shared.llk_k);
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
        if (!shared.row_split && !shared.two_level) {
            tt::tt_metal::SetRuntimeArgs(
                program, shared.reader_kernel_id, assignment.core, {assignment.start_row, assignment.num_rows});
            tt::tt_metal::SetRuntimeArgs(program, shared.compute_kernel_id, assignment.core, {assignment.num_rows});
            tt::tt_metal::SetRuntimeArgs(
                program, shared.writer_kernel_id, assignment.core, {assignment.start_row, assignment.num_rows});
            continue;
        }
        // A two-level program always takes the split layout (zero without a split row), then its own slots.
        const uint32_t has_split = assignment.has_split ? 1u : 0u;
        const uint32_t second_half = assignment.second_half ? 1u : 0u;
        const auto partner = assignment.has_split
                                 ? device->worker_core_from_logical_core(assignments[assignment.partner_index].core)
                                 : CoreCoord{0, 0};
        std::vector<uint32_t> reader_rt = {
            assignment.start_row, assignment.num_rows, assignment.split_row, has_split, second_half};
        std::vector<uint32_t> compute_rt = {assignment.num_rows, has_split, second_half};
        std::vector<uint32_t> writer_rt = {
            assignment.start_row,
            assignment.num_rows,
            has_split,
            second_half,
            static_cast<uint32_t>(partner.x),
            static_cast<uint32_t>(partner.y),
            assignment.split_row};
        if (shared.two_level) {
            reader_rt.push_back(i);
        }
        tt::tt_metal::SetRuntimeArgs(program, shared.reader_kernel_id, assignment.core, reader_rt);
        tt::tt_metal::SetRuntimeArgs(program, shared.compute_kernel_id, assignment.core, compute_rt);
        tt::tt_metal::SetRuntimeArgs(program, shared.writer_kernel_id, assignment.core, writer_rt);
    }
    shared.num_rows = runtime_args.num_rows;
    shared.num_chunks = num_chunks;
}

void set_two_level_common_args(
    tt::tt_metal::Program& program,
    TopkLargeIndicesSharedVariables& shared,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args) {
    if (!shared.two_level) {
        return;
    }
    const auto& input = tensor_args.input_tensor;
    const uint32_t num_rows = flattened_rows_excluding_last_dim(input.logical_shape());
    const uint32_t blocks = tt::div_up(input.logical_shape()[-1], two_level_block_keys);
    const uint32_t num_cores = static_cast<uint32_t>(shared.cores.size());
    // The maxima stage 2 reads: the caller's, else the caller's maxima output, else the program's own. The row count
    // and width are in the program hash, so the program's own is allocated once and keeps its address for as long as
    // the program is cached (a captured trace that runs the program needs it cached).
    const bool own_maxima = !tensor_args.block_max_tensor.has_value() && !tensor_args.block_max_output.has_value();
    if (own_maxima && !shared.two_level_maxima.has_value()) {
        shared.two_level_maxima = create_device_tensor(
            tt::tt_metal::TensorSpec(
                ttnn::Shape({num_rows, blocks}),
                tt::tt_metal::TensorLayout(
                    DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), DRAM_MEMORY_CONFIG)),
            input.device());
    }
    const uint32_t maxima_address =
        tensor_args.block_max_tensor.has_value()   ? tensor_args.block_max_tensor->mesh_buffer().address()
        : tensor_args.block_max_output.has_value() ? tensor_args.block_max_output->mesh_buffer().address()
                                                   : shared.two_level_maxima->mesh_buffer().address();
    const uint32_t row_bytes = blocks * input.element_size();
    auto& reader_args = tt::tt_metal::GetCommonRuntimeArgs(program, shared.reader_kernel_id);
    reader_args[topk_common_args::two_level_cutover] =
        attrs.two_level_cutover.value_or(two_level_default_cutover(num_cores));
    reader_args[topk_common_args::block_max_address] = maxima_address;
    reader_args[topk_common_args::two_level_max_row_bytes] = row_bytes;
    reader_args[topk_common_args::two_level_num_input_rows] = num_rows;
    reader_args[topk_common_args::two_level_num_cores] = num_cores;
    // The rows the single-level assignment halves across core pairs; stage 3 of each runs in halves on its pair.
    const uint32_t split_rows = shared.row_split && row_split_possible(num_rows, num_cores) ? num_rows % num_cores : 0;
    reader_args[topk_common_args::two_level_split_rows] = split_rows;
    auto& compute_args = tt::tt_metal::GetCommonRuntimeArgs(program, shared.compute_kernel_id);
    compute_args[topk_common_args::compute_two_level_num_input_rows] = num_rows;
    auto& writer_args = tt::tt_metal::GetCommonRuntimeArgs(program, shared.writer_kernel_id);
    writer_args[topk_common_args::two_level_max_address] = maxima_address;
    writer_args[topk_common_args::mode_output_address] =
        tensor_args.mode_output.has_value() ? tensor_args.mode_output->mesh_buffer().address() : 0u;
    writer_args[topk_common_args::writer_two_level_max_row_bytes] = row_bytes;
    writer_args[topk_common_args::writer_two_level_num_input_rows] = num_rows;
    const auto coordinator = input.device()->worker_core_from_logical_core(shared.cores.front());
    reader_args[topk_common_args::reader_coordinator_noc_x] = static_cast<uint32_t>(coordinator.x);
    reader_args[topk_common_args::reader_coordinator_noc_y] = static_cast<uint32_t>(coordinator.y);
    writer_args[topk_common_args::writer_two_level_num_cores] = num_cores;
    writer_args[topk_common_args::writer_two_level_split_rows] = split_rows;
    for (uint32_t core = 0; core < num_cores; ++core) {
        const auto noc = input.device()->worker_core_from_logical_core(shared.cores[core]);
        writer_args[topk_common_args::writer_two_level_core_coords + core] =
            static_cast<uint32_t>(noc.x) | (static_cast<uint32_t>(noc.y) << 16);
    }
}

}  // namespace

bool two_level_enabled(const operation_attributes_t& attrs, uint32_t input_last_dim) {
    return attrs.block_select.has_value() && attrs.k == to_uint32(LlkTargetK::K2048) &&
           input_last_dim % two_level_block_keys == 0 && input_last_dim >= two_level_min_width &&
           input_last_dim <= two_level_max_width;
}

static_assert(two_level_max_rows_per_core == topk_two_level::max_local_rows);

uint32_t two_level_default_cutover(uint32_t num_cores) {
    // Where the two modes cost the same for 160 rows on p150b: about 200K keys on 108 cores and 150K on 99, whose
    // single-level rows split less evenly.
    return num_cores >= 108 ? 196'608 : 147'456;
}

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
    auto program = tt::tt_metal::CreateProgram(*tensor_args.input_tensor.device());

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
    // A two-level program gathers its candidate chunks block by block; deeper buffering hides the read latency.
    const uint32_t input_cb_depth = two_level_enabled(operation_attributes, input.logical_shape()[-1]) ? 4 : cb_depth;
    const auto input_cb_config =
        tt::tt_metal::CircularBufferConfig(
            input_cb_depth * tiles_per_sequence * input_tile_bytes, {{cb_in, tt::DataFormat::Float16_b}})
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
    const uint32_t input_width = input.logical_shape()[-1];
    const bool two_level = two_level_enabled(operation_attributes, input_width);
    // A two-level program's compute reads its bounds and mode from the reader's mailbox in scalar mode too.
    const bool compute_mailbox = has_meta || two_level;
    if (compute_mailbox) {
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

    // Two-level CBs (see kernels/topk_large_indices_two_level.hpp).
    constexpr uint32_t cb_stage1 = tt::CBIndex::c_7;
    constexpr uint32_t cb_tiled = tt::CBIndex::c_8;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_9;
    constexpr uint32_t cb_red = tt::CBIndex::c_10;
    constexpr uint32_t cb_max = tt::CBIndex::c_11;
    constexpr uint32_t cb_table = tt::CBIndex::c_12;
    constexpr uint32_t cb_bitmap = tt::CBIndex::c_13;
    constexpr uint32_t cb_own_table = tt::CBIndex::c_14;
    constexpr uint32_t cb_writer_mailbox = tt::CBIndex::c_15;
    constexpr uint32_t stage1_batch = 32;
    if (two_level) {
        // Stage 3 reads 64 B blocks straight into candidate slots: the DRAM read rule needs the source and the L1
        // destination congruent modulo 64. Slots are multiples of 64 B from a CB base that the allocator aligns to
        // the DRAM alignment.
        const uint32_t cb_alignment = input.device()->allocator()->get_alignment(tt::tt_metal::BufferType::DRAM);
        TT_FATAL(
            cb_alignment % 64 == 0,
            "topk_large_indices two-level selection needs CB bases aligned to 64 B (allocator alignment {})",
            cb_alignment);
        const auto create_cb = [&](uint32_t cb, uint32_t pages, uint32_t page_bytes, tt::DataFormat format) {
            const auto config =
                tt::tt_metal::CircularBufferConfig(pages * page_bytes, {{cb, format}}).set_page_size(cb, page_bytes);
            tt::tt_metal::CreateCircularBuffer(program, all_cores, config);
        };
        const uint32_t table_bytes = 2048 * sizeof(uint32_t);
        // Two units, each a header page (the unit's number) and its tiles.
        create_cb(cb_stage1, 2 * (1 + stage1_batch), input_tile_bytes, tt::DataFormat::Float16_b);
        create_cb(cb_tiled, stage1_batch, input_tile_bytes, tt::DataFormat::Float16_b);
        create_cb(cb_scaler, 1, input_tile_bytes, tt::DataFormat::Float16_b);
        // The compute packs eight reduced tiles per DEST acquire; the writer drains them one at a time.
        create_cb(cb_red, 16, input_tile_bytes, tt::DataFormat::Float16_b);
        // The writer's phase-1 staging: one unit of block maxima.
        create_cb(cb_max, 1, stage1_batch * 32 * input.element_size(), tt::DataFormat::Float16_b);
        create_cb(cb_table, 1, table_bytes, tt::DataFormat::UInt32);
        const uint32_t bitmap_bytes =
            tt::round_up(tt::div_up(tt::div_up(input_width, two_level_block_keys), 32) * sizeof(uint32_t), 64);
        create_cb(cb_bitmap, 1, bitmap_bytes, tt::DataFormat::UInt32);
        // The writer's copy of its row's table, then (with split rows) the half of a split row's table its pair's
        // first core sends.
        create_cb(cb_own_table, 1, row_split ? table_bytes + table_bytes / 2 : table_bytes, tt::DataFormat::UInt32);
        // The mode and search length, then a header per item (see topk_two_level::header_row).
        create_cb(cb_writer_mailbox, 4, 64, tt::DataFormat::UInt32);
    }
    // unit: on the coordinator, the next phase-1 unit to claim. claim: where a core's claims return the counter.
    // ready: a core's count of finished phase-1 units per local row.
    const uint32_t unit_semaphore_id = two_level ? tt::tt_metal::CreateSemaphore(program, all_cores, 0) : 0;
    const uint32_t claim_semaphore_id = two_level ? tt::tt_metal::CreateSemaphore(program, all_cores, 0) : 0;
    const uint32_t ready_semaphore_id = two_level ? tt::tt_metal::CreateSemaphore(program, all_cores, 0) : 0;
    // half: on the second core of a split pair, set once its half of the split row's table has arrived.
    const uint32_t half_semaphore_id = two_level ? tt::tt_metal::CreateSemaphore(program, all_cores, 0) : 0;
    const std::vector<std::pair<std::string, uint32_t>> two_level_named_args = {
        {"tl_unit_sem", unit_semaphore_id},
        {"tl_claim_sem", claim_semaphore_id},
        {"tl_ready_sem", ready_semaphore_id},
        {"tl_half_sem", half_semaphore_id},
        {"tl_cb_stage1", cb_stage1},
        {"tl_cb_tiled", cb_tiled},
        {"tl_cb_scaler", cb_scaler},
        {"tl_cb_red", cb_red},
        {"tl_cb_max", cb_max},
        {"tl_cb_table", cb_table},
        {"tl_cb_bitmap", cb_bitmap},
        {"tl_cb_own_table", cb_own_table},
        {"tl_cb_writer_mailbox", cb_writer_mailbox},
        {"tl_stage1_batch", stage1_batch},
        {"tl_block_max_input", tensor_args.block_max_tensor.has_value() ? 1u : 0u},
        {"tl_mode_output", tensor_args.mode_output.has_value() ? 1u : 0u},
        {"tl_page_bytes", input_tile_bytes},
        {"tl_chunk_keys", llk_k}};
    // The kernels compile their two-level code only under this define, so single-level binaries are unchanged.
    const std::map<std::string, std::string> two_level_defines =
        two_level ? std::map<std::string, std::string>{{"TOPK_TWO_LEVEL", "1"}} : std::map<std::string, std::string>{};
    // The split LLK's stamp offset (split-row halves stamped as their row) exists only in two-level compute builds.
    std::map<std::string, std::string> compute_defines = two_level_defines;
    if (two_level) {
        compute_defines["TOPK_XL_SPLIT_STAMP_OFFSET"] = "1";
    }
    const auto with_two_level = [&](std::unordered_map<std::string, uint32_t> named) {
        if (two_level) {
            for (const auto& [name, value] : two_level_named_args) {
                named[name] = value;
            }
        }
        return named;
    };

    std::vector<uint32_t> reader_compile_args = {cb_in, input_chunk_bytes, input_tile_bytes, tiles_per_sequence};
    interleaved_accessor_args(input).append_to(reader_compile_args);
    // Keep this block fixed-width so the kernel can use direct compile-argument offsets. The scalar path's
    // placeholders and duplicate input accessor are compiled out.
    reader_compile_args.push_back(has_meta ? 1u : 0u);
    reader_compile_args.push_back(compute_mailbox ? cb_meta : 0u);
    reader_compile_args.push_back(has_meta ? operation_attributes.valid_length_offset : 0u);
    interleaved_accessor_args(has_meta ? *tensor_args.valid_length_tensor : input).append_to(reader_compile_args);
    // Real-token-end block: accessor only (placeholder when absent). No presence flag -- metadata mode is
    // one flag now, and the kernel reads presence from the common arg's VALUE (0 = uncapped).
    const bool has_vend = tensor_args.has_valid_end_metadata();
    interleaved_accessor_args(has_vend ? *tensor_args.valid_end_tensor : input).append_to(reader_compile_args);
    if (two_level) {
        // The maxima tensor: the caller's or the program's own, DRAM interleaved either way.
        tt::tt_metal::TensorAccessorArgs::create_dram_interleaved().append_to(reader_compile_args);
    }

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
            .defines = two_level_defines,
            .named_compile_args = with_two_level({{"row_split", row_split}})});

    const auto body_mode = compute_body_mode(k, input.logical_shape()[-1]);
    const bool rows_are_multi_chunk = multi_chunk_rows(k, input.logical_shape()[-1]);
    // A two-level program runs its stages on the whole-row split body, so it always carries it.
    const bool whole_rows_split = two_level || split_whole_rows(
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
    compute_compile_args.push_back(compute_mailbox ? 1u : 0u);
    compute_compile_args.push_back(compute_mailbox ? cb_meta : 0u);
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
            .defines = compute_defines,
            .named_compile_args = with_two_level(
                {{"row_split", row_split},
                 {"split_whole_rows", whole_rows_split ? 1u : 0u},
                 {"long_rows", long_rows(k, input.logical_shape()[-1]) ? 1u : 0u}})});

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
    std::unordered_map<std::string, uint32_t> writer_named = {{"row_split", row_split}};
    if (two_level) {
        // Test outputs; the indices are the placeholder when absent.
        writer_named["tl_max_out_args_offset"] = static_cast<uint32_t>(writer_compile_args.size());
        tt::tt_metal::TensorAccessorArgs::create_dram_interleaved().append_to(writer_compile_args);
        writer_named["tl_mode_args_offset"] = static_cast<uint32_t>(writer_compile_args.size());
        interleaved_accessor_args(tensor_args.mode_output.value_or(indices)).append_to(writer_compile_args);
    }

    auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/writer.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_noc,
            .compile_args = writer_compile_args,
            .defines = two_level_defines,
            .named_compile_args = with_two_level(writer_named)});

    TopkLargeIndicesSharedVariables shared{
        .reader_kernel_id = reader_kernel,
        .compute_kernel_id = compute_kernel,
        .writer_kernel_id = writer_kernel,
        .core_grid = all_cores,
        .cores = cores,
        .input_shape = input.logical_shape(),
        .valid_length = operation_attributes.valid_length,
        .llk_k = llk_k,
        .row_split = row_split != 0,
        .two_level = two_level};
    const uint32_t meta_addr =
        tensor_args.has_valid_length_metadata() ? tensor_args.valid_length_tensor->mesh_buffer().address() : 0u;
    const uint32_t valid_end_addr =
        tensor_args.has_valid_end_metadata() ? tensor_args.valid_end_tensor->mesh_buffer().address() : 0u;
    std::vector<uint32_t> reader_common = {input.mesh_buffer().address(), meta_addr, 0, 0, valid_end_addr};
    std::vector<uint32_t> writer_common = {indices.mesh_buffer().address()};
    if (two_level) {
        reader_common.resize(topk_common_args::two_level_split_rows + 1, 0);
        writer_common.resize(topk_common_args::writer_two_level_core_coords + cores.size(), 0);
    }
    tt::tt_metal::SetCommonRuntimeArgs(program, reader_kernel, reader_common);
    tt::tt_metal::SetCommonRuntimeArgs(
        program,
        compute_kernel,
        std::vector<uint32_t>(two_level ? topk_common_args::compute_two_level_num_input_rows + 1 : 1, 0));
    tt::tt_metal::SetCommonRuntimeArgs(program, writer_kernel, writer_common);
    set_runtime_args(program, shared, input, operation_attributes.valid_length);
    set_two_level_common_args(program, shared, operation_attributes, tensor_args);

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
    reader_args[topk_common_args::input_address] = input.mesh_buffer().address();
    reader_args[topk_common_args::metadata_address] =
        tensor_args.has_valid_length_metadata() ? tensor_args.valid_length_tensor->mesh_buffer().address() : 0u;
    // Refreshed alongside metadata_address: the bound lives in a device tensor whose buffer can move
    // between dispatches, and a stale address here would silently clamp the search to garbage.
    reader_args[topk_common_args::valid_end_address] =
        tensor_args.has_valid_end_metadata() ? tensor_args.valid_end_tensor->mesh_buffer().address() : 0u;
    writer_args[topk_common_args::output_address] = tensor_return_value.mesh_buffer().address();
    set_two_level_common_args(cached_program.program, shared, operation_attributes, tensor_args);

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
