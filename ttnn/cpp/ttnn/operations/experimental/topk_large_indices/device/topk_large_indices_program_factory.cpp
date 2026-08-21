// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_program_factory.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <algorithm>
#include <map>
#include <optional>
#include <string>
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

// Double-buffered input chunk-staging CB, identical in both factories.
void create_input_cb(
    tt::tt_metal::Program& program,
    const CoreRangeSet& cores,
    uint32_t cb_depth,
    uint32_t tiles_per_sequence,
    uint32_t input_tile_bytes,
    uint32_t cb_in) {
    const auto input_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_depth * tiles_per_sequence * input_tile_bytes, {{cb_in, tt::DataFormat::Float16_b}})
            .set_page_size(cb_in, input_tile_bytes);
    tt::tt_metal::CreateCircularBuffer(program, cores, input_cb_config);
}

// Output-side CBs (materialized index row + face-reorder scratch), identical
// in both factories; the tree factory creates them on the root cores only.
// The indices CB carries the raw 32-bit index words the packer produces.
void create_indices_output_cbs(
    tt::tt_metal::Program& program,
    const CoreRangeSet& cores,
    LlkTargetK llk_target_k,
    uint32_t cb_depth,
    uint32_t indices_cb_row_bytes,
    uint32_t indices_row_bytes,
    uint32_t cb_indices,
    uint32_t cb_indices_scratch) {
    auto indices_cb_config =
        tt::tt_metal::CircularBufferConfig(cb_depth * indices_cb_row_bytes, {{cb_indices, tt::DataFormat::Float32}})
            .set_page_size(cb_indices, indices_cb_row_bytes);
    if (llk_target_k == LlkTargetK::K512) {
        indices_cb_config.set_unpack_face_geometry(cb_indices, tt::constants::FACE_HEIGHT, 2);
    }
    tt::tt_metal::CreateCircularBuffer(program, cores, indices_cb_config);

    if (llk_target_k != LlkTargetK::K512) {
        const auto indices_scratch_cb_config =
            tt::tt_metal::CircularBufferConfig(indices_row_bytes, {{cb_indices_scratch, tt::DataFormat::Float32}})
                .set_page_size(cb_indices_scratch, indices_row_bytes);
        tt::tt_metal::CreateCircularBuffer(program, cores, indices_scratch_cb_config);
    }
}

// Chunked row reader (kernels/reader.cpp), shared source for both factories;
// the tree factory passes TOPK_TREE to enable the per-core slice offset.
tt::tt_metal::KernelHandle create_reader_kernel(
    tt::tt_metal::Program& program,
    const CoreRangeSet& cores,
    const Tensor& input,
    uint32_t cb_in,
    uint32_t input_chunk_bytes,
    uint32_t input_tile_bytes,
    uint32_t tiles_per_sequence,
    std::map<std::string, std::string> defines) {
    std::vector<uint32_t> reader_compile_args = {cb_in, input_chunk_bytes, input_tile_bytes, tiles_per_sequence};
    interleaved_accessor_args(input).append_to(reader_compile_args);
    return tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/reader.cpp",
        cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args, std::move(defines)));
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
    const operation_attributes_t& attrs) {
    const LlkTargetK llk_target_k = snap_to_llk_target_k(attrs.k);
    const std::optional<uint32_t> valid_length = attrs.valid_length;
    const auto runtime_args = get_runtime_shape_args(input, llk_target_k, valid_length);
    // Row window (hybrid wrapper): readers index global input rows from row_base;
    // writers stay output-relative (the window's output has its own row 0).
    const uint32_t row_base = attrs.row_start.value_or(0);
    const uint32_t effective_rows = attrs.row_count.value_or(runtime_args.num_rows);
    const auto work_split =
        tt::tt_metal::split_work_to_cores(input.device()->compute_with_storage_grid_size(), effective_rows, true);
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

        const uint32_t input_row = row_base + start_row;
        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.reader_kernel_id,
            core,
            {input.buffer()->address(),
             input_row,
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
        start_row == effective_rows, "topk_large_indices assigned {} rows, expected {}", start_row, effective_rows);
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
    // The indices CB carries the raw 32-bit index words the packer produces.
    const uint32_t indices_cb_row_bytes = llk_k * static_cast<uint32_t>(sizeof(uint32_t));

    const uint32_t cb_depth = 2;
    create_input_cb(program, all_cores, cb_depth, tiles_per_sequence, input_tile_bytes, cb_in);
    create_indices_output_cbs(
        program,
        all_cores,
        llk_target_k,
        cb_depth,
        indices_cb_row_bytes,
        indices_row_bytes,
        cb_indices,
        cb_indices_scratch);

    auto reader_kernel = create_reader_kernel(
        program, all_cores, input, cb_in, input_chunk_bytes, input_tile_bytes, tiles_per_sequence, /*defines=*/{});

    // Row-reduction body (classic / fused end-to-end / fused segmented),
    // selected host-side from (k, physical last dim) and passed as a
    // compile-time arg the kernel dispatches on with if constexpr. Derived
    // from shape, so it is mirrored into compute_program_hash.
    const auto body_mode = compute_body_mode(k, input.logical_shape()[-1]);
    std::vector<uint32_t> compute_compile_args = {cb_in, cb_indices, llk_k, static_cast<uint32_t>(body_mode)};
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
    set_runtime_args(program, shared, input, tensor_return_value, operation_attributes);

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
        operation_attributes);
}

// ---------------------------------------------------------------------------
// Column-parallel (intra-row multi-core) path
// ---------------------------------------------------------------------------

// Slice-count cap: the in-place merge tree needs only one 2-sequence recv CB
// per core (16 KB at K=2048), so L1 does not bind P; the envelope is 7 tree
// levels (128 slices). In practice P is bound first by the worker-grid
// rectangle capacity (e.g. 13x10 = 130 on P150) and the chunk count.
constexpr uint32_t max_column_slices = 128;

namespace {

// ceil(log2(p)) for p >= 1: number of pairwise merge-tree levels.
uint32_t tree_levels(uint32_t p) {
    uint32_t levels = 0;
    while ((1u << levels) < p) {
        ++levels;
    }
    return levels;
}

// The built-in cost-model pick (no user override).
ColumnSplitConfig compute_model_column_split_config(
    uint32_t k, uint32_t n, uint32_t num_rows, const CoreCoord& grid, bool allow_multi_row) {
    constexpr uint32_t max_slices = max_column_slices;

    ColumnSplitConfig config{};
    // Intra-row parallelism only pays off when rows cannot saturate the grid.
    // Single row: the classic tree. Multiple rows (allow_multi_row): the
    // multi-rectangle form, considered only when EVERY row gets its own
    // concurrent rectangle (one rect-wave) — then the per-row cost comparison
    // below is exact and, at any runtime valid_length, the rect's saturating
    // makespan bounds the loss to one merge term while row-parallel would run
    // the same single wave. Rows beyond every rectangle capacity keep the
    // row-parallel factory (and its shape-free program hash) unchanged; the
    // hybrid wrapper handles those by splitting off a remainder window.
    if (num_rows != 1 && !allow_multi_row) {
        return config;
    }
    if (static_cast<uint32_t>(grid.x * grid.y) < 2) {
        return config;
    }
    if (num_rows == 0) {
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

    // Cost model in merge units: processing one chunk locally
    // (copy + lsb + fused sort + index split + merge + rebuild) ~ 2 units,
    // one tree merge+rebuild ~ 1 unit. With the log-tree the serial term is
    // ceil(log2 P), so cost(P) = 2*ceil(chunks/P) + ceil(log2 P). Tree cores
    // must form a rectangle, so search every a x b rectangle that fits the
    // grid (a <= grid.x, b <= grid.y, a*b <= min(chunks, max_slices)) and
    // take the cheapest P; ties prefer fewer cores. This beats the previous
    // full-rows-only fit (e.g. 32 chunks on a 13x10 grid: 8x4 = 32 slices vs
    // 13x2 = 26; 128+ chunks: 8x8 = 64 vs 13x4 = 52).
    const uint32_t slice_ceiling = std::min(num_chunks, max_slices);
    uint32_t num_slices = 0;
    uint32_t local_grid_x = 0;
    uint32_t local_grid_y = 0;
    uint32_t best_cost = std::numeric_limits<uint32_t>::max();
    for (uint32_t a = 1; a <= static_cast<uint32_t>(grid.x); ++a) {
        for (uint32_t b = 1; b <= static_cast<uint32_t>(grid.y); ++b) {
            const uint32_t p = a * b;
            if (p < 2 || p > slice_ceiling) {
                continue;
            }
            // Multi-row: every row must own a concurrent rectangle, so the
            // candidate's grid-tiling capacity must cover the row count.
            const uint32_t capacity = (static_cast<uint32_t>(grid.x) / a) * (static_cast<uint32_t>(grid.y) / b);
            if (num_rows > capacity) {
                continue;
            }
            const uint32_t cost = 2 * tt::div_up(num_chunks, p) + tree_levels(p);
            if (cost < best_cost || (cost == best_cost && p < num_slices)) {
                best_cost = cost;
                num_slices = p;
                local_grid_x = a;
                local_grid_y = b;
            }
        }
    }
    if (num_slices < 2) {
        return config;
    }

    const uint32_t cost_row = 2 * num_chunks;  // one row per core on the row-parallel path (single wave)
    if (best_cost >= cost_row) {
        return config;
    }
    // Multi-row acceptance needs a margin over row-parallel: the hybrid
    // wrapper's remainder window adds a concat + a second dispatch, and bare
    // multi-row calls (the device op auto-routes them here too — see
    // column_split_config_for) shouldn't flip engines on a marginal modeled
    // win the merge-unit model can't guarantee on silicon. Demand a >= 12.5%
    // modeled win to keep marginal picks from netting negative.
    if (num_rows > 1 && best_cost + std::max(2u, cost_row / 8) > cost_row) {
        return config;
    }

    config.enabled = true;
    config.num_slices = num_slices;
    config.local_grid_x = local_grid_x;
    config.local_grid_y = local_grid_y;
    // Single row keeps the classic one-rectangle program (byte-identical to
    // before multi-rect existed). Multi-row tiles at full capacity: rectangles
    // beyond the runtime row count run zero rows, so the program (and its
    // hash) is row-count-free within this mode — one cached program serves
    // any rows in [1, capacity].
    config.num_rects = (num_rows == 1) ? 1
                                       : (static_cast<uint32_t>(grid.x) / local_grid_x) *
                                             (static_cast<uint32_t>(grid.y) / local_grid_y);
    return config;
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
    return physical_chunks <= 32 ? ComputeBodyMode::FusedEndToEnd : ComputeBodyMode::Classic;
}

ColumnSplitConfig compute_column_split_config(
    uint32_t k,
    uint32_t n,
    uint32_t num_rows,
    const CoreCoord& grid,
    std::optional<uint32_t> num_slices_override,
    bool allow_multi_row) {
    ColumnSplitConfig config = compute_model_column_split_config(k, n, num_rows, grid, allow_multi_row);
    if (!num_slices_override.has_value()) {
        return config;
    }

    // Explicit num_slices selects the tree path directly. Single row: the
    // classic column-parallel tree. Multiple rows: the multi-rectangle
    // variant — one P-core tree per rectangle, rows split contiguously across
    // as many rectangles as tile the grid, all running concurrently. (The
    // built-in cost model can also auto-select the multi-row form when it
    // models a win; the override bypasses the model and pins an exact P —
    // the hybrid wrapper uses it to carry a P modeled on the searched width.)
    TT_FATAL(
        static_cast<uint32_t>(grid.x * grid.y) >= 2,
        "topk_large_indices num_slices={} needs a worker grid of at least 2 cores",
        *num_slices_override);

    const uint32_t requested = *num_slices_override;
    TT_FATAL(
        requested >= 2 && requested <= max_column_slices,
        "topk_large_indices num_slices must be in [2, {}], got {}",
        max_column_slices,
        requested);
    const uint32_t llk_k = to_uint32(snap_to_llk_target_k(k));
    const uint32_t num_chunks = tt::div_up(n, llk_k);
    TT_FATAL(
        requested <= num_chunks,
        "topk_large_indices num_slices={} exceeds the row's chunk count {} (last dim {} / LLK window {}); every "
        "slice must own at least one chunk",
        requested,
        num_chunks,
        n,
        llk_k);

    // Clamp only against the physical grid (rectangle capacity), with a warning.
    // Honor the requested count with an exact a x b rectangle when one fits
    // the grid (e.g. 32 = 8x4 on 13x10); otherwise fall back to the largest
    // achievable product <= requested, with a warning. The old full-rows-only
    // fit silently clamped 16->13, 24->13, 32->26, mislabeling P-sweeps.
    uint32_t num_slices = 0;
    uint32_t local_grid_x = 0;
    uint32_t local_grid_y = 0;
    // Among rectangles of equal core count, prefer the shape that tiles the
    // grid the most times: with rows > 1 the tiling capacity IS the rectangle
    // concurrency (a 2x2 fit tiles a 13x10 grid 30 times; a 1x4 fit only 26,
    // silently doubling some rectangles' row load).
    uint32_t best_capacity = 0;
    for (uint32_t a = 1; a <= static_cast<uint32_t>(grid.x); ++a) {
        for (uint32_t b = 1; b <= static_cast<uint32_t>(grid.y); ++b) {
            const uint32_t p = a * b;
            if (p > requested) {
                continue;
            }
            const uint32_t capacity = (static_cast<uint32_t>(grid.x) / a) * (static_cast<uint32_t>(grid.y) / b);
            if (p > num_slices || (p == num_slices && capacity > best_capacity)) {
                num_slices = p;
                local_grid_x = a;
                local_grid_y = b;
                best_capacity = capacity;
            }
        }
    }
    if (num_slices != requested) {
        log_warning(
            tt::LogOp,
            "topk_large_indices num_slices={} does not fit the {}x{} worker grid's local-core rectangle; "
            "clamped to {} ({}x{} tree cores, in-place root)",
            requested,
            grid.x,
            grid.y,
            num_slices,
            local_grid_x,
            local_grid_y);
    }

    config.enabled = true;
    config.num_slices = num_slices;
    config.local_grid_x = local_grid_x;
    config.local_grid_y = local_grid_y;
    // Single row keeps the classic one-rectangle program; multi-row tiles at
    // full capacity (rectangles beyond the runtime row count run zero rows):
    // row distribution is pure runtime args, so one cached program serves any
    // row count with this layout.
    config.num_rects =
        (num_rows == 1)
            ? 1
            : std::max(
                  1u, (static_cast<uint32_t>(grid.x) / local_grid_x) * (static_cast<uint32_t>(grid.y) / local_grid_y));
    return config;
}

namespace {

struct SliceRuntime {
    uint32_t start_chunk = 0;    // index of the slice's first chunk within the row
    uint32_t start_element = 0;  // start_chunk * llk_k
    uint32_t num_chunks = 0;     // active chunks after the valid_length cut (0 = empty slice)
    uint32_t tail_elements = 0;  // active elements in the slice's last chunk
};

// Splits the VALID chunk range (bounded by the runtime search width) evenly
// over the slices. Balancing on valid rather than physical chunks keeps every
// tree working when a preallocated buffer carries a short valid prefix (the
// DSA indexer grows valid_length inside a fixed 1M buffer): position-based
// physical splitting emptied the trailing slices while the busy ones kept
// near-full per-slice work — measured 1424us vs 720us for a 30-row remainder
// window at buf=1M/valid=512k k=2048. Slices beyond the valid chunk count
// come back empty and are serviced by the writer's -inf fill; everything here
// is runtime-only, so one cached program serves any valid_length with
// freshly balanced slices.
std::vector<SliceRuntime> compute_slice_runtime(
    uint32_t n, uint32_t llk_k, uint32_t num_slices, std::optional<uint32_t> valid_length) {
    const uint32_t search_len = std::min(valid_length.value_or(n), n);
    const uint32_t num_chunks_valid = tt::div_up(search_len, llk_k);
    const uint32_t base_chunks = num_chunks_valid / num_slices;
    const uint32_t extra_chunks = num_chunks_valid % num_slices;

    std::vector<SliceRuntime> slices(num_slices);
    uint32_t start_chunk = 0;
    for (uint32_t s = 0; s < num_slices; ++s) {
        const uint32_t chunk_count = base_chunks + (s < extra_chunks ? 1 : 0);
        const uint32_t start_element = start_chunk * llk_k;
        const uint32_t active_end = std::min((start_chunk + chunk_count) * llk_k, search_len);

        SliceRuntime& slice = slices[s];
        slice.start_chunk = start_chunk;
        slice.start_element = start_element;
        if (chunk_count > 0 && active_end > start_element) {
            const uint32_t active_len = active_end - start_element;
            slice.num_chunks = tt::div_up(active_len, llk_k);
            slice.tail_elements = active_len - ((slice.num_chunks - 1) * llk_k);
        }
        start_chunk += chunk_count;
    }
    TT_FATAL(
        start_chunk == num_chunks_valid,
        "topk_large_indices column split assigned {} chunks, expected {}",
        start_chunk,
        num_chunks_valid);
    return slices;
}

void set_runtime_args_multi_core(
    tt::tt_metal::Program& program,
    const TopkLargeIndicesMultiCoreSharedVariables& shared,
    const Tensor& input,
    const Tensor& indices,
    const operation_attributes_t& attrs) {
    const LlkTargetK llk_target_k = snap_to_llk_target_k(attrs.k);
    const std::optional<uint32_t> valid_length = attrs.valid_length;
    const uint32_t llk_k = to_uint32(llk_target_k);
    const auto& shape = input.logical_shape();
    const uint32_t n = shape[shape.rank() - 1];
    const uint32_t num_rows = flattened_rows_excluding_last_dim(shape);
    const uint32_t input_row_bytes = n * input.element_size();
    const uint32_t num_rects = static_cast<uint32_t>(shared.rect_cores.size());
    TT_FATAL(num_rects >= 1, "topk_large_indices multi-core program has no rectangles");
    const uint32_t num_slices = static_cast<uint32_t>(shared.rect_cores[0].size());
    const uint32_t num_levels = tree_levels(num_slices);
    TT_FATAL(
        num_levels <= 7, "topk_large_indices merge tree supports at most 7 levels (128 slices), got {}", num_levels);

    const auto slices = compute_slice_runtime(n, llk_k, num_slices, valid_length);
    auto* device = input.device();

    // Rows split contiguously across the leading rectangles (runtime-only,
    // like valid_length: a different row count reuses this cached program;
    // rectangles beyond the row count run zero rows and exit immediately).
    // Row window (hybrid wrapper): readers index global input rows from
    // row_base; writers stay output-relative.
    const uint32_t row_base = attrs.row_start.value_or(0);
    const uint32_t effective_rows = attrs.row_count.value_or(num_rows);
    const uint32_t rects_used = std::max(1u, std::min(effective_rows, num_rects));
    const uint32_t base_rows = effective_rows / rects_used;
    const uint32_t extra_rows = effective_rows % rects_used;

    uint32_t rect_start_row = 0;
    for (uint32_t r = 0; r < num_rects; ++r) {
        const auto& rect = shared.rect_cores[r];
        const uint32_t rect_rows = (r < rects_used) ? base_rows + (r < extra_rows ? 1 : 0) : 0;
        const uint32_t start_row = rect_start_row;
        rect_start_row += rect_rows;

        for (uint32_t i = 0; i < num_slices; ++i) {
            const auto& core = rect[i];
            const auto& slice = slices[i];
            const uint32_t input_row = row_base + start_row;
            tt::tt_metal::SetRuntimeArgs(
                program,
                shared.reader_kernel_id,
                core,
                {input.buffer()->address(),
                 input_row,
                 rect_rows,
                 slice.num_chunks,
                 slice.tail_elements * input.element_size(),
                 input_row_bytes,
                 slice.start_element * input.element_size()});

            // Tree schedule for slice i: it wins levels [0, t) where t is the
            // index of i's lowest set bit (root i=0 wins every level), merging
            // partner i + 2^level whenever that slice exists (byes otherwise),
            // then ships its survivor to i with the lowest set bit cleared.
            const bool is_root = (i == 0);
            uint32_t winning_levels = num_levels;
            if (!is_root) {
                winning_levels = 0;
                while (((i >> winning_levels) & 1u) == 0) {
                    ++winning_levels;
                }
            }
            uint32_t num_merges = 0;
            // 7 (x, y) pairs — must match the tree writer's partner_x/y[7] and
            // the positional offsets of the args that follow (do_ship at 16).
            std::vector<uint32_t> partner_coords(14, 0);
            for (uint32_t level = 0; level < winning_levels; ++level) {
                const uint32_t partner = i + (1u << level);
                if (partner < num_slices) {
                    const CoreCoord partner_physical = device->worker_core_from_logical_core(rect[partner]);
                    partner_coords[2 * num_merges] = static_cast<uint32_t>(partner_physical.x);
                    partner_coords[2 * num_merges + 1] = static_cast<uint32_t>(partner_physical.y);
                    ++num_merges;
                }
            }

            tt::tt_metal::SetRuntimeArgs(
                program,
                is_root ? shared.compute_root_kernel_id : shared.compute_node_kernel_id,
                core,
                {rect_rows, slice.num_chunks, slice.tail_elements, slice.start_chunk, num_merges});

            uint32_t winner_x = 0;
            uint32_t winner_y = 0;
            if (!is_root) {
                const uint32_t winner = i & (i - 1);  // clear the lowest set bit
                const CoreCoord winner_physical = device->worker_core_from_logical_core(rect[winner]);
                winner_x = static_cast<uint32_t>(winner_physical.x);
                winner_y = static_cast<uint32_t>(winner_physical.y);
            }
            // A shipping core with an empty slice AND no adopted partners has no
            // survivor in DST; its writer sends the prefilled -inf sequence.
            const uint32_t is_empty_ship = (!is_root && slice.num_chunks == 0 && num_merges == 0) ? 1u : 0u;

            std::vector<uint32_t> writer_args;
            writer_args.reserve(22);
            writer_args.push_back(rect_rows);
            writer_args.push_back(num_merges);
            for (uint32_t v : partner_coords) {
                writer_args.push_back(v);
            }
            writer_args.push_back(is_root ? 0u : 1u);  // do_ship
            writer_args.push_back(winner_x);
            writer_args.push_back(winner_y);
            writer_args.push_back(is_empty_ship);
            writer_args.push_back(indices.buffer()->address());
            // The tree writer reads start_row at arg 21, directly after the indices address.
            writer_args.push_back(start_row);
            tt::tt_metal::SetRuntimeArgs(program, shared.writer_kernel_id, core, writer_args);
        }
    }
    TT_FATAL(
        rect_start_row == effective_rows,
        "topk_large_indices multi-rect assigned {} rows, expected {}",
        rect_start_row,
        effective_rows);
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
    const uint32_t num_rows = operation_attributes.row_count.value_or(flattened_rows_excluding_last_dim(shape));
    const auto grid = input.device()->compute_with_storage_grid_size();
    const auto config =
        compute_column_split_config(k, n, num_rows, grid, operation_attributes.num_slices, /*allow_multi_row=*/true);
    TT_FATAL(config.enabled, "topk_large_indices multi-core factory selected for a shape it does not support");
    const uint32_t num_slices = config.num_slices;

    // The merge tree lives IN PLACE on each slice rectangle: slice i is
    // rectangle-local core (i % sx, i / sx) in row-major order; slice 0 (the
    // rectangle origin) is that tree's root and produces its rows' output.
    // Winners keep their survivor in DST across levels — only the shipped
    // operand crosses the NoC, once, per losing core. Multi-rectangle
    // (num_rects > 1): disjoint rectangles tile the grid and run their trees
    // concurrently on contiguous row ranges.
    const uint32_t num_rects = config.num_rects;
    TT_FATAL(num_rects >= 1, "topk_large_indices multi-core factory needs at least one rectangle");
    const uint32_t rects_x = static_cast<uint32_t>(grid.x) / config.local_grid_x;
    std::vector<CoreRange> all_ranges;
    std::vector<CoreRange> root_ranges;
    std::vector<CoreRange> node_ranges;
    std::vector<std::vector<CoreCoord>> rect_cores(num_rects);
    for (uint32_t r = 0; r < num_rects; ++r) {
        const uint32_t ox = (r % rects_x) * config.local_grid_x;
        const uint32_t oy = (r / rects_x) * config.local_grid_y;
        const CoreRange rect(CoreCoord(ox, oy), CoreCoord(ox + config.local_grid_x - 1, oy + config.local_grid_y - 1));
        all_ranges.push_back(rect);
        rect_cores[r] = corerange_to_cores(CoreRangeSet(rect), std::nullopt, true);
        TT_FATAL(
            rect_cores[r].size() == num_slices,
            "topk_large_indices column split expected {} tree cores per rectangle, got {}",
            num_slices,
            rect_cores[r].size());
        root_ranges.emplace_back(rect_cores[r].front(), rect_cores[r].front());
        // All rectangle cores except the root run the node compute kernel.
        if (config.local_grid_x > 1) {
            node_ranges.emplace_back(CoreCoord(ox + 1, oy), CoreCoord(ox + config.local_grid_x - 1, oy));
        }
        if (config.local_grid_y > 1) {
            node_ranges.emplace_back(
                CoreCoord(ox, oy + 1), CoreCoord(ox + config.local_grid_x - 1, oy + config.local_grid_y - 1));
        }
    }
    TT_FATAL(!node_ranges.empty(), "topk_large_indices merge tree needs at least 2 slices");
    const CoreRangeSet all_cores(all_ranges);
    const CoreRangeSet root_core_set(root_ranges);
    const CoreRangeSet node_cores_set(node_ranges);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_indices_out = tt::CBIndex::c_1;
    constexpr uint32_t cb_indices_scratch = tt::CBIndex::c_2;
    constexpr uint32_t cb_recv = tt::CBIndex::c_3;
    constexpr uint32_t cb_ship_values = tt::CBIndex::c_5;
    constexpr uint32_t cb_ship_indices = tt::CBIndex::c_6;
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
    // The indices CB carries the raw 32-bit index words the packer produces.
    const uint32_t indices_cb_row_bytes = llk_k * static_cast<uint32_t>(sizeof(uint32_t));
    constexpr uint32_t cb_depth = 2;

    // The recv CB spans the whole rectangle so every core sees the same L1
    // address (a shipping core derives its winner's destination from its own
    // copy). Sized to exactly one sequence: capacity IS the level-to-level
    // backpressure. Created FIRST so the multi-range address has no gaps.
    const auto recv_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * sequence_bytes, {{cb_recv, tt::DataFormat::UInt32}})
            .set_page_size(cb_recv, tile32_bytes);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, recv_cb_config);

    create_input_cb(program, all_cores, cb_depth, tiles_per_sequence, input_tile_bytes, cb_in);

    const auto ship_values_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_depth * tiles_per_sequence * tile32_bytes, {{cb_ship_values, tt::DataFormat::UInt32}})
            .set_page_size(cb_ship_values, tile32_bytes);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, ship_values_cb_config);

    const auto ship_indices_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_depth * tiles_per_sequence * tile32_bytes, {{cb_ship_indices, tt::DataFormat::UInt32}})
            .set_page_size(cb_ship_indices, tile32_bytes);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, ship_indices_cb_config);

    // Writer-owned scratch for the empty-slice -inf sequence (values + indices).
    const auto neginf_scratch_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * sequence_bytes, {{cb_neginf_scratch, tt::DataFormat::UInt32}})
            .set_page_size(cb_neginf_scratch, tile32_bytes);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, neginf_scratch_cb_config);

    // Root-only output CBs: identical shape to the row-parallel factory's.
    create_indices_output_cbs(
        program,
        root_core_set,
        llk_target_k,
        cb_depth,
        indices_cb_row_bytes,
        indices_row_bytes,
        cb_indices_out,
        cb_indices_scratch);

    // Pairwise tree flow control (see the TOPK_TREE writer role in kernels/writer.cpp):
    // ready = "my winner's recv slot is free", data = "my current partner's sequence landed".
    const uint32_t ready_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    const uint32_t data_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);

    // Role selection for the unified kernel sources: TOPK_TREE marks a tree
    // member (leaf slice reduction + in-DST pairwise merges); TOPK_TREE_ROOT
    // additionally selects the root's materializing epilogue.
    const std::map<std::string, std::string> tree_defines = {{"TOPK_TREE", "1"}};
    const std::map<std::string, std::string> tree_root_defines = {{"TOPK_TREE", "1"}, {"TOPK_TREE_ROOT", "1"}};

    // Leaf reader: the row-parallel reader plus a per-core slice offset
    // (TOPK_TREE enables the extra slice_offset_bytes runtime arg).
    auto reader_kernel = create_reader_kernel(
        program, all_cores, input, cb_in, input_chunk_bytes, input_tile_bytes, tiles_per_sequence, tree_defines);

    const std::vector<uint32_t> compute_node_compile_args = {cb_in, cb_ship_values, cb_ship_indices, cb_recv, llk_k};
    auto compute_node_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/compute.cpp",
        node_cores_set,
        tt::tt_metal::ComputeConfig{// Same DST configuration as the row-parallel compute role: the
                                    // unfused K=2048 merge occupies DEST slots 0..7 (FP32, full sync).
                                    .fp32_dest_acc_en = true,
                                    .dst_full_sync_en = true,
                                    .compile_args = compute_node_compile_args,
                                    .defines = tree_defines});

    const std::vector<uint32_t> compute_root_compile_args = {cb_in, cb_indices_out, cb_recv, llk_k};
    auto compute_root_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/compute.cpp",
        root_core_set,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .compile_args = compute_root_compile_args,
            .defines = tree_root_defines});

    std::vector<uint32_t> writer_compile_args = {
        cb_ship_values,
        cb_ship_indices,
        cb_neginf_scratch,
        cb_recv,
        ready_sem_id,
        data_sem_id,
        tiles_per_sequence,
        tile32_bytes,
        cb_indices_out,
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
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args, tree_defines));

    TopkLargeIndicesMultiCoreSharedVariables shared{
        .reader_kernel_id = reader_kernel,
        .compute_node_kernel_id = compute_node_kernel,
        .compute_root_kernel_id = compute_root_kernel,
        .writer_kernel_id = writer_kernel,
        .rect_cores = std::move(rect_cores)};
    set_runtime_args_multi_core(program, shared, input, tensor_return_value, operation_attributes);

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
        operation_attributes);
}

}  // namespace ttnn::operations::experimental::topk_large_indices::program
