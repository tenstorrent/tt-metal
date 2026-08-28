// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
#include <string>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/transformer/sdpa/device/kernels/neighborhood_kernel_args.hpp"
#include "ttnn/operations/transformer/sdpa/device/neighborhood_sdpa_device_operation.hpp"

namespace ttnn::prim {

namespace neighborhood = ttnn::transformer::neighborhood;
namespace kernel_args = ttnn::transformer::neighborhood::kernel_args;

namespace {

uint32_t ceil_div(uint32_t numerator, uint32_t denominator) { return (numerator + denominator - 1) / denominator; }

// The widest subblock that divides `width_tiles` and still fits in DST. Must divide exactly,
// so every subblock is the same shape.
uint32_t widest_subblock(uint32_t width_tiles, uint32_t dst_capacity_tiles) {
    for (uint32_t candidate = std::min(width_tiles, dst_capacity_tiles); candidate > 1; --candidate) {
        if (width_tiles % candidate == 0) {
            return candidate;
        }
    }
    return 1;
}

}  // namespace

tt::tt_metal::ProgramDescriptor NeighborhoodSDPAOperation::NeighborhoodSDPAProgramFactory::create_descriptor(
    const NeighborhoodSDPAParams& attributes, const NeighborhoodSDPAInputs& tensors, Tensor& output) {
    tt::tt_metal::ProgramDescriptor descriptor;

    const neighborhood::NeighborhoodPlan plan = neighborhood::build_plan(attributes.config);
    const auto& config = attributes.config;

    const auto query_shape = tensors.query_tensor.logical_shape();
    // Site-major: [batch, 1, brick_count * 32 sites, head_count * head_dim]. See chunk_layout.
    const uint32_t batch_count = query_shape[0];
    const uint32_t head_count = attributes.head_count;
    const uint32_t head_dim_tiles = query_shape[3] / head_count / tt::constants::TILE_WIDTH;
    const uint32_t brick_count = plan.brick_count;
    const uint32_t query_tile_rows = config.bricks_per_query_chunk();

    const uint32_t tiles_per_kv_chunk = attributes.tiles_per_kv_chunk;
    const uint32_t kv_chunk_count = ceil_div(plan.gather_brick_count, tiles_per_kv_chunk);

    // A chunk WIDER than the stride means its bricks do not share a context window, so the mask
    // cannot be one tile per slot broadcast down the query rows -- each brick needs its own. That
    // is only reachable via DIFFVAE_NA_UNSAFE_CHUNK today, so this follows the same switch rather
    // than costing anything on the shipped path.
    const bool chunk_exceeds_stride = query_tile_rows > 1 && !(config.query_chunk_sites() == config.stride);
    const char* per_brick_env = std::getenv("DIFFVAE_NA_PER_BRICK_MASK");
    const bool per_brick_mask = per_brick_env != nullptr ? per_brick_env[0] == '1' : chunk_exceeds_stride;
    const uint32_t mask_tiles_per_kv_chunk = per_brick_mask ? query_tile_rows * tiles_per_kv_chunk : tiles_per_kv_chunk;

    // The relative mask table makes every unclamped query brick want the SAME tiles in the same
    // order, so sizing cb_mask to a whole work item (rather than double-buffering one kv chunk)
    // makes its pages cycle back to the same addresses every item -- and the reader can then skip
    // writing them entirely for a run of unclamped bricks -- worth 15.2 s against 15.6 s at 145
    // frames. (Less than the traffic it removes would suggest: what the op is actually bound by is
    // the number of score tiles the compute kernel walks. See FINDINGS section 10.)
    const bool relative_mask_table = config.stride.time() == 1 && config.stride.height() == 1 &&
                                     config.stride.width() == 1 && tensors.interior_mask.has_value();
    // Must match `interior_table_supported` in neighborhood_reader.cpp exactly: the reader skips
    // rewriting the mask on the strength of the pages cycling, which only holds at this size.
    const bool persistent_mask = relative_mask_table && !per_brick_mask;
    const uint32_t mask_cb_pages =
        persistent_mask ? mask_tiles_per_kv_chunk * kv_chunk_count : mask_tiles_per_kv_chunk * 2;

    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(tensors.query_tensor.dtype());
    const uint32_t tile_bytes = tt::tile_size(data_format);
    constexpr tt::DataFormat bfloat16_format = tt::DataFormat::Float16_b;
    const uint32_t bfloat16_tile_bytes = tt::tile_size(bfloat16_format);

    auto* device = tensors.query_tensor.device();
    const tt::tt_metal::CoreCoord worker_grid = device->compute_with_storage_grid_size();
    const auto worker_core_range =
        tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {worker_grid.x - 1, worker_grid.y - 1}));
    const uint32_t worker_core_count = worker_grid.x * worker_grid.y;

    // One work item is one (batch, head, query brick). Bricks vary fastest inside a work item
    // index, so a core's items are spatially adjacent bricks -- which is what will later let
    // their overlapping context windows share resident K/V.
    const uint32_t work_item_count = batch_count * head_count * plan.chunk_count;
    const uint32_t work_items_per_core = work_item_count / worker_core_count;
    const uint32_t cores_with_one_extra = work_item_count % worker_core_count;

    // ---- circular buffers ----
    const auto add_circular_buffer = [&](uint32_t buffer_index,
                                         uint32_t page_bytes,
                                         uint32_t page_count,
                                         tt::DataFormat format) {
        descriptor.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = page_bytes * page_count,
            .core_ranges = worker_core_range,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(buffer_index), .data_format = format, .page_size = page_bytes}}},
        });
    };

    constexpr uint32_t DOUBLE_BUFFERED = 2;
    add_circular_buffer(
        kernel_args::cb_query, tile_bytes, head_dim_tiles * query_tile_rows * DOUBLE_BUFFERED, data_format);
    add_circular_buffer(
        kernel_args::cb_key, tile_bytes, tiles_per_kv_chunk * head_dim_tiles * DOUBLE_BUFFERED, data_format);
    add_circular_buffer(
        kernel_args::cb_value, tile_bytes, tiles_per_kv_chunk * head_dim_tiles * DOUBLE_BUFFERED, data_format);
    // The mask is always bfloat16: the reader writes {0, -inf} bit patterns into it directly.
    // One tile per gather slot, NOT per query row: with one query group per chunk every row
    // shares the window, so the mask broadcasts down the chunk exactly as the reference's does.
    add_circular_buffer(kernel_args::cb_mask, bfloat16_tile_bytes, mask_cb_pages, bfloat16_format);
    add_circular_buffer(kernel_args::cb_reduce_scalar, bfloat16_tile_bytes, 1, bfloat16_format);
    add_circular_buffer(kernel_args::cb_zero, bfloat16_tile_bytes, 1, bfloat16_format);
    add_circular_buffer(kernel_args::cb_column_identity, bfloat16_tile_bytes, 1, bfloat16_format);
    add_circular_buffer(
        kernel_args::cb_scores, bfloat16_tile_bytes, tiles_per_kv_chunk * query_tile_rows, bfloat16_format);

    // Running statistics: one tile row of queries, so one tile each.
    for (uint32_t statistic_buffer :
         {kernel_args::cb_row_max_current,
          kernel_args::cb_row_max_previous,
          kernel_args::cb_row_sum_current,
          kernel_args::cb_row_sum_previous,
          kernel_args::cb_exp_max_difference}) {
        add_circular_buffer(statistic_buffer, bfloat16_tile_bytes, query_tile_rows * DOUBLE_BUFFERED, bfloat16_format);
    }
    // Single-buffered on purpose: these accumulate in place across KV chunks.
    add_circular_buffer(
        kernel_args::cb_output_accumulator_current,
        bfloat16_tile_bytes,
        head_dim_tiles * query_tile_rows,
        bfloat16_format);
    add_circular_buffer(
        kernel_args::cb_output_accumulator_previous,
        bfloat16_tile_bytes,
        head_dim_tiles * query_tile_rows,
        bfloat16_format);
    add_circular_buffer(
        kernel_args::cb_output, tile_bytes, head_dim_tiles * query_tile_rows * DOUBLE_BUFFERED, data_format);
    add_circular_buffer(
        kernel_args::cb_gather_origin, kernel_args::GATHER_ORIGIN_ROW_BYTES, DOUBLE_BUFFERED, tt::DataFormat::UInt32);
    // Writer-private copy of the same origin row. Reader and writer run in parallel, so they
    // cannot share cb_gather_origin; the writer needs the host-stamped interior bit to skip
    // bricks this launch does not own.
    add_circular_buffer(
        kernel_args::cb_writer_origin, kernel_args::GATHER_ORIGIN_ROW_BYTES, DOUBLE_BUFFERED, tt::DataFormat::UInt32);
    // Holds ONE regime's whole mask set. Every chunk in a regime wants the same patterns, so
    // fetching them per chunk re-reads the same tiles from DRAM thousands of times; this keeps
    // them on the core and turns the per-chunk cost into a local copy.
    // Per-brick masks never read this set -- it holds ONE window's patterns, keyed on the chunk's
    // regime, which is the wrong window for every brick but the first. Allocating it anyway costs
    // gather_brick_count tiles of L1 (590 KB at a 288-brick gather) next to a mask CB that per-brick
    // mode has already made query_tile_rows times bigger, and the reader would copy all of it in
    // per chunk for nothing.
    const bool uses_resident_mask = tensors.interior_mask.has_value() && !per_brick_mask && !relative_mask_table;
    if (uses_resident_mask) {
        add_circular_buffer(
            kernel_args::cb_resident_mask, bfloat16_tile_bytes, plan.gather_brick_count, bfloat16_format);
    }

    // ---- compile-time arguments, written by name ----
    std::vector<uint32_t> reader_compile_args(kernel_args::reader_arg::COUNT);
    reader_compile_args[kernel_args::reader_arg::head_count] = head_count;
    reader_compile_args[kernel_args::reader_arg::brick_count] = brick_count;
    reader_compile_args[kernel_args::reader_arg::head_dim_tiles] = head_dim_tiles;
    reader_compile_args[kernel_args::reader_arg::query_chunk_bricks_time] = config.query_chunk_bricks.time();
    reader_compile_args[kernel_args::reader_arg::query_chunk_bricks_height] = config.query_chunk_bricks.height();
    reader_compile_args[kernel_args::reader_arg::query_chunk_bricks_width] = config.query_chunk_bricks.width();
    reader_compile_args[kernel_args::reader_arg::bricks_per_query_chunk] = query_tile_rows;
    reader_compile_args[kernel_args::reader_arg::volume_chunks_time] = plan.volume_chunks.time();
    reader_compile_args[kernel_args::reader_arg::volume_chunks_height] = plan.volume_chunks.height();
    reader_compile_args[kernel_args::reader_arg::volume_chunks_width] = plan.volume_chunks.width();
    reader_compile_args[kernel_args::reader_arg::tiles_per_kv_chunk] = tiles_per_kv_chunk;
    reader_compile_args[kernel_args::reader_arg::per_brick_mask] = per_brick_mask ? 1u : 0u;
    reader_compile_args[kernel_args::reader_arg::mask_memset_only] = attributes.probe == 2 ? 1u : 0u;
    // 1 = skip K/V DMA (probe 1). 2 = skip the whole gather-slot loop (probes 7/8).
    reader_compile_args[kernel_args::reader_arg::skip_kv] = attributes.probe == 1                              ? 1u
                                                            : (attributes.probe == 7 || attributes.probe == 8) ? 2u
                                                                                                               : 0u;
    reader_compile_args[kernel_args::reader_arg::kv_chunk_count] = kv_chunk_count;
    reader_compile_args[kernel_args::reader_arg::gather_brick_count] = plan.gather_brick_count;
    reader_compile_args[kernel_args::reader_arg::volume_bricks_time] = plan.volume_bricks.time();
    reader_compile_args[kernel_args::reader_arg::volume_bricks_height] = plan.volume_bricks.height();
    reader_compile_args[kernel_args::reader_arg::volume_bricks_width] = plan.volume_bricks.width();
    reader_compile_args[kernel_args::reader_arg::gather_bricks_time] = plan.gather_bricks.time();
    reader_compile_args[kernel_args::reader_arg::gather_bricks_height] = plan.gather_bricks.height();
    reader_compile_args[kernel_args::reader_arg::gather_bricks_width] = plan.gather_bricks.width();
    reader_compile_args[kernel_args::reader_arg::query_bricks_time] = plan.query_bricks.time();
    reader_compile_args[kernel_args::reader_arg::query_bricks_height] = plan.query_bricks.height();
    reader_compile_args[kernel_args::reader_arg::query_bricks_width] = plan.query_bricks.width();
    reader_compile_args[kernel_args::reader_arg::query_brick_count] = plan.query_brick_count;
    reader_compile_args[kernel_args::reader_arg::query_origin_bricks_time] = plan.query_origin_bricks.time();
    reader_compile_args[kernel_args::reader_arg::query_origin_bricks_height] = plan.query_origin_bricks.height();
    reader_compile_args[kernel_args::reader_arg::query_origin_bricks_width] = plan.query_origin_bricks.width();
    reader_compile_args[kernel_args::reader_arg::brick_sites_time] = config.brick.time();
    reader_compile_args[kernel_args::reader_arg::brick_sites_height] = config.brick.height();
    reader_compile_args[kernel_args::reader_arg::brick_sites_width] = config.brick.width();
    reader_compile_args[kernel_args::reader_arg::context_window_time] = config.context_window.time();
    reader_compile_args[kernel_args::reader_arg::context_window_height] = config.context_window.height();
    reader_compile_args[kernel_args::reader_arg::context_window_width] = config.context_window.width();
    reader_compile_args[kernel_args::reader_arg::stride_time] = config.stride.time();
    reader_compile_args[kernel_args::reader_arg::stride_height] = config.stride.height();
    reader_compile_args[kernel_args::reader_arg::stride_width] = config.stride.width();
    reader_compile_args[kernel_args::reader_arg::volume_time] = config.volume.time();
    reader_compile_args[kernel_args::reader_arg::volume_height] = config.volume.height();
    reader_compile_args[kernel_args::reader_arg::volume_width] = config.volume.width();
    const neighborhood::Extent3 resident = config.resident_extent();
    reader_compile_args[kernel_args::reader_arg::resident_time] = resident.time();
    reader_compile_args[kernel_args::reader_arg::resident_height] = resident.height();
    reader_compile_args[kernel_args::reader_arg::resident_width] = resident.width();
    // A stride-1 table is relative; a GNA one is per-regime. The kernel indexes them differently.
    const bool relative_mask = config.stride.time() == 1 && config.stride.height() == 1 && config.stride.width() == 1;
    reader_compile_args[kernel_args::reader_arg::relative_mask] = relative_mask ? 1u : 0u;
    const char* always_env = std::getenv("DIFFVAE_NA_TABLE_ALWAYS");
    reader_compile_args[kernel_args::reader_arg::table_always] =
        (always_env != nullptr && always_env[0] == '1') ? 1u : 0u;
    // The relative table is read straight from DRAM per slot, so it needs no L1 staging -- and
    // staging is what forced the per-slot fill to be a word loop in the first place.
    reader_compile_args[kernel_args::reader_arg::has_interior_mask] =
        (tensors.interior_mask.has_value() && (relative_mask || uses_resident_mask)) ? 1u : 0u;
    reader_compile_args[kernel_args::reader_arg::path_mode] = attributes.path_mode;
    // Same compare as skip_kv (if == 1): path_mode equality was DCE'd and skip never ran.
    const bool split_path = attributes.path_mode == 1 || attributes.path_mode == 2;
    reader_compile_args[kernel_args::reader_arg::skip_unowned] = split_path ? 1u : 0u;
    // 2/3, not 0/1: na_should_skip is `(2 + bit) == skip_if`. 0/1 never matches, so both
    // programs walked. Also changes the compile-arg hash so JIT cannot reuse a stale ELF.
    reader_compile_args[kernel_args::reader_arg::skip_if_bit] = attributes.path_mode == 2   ? 3u
                                                                : attributes.path_mode == 1 ? 2u
                                                                                            : 0u;

    // Accessor args come after the named block, in the order the reader constructs them.
    tt::tt_metal::TensorAccessorArgs(tensors.query_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensors.key_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensors.value_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensors.gather_origin_table.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensors.interior_mask.has_value() ? tensors.interior_mask->buffer() : nullptr)
        .append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args(kernel_args::writer_arg::COUNT);
    writer_compile_args[kernel_args::writer_arg::head_count] = head_count;
    writer_compile_args[kernel_args::writer_arg::brick_count] = plan.query_brick_count;
    writer_compile_args[kernel_args::writer_arg::head_dim_tiles] = head_dim_tiles;
    writer_compile_args[kernel_args::writer_arg::query_chunk_bricks_time] = config.query_chunk_bricks.time();
    writer_compile_args[kernel_args::writer_arg::query_chunk_bricks_height] = config.query_chunk_bricks.height();
    writer_compile_args[kernel_args::writer_arg::query_chunk_bricks_width] = config.query_chunk_bricks.width();
    writer_compile_args[kernel_args::writer_arg::bricks_per_query_chunk] = query_tile_rows;
    writer_compile_args[kernel_args::writer_arg::volume_chunks_time] = plan.volume_chunks.time();
    writer_compile_args[kernel_args::writer_arg::volume_chunks_height] = plan.volume_chunks.height();
    writer_compile_args[kernel_args::writer_arg::volume_chunks_width] = plan.volume_chunks.width();
    // The output is query-sized, so the writer's brick grid is the QUERY grid.
    writer_compile_args[kernel_args::writer_arg::volume_bricks_time] = plan.query_bricks.time();
    writer_compile_args[kernel_args::writer_arg::volume_bricks_height] = plan.query_bricks.height();
    writer_compile_args[kernel_args::writer_arg::volume_bricks_width] = plan.query_bricks.width();
    writer_compile_args[kernel_args::writer_arg::path_mode] = attributes.path_mode;
    writer_compile_args[kernel_args::writer_arg::skip_unowned] = split_path ? 1u : 0u;
    writer_compile_args[kernel_args::writer_arg::skip_if_bit] = attributes.path_mode == 2   ? 3u
                                                                : attributes.path_mode == 1 ? 2u
                                                                                            : 0u;
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensors.gather_origin_table.buffer()).append_to(writer_compile_args);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_enabled, packer_l1_accumulate, dst_full_sync_enabled] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), attributes.compute_kernel_config);
    (void)packer_l1_accumulate;

    std::vector<uint32_t> compute_compile_args(kernel_args::compute_arg::COUNT);
    compute_compile_args[kernel_args::compute_arg::head_dim_tiles] = head_dim_tiles;
    compute_compile_args[kernel_args::compute_arg::query_tile_rows] = query_tile_rows;
    compute_compile_args[kernel_args::compute_arg::tiles_per_kv_chunk] = tiles_per_kv_chunk;
    compute_compile_args[kernel_args::compute_arg::kv_chunk_count] = kv_chunk_count;
    compute_compile_args[kernel_args::compute_arg::work_item_count] = 0;  // per-core, set as a runtime arg
    compute_compile_args[kernel_args::compute_arg::scale_as_float_bits] = std::bit_cast<uint32_t>(attributes.scale);

    // One query tile row means subblock_h is always 1, so DST capacity bounds the WIDTH alone.
    // subblock_h stays 1: matmul_blocks re-reads the mask from the CB front for every in0
    // subblock, which is only correct when each subblock is one query row sharing the window.
    const uint32_t dst_capacity_tiles = fp32_dest_acc_enabled ? 4u : 8u;
    const uint32_t scores_subblock_width = widest_subblock(tiles_per_kv_chunk, dst_capacity_tiles);
    const uint32_t output_subblock_width = widest_subblock(head_dim_tiles, dst_capacity_tiles);
    compute_compile_args[kernel_args::compute_arg::scores_subblock_width] = scores_subblock_width;
    compute_compile_args[kernel_args::compute_arg::scores_subblock_count] = tiles_per_kv_chunk / scores_subblock_width;
    compute_compile_args[kernel_args::compute_arg::output_subblock_width] = output_subblock_width;
    compute_compile_args[kernel_args::compute_arg::output_subblock_count] = head_dim_tiles / output_subblock_width;
    compute_compile_args[kernel_args::compute_arg::mask_subblock_stride] = per_brick_mask ? tiles_per_kv_chunk : 0u;

    // ---- kernels ----
    const std::string kernel_directory = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";

    // Separate translation units for interior vs edge skip. Factory KernelDescriptor.defines
    // and compile-arg compares did not produce a new JIT kernel (1484/1484 hits).
    std::string reader_source = kernel_directory + "dataflow/neighborhood_reader.cpp";
    std::string writer_source = kernel_directory + "dataflow/neighborhood_writer.cpp";
    if (attributes.path_mode == 1) {
        reader_source = kernel_directory + "dataflow/neighborhood_reader_interior.cpp";
        writer_source = kernel_directory + "dataflow/neighborhood_writer_interior.cpp";
    } else if (attributes.path_mode == 2) {
        reader_source = kernel_directory + "dataflow/neighborhood_reader_edge.cpp";
        writer_source = kernel_directory + "dataflow/neighborhood_writer_edge.cpp";
    }

    tt::tt_metal::KernelDescriptor reader_descriptor;
    reader_descriptor.kernel_source = reader_source;
    reader_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_descriptor.core_ranges = worker_core_range;
    reader_descriptor.compile_time_args = reader_compile_args;
    reader_descriptor.defines = {
        {"NA_PATH_KIND", std::to_string(attributes.path_mode)},
        {"NA_SKIP_IF",
         attributes.path_mode == 2   ? "3"
         : attributes.path_mode == 1 ? "2"
                                     : "0"},
    };
    reader_descriptor.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_descriptor;
    writer_descriptor.kernel_source = writer_source;
    writer_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_descriptor.core_ranges = worker_core_range;
    writer_descriptor.compile_time_args = writer_compile_args;
    writer_descriptor.defines = {
        {"NA_PATH_KIND", std::to_string(attributes.path_mode)},
        {"NA_SKIP_IF",
         attributes.path_mode == 2   ? "3"
         : attributes.path_mode == 1 ? "2"
                                     : "0"},
    };
    writer_descriptor.config = tt::tt_metal::WriterConfigDescriptor{};

    tt::tt_metal::KernelDescriptor compute_descriptor;
    compute_descriptor.kernel_source = kernel_directory + "compute/neighborhood_sdpa.cpp";
    compute_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_descriptor.core_ranges = worker_core_range;
    compute_descriptor.compile_time_args = compute_compile_args;
    // compute_common.hpp reads these as macros rather than template parameters, so the
    // including kernel has to be told about them here.
    // compute_common.hpp defines REDUCE_OP and REDUCE_DIM itself, but reads EXP_APPROX_MODE as
    // a macro the including build has to supply.
    // Probes 1/2/7 only change the reader; TRISC stays the shipped flash. Probe 8 is skip_slots
    // plus drain, so compute can show up once the slot walk is gone.
    uint32_t compute_probe = attributes.probe;
    if (attributes.probe == 1 || attributes.probe == 2 || attributes.probe == 7) {
        compute_probe = 0u;
    } else if (attributes.probe == 8) {
        compute_probe = 3u;
    }
    compute_descriptor.defines = {
        {"EXP_APPROX_MODE", std::to_string(static_cast<int>(math_approx_mode))},
        {"NEIGHBORHOOD_SDPA_PROBE", std::to_string(compute_probe)}};
    compute_descriptor.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_enabled,
        .dst_full_sync_en = dst_full_sync_enabled,
        .math_approx_mode = math_approx_mode};

    auto* query_buffer = tensors.query_tensor.buffer();
    auto* key_buffer = tensors.key_tensor.buffer();
    auto* value_buffer = tensors.value_tensor.buffer();
    auto* gather_origin_buffer = tensors.gather_origin_table.buffer();
    auto* interior_mask_buffer = tensors.interior_mask.has_value() ? tensors.interior_mask->buffer() : nullptr;
    auto* output_buffer = output.buffer();

    for (uint32_t worker_index = 0; worker_index < worker_core_count; ++worker_index) {
        const tt::tt_metal::CoreCoord worker_core = {worker_index % worker_grid.x, worker_index / worker_grid.x};

        const uint32_t work_item_start =
            worker_index * work_items_per_core + std::min(worker_index, cores_with_one_extra);
        const uint32_t core_work_item_count = work_items_per_core + (worker_index < cores_with_one_extra ? 1u : 0u);

        const uint32_t skip_if_packed = reader_compile_args[kernel_args::reader_arg::skip_if_bit];
        const uint32_t writer_skip_if = writer_compile_args[kernel_args::writer_arg::skip_if_bit];
        reader_descriptor.emplace_runtime_args(
            worker_core,
            {query_buffer,
             key_buffer,
             value_buffer,
             gather_origin_buffer,
             interior_mask_buffer,
             work_item_start,
             core_work_item_count,
             tile_bytes | (skip_if_packed << 16)});
        writer_descriptor.emplace_runtime_args(
            worker_core,
            {output_buffer,
             gather_origin_buffer,
             work_item_start,
             core_work_item_count,
             tile_bytes | (writer_skip_if << 16)});
        compute_descriptor.emplace_runtime_args(worker_core, {core_work_item_count});
    }

    descriptor.kernels.push_back(std::move(reader_descriptor));
    descriptor.kernels.push_back(std::move(writer_descriptor));
    descriptor.kernels.push_back(std::move(compute_descriptor));
    return descriptor;
}

}  // namespace ttnn::prim
