// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "hostdevcommon/kernel_structs.h"

#include "fused_msda_device_operation.hpp"

namespace ttnn::operations::experimental::fused_msda {

using namespace tt::tt_metal;

namespace {

constexpr std::string_view kKernelDir = "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/";

uint32_t aligned_page_size(uint32_t raw_bytes, BufferType buffer_type) {
    const uint32_t alignment = buffer_type == BufferType::DRAM ? tt::tt_metal::hal::get_dram_alignment()
                                                               : tt::tt_metal::hal::get_l1_alignment();
    return tt::round_up(raw_bytes, alignment);
}

// One unit of work: up to 32 consecutive queries of a single (batch, head).
// Queries and heads are independent and the whole 4 * L * P reduction for a
// unit stays on the core that owns it, so there is no cross-core reduction.
struct TileAssignment {
    uint32_t batch;
    uint32_t head;
    uint32_t q_start;
    uint32_t v_rows;
};

constexpr uint32_t TILE_MAX_ROWS = 32;

// Work split shared by create_descriptor (cache miss) and override_runtime_arguments
// (cache hit). The core set and the tile slots are sized for every query, so they
// do not depend on num_valid_queries, which is excluded from the program hash. Only
// which tiles fill the slots, and how many, changes between calls.
//
// Every core gets the same number of slots: a cached program reports each core's
// runtime-arg count as the kernel's longest, so per-core lengths could not be
// matched on a hit.
struct MSDAWorkSplit {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    uint32_t slots = 0;                 // tile slots per core
    std::vector<uint32_t> counts;       // tiles this call runs per core
    std::vector<TileAssignment> tiles;  // tiles this call runs, in core order
};

MSDAWorkSplit msda_work_split(
    const FusedMSDAOperation::operation_attributes_t& attrs, const MSDAShapes& s, const CoreCoord& grid) {
    const uint32_t q_blocks = (s.num_queries + TILE_MAX_ROWS - 1) / TILE_MAX_ROWS;
    const uint32_t total_slots = s.batch * s.num_heads * q_blocks;

    MSDAWorkSplit split;
    auto [num_cores, all_cores, core_group_1, core_group_2, slots_group_1, slots_group_2] =
        tt::tt_metal::split_work_to_cores(grid, total_slots);
    split.all_cores = all_cores;
    split.cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    // split_work_to_cores puts the larger per-core count in group 1; the groups
    // themselves are not used, since every core gets that many slots.
    split.slots = slots_group_1;

    for (uint32_t b = 0; b < s.batch; ++b) {
        // Rows past the batch's valid count are neither computed nor written.
        const uint32_t q_end = attrs.num_valid_queries.empty() ? s.num_queries : attrs.num_valid_queries[b];
        for (uint32_t h = 0; h < s.num_heads; ++h) {
            for (uint32_t q_start = 0; q_start < q_end; q_start += TILE_MAX_ROWS) {
                split.tiles.push_back({b, h, q_start, std::min(TILE_MAX_ROWS, q_end - q_start)});
            }
        }
    }

    // Spread the tiles evenly over every core. ceil(active / n) never exceeds the
    // slots, which are ceil(total / n) for active <= total.
    const auto n = static_cast<uint32_t>(split.cores.size());
    const auto active = static_cast<uint32_t>(split.tiles.size());
    split.counts.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        split.counts[i] = active / n + (i < active % n ? 1u : 0u);
    }
    return split;
}

// Per-core runtime args of the three kernels. Slots past a core's count are
// zero-filled so the arg count, and with it the cached program, stays fixed.
struct MSDARuntimeArgs {
    std::vector<std::vector<uint32_t>> reader;
    std::vector<std::vector<uint32_t>> compute;
    std::vector<std::vector<uint32_t>> writer;
};

MSDARuntimeArgs msda_runtime_args(
    const FusedMSDAOperation::operation_attributes_t& attrs,
    const FusedMSDAOperation::tensor_args_t& tensor_args,
    const Tensor& output,
    const MSDAShapes& s,
    const MSDAWorkSplit& split) {
    const Tensor& loc = attrs.from_offsets ? *tensor_args.sampling_offsets : *tensor_args.sampling_locations;
    const uint32_t ref_addr = attrs.from_offsets ? tensor_args.reference_points->buffer()->address() : 0u;

    MSDARuntimeArgs args;
    uint32_t tile_cursor = 0;
    const uint32_t slots = split.slots;
    for (size_t i = 0; i < split.cores.size(); ++i) {
        const uint32_t count = split.counts[i];

        auto& reader = args.reader.emplace_back();
        reader.reserve(5 + 3 * s.num_levels + slots * 4);
        reader.push_back(tensor_args.value.buffer()->address());
        reader.push_back(tensor_args.attention_weights.buffer()->address());
        reader.push_back(loc.buffer()->address());
        reader.push_back(ref_addr);
        reader.push_back(count);
        uint32_t level_start = 0;
        for (uint32_t l = 0; l < s.num_levels; ++l) {
            const uint32_t h = attrs.spatial_shapes_hw[2 * l];
            const uint32_t w = attrs.spatial_shapes_hw[2 * l + 1];
            reader.push_back(h);
            reader.push_back(w);
            reader.push_back(level_start);
            level_start += h * w;
        }

        auto& writer = args.writer.emplace_back();
        writer.reserve(2 + slots * 3);
        writer.push_back(output.buffer()->address());
        writer.push_back(count);

        for (uint32_t t = 0; t < count; ++t) {
            const auto& asn = split.tiles[tile_cursor + t];
            reader.insert(reader.end(), {asn.batch, asn.head, asn.q_start, asn.v_rows});
            // Output is (B, Q, H*D): one page per (b, q), head placed by offset.
            writer.insert(writer.end(), {asn.batch * s.num_queries + asn.q_start, asn.head, asn.v_rows});
        }
        reader.resize(5 + 3 * s.num_levels + slots * 4, 0u);
        writer.resize(2 + slots * 3, 0u);

        args.compute.push_back({count});
        tile_cursor += count;
    }
    return args;
}

// Kernel order in create_descriptor's descriptor.kernels; override_runtime_arguments
// patches by these indices.
constexpr uint32_t kReaderKernel = 0;
constexpr uint32_t kComputeKernel = 1;
constexpr uint32_t kWriterKernel = 2;

}  // namespace

ProgramDescriptor FusedMSDAOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, Tensor& output) {
    ProgramDescriptor descriptor{};

    const auto s = derive_shapes(attrs, tensor_args);

    const auto& value = tensor_args.value;
    const auto& attn = tensor_args.attention_weights;
    const Tensor& loc = attrs.from_offsets ? *tensor_args.sampling_offsets : *tensor_args.sampling_locations;

    auto* device = value.device();
    const auto compute_grid = device->compute_with_storage_grid_size();

    constexpr uint32_t TILE_WIDTH = 32;
    const uint32_t n_d_tiles = (s.head_dim + TILE_WIDTH - 1) / TILE_WIDTH;
    const uint32_t reduction_size = 4 * s.num_levels * s.num_points;

    const auto split = msda_work_split(attrs, s, compute_grid);
    const auto& all_cores = split.all_cores;

    // ---- page / stick sizes ----
    const uint32_t attn_sticks_per_row = s.weights_packed ? 1u : s.num_levels;
    const uint32_t loc_sticks_per_row = s.locations_packed ? 1u : (s.num_levels * s.num_points);

    const uint32_t value_stick_raw = s.head_dim * value.element_size();
    const uint32_t attn_stick_raw =
        (s.weights_packed ? s.num_levels * s.num_points : s.num_points) * attn.element_size();
    const uint32_t loc_stick_raw = (s.locations_packed ? s.num_levels * s.num_points * 2u : 2u) * loc.element_size();
    const uint32_t output_page_raw = s.num_heads * s.head_dim * output.element_size();

    const uint32_t value_stick_aligned = aligned_page_size(value_stick_raw, value.buffer()->buffer_type());
    const uint32_t attn_stick_aligned = aligned_page_size(attn_stick_raw, attn.buffer()->buffer_type());
    const uint32_t loc_stick_aligned = aligned_page_size(loc_stick_raw, loc.buffer()->buffer_type());
    const uint32_t output_page_aligned = aligned_page_size(output_page_raw, output.buffer()->buffer_type());
    // Writer scratch holds one head's D-wide stick, not the whole output page.
    const uint32_t head_stick_aligned = aligned_page_size(value_stick_raw, output.buffer()->buffer_type());

    // The writer places head h at byte offset h * D * elem_size inside the
    // output page, so that offset has to be a legal NoC destination address.
    // D % 16 == 0 makes it a multiple of 32, which covers Wormhole but not a
    // 64-byte-aligned target at D == 16. Check it rather than silently
    // scribbling into the neighbouring head.
    if (s.num_heads > 1) {
        const uint32_t out_alignment = output.buffer()->buffer_type() == BufferType::DRAM
                                           ? tt::tt_metal::hal::get_dram_alignment()
                                           : tt::tt_metal::hal::get_l1_alignment();
        TT_FATAL(
            value_stick_raw % out_alignment == 0,
            "fused_msda: with num_heads > 1 the per-head output stride (head_dim * element_size = {} B) must be a "
            "multiple of this device's output alignment ({} B). head_dim {} is too small on this architecture; use a "
            "head_dim of at least {}",
            value_stick_raw,
            out_alignment,
            s.head_dim,
            out_alignment / output.element_size());
        if (s.value_packed) {
            TT_FATAL(
                value_stick_aligned == value_stick_raw,
                "fused_msda: packed value (B, S, H*D) stores heads without per-head page padding, so the D-wide NoC "
                "read ({} B aligned) must equal the raw head stick ({} B); otherwise the read would overlap the next "
                "head",
                value_stick_aligned,
                value_stick_raw);
        }
    }

    uint32_t ref_stick_aligned = 0;
    if (attrs.from_offsets) {
        const auto& ref = *tensor_args.reference_points;
        ref_stick_aligned = aligned_page_size(2u * ref.element_size(), ref.buffer()->buffer_type());
    }

    const auto data_format = datatype_to_dataformat_converter(DataType::BFLOAT16);
    const uint32_t tile_nbytes = tt::tile_size(data_format);

    // ---- circular buffers (see README.md §6) ----
    constexpr uint8_t value_scratch_cb = tt::CBIndex::c_0;
    constexpr uint8_t attn_scratch_cb = tt::CBIndex::c_1;
    constexpr uint8_t loc_scratch_cb = tt::CBIndex::c_2;
    constexpr uint8_t input_tile_cb = tt::CBIndex::c_3;
    constexpr uint8_t scalar_tile_cb = tt::CBIndex::c_4;
    constexpr uint8_t output_scratch_cb = tt::CBIndex::c_5;
    constexpr uint8_t ref_scratch_cb = tt::CBIndex::c_6;
    constexpr uint8_t output_tile_cb = tt::CBIndex::c_16;

    auto push_cb = [&](uint8_t idx, uint32_t pages, uint32_t page_size, tt::DataFormat fmt) {
        descriptor.cbs.push_back(CBDescriptor{
            .total_size = pages * page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = idx,
                .data_format = fmt,
                .page_size = page_size,
            }}},
        });
    };

    const auto value_fmt = datatype_to_dataformat_converter(value.dtype());
    const auto attn_fmt = datatype_to_dataformat_converter(attn.dtype());
    const auto loc_fmt = datatype_to_dataformat_converter(loc.dtype());
    const auto output_fmt = datatype_to_dataformat_converter(output.dtype());

    // Reader-only staging arenas: reserved once and treated as flat L1 space.
    push_cb(value_scratch_cb, TILE_MAX_ROWS, value_stick_aligned, value_fmt);
    push_cb(attn_scratch_cb, TILE_MAX_ROWS * attn_sticks_per_row, attn_stick_aligned, attn_fmt);
    push_cb(loc_scratch_cb, TILE_MAX_ROWS * loc_sticks_per_row, loc_stick_aligned, loc_fmt);
    if (attrs.from_offsets) {
        push_cb(ref_scratch_cb, TILE_MAX_ROWS * s.num_refs, ref_stick_aligned, loc_fmt);
    }
    // Reader -> compute, double-buffered.
    push_cb(input_tile_cb, 2 * n_d_tiles, tile_nbytes, data_format);
    push_cb(scalar_tile_cb, 2, tile_nbytes, data_format);
    // Compute -> writer, double-buffered.
    push_cb(output_tile_cb, 2 * n_d_tiles, tile_nbytes, output_fmt);
    // Writer-only scratch.
    push_cb(output_scratch_cb, 1, head_stick_aligned, output_fmt);

    // ---- reader ----
    // Compile-time arg order is fixed by fused_msda_reader_common.hpp; both
    // readers read the same indices.
    KernelDescriptor::CompileTimeArgs reader_ct{
        value_scratch_cb,
        attn_scratch_cb,
        loc_scratch_cb,
        input_tile_cb,
        scalar_tile_cb,
        attrs.from_offsets ? static_cast<uint32_t>(ref_scratch_cb) : 0u,
        s.head_dim,
        s.num_queries,
        s.num_heads,
        s.num_levels,
        s.num_points,
        s.num_keys,
        s.num_refs,
        value_stick_aligned,
        attn_stick_aligned,
        loc_stick_aligned,
        ref_stick_aligned,
        static_cast<uint32_t>(attrs.align_corners),
        static_cast<uint32_t>(attrs.locations_in_grid_space),
        static_cast<uint32_t>(s.locations_packed),
        static_cast<uint32_t>(s.weights_packed),
        static_cast<uint32_t>(attrs.reference_mode == MSDAReferenceMode::Level ? 0 : 1),
        static_cast<uint32_t>(s.value_packed),
        // Must match the tensor's DRAM page size, not the D-wide scratch stick.
        // Packed value is one (B, S) stick of H*D; canonical is one (B, S, H) stick of D.
        static_cast<uint32_t>(value.buffer()->page_size()),
    };
    TensorAccessorArgs(*value.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*attn.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*loc.buffer()).append_to(reader_ct);
    if (attrs.from_offsets) {
        TensorAccessorArgs(*tensor_args.reference_points->buffer()).append_to(reader_ct);
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        std::string(kKernelDir) + (attrs.from_offsets ? "dataflow/reader_msda_v2.cpp" : "dataflow/reader_msda_v1.cpp");
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.config = ReaderConfigDescriptor{};

    // ---- compute (shared by V1 and V2) ----
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = std::string(kKernelDir) + "compute/compute_msda.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {input_tile_cb, scalar_tile_cb, output_tile_cb, reduction_size, n_d_tiles};
    compute_desc.config = ComputeConfigDescriptor{};

    // ---- writer (shared by V1 and V2) ----
    KernelDescriptor::CompileTimeArgs writer_ct{
        output_tile_cb,
        output_scratch_cb,
        output_page_aligned,
        s.head_dim,
    };
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = std::string(kKernelDir) + "dataflow/writer_msda.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    // ---- per-core runtime args ----
    // Plain addresses, not Buffer* bindings: override_runtime_arguments rewrites
    // every arg on a cache hit, addresses included.
    auto rt = msda_runtime_args(attrs, tensor_args, output, s, split);
    for (size_t i = 0; i < split.cores.size(); ++i) {
        const auto& core = split.cores[i];
        reader_desc.runtime_args.emplace_back(core, std::move(rt.reader[i]));
        compute_desc.runtime_args.emplace_back(core, std::move(rt.compute[i]));
        writer_desc.runtime_args.emplace_back(core, std::move(rt.writer[i]));
    }

    // Order must match kReaderKernel, kComputeKernel, kWriterKernel.
    descriptor.kernels.push_back(std::move(reader_desc));
    descriptor.kernels.push_back(std::move(compute_desc));
    descriptor.kernels.push_back(std::move(writer_desc));

    return descriptor;
}

void FusedMSDAOperation::ProgramFactory::override_runtime_arguments(
    Program& program,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const auto s = derive_shapes(attrs, tensor_args);
    const auto split = msda_work_split(attrs, s, tensor_args.value.device()->compute_with_storage_grid_size());
    const auto rt = msda_runtime_args(attrs, tensor_args, output, s, split);
    const auto patch = [&](uint32_t kernel, const CoreCoord& core, const std::vector<uint32_t>& values) {
        auto& args = GetRuntimeArgs(program, kernel, core);
        TT_FATAL(
            args.size() == values.size(),
            "fused_msda: internal error, runtime-arg count changed on a cache hit (kernel {}, core {}: {} cached, {} "
            "now)",
            kernel,
            core,
            args.size(),
            values.size());
        std::copy(values.begin(), values.end(), args.data());
    };
    for (size_t i = 0; i < split.cores.size(); ++i) {
        patch(kReaderKernel, split.cores[i], rt.reader[i]);
        patch(kComputeKernel, split.cores[i], rt.compute[i]);
        patch(kWriterKernel, split.cores[i], rt.writer[i]);
    }
}

}  // namespace ttnn::operations::experimental::fused_msda
