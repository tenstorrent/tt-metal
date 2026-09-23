// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
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

// The normalized -> pixel mapping for one axis of one level, folded into the
// three constants the SFPU geometry applies:
//
//     px = primary * primary_scale + secondary * secondary_scale + bias
//
// `primary` is the sampling location (V1) or the reference point (V2);
// `secondary` is the raw sampling offset and only exists for V2. Folding
// align_corners and locations_in_grid_space in here keeps the compute kernel
// branch-free on them and keeps the int -> float conversion of the extent off
// the device entirely.
//
// The four (locations_in_grid_space, align_corners) cases are README.md §3;
// V2's two rows follow from substituting loc = ref + off / extent into the
// [0, 1] rows, which is why `secondary_scale` is exactly 1 in the common case.
//
// Must stay in step with `AxisConstants` in msda_geometry.hpp, which consumes
// them, and with kAxisConstantsPerLevel below, which is how many uint32 one
// level ships as.
struct AxisConstants {
    uint32_t primary_scale;    // fp32 bit pattern
    uint32_t secondary_scale;  // fp32 bit pattern
    uint32_t bias;             // fp32 bit pattern
};

constexpr uint32_t kAxisConstantsPerLevel = 6;  // (primary, secondary, bias) x (x, y)

AxisConstants axis_constants(uint32_t extent, bool align_corners, bool grid_space, bool from_offsets) {
    // The from-offsets branch has no grid-space row. Today that is unreachable
    // because the V2 prim hardcodes locations_in_grid_space = false, but this
    // function is the single source of truth for the whole coordinate mapping,
    // and silently applying the [0, 1] mapping to a grid-space caller would be
    // a wrong sample rather than a failure.
    TT_FATAL(
        !(from_offsets && grid_space),
        "fused_msda: grid-space sampling offsets are not supported; there is no from-offsets row for "
        "locations_in_grid_space=true in the coordinate table");

    const float e = static_cast<float>(extent);
    float primary_scale = 0.0f;
    float secondary_scale = 0.0f;
    float bias = 0.0f;
    if (from_offsets) {
        // loc = ref + off / extent, then the [0, 1] mapping.
        if (align_corners) {
            primary_scale = e - 1.0f;
            secondary_scale = (e - 1.0f) / e;
        } else {
            primary_scale = e;
            secondary_scale = 1.0f;
            bias = -0.5f;
        }
    } else if (grid_space) {
        if (align_corners) {
            primary_scale = 0.5f * (e - 1.0f);
            bias = 0.5f * (e - 1.0f);
        } else {
            primary_scale = 0.5f * e;
            bias = 0.5f * e - 0.5f;
        }
    } else {
        if (align_corners) {
            primary_scale = e - 1.0f;
        } else {
            primary_scale = e;
            bias = -0.5f;
        }
    }
    return {std::bit_cast<uint32_t>(primary_scale), std::bit_cast<uint32_t>(secondary_scale), std::bit_cast<uint32_t>(bias)};
}

// CB depths for the geometry pipeline. Both the reader and the compute kernel
// run one sampling point ahead of the work they are feeding, so every pipe
// between them has to hold two points' worth plus the one in flight.
//
// kScalarCbPages is a hard deadlock floor: at the fourth corner_weight of point
// j+1 the PACK thread holds 7 pages and reserves the 8th, and UNPACK cannot free
// one until the reduction of point j runs, so 7 hangs the kernel.
// kInputCbGroups is only a throughput floor — the reader's pushes are never
// gated behind work the compute kernel has not already issued, so halving it
// costs overlap but cannot hang. kCornerCbPages and kFracCbPages each carry one
// page of slack over their maximum outstanding (2 and 1).
// Deepening the lookahead means scaling all of them.
constexpr uint32_t kGeomCbPages = 4;    // reader -> compute operand tiles
constexpr uint32_t kCornerCbPages = 3;  // compute -> reader floored corners
constexpr uint32_t kFracCbPages = 2;    // compute -> compute, one point at a time
constexpr uint32_t kScalarCbPages = 8;  // two points of four corners each
constexpr uint32_t kInputCbGroups = 8;  // ditto, in corner groups of n_d_tiles

}  // namespace

ProgramDescriptor FusedMSDAOperation::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, Tensor& output) {
    ProgramDescriptor descriptor{};

    const auto s = derive_shapes(attrs, tensor_args);

    const auto& value = tensor_args.value;
    const auto& attn = tensor_args.attention_weights;
    const Tensor& loc = attrs.from_offsets ? *tensor_args.sampling_offsets : *tensor_args.sampling_locations;

    auto* device = value.device();
    const auto compute_grid = device->compute_with_storage_grid_size();

    constexpr uint32_t TILE_MAX_ROWS = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    const uint32_t n_d_tiles = (s.head_dim + TILE_WIDTH - 1) / TILE_WIDTH;

    // ---- work units ----
    const uint32_t q_blocks = (s.num_queries + TILE_MAX_ROWS - 1) / TILE_MAX_ROWS;
    std::vector<TileAssignment> tiles;
    tiles.reserve(static_cast<size_t>(s.batch) * s.num_heads * q_blocks);
    for (uint32_t b = 0; b < s.batch; ++b) {
        for (uint32_t h = 0; h < s.num_heads; ++h) {
            for (uint32_t q_start = 0; q_start < s.num_queries; q_start += TILE_MAX_ROWS) {
                tiles.push_back({b, h, q_start, std::min(TILE_MAX_ROWS, s.num_queries - q_start)});
            }
        }
    }
    const uint32_t total_output_tiles = static_cast<uint32_t>(tiles.size());

    // ---- per-level geometry (host-side; shipped as reader runtime args) ----
    std::vector<uint32_t> level_h(s.num_levels);
    std::vector<uint32_t> level_w(s.num_levels);
    std::vector<uint32_t> level_start(s.num_levels);
    // The SFPU geometry's per-level constants go with them: same source, same
    // place, so the two cannot drift.
    std::vector<AxisConstants> level_kx(s.num_levels);
    std::vector<AxisConstants> level_ky(s.num_levels);
    uint32_t running = 0;
    for (uint32_t l = 0; l < s.num_levels; ++l) {
        level_h[l] = attrs.spatial_shapes_hw[2 * l];
        level_w[l] = attrs.spatial_shapes_hw[2 * l + 1];
        level_start[l] = running;
        running += level_h[l] * level_w[l];
        level_kx[l] =
            axis_constants(level_w[l], attrs.align_corners, attrs.locations_in_grid_space, attrs.from_offsets);
        level_ky[l] =
            axis_constants(level_h[l], attrs.align_corners, attrs.locations_in_grid_space, attrs.from_offsets);
    }

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

    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_group_1, tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_grid, total_output_tiles);

    // ---- circular buffers (see README.md §6) ----
    constexpr uint8_t value_scratch_cb = tt::CBIndex::c_0;
    constexpr uint8_t attn_scratch_cb = tt::CBIndex::c_1;
    constexpr uint8_t loc_scratch_cb = tt::CBIndex::c_2;
    constexpr uint8_t input_tile_cb = tt::CBIndex::c_3;
    constexpr uint8_t scalar_tile_cb = tt::CBIndex::c_4;
    constexpr uint8_t output_scratch_cb = tt::CBIndex::c_5;
    constexpr uint8_t ref_scratch_cb = tt::CBIndex::c_6;
    // The geometry pipeline (README.md §6). geom/offset/attn go reader ->
    // compute, x0/y0 come back compute -> reader, frac never leaves compute.
    constexpr uint8_t geom_x_cb = tt::CBIndex::c_7;
    constexpr uint8_t geom_y_cb = tt::CBIndex::c_8;
    constexpr uint8_t offset_x_cb = tt::CBIndex::c_9;
    constexpr uint8_t offset_y_cb = tt::CBIndex::c_10;
    constexpr uint8_t attn_tile_cb = tt::CBIndex::c_11;
    constexpr uint8_t x0_cb = tt::CBIndex::c_12;
    constexpr uint8_t y0_cb = tt::CBIndex::c_13;
    constexpr uint8_t frac_x_cb = tt::CBIndex::c_14;
    constexpr uint8_t frac_y_cb = tt::CBIndex::c_15;
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
    // The sample stream, sized so that reader and compute can each run one
    // sampling point ahead of the other (see kGeomCbPages and friends).
    push_cb(input_tile_cb, kInputCbGroups * n_d_tiles, tile_nbytes, data_format);
    push_cb(scalar_tile_cb, kScalarCbPages, tile_nbytes, data_format);
    // The geometry pipeline. Always allocated, including the offset pipes on
    // V1 where the compute kernel names but never reads them: 8 KB of L1
    // against a conditional CB index the kernel would have to guard.
    push_cb(geom_x_cb, kGeomCbPages, tile_nbytes, data_format);
    push_cb(geom_y_cb, kGeomCbPages, tile_nbytes, data_format);
    push_cb(offset_x_cb, kGeomCbPages, tile_nbytes, data_format);
    push_cb(offset_y_cb, kGeomCbPages, tile_nbytes, data_format);
    push_cb(attn_tile_cb, kGeomCbPages, tile_nbytes, data_format);
    push_cb(x0_cb, kCornerCbPages, tile_nbytes, data_format);
    push_cb(y0_cb, kCornerCbPages, tile_nbytes, data_format);
    push_cb(frac_x_cb, kFracCbPages, tile_nbytes, data_format);
    push_cb(frac_y_cb, kFracCbPages, tile_nbytes, data_format);
    // Compute -> writer, double-buffered.
    push_cb(output_tile_cb, 2 * n_d_tiles, tile_nbytes, output_fmt);
    // Writer-only scratch.
    push_cb(output_scratch_cb, 1, head_stick_aligned, output_fmt);

    // ---- reader ----
    // Compile-time arg order is fixed by fused_msda_reader_common.hpp; both
    // readers read the same indices.
    // The pixel mapping lives in the compute kernel's per-level constants, so
    // no coordinate-convention flags are passed here.
    KernelDescriptor::CompileTimeArgs reader_ct{
        value_scratch_cb,
        attn_scratch_cb,
        loc_scratch_cb,
        input_tile_cb,
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
        static_cast<uint32_t>(s.locations_packed),
        static_cast<uint32_t>(s.weights_packed),
        static_cast<uint32_t>(attrs.reference_mode == MSDAReferenceMode::Level ? 0 : 1),
        static_cast<uint32_t>(s.value_packed),
        // Must match the tensor's DRAM page size, not the D-wide scratch stick.
        // Packed value is one (B, S) stick of H*D; canonical is one (B, S, H) stick of D.
        static_cast<uint32_t>(value.buffer()->page_size()),
        static_cast<uint32_t>(attrs.from_offsets),
        geom_x_cb,
        geom_y_cb,
        offset_x_cb,
        offset_y_cb,
        attn_tile_cb,
        x0_cb,
        y0_cb,
        kGeomCbPages,
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
    compute_desc.compile_time_args = {
        input_tile_cb,
        scalar_tile_cb,
        output_tile_cb,
        n_d_tiles,
        geom_x_cb,
        geom_y_cb,
        offset_x_cb,
        offset_y_cb,
        attn_tile_cb,
        x0_cb,
        y0_cb,
        frac_x_cb,
        frac_y_cb,
        s.num_levels,
        s.num_points,
        static_cast<uint32_t>(attrs.from_offsets),
    };
    // fp32 destination: the geometry builds px, which reaches the feature map's
    // extent. In a 16-bit destination the ulp at 200 is 1.0, so floor(px) and
    // the fraction it feeds are lost outright on the larger levels and bilinear
    // degrades to nearest-neighbour. The from-offsets path uses all four DST
    // slots a 32-bit destination allows; there is no spare.
    //
    // This kernel also drives packer L1 accumulation, which #28800 records as
    // unsupported alongside a 32-bit destination (docs/L1_ACCUMULATION_FP32_ANALYSIS.md;
    // sdpa_fw disables L1 acc when the flag is on). The combination is measured
    // good on Blackhole here; it has not been checked on Wormhole.
    ComputeConfigDescriptor compute_cfg{};
    compute_cfg.fp32_dest_acc_en = true;
    compute_desc.config = compute_cfg;

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
    // Buffer* entries auto-register as buffer bindings, so the framework patches
    // addresses on program-cache hits (no override_runtime_arguments needed).
    const auto logical_cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    uint32_t tile_cursor = 0;
    for (const auto& core : logical_cores) {
        uint32_t tiles_here = 0;
        if (core_group_1.contains(core)) {
            tiles_here = tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            tiles_here = tiles_per_core_group_2;
        } else {
            continue;
        }

        KernelDescriptor::RTArgList reader_args;
        reader_args.reserve(5 + 3 * s.num_levels + tiles_here * 4);
        reader_args.push_back(value.buffer());
        reader_args.push_back(attn.buffer());
        reader_args.push_back(loc.buffer());
        if (attrs.from_offsets) {
            reader_args.push_back(tensor_args.reference_points->buffer());
        } else {
            reader_args.push_back(uint32_t{0});
        }
        reader_args.push_back(tiles_here);
        for (uint32_t l = 0; l < s.num_levels; ++l) {
            reader_args.push_back(level_h[l]);
            reader_args.push_back(level_w[l]);
            reader_args.push_back(level_start[l]);
        }

        KernelDescriptor::RTArgList writer_args;
        writer_args.reserve(2 + tiles_here * 3);
        writer_args.push_back(output.buffer());
        writer_args.push_back(tiles_here);

        for (uint32_t i = 0; i < tiles_here; ++i) {
            const auto& asn = tiles[tile_cursor + i];
            reader_args.push_back(asn.batch);
            reader_args.push_back(asn.head);
            reader_args.push_back(asn.q_start);
            reader_args.push_back(asn.v_rows);
            // Output is (B, Q, H*D): one page per (b, q), head placed by offset.
            writer_args.push_back(asn.batch * s.num_queries + asn.q_start);
            writer_args.push_back(asn.head);
            writer_args.push_back(asn.v_rows);
        }

        // `mul_unary_tile` takes its scalar as a runtime uint32 and the level
        // index is a runtime value, so the per-level constants are runtime args.
        KernelDescriptor::CoreRuntimeArgs compute_args;
        compute_args.reserve(1 + kAxisConstantsPerLevel * s.num_levels);
        compute_args.push_back(tiles_here);
        for (uint32_t l = 0; l < s.num_levels; ++l) {
            compute_args.push_back(level_kx[l].primary_scale);
            compute_args.push_back(level_kx[l].secondary_scale);
            compute_args.push_back(level_kx[l].bias);
            compute_args.push_back(level_ky[l].primary_scale);
            compute_args.push_back(level_ky[l].secondary_scale);
            compute_args.push_back(level_ky[l].bias);
        }

        reader_desc.emplace_runtime_args(core, reader_args);
        compute_desc.runtime_args.emplace_back(core, std::move(compute_args));
        writer_desc.emplace_runtime_args(core, writer_args);

        tile_cursor += tiles_here;
    }

    descriptor.kernels.push_back(std::move(reader_desc));
    descriptor.kernels.push_back(std::move(compute_desc));
    descriptor.kernels.push_back(std::move(writer_desc));

    return descriptor;
}

}  // namespace ttnn::operations::experimental::fused_msda
