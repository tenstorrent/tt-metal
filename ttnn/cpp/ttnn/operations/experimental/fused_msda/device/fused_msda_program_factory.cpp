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
#include "ttnn/operations/experimental/fused_msda/device/kernels/msda_tiled_geometry.hpp"

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
// V2's two rows follow from substituting loc = ref + off * scale_l into the
// [0, 1] rows, which is why `secondary_scale` is 1 except on a singleton
// align_corners axis.
//
// `extent` is the axis' W or H, which bounds the floored corner on the SFPU
// (kernels/msda_corner_word.hpp).
//
// Must stay in step with `AxisConstants` in msda_geometry.hpp, which consumes
// them, and with kAxisConstantsPerLevel below, which is how many uint32 one
// level ships as.
struct AxisConstants {
    uint32_t primary_scale;    // fp32 bit pattern
    uint32_t secondary_scale;  // fp32 bit pattern
    uint32_t bias;             // fp32 bit pattern
    uint32_t extent;           // fp32 bit pattern
};

constexpr uint32_t kAxisConstantsPerLevel = 8;  // (primary, secondary, bias, extent) x (x, y)
static_assert(kAxisConstantsPerLevel == fused_msda_tiled_geometry::CONSTANTS_PER_LEVEL);

// Appends each level's constants in kAxisConstantsPerLevel order, the layout
// both the compute kernel and the tiled-geometry reader read.
template <typename Args>
void append_axis_constants(Args& args, const std::vector<AxisConstants>& kx, const std::vector<AxisConstants>& ky) {
    for (size_t l = 0; l < kx.size(); ++l) {
        for (const auto& k : {kx[l], ky[l]}) {
            args.push_back(k.primary_scale);
            args.push_back(k.secondary_scale);
            args.push_back(k.bias);
            args.push_back(k.extent);
        }
    }
}

AxisConstants axis_constants(uint32_t extent, bool align_corners, bool grid_space, bool from_offsets) {
    // The from-offsets branch has no grid-space row. Silently applying the
    // [0, 1] mapping to a grid-space caller would sample the wrong pixel.
    TT_FATAL(
        !(from_offsets && grid_space),
        "fused_msda: grid-space sampling offsets are not supported; there is no from-offsets row for "
        "locations_in_grid_space=true in the coordinate table");

    const float e = static_cast<float>(extent);
    float primary_scale = 0.0f;
    float secondary_scale = 0.0f;
    float bias = 0.0f;
    if (from_offsets) {
        // loc = ref + off * scale_l, then the [0, 1] mapping. scale_l is the
        // mapping's pixel spacing, so one offset unit is one pixel; a singleton
        // axis under align_corners collapses onto pixel 0 and scale_l is 0.
        if (align_corners) {
            primary_scale = e - 1.0f;
            secondary_scale = extent > 1u ? 1.0f : 0.0f;
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
    return {
        std::bit_cast<uint32_t>(primary_scale),
        std::bit_cast<uint32_t>(secondary_scale),
        std::bit_cast<uint32_t>(bias),
        std::bit_cast<uint32_t>(e)};
}

// CB depths for the geometry pipeline. Both the reader and the compute kernel
// run at most one sampling point ahead of the work they are feeding (where the
// reader's push sits depends on the split gather; see the pipelining note in
// fused_msda_reader_common.hpp), so the pipes between them hold up to two
// points' worth; each depth below is that pipe's floor plus any slack, as noted.
//
// kScalarCbPages is a hard deadlock floor: the geometry of point j+1 reserves
// its four scalar pages at once while point j's four are still outstanding,
// and UNPACK cannot free one until the reduction of point j runs, so 7 hangs
// the kernel.
// kInputCbGroups is only a throughput floor — the reader's pushes are never
// gated behind work the compute kernel has not already issued, so halving it
// costs overlap but cannot hang. kCornerCbPages carries one page of slack over
// its maximum outstanding (2).
// Deepening the lookahead means scaling all of them.
constexpr uint32_t kGeomCbPages = 4;    // reader -> compute operand tiles
constexpr uint32_t kCornerCbPages = 3;  // compute -> reader corner words
constexpr uint32_t kScalarCbPages = 8;  // two points of four corners each
constexpr uint32_t kInputCbGroups = 8;  // ditto, in corner groups of n_d_tiles
// The tiled geometry's depths are in msda_tiled_geometry.hpp, which the
// kernels read too.

bool bf16_exact(uint32_t fp32_bits) { return (fp32_bits & 0xFFFFu) == 0; }

constexpr uint32_t kTileMaxRows = 32;
constexpr uint32_t kTileWidth = 32;

constexpr uint8_t value_scratch_cb = tt::CBIndex::c_0;
constexpr uint8_t attn_scratch_cb = tt::CBIndex::c_1;
constexpr uint8_t loc_scratch_cb = tt::CBIndex::c_2;
constexpr uint8_t input_tile_cb = tt::CBIndex::c_3;
constexpr uint8_t scalar_tile_cb = tt::CBIndex::c_4;
constexpr uint8_t output_scratch_cb = tt::CBIndex::c_5;
constexpr uint8_t ref_scratch_cb = tt::CBIndex::c_6;
// The geometry pipeline (README.md §6). geom/attn go reader -> compute, the
// corner words come back compute -> reader.
constexpr uint8_t geom_x_cb = tt::CBIndex::c_7;
constexpr uint8_t geom_y_cb = tt::CBIndex::c_8;
constexpr uint8_t attn_tile_cb = tt::CBIndex::c_11;
constexpr uint8_t corner_cb = tt::CBIndex::c_12;
constexpr uint8_t output_tile_cb = tt::CBIndex::c_16;
constexpr uint8_t tiled_input_cb = tt::CBIndex::c_17;
constexpr uint8_t gather_mailbox_cb = tt::CBIndex::c_18;
// Tiled geometry only, in place of geom_x/geom_y/attn_tile and the offset/attn
// arenas (kernels/msda_tiled_geometry.hpp).
constexpr uint8_t geom_rm_cb = tt::CBIndex::c_9;
constexpr uint8_t geom_operand_cb = tt::CBIndex::c_10;
constexpr uint8_t ref_tile_cb = tt::CBIndex::c_14;
constexpr uint8_t const_cb = tt::CBIndex::c_15;
constexpr uint8_t weight_cb = tt::CBIndex::c_19;

// Row-major staging: when D is a whole number of tile widths, the reader
// lands the four corners of a point side by side as a row-major block and
// compute tilizes it on the unpacker, so the reader copies nothing. Other
// D fall back to the reader scattering sticks into tile faces.
bool uses_rm_staging(const MSDAShapes& s) { return s.head_dim % kTileWidth == 0; }

// Split gather (msda_gather_rows.hpp): the writer RISC takes part of every
// row-major block. Blackhole only, as Wormhole has not been measured; the
// scatter path (D % 32 != 0) is always reader-only. Decided only here, by the
// CB plan; create_descriptor reads the decision back from the plan.
bool uses_split_gather(const MSDAShapes& s, tt::ARCH arch) { return uses_rm_staging(s) && arch == tt::ARCH::BLACKHOLE; }

struct LevelConstants {
    std::vector<AxisConstants> kx;
    std::vector<AxisConstants> ky;
};

LevelConstants level_constants(const FusedMSDAOperation::operation_attributes_t& attrs, const MSDAShapes& s) {
    LevelConstants c;
    for (uint32_t l = 0; l < s.num_levels; ++l) {
        c.kx.push_back(axis_constants(
            attrs.spatial_shapes_hw[2 * l + 1],
            attrs.align_corners,
            attrs.locations_in_grid_space,
            attrs.from_offsets));
        c.ky.push_back(axis_constants(
            attrs.spatial_shapes_hw[2 * l], attrs.align_corners, attrs.locations_in_grid_space, attrs.from_offsets));
    }
    return c;
}

// Tiled geometry (kernels/msda_tiled_geometry.hpp): compute solves a whole
// block from tiles, so the reader hands the operands over once per block. The
// layout puts point j in tile column j, which fixes L * P at 16, and the FPU
// applies the per-level constants as bf16 matrix entries, so each must be exact
// in bf16. Every staged offsets column must be one the layout names, so the
// offsets stick must be unpadded. bf16 reference points only: the reference
// tile is bf16. Blackhole only: like the split gather, it has been validated
// and measured nowhere else.
bool uses_tiled_geometry(
    const FusedMSDAOperation::operation_attributes_t& attrs,
    const FusedMSDAOperation::tensor_args_t& args,
    const MSDAShapes& s,
    uint32_t loc_stick_aligned,
    uint32_t attn_stick_aligned) {
    namespace tg = fused_msda_tiled_geometry;
    if (!attrs.from_offsets || args.reference_points->dtype() != DataType::BFLOAT16 ||
        args.value.device()->arch() != tt::ARCH::BLACKHOLE) {
        return false;
    }
    const auto c = level_constants(attrs, s);
    bool constants_exact = true;
    for (uint32_t l = 0; l < s.num_levels; ++l) {
        for (const auto& k : {c.kx[l], c.ky[l]}) {
            constants_exact = constants_exact && bf16_exact(k.primary_scale) && bf16_exact(k.secondary_scale) &&
                              bf16_exact(k.bias) && bf16_exact(k.extent);
        }
    }
    return s.locations_packed && s.weights_packed && s.num_levels * s.num_points == tg::POINTS &&
           s.num_refs <= tg::MAX_REFS && loc_stick_aligned == tg::RM_ATTN_OFFSET &&
           attn_stick_aligned <= tg::RM_ROW_NBYTES - tg::RM_ATTN_OFFSET && constants_exact;
}

uint32_t page_size_for_cb(const MSDACircularBufferPlan& plan, uint8_t index) {
    const auto *const it = std::find_if(plan.begin(), plan.end(), [index](const auto& spec) { return spec.index == index; });
    TT_FATAL(it != plan.end(), "fused_msda: internal error, missing circular buffer {}", index);
    return it->page_size;
}

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
    const uint32_t q_blocks = (s.num_queries + kTileMaxRows - 1) / kTileMaxRows;
    const uint32_t total_slots = s.batch * s.num_heads * q_blocks;

    MSDAWorkSplit split;
    auto [num_cores, all_cores, core_group_1, core_group_2, slots_group_1, slots_group_2] =
        tt::tt_metal::split_work_to_cores(grid, total_slots);
    (void)core_group_1;
    (void)core_group_2;
    (void)slots_group_2;
    split.all_cores = all_cores;
    split.cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    // split_work_to_cores puts the larger per-core count in group 1; the groups
    // themselves are not used, since every core gets that many slots.
    split.slots = slots_group_1;

    // Heads innermost: each core takes a contiguous run of tiles, so most of
    // its consecutive tiles share (b, q_start), and the V2 reader stages the
    // head-invariant reference points once per query block in its run rather
    // than once per tile.
    for (uint32_t b = 0; b < s.batch; ++b) {
        const uint32_t q_end = attrs.num_valid_queries.empty() ? s.num_queries : attrs.num_valid_queries[b];
        for (uint32_t q_start = 0; q_start < q_end; q_start += kTileMaxRows) {
            for (uint32_t h = 0; h < s.num_heads; ++h) {
                split.tiles.push_back({b, h, q_start, std::min(kTileMaxRows, q_end - q_start)});
            }
        }
    }

    // ceil(active / n) never exceeds the slots: slots are ceil(total / n) and active <= total.
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
    const MSDAWorkSplit& split,
    bool tiled_geometry) {
    const Tensor& loc = attrs.from_offsets ? *tensor_args.sampling_offsets : *tensor_args.sampling_locations;
    // The tiled geometry's reader writes the per-level constants into its
    // matrices, so it gets them too, after the level table.
    const uint32_t reader_fixed = 5 + 3 * s.num_levels + (tiled_geometry ? kAxisConstantsPerLevel * s.num_levels : 0);
    const uint32_t ref_addr = attrs.from_offsets ? tensor_args.reference_points->buffer()->address() : 0u;

    // Same constants on every core.
    std::vector<uint32_t> compute_template;
    compute_template.reserve(1 + kAxisConstantsPerLevel * s.num_levels);
    compute_template.push_back(0);
    const auto level_k = level_constants(attrs, s);
    append_axis_constants(compute_template, level_k.kx, level_k.ky);

    MSDARuntimeArgs args;
    uint32_t tile_cursor = 0;
    const uint32_t slots = split.slots;
    for (size_t i = 0; i < split.cores.size(); ++i) {
        const uint32_t count = split.counts[i];

        auto& reader = args.reader.emplace_back();
        reader.reserve(reader_fixed + slots * 4);
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
        if (tiled_geometry) {
            append_axis_constants(reader, level_k.kx, level_k.ky);
        }

        auto& writer = args.writer.emplace_back();
        writer.reserve(3 + slots * 3);
        writer.push_back(output.buffer()->address());
        writer.push_back(count);
        writer.push_back(tensor_args.value.buffer()->address());

        for (uint32_t t = 0; t < count; ++t) {
            const auto& asn = split.tiles[tile_cursor + t];
            reader.insert(reader.end(), {asn.batch, asn.head, asn.q_start, asn.v_rows});
            // Output is (B, Q, H*D): one page per (b, q), head placed by offset.
            writer.insert(writer.end(), {asn.batch * s.num_queries + asn.q_start, asn.head, asn.v_rows});
        }
        reader.resize(reader_fixed + slots * 4, 0u);
        writer.resize(3 + slots * 3, 0u);

        // The per-level constants are runtime args. Their count is fixed by
        // num_levels, which is in the program hash.
        auto compute = compute_template;
        compute[0] = count;
        args.compute.push_back(std::move(compute));
        tile_cursor += count;
    }
    return args;
}

// Kernel order in create_descriptor's descriptor.kernels; override_runtime_arguments
// patches by these indices.
constexpr uint32_t kReaderKernel = 0;
constexpr uint32_t kComputeKernel = 1;
constexpr uint32_t kWriterKernel = 2;

// One split-gather point's arguments, reader -> writer. Both kernels get the
// allocated page size as a compile-time arg and static_assert that PointArgs
// fits it.
constexpr uint32_t kGatherMailboxNbytes = 64;

}  // namespace

// Every CB the program allocates. can_use_fused_msda_from_offsets_l1_value sums
// this same plan, so a CB added anywhere else is invisible to that L1 check.
MSDACircularBufferPlan derive_circular_buffer_plan(
    const FusedMSDAOperation::operation_attributes_t& attrs,
    const FusedMSDAOperation::tensor_args_t& args,
    const MSDAShapes& s,
    BufferType value_buffer_type,
    BufferType output_buffer_type,
    DataType output_dtype) {
    const auto& value = args.value;
    const auto& attn = args.attention_weights;
    const Tensor& loc = attrs.from_offsets ? *args.sampling_offsets : *args.sampling_locations;

    const uint32_t n_d_tiles = (s.head_dim + kTileWidth - 1) / kTileWidth;
    const uint32_t attn_sticks_per_row = s.weights_packed ? 1u : s.num_levels;
    const uint32_t loc_sticks_per_row = s.locations_packed ? 1u : (s.num_levels * s.num_points);

    const uint32_t value_stick_raw = s.head_dim * value.element_size();
    const uint32_t attn_stick_raw =
        (s.weights_packed ? s.num_levels * s.num_points : s.num_points) * attn.element_size();
    const uint32_t loc_stick_raw = (s.locations_packed ? s.num_levels * s.num_points * 2u : 2u) * loc.element_size();

    const uint32_t value_stick_aligned = aligned_page_size(value_stick_raw, value_buffer_type);
    const uint32_t attn_stick_aligned = aligned_page_size(attn_stick_raw, attn.buffer()->buffer_type());
    const uint32_t loc_stick_aligned = aligned_page_size(loc_stick_raw, loc.buffer()->buffer_type());
    // Writer scratch holds one head's D-wide stick, not the whole output page.
    const uint32_t head_stick_aligned = aligned_page_size(value_stick_raw, output_buffer_type);

    uint32_t ref_stick_aligned = 0;
    if (attrs.from_offsets) {
        const auto& ref = *args.reference_points;
        ref_stick_aligned = aligned_page_size(2u * ref.element_size(), ref.buffer()->buffer_type());
    }

    const auto data_format = datatype_to_dataformat_converter(DataType::BFLOAT16);
    const auto output_format = datatype_to_dataformat_converter(output_dtype);
    const uint32_t tile_nbytes = tt::tile_size(data_format);

    const bool rm_staging = uses_rm_staging(s);
    // Row-major staging packs the four corner slots of a row back to back, so a
    // slot must be exactly D*2 bytes; alignment padding would shift the tile
    // columns compute tilizes.
    TT_FATAL(
        !rm_staging || value_stick_aligned == value_stick_raw,
        "fused_msda: row-major staging needs an unpadded value stick ({} B aligned vs {} B raw)",
        value_stick_aligned,
        value_stick_raw);

    MSDACircularBufferPlan plan;
    auto push = [&](uint8_t index, uint32_t pages, uint32_t page_size, tt::DataFormat data_format) {
        plan.push_back(MSDACircularBufferSpec{
            .index = index,
            .total_size = pages * page_size,
            .page_size = page_size,
            .data_format = data_format,
        });
    };

    const bool tiled_geometry = uses_tiled_geometry(attrs, args, s, loc_stick_aligned, attn_stick_aligned);

    // Reader-only staging arenas: reserved once and treated as flat L1 space.
    // The tiled geometry stages offsets and attn into its own row-major page.
    push(value_scratch_cb, kTileMaxRows, value_stick_aligned, datatype_to_dataformat_converter(value.dtype()));
    if (!tiled_geometry) {
        push(
            attn_scratch_cb,
            kTileMaxRows * attn_sticks_per_row,
            attn_stick_aligned,
            datatype_to_dataformat_converter(attn.dtype()));
        push(
            loc_scratch_cb,
            kTileMaxRows * loc_sticks_per_row,
            loc_stick_aligned,
            datatype_to_dataformat_converter(loc.dtype()));
    }
    if (attrs.from_offsets) {
        push(
            ref_scratch_cb,
            kTileMaxRows * s.num_refs,
            ref_stick_aligned,
            datatype_to_dataformat_converter(args.reference_points->dtype()));
    }
    // The sample stream, sized so that reader and compute can each run one
    // sampling point ahead of the other (see kGeomCbPages and friends).
    push(input_tile_cb, kInputCbGroups * n_d_tiles, tile_nbytes, data_format);
    if (rm_staging) {
        push(tiled_input_cb, kInputCbGroups * n_d_tiles, tile_nbytes, data_format);
    }
    push(scalar_tile_cb, kScalarCbPages, tile_nbytes, data_format);
    // The geometry pipeline. On V2 the sampling offset rides in column 16 of
    // the geom tiles.
    //
    // float32 reference points use float32 geom_x/geom_y tiles. Packing them
    // to bf16 first would round the point before the SFPU multiply.
    const bool ref_fp32 = attrs.from_offsets && args.reference_points->dtype() == DataType::FLOAT32;
    const auto primary_format = ref_fp32 ? datatype_to_dataformat_converter(DataType::FLOAT32) : data_format;
    const uint32_t primary_tile_nbytes = tt::tile_size(primary_format);
    // The corner word needs 24 exact bits (kernels/msda_corner_word.hpp).
    const auto corner_format = datatype_to_dataformat_converter(DataType::FLOAT32);
    const uint32_t corner_tile_nbytes = tt::tile_size(corner_format);
    if (tiled_geometry) {
        namespace tg = fused_msda_tiled_geometry;
        push(geom_rm_cb, tg::RM_CB_PAGES, tile_nbytes, data_format);
        push(geom_operand_cb, tg::RM_TILES, tile_nbytes, data_format);
        push(ref_tile_cb, tg::REF_TILE_CB_PAGES, tile_nbytes, data_format);
        push(const_cb, tg::CONST_TILES, tile_nbytes, data_format);
        push(weight_cb, tg::WEIGHT_CB_PAGES, tile_nbytes, data_format);
        push(corner_cb, tg::CORNER_CB_PAGES, corner_tile_nbytes, corner_format);
    } else {
        push(geom_x_cb, kGeomCbPages, primary_tile_nbytes, primary_format);
        push(geom_y_cb, kGeomCbPages, primary_tile_nbytes, primary_format);
        push(attn_tile_cb, kGeomCbPages, tile_nbytes, data_format);
        push(corner_cb, kCornerCbPages, corner_tile_nbytes, corner_format);
    }
    // Compute -> writer, double-buffered.
    push(output_tile_cb, 2 * n_d_tiles, tile_nbytes, output_format);
    // Writer-only scratch.
    push(output_scratch_cb, 1, head_stick_aligned, output_format);
    if (uses_split_gather(s, value.device()->arch())) {
        // Reader -> writer, one point's gather arguments at a time. Raw uint32
        // words; the format only satisfies the CB config.
        push(gather_mailbox_cb, 1, kGatherMailboxNbytes, tt::DataFormat::UInt32);
    }
    return plan;
}

uint64_t circular_buffer_bytes_per_core(const MSDACircularBufferPlan& plan) {
    uint64_t total = 0;
    for (const auto& spec : plan) {
        total += spec.total_size;
    }
    return total;
}

ProgramDescriptor FusedMSDAOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, Tensor& output) {
    ProgramDescriptor descriptor{};

    const auto s = derive_shapes(attrs, tensor_args);

    const auto& value = tensor_args.value;
    const auto& attn = tensor_args.attention_weights;
    const Tensor& loc = attrs.from_offsets ? *tensor_args.sampling_offsets : *tensor_args.sampling_locations;

    auto* device = value.device();
    const auto compute_grid = device->compute_with_storage_grid_size();

    const uint32_t n_d_tiles = (s.head_dim + kTileWidth - 1) / kTileWidth;

    const auto split = msda_work_split(attrs, s, compute_grid);
    const auto& all_cores = split.all_cores;

    // ---- page / stick sizes ----
    const auto cb_plan = derive_circular_buffer_plan(
        attrs, tensor_args, s, value.buffer()->buffer_type(), output.buffer()->buffer_type(), output.dtype());
    const uint32_t value_stick_raw = s.head_dim * value.element_size();
    const uint32_t output_page_raw = s.num_heads * s.head_dim * output.element_size();
    const uint32_t value_stick_aligned = page_size_for_cb(cb_plan, value_scratch_cb);
    // Not read back from the plan: the tiled geometry allocates no attn / offset
    // arena, but its reader still reads sticks of these sizes.
    const uint32_t attn_stick_aligned = aligned_page_size(
        (s.weights_packed ? s.num_levels * s.num_points : s.num_points) * attn.element_size(),
        attn.buffer()->buffer_type());
    const uint32_t loc_stick_aligned = aligned_page_size(
        (s.locations_packed ? s.num_levels * s.num_points * 2u : 2u) * loc.element_size(), loc.buffer()->buffer_type());
    const uint32_t ref_stick_aligned = attrs.from_offsets ? page_size_for_cb(cb_plan, ref_scratch_cb) : 0;
    const uint32_t output_page_aligned = aligned_page_size(output_page_raw, output.buffer()->buffer_type());

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

    // ---- circular buffers (see README.md §6) ----
    for (const auto& spec : cb_plan) {
        descriptor.cbs.push_back(CBDescriptor{
            .total_size = spec.total_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = spec.index,
                .data_format = spec.data_format,
                .page_size = spec.page_size,
            }}},
        });
    }
    const bool rm_staging = uses_rm_staging(s);
    const bool ref_fp32 = attrs.from_offsets && tensor_args.reference_points->dtype() == DataType::FLOAT32;
    // The mailbox CB, the semaphores and both kernels' SPLIT_GATHER must agree,
    // so all of them follow the CB plan.
    const bool split_gather =
        std::any_of(cb_plan.begin(), cb_plan.end(), [](const auto& spec) { return spec.index == gather_mailbox_cb; });
    const uint32_t gather_mailbox_nbytes = split_gather ? page_size_for_cb(cb_plan, gather_mailbox_cb) : 0u;
    constexpr uint32_t gather_ready_sem = 0;
    constexpr uint32_t gather_done_sem = 1;
    // The tiled geometry's CBs, decided by the plan like the split gather.
    const bool tiled_geometry =
        std::any_of(cb_plan.begin(), cb_plan.end(), [](const auto& spec) { return spec.index == geom_rm_cb; });
    if (split_gather) {
        descriptor.semaphores.push_back(SemaphoreDescriptor{
            .id = gather_ready_sem, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
        descriptor.semaphores.push_back(SemaphoreDescriptor{
            .id = gather_done_sem, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    }

    // ---- reader ----
    // Compile-time arg order is fixed by fused_msda_reader_common.hpp; both
    // readers read the same indices.
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
        attn_tile_cb,
        corner_cb,
        kGeomCbPages,
        static_cast<uint32_t>(rm_staging),
        static_cast<uint32_t>(split_gather),
        gather_ready_sem,
        gather_done_sem,
        gather_mailbox_cb,
        gather_mailbox_nbytes,
        static_cast<uint32_t>(tiled_geometry),
        geom_rm_cb,
        ref_tile_cb,
        const_cb,
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
    if (ref_fp32) {
        reader_desc.defines = {{"REF_FP32", "1"}};
    }
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
        attn_tile_cb,
        corner_cb,
        s.num_levels,
        s.num_points,
        static_cast<uint32_t>(attrs.from_offsets),
        static_cast<uint32_t>(rm_staging),
        tiled_input_cb,
        static_cast<uint32_t>(tiled_geometry),
        geom_rm_cb,
        geom_operand_cb,
        ref_tile_cb,
        const_cb,
        weight_cb,
    };
    // fp32 destination: the geometry builds px, which reaches the feature map's
    // extent. In a 16-bit destination the ulp at 200 is 1.0, so floor(px) and
    // the fraction it feeds are lost outright on the larger levels and bilinear
    // degrades to nearest-neighbour. The geometry uses all four DST slots a
    // 32-bit destination allows; there is no spare.
    //
    // This kernel also drives packer L1 accumulation, which #28800 records as
    // unsupported alongside a 32-bit destination (docs/L1_ACCUMULATION_FP32_ANALYSIS.md;
    // sdpa_fw disables L1 acc when the flag is on). Validated on Blackhole;
    // Wormhole is unverified.
    ComputeConfigDescriptor compute_cfg{};
    compute_cfg.fp32_dest_acc_en = true;
    // copy_tile through SrcA keeps 10 of a float32 reference point's 23 mantissa
    // bits; unpacking straight into the 32-bit destination keeps all of them.
    if (ref_fp32) {
        compute_cfg.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
        compute_cfg.unpack_to_dest_mode[geom_x_cb] = UnpackToDestMode::UnpackToDestFp32;
        compute_cfg.unpack_to_dest_mode[geom_y_cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    // HiFi4 (the default) is required by the tiled geometry: its matmuls must
    // reproduce bf16 data exactly, and only HiFi4 multiplies every mantissa bit
    // of both sources: in0 (the data) is SrcB, in1 (the bf16-exact constants)
    // is SrcA (tech_reports/matrix_engine/matrix_engine.md).
    compute_cfg.math_fidelity = MathFidelity::HiFi4;
    compute_desc.config = compute_cfg;

    // ---- writer (shared by V1 and V2) ----
    // Compile-time arg order is fixed by writer_msda.cpp.
    KernelDescriptor::CompileTimeArgs writer_ct{
        output_tile_cb,
        output_scratch_cb,
        output_page_aligned,
        s.head_dim,
        static_cast<uint32_t>(split_gather),
        gather_ready_sem,
        gather_done_sem,
        gather_mailbox_cb,
        gather_mailbox_nbytes,
        s.num_levels * s.num_points,
        s.num_heads,
        s.num_keys,
        static_cast<uint32_t>(s.value_packed),
        value_stick_aligned,
        static_cast<uint32_t>(value.buffer()->page_size()),
    };
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*value.buffer()).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = std::string(kKernelDir) + "dataflow/writer_msda.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    // ---- per-core runtime args ----
    // Plain addresses, not Buffer* bindings: override_runtime_arguments rewrites
    // every arg on a cache hit, addresses included. num_valid_queries changes the
    // tile lists and is excluded from the program hash, so buffer bindings alone
    // cannot refresh a hit.
    // Order must match kReaderKernel, kComputeKernel, kWriterKernel.
    auto rt = msda_runtime_args(attrs, tensor_args, output, s, split, tiled_geometry);
    for (size_t i = 0; i < split.cores.size(); ++i) {
        const auto& core = split.cores[i];
        reader_desc.runtime_args.emplace_back(core, std::move(rt.reader[i]));
        compute_desc.runtime_args.emplace_back(core, std::move(rt.compute[i]));
        writer_desc.runtime_args.emplace_back(core, std::move(rt.writer[i]));
    }

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
    const auto cb_plan = derive_circular_buffer_plan(
        attrs,
        tensor_args,
        s,
        tensor_args.value.buffer()->buffer_type(),
        output.buffer()->buffer_type(),
        output.dtype());
    const bool tiled_geometry =
        std::any_of(cb_plan.begin(), cb_plan.end(), [](const auto& spec) { return spec.index == geom_rm_cb; });
    const auto rt = msda_runtime_args(attrs, tensor_args, output, s, split, tiled_geometry);
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
