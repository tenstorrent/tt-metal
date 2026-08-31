// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_tiled_unaligned_program_factory.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

struct BandGeometry {
    std::vector<uint32_t> in_wt;       // tiles per band per input
    std::vector<uint32_t> in_w_bytes;  // logical row bytes per input
    uint32_t total_in_wt = 0;
    uint32_t out_wt = 0;
    uint32_t num_bands = 0;
    uint32_t element_size = 0;
};

BandGeometry compute_band_geometry(const std::vector<Tensor>& input_tensors) {
    BandGeometry g;
    g.element_size = input_tensors[0].element_size();
    uint32_t out_w_logical = 0;
    g.in_wt.reserve(input_tensors.size());
    g.in_w_bytes.reserve(input_tensors.size());
    for (const auto& t : input_tensors) {
        const uint32_t wt = t.padded_shape()[-1] / TILE_WIDTH;
        g.in_wt.push_back(wt);
        g.in_w_bytes.push_back(t.logical_shape()[-1] * g.element_size);
        g.total_in_wt += wt;
        out_w_logical += t.logical_shape()[-1];
    }
    g.out_wt = tt::div_up(out_w_logical, TILE_WIDTH);
    const auto& first = input_tensors[0];
    const uint32_t padded_rows = first.physical_volume() / first.padded_shape()[-1];
    g.num_bands = padded_rows / TILE_HEIGHT;
    return g;
}

// Per-core CB bytes the factory allocates; keep in sync with create_descriptor.
uint64_t required_cb_bytes(const BandGeometry& g, uint32_t tile_bytes) {
    // c_in double-buffered, c_rm and c_asm single (whole band each), c_out double-buffered.
    return static_cast<uint64_t>(3 * g.total_in_wt + 3 * g.out_wt) * tile_bytes;
}

// Static conditions only (layout, dtype, dims, alignment); no L1 budget. Shared by the
// eligibility check and the post-selection sanity check in create_descriptor, which runs
// after the output buffer is allocated and therefore must not re-evaluate the budget.
bool is_supported_tiled_unaligned_config(
    const std::vector<Tensor>& input_tensors,
    uint32_t normalized_dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    if (groups != 1 || input_tensors.size() < 2) {
        return false;
    }
    // Runtime args stay well within limits for any practical input count; cap defensively.
    if (input_tensors.size() > 32) {
        return false;
    }
    if (output_mem_config.is_sharded()) {
        return false;
    }
    const auto& first = input_tensors[0];
    // The writer kernel's row assembly (tt_memmove's CPU fallback + tail stores) isn't
    // cache-coherent on Quasar (see the TODO(ARCH_QUASAR) note in common/kernels/common.hpp),
    // so gate this factory off until that's resolved.
    if (first.device()->arch() == tt::ARCH::QUASAR) {
        return false;
    }
    const uint32_t rank = first.logical_shape().rank();
    if (rank < 2 || normalized_dim != rank - 1) {
        return false;
    }
    const DataType dtype = first.dtype();
    if (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT32) {
        return false;
    }
    bool any_unaligned = false;
    for (const auto& t : input_tensors) {
        if (t.layout() != Layout::TILE || t.is_sharded() || t.storage_type() != StorageType::DEVICE) {
            return false;
        }
        const auto& tile = t.tensor_spec().tile();
        if (tile.get_height() != TILE_HEIGHT || tile.get_width() != TILE_WIDTH) {
            return false;
        }
        any_unaligned |= (t.logical_shape()[-1] != t.padded_shape()[-1]);
    }
    if (!any_unaligned) {
        // Tile-aligned width concat is already handled natively by ConcatProgramFactory.
        return false;
    }

    return compute_band_geometry(input_tensors).num_bands > 0;
}

}  // namespace

bool can_use_tiled_unaligned_concat(
    const std::vector<Tensor>& input_tensors,
    uint32_t normalized_dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool output_already_allocated) {
    using namespace ttnn::operations::data_movement;

    if (!is_supported_tiled_unaligned_config(input_tensors, normalized_dim, groups, output_mem_config)) {
        return false;
    }

    const BandGeometry g = compute_band_geometry(input_tensors);
    const auto& first = input_tensors[0];
    const DataType dtype = first.dtype();
    const tt::DataFormat df = datatype_to_dataformat_converter(dtype);
    const uint32_t tile_bytes = tt::tile_size(df);

    // The CBs must fit in the L1 window that is actually free right now: get_max_l1_space
    // reads the allocator's lowest occupied compute address, which accounts for every live
    // L1 buffer -- this op's inputs as well as unrelated tensors a model keeps resident.
    // When called before the output buffer exists (the routing hook in ttnn::concat), its
    // worst-case per-bank footprint must be reserved on top; inside the device op the
    // launch infra has already allocated it, so the free window accounts for it and
    // reserving it again would double-count (and mis-route eligible concats).
    const uint64_t free_l1 = get_max_l1_space(first);
    uint64_t pending_output_bytes = 0;
    if (!output_already_allocated) {
        ttnn::Shape out_padded_shape = first.padded_shape();
        out_padded_shape[-1] = g.out_wt * TILE_WIDTH;
        pending_output_bytes = get_pending_l1_output_reservation(
            first, out_padded_shape, output_mem_config, dtype, Layout::TILE, /*require_constructible=*/true);
    }
    return required_cb_bytes(g, tile_bytes) + pending_output_bytes <= free_l1;
}

ProgramDescriptor ConcatTiledUnalignedProgramFactory::create_descriptor(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input_tensors = tensor_args.input_tensors;
    Tensor& output = tensor_return_value;
    const uint32_t num_tensors = input_tensors.size();

    TT_FATAL(
        is_supported_tiled_unaligned_config(
            input_tensors,
            operation_attributes.dim,
            operation_attributes.groups,
            operation_attributes.output_mem_config),
        "ConcatTiledUnalignedProgramFactory selected for an unsupported configuration");

    const BandGeometry g = compute_band_geometry(input_tensors);
    const tt::DataFormat df = datatype_to_dataformat_converter(input_tensors[0].dtype());
    const uint32_t tile_bytes = tt::tile_size(df);
    const uint32_t tile_row_bytes = TILE_WIDTH * g.element_size;
    const uint32_t out_row_bytes = g.out_wt * TILE_WIDTH * g.element_size;
    uint32_t logical_row_bytes = 0;
    for (uint32_t b : g.in_w_bytes) {
        logical_row_bytes += b;
    }
    const uint32_t tail_bytes = out_row_bytes - logical_row_bytes;
    const bool fp32_lossless = input_tensors[0].dtype() == DataType::FLOAT32;

    TT_FATAL(
        output.padded_shape()[-1] == g.out_wt * TILE_WIDTH,
        "Output padded width {} does not match expected {}",
        output.padded_shape()[-1],
        g.out_wt * TILE_WIDTH);

    auto* device = output.device();
    const auto grid_size = device->compute_with_storage_grid_size();
    const CoreRangeSet default_grid(CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1}));
    const CoreRangeSet available_grid = operation_attributes.sub_core_grids.value_or(default_grid);
    auto [ncores, all_cores, core_range, core_range_cliff, nbands_per_core, nbands_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, g.num_bands);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;    // tiled input bands, all tensors sequentially
    constexpr uint32_t cb_rm = tt::CBIndex::c_1;    // untilized 32x32 blocks, one page per input tile
    constexpr uint32_t cb_asm = tt::CBIndex::c_2;   // assembled row-major output band
    constexpr uint32_t cb_out = tt::CBIndex::c_16;  // tiled output band

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t index, uint32_t num_pages) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_pages * tile_bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index),
                .data_format = df,
                .page_size = tile_bytes,
            }}},
        });
    };
    add_cb(cb_in, 2 * g.total_in_wt);
    add_cb(cb_rm, g.total_in_wt);
    add_cb(cb_asm, g.out_wt);
    add_cb(cb_out, 2 * g.out_wt);

    // Reader: per band, each input's tile-row back to back.
    KernelDescriptor reader_desc;
    {
        KernelDescriptor::CompileTimeArgs ct_args = {cb_in, num_tensors};
        for (const auto& t : input_tensors) {
            TensorAccessorArgs(*t.buffer()).append_to(ct_args);
        }
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_concat_tiled_unaligned.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = all_cores;
        reader_desc.compile_time_args = std::move(ct_args);
        reader_desc.config = ReaderConfigDescriptor{};
    }

    // Writer doubles as the band assembler: gathers logical rows out of the untilized blocks,
    // zero-fills the output width padding, then ships the retilized band.
    KernelDescriptor writer_desc;
    {
        KernelDescriptor::CompileTimeArgs ct_args = {
            cb_rm, cb_asm, cb_out, num_tensors, g.out_wt, g.total_in_wt, tile_row_bytes, out_row_bytes, tail_bytes};
        TensorAccessorArgs(*output.buffer()).append_to(ct_args);
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "writer_concat_tiled_unaligned.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = all_cores;
        writer_desc.compile_time_args = std::move(ct_args);
        writer_desc.config = WriterConfigDescriptor{};
    }

    KernelDescriptor compute_desc;
    {
        std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
        if (fp32_lossless) {
            unpack_to_dest_mode[cb_in] = UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[cb_asm] = UnpackToDestMode::UnpackToDestFp32;
        }
        compute_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/compute/"
            "concat_untilize_retilize.cpp";
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = all_cores;
        compute_desc.compile_time_args = {
            cb_in, cb_rm, cb_asm, cb_out, g.total_in_wt, g.out_wt, static_cast<uint32_t>(fp32_lossless)};
        compute_desc.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_lossless,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        };
    }

    const bool has_cliff = !core_range_cliff.empty();
    const auto cores = corerange_to_cores(all_cores);
    uint32_t band_start = 0;
    for (uint32_t i = 0; i < ncores; ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t nbands = (has_cliff && i == ncores - 1) ? nbands_per_core_cliff : nbands_per_core;

        KernelDescriptor::RTArgList reader_args;
        reader_args.reserve(2 + 2 * num_tensors);
        reader_args.push_back(nbands);
        reader_args.push_back(band_start);
        for (const auto& t : input_tensors) {
            reader_args.push_back(t.buffer());
        }
        reader_args.append(g.in_wt);
        reader_desc.emplace_runtime_args(core, reader_args);

        KernelDescriptor::RTArgList writer_args;
        writer_args.reserve(3 + 2 * num_tensors);
        writer_args.push_back(output.buffer());
        writer_args.push_back(nbands);
        writer_args.push_back(band_start);
        writer_args.append(g.in_wt);
        writer_args.append(g.in_w_bytes);
        writer_desc.emplace_runtime_args(core, writer_args);

        compute_desc.emplace_runtime_args(core, {nbands});

        band_start += nbands;
    }
    TT_FATAL(band_start == g.num_bands, "Band split covered {} of {} bands", band_start, g.num_bands);

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::prim
