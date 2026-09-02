// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// TILE-layout last-dim argmax on the SFPU — Blackhole, single- or multi-core.
// Selected by ArgMaxPath::Sfpu; ttnn::argmax decides that on its own (see
// select_argmax_path in argmax.cpp).
// See kernels/argmax_sfpu_tile_compute.cpp for the algorithm and
// the (documented, silicon-measured) special-value semantics.
//
// Measurements and the routing-threshold rationale live next to kSfpuMinRows
// in argmax.cpp.
//
// Work split: phase 1 reduces all 32 rows of a tile-row lane-parallel, so a
// tile-row pass costs the same whether 1 or 32 rows are valid — the batch-
// shape win. Multicore splits the REDUCTION dim's tiles across cores: core j
// reduces slice [w_start_j, w_start_j + w_count_j) of every tile-row pass
// and finishes it with a per-row phase 2 on its dataflow RISC; the gather
// core (core 0) then merges the per-core per-row candidates with the same
// lexicographic rule. The cross-core traffic is 256 B per core per pass —
// per-row scalar candidates, never tiles.
//
// Core-count heuristic: per-core phase 1 is a fixed ~0.60 us/tile that does
// NOT depend on H (the flat-in-H measurement in argmax.cpp), while every extra
// core adds a gather-merge pass and ~0.44 us of per-program dispatch. The
// optimum therefore sits near sqrt(w_tiles) and does not move with H; we use
// ceil(sqrt(1.5 * w_tiles)) capped by the grid and by w_tiles, which lands
// within 0.87x-1.04x of this path's own per-shape optimum at every point
// swept.
//
// This is deliberately NOT the RVV path's formula: that scan costs per ROW,
// so its optimum grows with H and it fits ceil(sqrt(w_tiles * (H + 2)) / 3)
// instead (see argmax_rvv_tile_program_factory.cpp).
//
// An explicit sub_core_grids overrides the heuristic (capped by w_tiles
// only) — pass a single-core grid to force the single-core variant.

#include "argmax_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <algorithm>
#include <cmath>

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor ArgMaxSfpuTileProgramFactory::create_descriptor(
    const ArgmaxParams& operation_attributes, const ArgmaxInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    const bool has_maxval = tensor_args.optional_maxval_tensor.has_value();

    const auto& padded_shape = input.padded_shape();
    const uint32_t rank = padded_shape.size();
    const uint32_t logical_rank = input.logical_shape().size();
    const uint32_t tile_width = input.tensor_spec().tile().get_width();
    const uint32_t tile_height = input.tensor_spec().tile().get_height();

    const uint32_t w_tiles = padded_shape[rank - 1] / tile_width;
    const uint32_t h_tiles = padded_shape[rank - 2] / tile_height;
    const uint32_t w_logical = input.logical_shape()[logical_rank - 1];
    const uint32_t h_logical = logical_rank > 1 ? input.logical_shape()[logical_rank - 2] : 1;
    const uint32_t outer_dim_units = input.logical_volume() / (h_logical * w_logical);
    const uint32_t num_passes = outer_dim_units * h_tiles;

    const uint32_t src_page_size = input.tensor_spec().compute_page_size_bytes();
    const uint32_t dst_page_size = output.tensor_spec().compute_page_size_bytes();
    const uint32_t val_page_size =
        has_maxval ? tensor_args.optional_maxval_tensor->mesh_tensor().tensor_spec().compute_page_size_bytes()
                   : dst_page_size;
    const uint32_t out_page_elems = dst_page_size / sizeof(uint32_t);

    // ---- core selection -----------------------------------------------------
    const IDevice* device = &output.mutable_device();
    std::vector<CoreCoord> cores;
    uint32_t num_cores = 0;
    if (operation_attributes.sub_core_grids.has_value()) {
        const auto& grids = operation_attributes.sub_core_grids.value();
        cores = corerange_to_cores(grids, std::nullopt, true);
        num_cores = std::min<uint32_t>(static_cast<uint32_t>(cores.size()), w_tiles);
    } else {
        const auto grid = device->compute_with_storage_grid_size();
        const CoreRangeSet full_grid(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
        cores = corerange_to_cores(full_grid, std::nullopt, true);
        const uint32_t want = static_cast<uint32_t>(std::ceil(std::sqrt(1.5 * static_cast<double>(w_tiles))));
        num_cores = std::min<uint32_t>({want, static_cast<uint32_t>(cores.size()), w_tiles});
    }
    TT_FATAL(num_cores >= 1, "the argmax SFPU path requires at least one core");
    cores.resize(num_cores);

    std::vector<CoreRange> core_ranges_vec;
    core_ranges_vec.reserve(num_cores);
    for (const auto& c : cores) {
        core_ranges_vec.emplace_back(c, c);
    }
    const CoreRangeSet all_cores(core_ranges_vec);

    // Contiguous slices of the reduction dim's tiles, remainder spread over
    // the leading cores.
    const uint32_t w_base = w_tiles / num_cores;
    const uint32_t w_rem = w_tiles % num_cores;
    auto slice_count = [&](uint32_t j) { return w_base + (j < w_rem ? 1u : 0u); };
    auto slice_start = [&](uint32_t j) { return j * w_base + std::min(j, w_rem); };
    const uint32_t w_count_max = slice_count(0);

    // Chunked double-buffered input streaming: the SFPU scan of chunk k
    // overlaps the NOC staging of chunk k+1. Uniform across cores so the CB
    // layout (and therefore the exchange-buffer address) is identical
    // everywhere.
    const uint32_t chunk_pages = std::min<uint32_t>(64, w_count_max);
    const uint32_t in_cb_pages = 2 * chunk_pages;

    ProgramDescriptor desc;

    constexpr auto cb_in = tt::CBIndex::c_0;         // input tiles (ring)
    constexpr auto cb_res_val = tt::CBIndex::c_1;    // phase-1 max-val candidate tile (bf16)
    constexpr auto cb_res_idx = tt::CBIndex::c_2;    // phase-1 win-tile candidate tile (u32)
    constexpr auto cb_xchg = tt::CBIndex::c_3;       // cross-core candidate exchange (raw scratch)
    constexpr auto cb_stage_idx = tt::CBIndex::c_4;  // output-page staging (indices)
    constexpr auto cb_stage_val = tt::CBIndex::c_5;  // output-page staging (max values)

    const tt::DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_cb_pages * src_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_in),
            .data_format = in_df,
            .page_size = src_page_size,
        }}},
    });
    const uint32_t res_val_page = tile_width * tile_height * sizeof(uint16_t);
    const uint32_t res_idx_page = tile_width * tile_height * sizeof(uint32_t);
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * res_val_page,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_res_val),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = res_val_page,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * res_idx_page,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_res_idx),
            .data_format = tt::DataFormat::UInt32,
            .page_size = res_idx_page,
        }}},
    });
    // Exchange buffer: one 256 B slot per core (u32 idx[32] + u32 val[32]).
    // Allocated identically on every core so a worker's local address equals
    // the gather core's.
    constexpr uint32_t xchg_slot_bytes = 64 * sizeof(uint32_t);
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cores * xchg_slot_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_xchg),
            .data_format = tt::DataFormat::UInt32,
            .page_size = num_cores * xchg_slot_bytes,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = dst_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_stage_idx),
            .data_format = tt::DataFormat::UInt32,
            .page_size = dst_page_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = val_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_stage_val),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = val_page_size,
        }}},
    });

    // Semaphores (multicore only): done = workers -> gather (cumulative),
    // start = gather -> workers slot-reuse credit (cumulative).
    constexpr uint32_t done_sem_id = 0;
    constexpr uint32_t start_sem_id = 1;
    if (num_cores > 1) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = done_sem_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = start_sem_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
    }

    const CoreCoord gather_phys = device->worker_core_from_logical_core(cores[0]);

    // ---- reader (dataflow RISC): streaming + phase 2 + gather merge ---------
    std::vector<uint32_t> reader_ct_args = {
        static_cast<uint32_t>(cb_in),
        static_cast<uint32_t>(cb_res_val),
        static_cast<uint32_t>(cb_res_idx),
        static_cast<uint32_t>(cb_xchg),
        static_cast<uint32_t>(cb_stage_idx),
        static_cast<uint32_t>(cb_stage_val),
        src_page_size,
        chunk_pages,
        in_cb_pages,
        h_tiles,
        h_logical,
        outer_dim_units,
        out_page_elems,
        dst_page_size,
        val_page_size,
        static_cast<uint32_t>(has_maxval),
        num_cores,
        static_cast<uint32_t>(gather_phys.x),
        static_cast<uint32_t>(gather_phys.y),
        done_sem_id,
        start_sem_id,
        w_tiles,
    };
    TensorAccessorArgs(input).append_to(reader_ct_args);
    TensorAccessorArgs(output).append_to(reader_ct_args);
    if (has_maxval) {
        TensorAccessorArgs(tensor_args.optional_maxval_tensor->mesh_tensor()).append_to(reader_ct_args);
    } else {
        // Placeholder — contract with reader_argmax_sfpu_tile.cpp: the kernel
        // unconditionally parses exactly THREE TensorAccessorArgs blocks at
        // compile time (src, dst, val), because the constexpr offset chain
        // (next_compile_time_args_offset) cannot be made conditional. When no
        // maxval tensor is supplied, this copy of the output's accessor args
        // fills the third slot so the arg counts line up; the has_maxval
        // compile-time arg (== false) guards every use of the resulting
        // accessor, so it is never dereferenced. Keep the two sides in sync.
        TensorAccessorArgs(output).append_to(reader_ct_args);
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_sfpu_tile.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};
    for (uint32_t j = 0; j < num_cores; ++j) {
        KernelDescriptor::RTArgList args;
        args.push_back(input);
        args.push_back(output);
        if (has_maxval) {
            args.push_back(tensor_args.optional_maxval_tensor->mesh_tensor());
        }
        args.push_back(j);
        args.push_back(slice_start(j));
        args.push_back(slice_count(j));
        if (j == 0) {
            // Gather core only: physical coords of every core (slot order),
            // used to return per-pass slot-reuse credits.
            for (uint32_t k = 0; k < num_cores; ++k) {
                const CoreCoord phys = device->worker_core_from_logical_core(cores[k]);
                args.push_back(static_cast<uint32_t>(phys.x));
                args.push_back(static_cast<uint32_t>(phys.y));
            }
        }
        reader_desc.emplace_runtime_args(cores[j], args);
    }
    desc.kernels.push_back(std::move(reader_desc));

    // ---- compute (SFPU): phase 1 --------------------------------------------
    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/argmax_sfpu_tile_compute.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {
        static_cast<uint32_t>(cb_in),
        static_cast<uint32_t>(cb_res_val),
        static_cast<uint32_t>(cb_res_idx),
        num_passes,
    };
    // fp32 DST accumulation: bf16 inputs widen exactly into fp32 DST slots,
    // and the uint32 win-tile accumulator needs 32-bit DST.
    compute_desc.config = ComputeConfigDescriptor{.fp32_dest_acc_en = true};
    for (uint32_t j = 0; j < num_cores; ++j) {
        compute_desc.emplace_runtime_args(cores[j], {slice_count(j)});
    }
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
