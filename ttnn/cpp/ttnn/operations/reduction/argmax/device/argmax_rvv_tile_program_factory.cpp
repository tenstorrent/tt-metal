// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// TILE-layout last-dim argmax on the pack RISC's RVV (Zve32f) unit — Blackhole
// only, single- or multi-core, selected by ArgMaxPath::Rvv (select_argmax_path
// in argmax.cpp, where the routing measurements sit next to kSfpuMinRows).
// kernels/argmax_rvv_tile_compute.cpp has the algorithm and semantics. Unlike
// the other argmax paths this one launches a compute kernel: unpack/math are
// no-ops and the pack thread runs the whole scan, so the dataflow RISC only
// streams tiles and writes results.
//
// The RVV scan visits every tile once PER VALID ROW, so a pass is linear in
// both w_tiles and H: one core over an 8192-tile row measures ~0.043 us/tile
// for the first row plus ~0.019 us/tile per further row. The work split is the
// SFPU path's (described in argmax_sfpu_tile_program_factory.cpp), except that
// core j emits one (global index, max value) candidate per valid row instead
// of a per-column candidate tile.
//
// The merge reuses that path's exchange PROTOCOL (per-core slots plus two
// cumulative semaphores) but NOT its comparator. This path exists to be
// bit-identical to the scalar readers, so the merge runs bfloat16_greater's
// sign-magnitude BIT-PATTERN total order with a smallest-global-index
// tie-break, never an IEEE compare (reader_argmax_rvv_tile.cpp). Unifying the
// two merges would silently break bit-exactness.
//
//     num_cores = ceil(sqrt(w_tiles * (h_logical + 2)) / 3)
//
// Both paths pay ~0.44 us/core of per-program dispatch, so both optima have
// the form sqrt(per-core-work / floor); only the work differs. The SFPU pass
// is flat in H, so sqrt(1.5 * w_tiles) suits it and is wrong for a per-row
// scan. That fixes the rule's SHAPE and nothing else: the 2 and the 3 are
// FITTED PARAMETERS from a grid search scored on the worst per-shape ratio to
// the measured optimum over V = 4096..262144 x H = 1/8/32, core counts pinned
// with sub_core_grids and swept 1..130.
// Do NOT "correct" them against the per-tile costs above — the closed form
// those imply, sqrt(w_tiles * (H + 1.26)) / 4.81, differs in both constants
// and does not track the measurements: at H == 1, V == 32768 it asks for 11
// cores where the swept optimum is 24.
//
// tests/ttnn/unit_tests/operations/reduce/test_argmax_path_crossover_bench.py
// re-measures those curves (V_SWEEP / H_SWEEP / CORE_SWEEP are the knobs).
// Every swept shape lands within 8% of its own optimum, worst at H == 32,
// which the automatic route sends to the SFPU anyway
// (V == 32768, H == 32: 63 cores / 84.6 us against an optimum of 44 / 78.6).
// The curves turn back upward well before the grid fills (V == 32768, H == 1:
// 11.3 us on 24 cores, 50.7 us on 111), so "use the whole grid" is not a safe
// default for either path.
//
// An explicit sub_core_grids overrides the heuristic (capped by w_tiles only);
// a single-core grid forces the single-core variant, which skips the exchange
// buffer entirely.

#include "argmax_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <algorithm>
#include <cmath>

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor ArgMaxRvvTileProgramFactory::create_descriptor(
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
        // Fitted, H-aware; see the header. Not the SFPU path's flat-in-H rule.
        const uint32_t want = static_cast<uint32_t>(
            std::ceil(std::sqrt(static_cast<double>(w_tiles) * (static_cast<double>(h_logical) + 2.0)) / 3.0));
        num_cores = std::min<uint32_t>({want, static_cast<uint32_t>(cores.size()), w_tiles});
    }
    TT_FATAL(num_cores >= 1, "the argmax RVV path requires at least one core");
    cores.resize(num_cores);

    std::vector<CoreRange> core_ranges_vec;
    core_ranges_vec.reserve(num_cores);
    for (const auto& c : cores) {
        core_ranges_vec.emplace_back(c, c);
    }
    const CoreRangeSet all_cores(core_ranges_vec);

    // Contiguous slices of the reduction dim's tiles, remainder over the leading cores.
    const uint32_t w_base = w_tiles / num_cores;
    const uint32_t w_rem = w_tiles % num_cores;
    auto slice_count = [&](uint32_t j) { return w_base + (j < w_rem ? 1u : 0u); };
    auto slice_start = [&](uint32_t j) { return j * w_base + std::min(j, w_rem); };
    const uint32_t w_count_max = slice_count(0);

    // Chunked double-buffered input streaming: the compute-side scan of chunk k
    // overlaps the NOC staging of chunk k+1. Uniform across cores so the CB layout
    // (and therefore the exchange-buffer address) is identical everywhere.
    const uint32_t chunk_pages = std::min<uint32_t>(64, w_count_max);
    const uint32_t in_cb_pages = 2 * chunk_pages;

    ProgramDescriptor desc;

    constexpr auto cb_in = tt::CBIndex::c_0;         // input tiles (ring)
    constexpr auto cb_res_idx = tt::CBIndex::c_1;    // per-pass index results (u32[32])
    constexpr auto cb_res_val = tt::CBIndex::c_2;    // per-pass maxval results (bf16[32])
    constexpr auto cb_stage_idx = tt::CBIndex::c_3;  // output-page staging (indices)
    constexpr auto cb_stage_val = tt::CBIndex::c_4;  // output-page staging (max values)
    constexpr auto cb_xchg = tt::CBIndex::c_5;       // cross-core candidate exchange (raw scratch)

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
    constexpr uint32_t res_idx_page = 32 * sizeof(uint32_t);
    constexpr uint32_t res_val_page = 32 * sizeof(uint16_t);
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * res_idx_page,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_res_idx),
            .data_format = tt::DataFormat::UInt32,
            .page_size = res_idx_page,
        }}},
    });
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
    // Exchange buffer: one 256 B slot per core (u32 idx[32] + u32 val[32]), allocated
    // identically everywhere so a worker's local address equals the gather core's, and
    // declared even single-core (unused there) so both variants share one CB layout.
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

    // Reader (data-movement RISC): streams tiles, then merges/writes results.
    std::vector<uint32_t> reader_ct_args = {
        static_cast<uint32_t>(cb_in),
        static_cast<uint32_t>(cb_res_idx),
        static_cast<uint32_t>(cb_res_val),
        static_cast<uint32_t>(cb_stage_idx),
        static_cast<uint32_t>(cb_stage_val),
        static_cast<uint32_t>(cb_xchg),
        src_page_size,
        chunk_pages,
        in_cb_pages,
        w_tiles,
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
    };
    TensorAccessorArgs(input).append_to(reader_ct_args);
    TensorAccessorArgs(output).append_to(reader_ct_args);
    if (has_maxval) {
        TensorAccessorArgs(tensor_args.optional_maxval_tensor->mesh_tensor()).append_to(reader_ct_args);
    } else {
        // Contract with reader_argmax_rvv_tile.cpp: it parses exactly THREE
        // TensorAccessorArgs blocks (src, dst, val) unconditionally, because the
        // constexpr offset chain (next_compile_time_args_offset) cannot be made
        // conditional. With no maxval tensor this duplicate of the output's args
        // fills the third slot; the has_maxval compile-time arg (== false) guards
        // every use of it, so it is never dereferenced. Keep the two sides in sync.
        TensorAccessorArgs(output).append_to(reader_ct_args);
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_rvv_tile.cpp";
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

    // Compute kernel: unpack/math no-op, pack thread runs the RVV scan.
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/argmax_rvv_tile_compute.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {
        static_cast<uint32_t>(cb_in),
        static_cast<uint32_t>(cb_res_idx),
        static_cast<uint32_t>(cb_res_val),
        chunk_pages,
        h_tiles,
        h_logical,
        outer_dim_units,
    };
    // enable_trisc2_rvv: compile TRISC2 (pack) with Zve32f — the in-tree opt-in that
    // makes the RVV scan build in a stock tree.
    compute_desc.config = ComputeConfigDescriptor{.enable_trisc2_rvv = true};
    for (uint32_t j = 0; j < num_cores; ++j) {
        compute_desc.emplace_runtime_args(cores[j], {slice_count(j)});
    }
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
