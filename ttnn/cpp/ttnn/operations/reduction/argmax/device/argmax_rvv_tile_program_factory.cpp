// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// TILE-layout last-dim argmax on the pack RISC's RVV (Zve32f) unit — Blackhole
// only, single- or multi-core. Selected by ArgMaxEngine::Rvv; ttnn::argmax
// decides that on its own (see select_argmax_engine in argmax.cpp). See
// kernels/argmax_rvv_tile_compute.cpp for the algorithm and semantics notes.
// Unlike the other argmax paths, this one launches a compute kernel: the
// unpack/math threads are no-ops and the pack thread does the whole scan, so
// the dataflow RISC only streams tiles and writes results.
//
// Work split: the RVV scan visits every tile once PER VALID ROW, so a pass is
// linear in both w_tiles and H (measured on one core over an 8192-tile row:
// ~0.043 us/tile for the first row plus ~0.019 us/tile per further row --
// see ArgMaxEngine::Rvv in argmax_device_operation_types.hpp). That is why
// the core-count heuristic below is H-aware and the SFPU engine's is not.
// Multicore splits the REDUCTION dim's tiles across cores exactly the way the
// SFPU engine does: core j scans slice
// [w_start_j, w_start_j + w_count_j) of every tile-row pass and produces one
// (global index, max value) candidate per valid row; the gather core (core 0)
// then merges the per-core per-row candidates. The cross-core traffic is
// 256 B per core per pass — per-row scalar candidates, never tiles.
//
// The merge reuses the SFPU engine's exchange PROTOCOL (per-core slots plus
// two cumulative semaphores) but NOT its comparator: this engine's whole
// reason for existing is that it is bit-identical to the scalar readers, so
// the cross-core merge runs bfloat16_greater's sign-magnitude BIT-PATTERN
// total order with a smallest-global-index tie-break, not an IEEE compare
// (see reader_argmax_rvv_tile.cpp).
//
// Core-count heuristic — DERIVED FROM MEASUREMENT, and deliberately NOT the
// SFPU engine's. Both engines pay a per-core cost that grows linearly in the
// core count (a shared ~0.44 us/core per-program dispatch floor, measured as
// the slope of every curve past its knee), so both optima have the form
// sqrt(per-core-work / floor). What differs is the work: the SFPU pass is
// flat in H, so sqrt(1.5 * w_tiles) is right for it, while this engine's scan
// is per ROW, so its optimum must grow with H too. Using the SFPU formula
// here cost up to 1.85x at H == 1 (see below).
//
// The fitted rule is
//
//     num_cores = ceil(sqrt(w_tiles * (h_logical + 2)) / 3)
//
// capped by the grid and by w_tiles. The `+ 2` is the per-tile cost that does
// NOT scale with H (NOC streaming and chunk-loop overhead) expressed in
// row-equivalents; the 9 = 3^2 in the denominator is the ratio of that
// per-row-per-tile scan cost to the per-core dispatch floor.
//
// Measured on a Blackhole p150 (13x10 = 130-core compute grid) by trace
// replay of throughput-mode captures -- per-op DEVICE time with host dispatch
// amortized away, NOT single-op latency; methodology and regeneration:
// tests/ttnn/unit_tests/operations/reduce/_argmax_engine_crossover_bench.py.
// Core counts were pinned with sub_core_grids and swept 1..130 at every
// (V, H) below; "opt" is the best time over that sweep, "old"/"new" are the
// times this factory's default lands on with the SFPU formula and with the
// formula above:
//
//   V         H    opt cores / us     OLD cores / us     NEW cores / us   new/opt
//   4096      1        6 /   4.4         14 /   8.2         7 /   4.5      1.02
//   4096      8       12 /  11.8         14 /  11.5        12 /  11.8      1.00
//   4096     32       16 /  38.6         14 /  39.5        22 /  39.9      1.03
//   16384     1       12 /   7.9         28 /  12.7        14 /   8.2      1.04
//   16384     8       28 /  18.6         28 /  18.6        24 /  19.2      1.03
//   16384    32       28 /  60.5         28 /  60.5        44 /  63.2      1.05
//   32768     1       24 /  11.3         40 /  18.5        19 /  11.5      1.02
//   32768     8       53 /  25.2         40 /  25.7        34 /  25.5      1.01
//   32768    32       44 /  78.6         40 /  79.1        63 /  84.6      1.08
//   131072    1       40 /  26.3         79 /  44.7        37 /  26.9      1.02
//   131072    8       80 /  54.8         79 /  54.9        68 /  56.6      1.03
//   131072   32       80 / 148.0        111 / 148.5       125 / 158.0      1.07
//   262144    1       28 /  47.3        111 /  74.3        53 /  48.2      1.02
//   262144    8       48 /  80.5        111 /  86.3        96 /  84.1      1.05
//   262144   32      130 / 215.2        111 / 215.9       130 / 215.2      1.00
//
// OLD = the SFPU formula this used to copy; NEW = the formula above, both
// re-measured. Worst case goes from 1.85x off the per-shape optimum to 1.08x,
// and every shape is now within 8%. What it buys is the H == 1 column
// (1.56x-1.85x recovered, which is the routing target -- H < 32 is exactly
// what select_argmax_engine sends here); what it costs is 4-8% at H == 32,
// where the fitted sqrt(H + 2) growth overshoots slightly and which the
// automatic route sends to the SFPU anyway.
//
// The optima themselves are worth reading: at H == 1, V == 4096 the best core
// count is SIX, and every curve turns back upward well before the grid is
// full (V = 32768, H = 1: 11.3 us on 24 cores, 50.7 us on 111). "Give it the
// whole grid" is not a safe default for either engine.
//
// An explicit sub_core_grids overrides the heuristic (capped by w_tiles only)
// — pass a single-core grid to force the single-core variant, which skips the
// exchange buffer entirely.

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
        // ceil(sqrt(w_tiles * (h_logical + 2)) / 3) — see the header comment for
        // the measurements this was fitted to. Deliberately not the SFPU
        // engine's flat-in-H formula: this scan costs per ROW.
        const uint32_t want = static_cast<uint32_t>(
            std::ceil(std::sqrt(static_cast<double>(w_tiles) * (static_cast<double>(h_logical) + 2.0)) / 3.0));
        num_cores = std::min<uint32_t>({want, static_cast<uint32_t>(cores.size()), w_tiles});
    }
    TT_FATAL(num_cores >= 1, "the argmax RVV engine requires at least one core");
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

    // Chunked double-buffered input streaming: the compute-side scan of chunk
    // k overlaps the NOC staging of chunk k+1. Uniform across cores so the CB
    // layout (and therefore the exchange-buffer address) is identical
    // everywhere.
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
    // Exchange buffer: one 256 B slot per core (u32 idx[32] + u32 val[32]).
    // Allocated identically on every core so a worker's local address equals
    // the gather core's. Single core needs no exchange at all, but the CB is
    // declared unconditionally so both variants share one CB layout (and one
    // reader kernel).
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
        // Placeholder — contract with reader_argmax_rvv_tile.cpp: the kernel
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
    // enable_trisc2_rvv: compile this kernel's TRISC2 (pack) TU with the Zve32f extension —
    // the in-tree opt-in that makes the RVV scan compile in a stock build.
    compute_desc.config = ComputeConfigDescriptor{.enable_trisc2_rvv = true};
    for (uint32_t j = 0; j < num_cores; ++j) {
        compute_desc.emplace_runtime_args(cores[j], {slice_count(j)});
    }
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
