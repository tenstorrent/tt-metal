// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/sp_matmul_fusion_common/sp_matmul_fusion_common.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <utility>
#include <tuple>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

namespace ttnn::experimental::ccl {

Tensor sub_batched_view(const Tensor& t, uint32_t num_slices) {
    const auto& shape = t.logical_shape();
    TT_FATAL(num_slices >= 1, "sub_batched_view: num_slices must be >= 1, got {}", num_slices);
    TT_FATAL(shape.rank() == 4, "sub_batched_view: expected a rank-4 [B,1,S,X] tensor, got {}", shape);
    TT_FATAL(shape[1] == 1, "sub_batched_view: expected dim 1 == 1 in [B,1,S,X], got {}", shape);
    TT_FATAL(t.layout() == Layout::TILE, "sub_batched_view: expected TILE layout, got {}", t.layout());
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "sub_batched_view: tensor must be on device");
    const uint32_t tile_h = t.tensor_spec().tile().get_height();
    TT_FATAL(
        shape[2] % (num_slices * tile_h) == 0,
        "sub_batched_view: S ({}) must be a multiple of num_slices ({}) x tile height ({})",
        shape[2],
        num_slices,
        tile_h);
    TT_FATAL(
        t.padded_shape()[2] == shape[2],
        "sub_batched_view: S must be tile aligned (no padding on dim 2), logical {} padded {}",
        shape[2],
        t.padded_shape()[2]);

    const Shape new_shape({shape[0] * num_slices, 1, shape[2] / num_slices, shape[3]});
    Tensor out = ttnn::view(t, new_shape);
    // ttnn::view wraps the same allocation in a new (non-owning) Buffer object, so compare the allocation, not the
    // Buffer pointer.
    TT_FATAL(
        out.buffer() != nullptr && t.buffer() != nullptr && out.buffer()->address() == t.buffer()->address() &&
            out.buffer()->size() == t.buffer()->size() && out.device() == t.device(),
        "sub_batched_view: view moved data (different allocation); this must be a metadata-only reshape");
    return out;
}

operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig sp_matmul_program_config(
    const Tensor& in0_view,
    const Tensor& in1,
    tt::tt_metal::CoreCoord grid,
    bool transpose_b,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using namespace operations::matmul;
    TT_FATAL(grid.x >= 1 && grid.y >= 1, "sp_matmul_program_config: empty core grid {}x{}", grid.x, grid.y);

    const auto a_shape_padded = utilities::get_matmul_tensor_padded_shape(in0_view, /*transpose=*/false);
    const auto b_shape_padded = utilities::get_matmul_tensor_padded_shape(in1, transpose_b);
    const auto in0_tile = utilities::get_matmul_tile(in0_view, /*transpose=*/false);
    const auto in1_tile = utilities::get_matmul_tile(in1, transpose_b);

    // Mt of ONE sub-batch: fuse_batch=false, the batch loop covers the B*T sub-batches.
    const uint32_t Mt = utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/false);
    const uint32_t Kt = utilities::get_K_dim(a_shape_padded, in0_tile);
    const uint32_t Nt = utilities::get_N_dim(b_shape_padded, in1_tile);
    TT_FATAL(
        a_shape_padded[-1] == b_shape_padded[-2],
        "sp_matmul_program_config: K mismatch, in0 {} vs in1 {} (transpose_b={})",
        a_shape_padded,
        b_shape_padded,
        transpose_b);

    const uint32_t per_core_M = static_cast<uint32_t>((Mt + grid.y - 1) / grid.y);
    const uint32_t min_per_core_N = std::max<uint32_t>(1, static_cast<uint32_t>((Nt + grid.x - 1) / grid.x));
    uint32_t in0_block_w = 4;
    while (Kt % in0_block_w != 0) {
        in0_block_w -= 1;
    }

    // L1 footprint check, same helper as create_matmul_program_config(); if the full per-core block does not fit,
    // split it into out blocks (largest area first, squarest on ties), keeping in0_block_w.
    const bool fp32_dest_acc_en = get_fp32_dest_acc_en(compute_kernel_config);
    const uint32_t interm_tile_size = utilities::estimate_interm_tile_size(compute_kernel_config, in0_view.dtype());
    const uint32_t max_l1_space = utilities::get_max_l1_space(in0_view);
    auto fits = [&](uint32_t m, uint32_t n) {
        return utilities::get_estimated_size_of_cbs(
                   m, n, in0_block_w, in0_view, in1, /*transpose_a=*/false, transpose_b, interm_tile_size, 0) <
               max_l1_space;
    };
    // Largest out block of a (per_core_M x per_core_N) core block that fits L1 (nullopt: none does).
    auto largest_fitting_out_block = [&](uint32_t pcm, uint32_t pcn) -> std::optional<std::pair<uint32_t, uint32_t>> {
        if (fits(pcm, pcn)) {
            return std::make_pair(pcm, pcn);
        }
        std::vector<uint32_t> m_factors;
        std::vector<uint32_t> n_factors;
        for (uint32_t f = pcm; f >= 1; --f) {
            if (pcm % f == 0) {
                m_factors.push_back(f);
            }
        }
        for (uint32_t f = pcn; f >= 1; --f) {
            if (pcn % f == 0) {
                n_factors.push_back(f);
            }
        }
        // area -> (m, n), keeping the squarest pair per area
        std::map<uint32_t, std::tuple<uint32_t, uint32_t>> by_area;
        for (uint32_t m : m_factors) {
            for (uint32_t n : n_factors) {
                const float ratio = static_cast<float>(std::max(m, n)) / static_cast<float>(std::min(m, n));
                auto it = by_area.find(m * n);
                if (it == by_area.end()) {
                    by_area[m * n] = {m, n};
                } else {
                    auto [em, en] = it->second;
                    const float eratio = static_cast<float>(std::max(em, en)) / static_cast<float>(std::min(em, en));
                    if (ratio < eratio) {
                        it->second = {m, n};
                    }
                }
            }
        }
        for (auto it = by_area.rbegin(); it != by_area.rend(); ++it) {
            auto [m, n] = it->second;
            if (fits(m, n)) {
                return std::make_pair(m, n);
            }
        }
        return std::nullopt;
    };

    // per_core_N: the smallest value ceil(Nt / grid.x) leaves the subblock shape to chance (a prime per_core_N, e.g.
    // 19 for gate_up and 11 for N=4096 on 12 columns, forces a 2x1 subblock: one in1 tile unpacked per two output
    // tiles). Widening the per-core N slightly (fewer columns busy, the last one partially filled) buys a wide
    // subblock. Pick the candidate that minimises the per-core cost model
    //     per_core_N * (subblock_h + subblock_w) / (subblock_h * subblock_w),
    // i.e. the tiles the busiest core has to produce weighted by the in0+in1 tiles unpacked per output tile; ties go
    // to the smaller per_core_N. Candidates: [ceil(Nt/grid.x), 2*ceil(Nt/grid.x)] capped at Nt.
    struct Candidate {
        uint32_t per_core_N = 0;
        uint32_t out_block_h = 0;
        uint32_t out_block_w = 0;
        uint32_t out_subblock_h = 0;
        uint32_t out_subblock_w = 0;
        float cost = 0.f;
    };
    // The widening only pays when the per-core in1 slab (Kt x per_core_N tiles) is L1-resident (the 2D-mcast
    // factory's SP_IN1_RESIDENT decision, estimated here the same way: slab + the other CBs + slack <= L1). A
    // streamed slab makes the sub-batched matmul DRAM-bound on the in1 re-read (gate_up: 57 MB per sub-batch), where
    // a wider column only adds work to the busiest core (measured +6% for gate_up at per_core_N 19 -> 20).
    const uint32_t in1_tile_size = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(in1.dtype()));
    auto slab_resident = [&](uint32_t pcn, uint32_t obh, uint32_t obw) {
        constexpr uint64_t kL1Slack = 32 * 1024;
        const uint64_t slab = static_cast<uint64_t>(Kt) * pcn * in1_tile_size;
        const uint64_t streamed_in1_cb = static_cast<uint64_t>(2) * in0_block_w * obw * in1_tile_size;
        const uint64_t others = utilities::get_estimated_size_of_cbs(
            obh, obw, in0_block_w, in0_view, in1, /*transpose_a=*/false, transpose_b, interm_tile_size, 0);
        return others - std::min(others, streamed_in1_cb) + slab + kL1Slack <= max_l1_space;
    };
    std::optional<Candidate> best;
    uint32_t max_per_core_N = min_per_core_N;
    if (const auto ob0 = largest_fitting_out_block(per_core_M, min_per_core_N);
        ob0.has_value() && slab_resident(min_per_core_N, ob0->first, ob0->second)) {
        max_per_core_N = std::min<uint32_t>(Nt, 2 * min_per_core_N);
    }
    for (uint32_t pcn = min_per_core_N; pcn <= max_per_core_N; ++pcn) {
        const auto ob = largest_fitting_out_block(per_core_M, pcn);
        if (!ob.has_value()) {
            continue;
        }
        if (pcn != min_per_core_N && !slab_resident(pcn, ob->first, ob->second)) {
            continue;
        }
        auto [obh, obw] = ob.value();
        auto [sbh, sbw] = bmm_op_utils::get_matmul_subblock_params(obh, obw, false, false, fp32_dest_acc_en);
        const float cost = static_cast<float>(pcn) * static_cast<float>(sbh + sbw) / static_cast<float>(sbh * sbw);
        if (!best.has_value() || cost < best->cost) {
            best = Candidate{
                .per_core_N = pcn,
                .out_block_h = obh,
                .out_block_w = obw,
                .out_subblock_h = sbh,
                .out_subblock_w = sbw,
                .cost = cost};
        }
    }
    TT_FATAL(
        best.has_value(),
        "sp_matmul_program_config: no out block of per_core_M={} x per_core_N in [{}, {}] (in0_block_w={}) fits L1 "
        "({} B)",
        per_core_M,
        min_per_core_N,
        max_per_core_N,
        in0_block_w,
        max_l1_space);
    const uint32_t per_core_N = best->per_core_N;
    const uint32_t out_block_h = best->out_block_h;
    const uint32_t out_block_w = best->out_block_w;
    const uint32_t out_subblock_h = best->out_subblock_h;
    const uint32_t out_subblock_w = best->out_subblock_w;

    MatmulMultiCoreReuseMultiCastProgramConfig config{
        .compute_with_storage_grid_size = grid,
        .in0_block_w = in0_block_w,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .out_block_h = out_block_h,
        .out_block_w = out_block_w,
        .per_core_M = per_core_M,
        .per_core_N = per_core_N,
        .transpose_mcast = false,
        .fused_activation = std::nullopt,
        .fuse_batch = false,
    };
    config.allowed_worker_cores =
        CoreRangeSet(CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(grid.x - 1, grid.y - 1)));
    return config;
}

std::vector<uint32_t> pack_sp_schedule(const std::vector<SpSubBatch>& order) {
    std::vector<uint32_t> words;
    words.reserve(order.size());
    for (const auto& e : order) {
        TT_FATAL(e.in0_idx < 256, "pack_sp_schedule: in0_idx {} does not fit 8 bits", e.in0_idx);
        TT_FATAL(e.out_idx < 256, "pack_sp_schedule: out_idx {} does not fit 8 bits", e.out_idx);
        TT_FATAL(e.wait_dir < 2, "pack_sp_schedule: wait_dir {} must be 0 or 1", e.wait_dir);
        TT_FATAL(e.wait_count < 256, "pack_sp_schedule: wait_count {} does not fit 8 bits", e.wait_count);
        words.push_back(
            e.in0_idx | (e.out_idx << 8) | (e.wait_dir << 16) | (static_cast<uint32_t>(e.is_local) << 17) |
            (e.wait_count << 24));
    }
    return words;
}

std::vector<SpSubBatch> sp_ag_schedule(ttnn::ccl::Topology topology, uint32_t T, uint32_t ring_index, uint32_t B) {
    TT_FATAL(T >= 2 && ring_index < T, "sp_ag_schedule: need T >= 2 and ring_index < T, got T={} r={}", T, ring_index);
    TT_FATAL(B >= 1, "sp_ag_schedule: need B >= 1");
    TT_FATAL(
        topology == ttnn::ccl::Topology::Ring || topology == ttnn::ccl::Topology::Linear,
        "sp_ag_schedule: topology must be Ring or Linear");
    // Same call as all_gather_async_default_program_factory.cpp (static_alternate=false).
    auto [num_targets_forward, num_targets_backward] =
        ttnn::ccl::get_forward_backward_line_mcast_distance(T, ring_index, topology, /*static_alternate=*/false);
    // minimal_default_reader.cpp: Linear dir1 <- num_targets_forward, dir0 <- num_targets_backward;
    //                             Ring   dir1 <- num_targets_backward, dir0 <- num_targets_forward.
    const uint32_t n_dir1 = topology == ttnn::ccl::Topology::Linear ? num_targets_forward : num_targets_backward;
    const uint32_t n_dir0 = topology == ttnn::ccl::Topology::Linear ? num_targets_backward : num_targets_forward;
    TT_FATAL(
        n_dir0 + n_dir1 == T - 1,
        "sp_ag_schedule: directions deliver {} + {} slices, expected T-1 = {}",
        n_dir0,
        n_dir1,
        T - 1);

    std::vector<SpSubBatch> order;
    order.reserve(static_cast<size_t>(B) * T);
    for (uint32_t b = 0; b < B; ++b) {
        order.push_back(
            SpSubBatch{.in0_idx = b, .out_idx = b * T + ring_index, .wait_dir = 0, .wait_count = 0, .is_local = true});
    }
    for (uint32_t k = 0; k < std::max(n_dir0, n_dir1); ++k) {
        if (k < n_dir0) {
            const uint32_t chip = (ring_index + T - (k + 1)) % T;  // dir0: my_chip_id - (k+1)
            for (uint32_t b = 0; b < B; ++b) {
                order.push_back(SpSubBatch{
                    .in0_idx = b * T + chip,
                    .out_idx = b * T + chip,
                    .wait_dir = 0,
                    .wait_count = k + 1,
                    .is_local = false});
            }
        }
        if (k < n_dir1) {
            const uint32_t chip = (ring_index + k + 1) % T;  // dir1: my_chip_id + (k+1)
            for (uint32_t b = 0; b < B; ++b) {
                order.push_back(SpSubBatch{
                    .in0_idx = b * T + chip,
                    .out_idx = b * T + chip,
                    .wait_dir = 1,
                    .wait_count = k + 2,
                    .is_local = false});
            }
        }
    }
    TT_FATAL(order.size() == static_cast<size_t>(B) * T, "sp_ag_schedule: produced {} entries", order.size());
    return order;
}

std::vector<SpSubBatch> sp_ag_schedule_serialized(
    ttnn::ccl::Topology topology, uint32_t T, uint32_t ring_index, uint32_t B) {
    const auto overlapped = sp_ag_schedule(topology, T, ring_index, B);
    uint32_t final_count[2] = {0, 0};
    for (const auto& e : overlapped) {
        if (!e.is_local) {
            final_count[e.wait_dir] = std::max(final_count[e.wait_dir], e.wait_count);
        }
    }
    std::vector<SpSubBatch> order;
    order.reserve(overlapped.size());
    for (const auto& e : overlapped) {
        if (!e.is_local) {
            SpSubBatch s = e;
            s.wait_count = final_count[e.wait_dir];
            order.push_back(s);
        }
    }
    for (const auto& e : overlapped) {
        if (e.is_local) {
            order.push_back(e);
        }
    }
    return order;
}

std::vector<uint32_t> sp_rs_ordinals(const std::vector<SpSubBatch>& order, uint32_t B, uint32_t T) {
    const uint32_t n = B * T;
    TT_FATAL(order.size() == n, "sp_rs_ordinals: schedule has {} entries, expected B*T = {}", order.size(), n);
    constexpr uint32_t UNSET = 0xFFFFFFFFu;
    std::vector<uint32_t> ordinals(n, UNSET);
    for (uint32_t j = 0; j < n; ++j) {
        const uint32_t out_idx = order[j].out_idx;
        TT_FATAL(out_idx < n, "sp_rs_ordinals: out_idx {} out of range [0, {})", out_idx, n);
        TT_FATAL(ordinals[out_idx] == UNSET, "sp_rs_ordinals: out_idx {} appears twice in the schedule", out_idx);
        ordinals[out_idx] = j;
    }
    return ordinals;
}

}  // namespace ttnn::experimental::ccl
