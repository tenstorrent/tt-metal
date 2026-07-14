// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "regime_a_matmul_config.hpp"

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/buffer.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

// Regime-A fixes the in1 DRAM shard to 8 banks (matches plan::Geometry::G).
constexpr uint32_t kNumBanks = 8u;

inline uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
inline uint32_t rup(uint32_t x, uint32_t y) { return cdiv(x, y) * y; }

}  // namespace

plan::PlanResult make_and_build_plan(
    IDevice* device, const Tensor& in0, const Tensor& in1, const RegimeAMatmulConfig& cfg) {
    // Tile counts from logical shapes (tile = 32).
    const auto& a_shape = in0.logical_shape();
    const auto& w_shape = in1.logical_shape();
    const uint32_t Mt = cdiv(static_cast<uint32_t>(a_shape[-2]), TILE_HEIGHT);
    const uint32_t Kt = cdiv(static_cast<uint32_t>(a_shape[-1]), TILE_WIDTH);
    const uint32_t Nt = cdiv(static_cast<uint32_t>(w_shape[-1]), TILE_WIDTH);

    const CoreCoord grid = device->compute_with_storage_grid_size();

    auto to_plan_xy = [](const std::vector<CoreCoord>& src) {
        std::vector<plan::PlanXY> out;
        out.reserve(src.size());
        for (const auto& c : src) {
            out.push_back(plan::PlanXY{static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)});
        }
        return out;
    };
    const auto opt0 = to_plan_xy(device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0));
    const auto opt1 = to_plan_xy(device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_1));

    plan::PlanInputs in;
    in.Mt = Mt;
    in.Kt = Kt;
    in.Nt = Nt;
    in.cfg = plan::RegimeAConfig{
        .k_slices = cfg.k_slices,
        .n_slices = cfg.n_slices,
        .m_slices = cfg.m_slices,
        .k_block_tiles = cfg.k_block_tiles,
        .n_subblock_tiles = cfg.n_subblock_tiles};
    in.grid_x = static_cast<uint32_t>(grid.x);
    in.grid_y = static_cast<uint32_t>(grid.y);
    in.opt0 = opt0;
    in.opt1 = opt1;
    in.holes = {};  // v1: no explicit grid holes; find_near just walks to the next free logical core.
    // BH usable L1 ~1440 KB; matches the validated prototype/sweep budget used by the picker.
    in.l1_budget_bytes = 1440u * 1024u;
    in.tb = 2048u;  // bf16 tile bytes
    in.tf = 4096u;  // fp32 tile bytes
    in.nn_chain = false;

    return plan::build_plan(in);
}

MemoryConfig create_regime_a_weight_memory_config(
    const ttnn::Shape& weight_shape, DataType /*dtype*/, IDevice* device) {
    const uint32_t K = static_cast<uint32_t>(weight_shape[-2]);
    const uint32_t N = static_cast<uint32_t>(weight_shape[-1]);
    const uint32_t Kt = cdiv(K, TILE_HEIGHT);
    const uint32_t Nt = cdiv(N, TILE_WIDTH);

    // CONFIG-INDEPENDENT padding: K rounded up to 8 tiles, N rounded up to 8 tiles. The shard spec is a
    // function of (K, N) only — no Pk/kb/Ns/nsb padding is baked into storage (see header note).
    const uint32_t Kt_pad = rup(Kt, kNumBanks);
    const uint32_t Nt_pad = rup(Nt, kNumBanks);

    // Shard shape in ELEMENTS: full K rows, N/8 columns per bank (width sharding across 8 banks).
    const std::array<uint32_t, 2> shard_shape = {Kt_pad * TILE_HEIGHT, (Nt_pad / kNumBanks) * TILE_WIDTH};

    // Shard grid = the first 8 DRAM banks (Regime-A fixes G=8). NOTE: this assumes the target device
    // exposes >= 8 DRAM banks along the DRAM grid row (BH p150b = 8). Guard against smaller grids.
    const CoreCoord dram_grid = device->dram_grid_size();
    TT_FATAL(
        static_cast<uint32_t>(dram_grid.x) * static_cast<uint32_t>(dram_grid.y) >= kNumBanks,
        "regime_a_matmul in1 width-shard needs >= {} DRAM banks, device exposes {}x{}",
        kNumBanks,
        dram_grid.x,
        dram_grid.y);
    const CoreRangeSet shard_grid(CoreRange(CoreCoord{0, 0}, CoreCoord{kNumBanks - 1, 0}));

    const ShardSpec shard_spec(shard_grid, shard_shape, ShardOrientation::ROW_MAJOR);
    return MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::DRAM, shard_spec);
}

}  // namespace ttnn::experimental::prim
