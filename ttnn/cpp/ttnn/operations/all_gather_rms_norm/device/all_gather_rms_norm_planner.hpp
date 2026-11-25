// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm planner: the C++ form of the Python build_plan / select_split / select_sharded_split / Walk /
// stage_plan / CB sizing. Integers in, integers out; memoized solves return const&.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "all_gather_rms_norm_device_operation_types.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace ttnn::operations::all_gather_rms_norm::planner {

// ---- knobs (the Python's single values; env overrides dropped: no contract test sets them) ----
constexpr uint32_t P_CAP = 128;
constexpr uint32_t COMB_TILE_COST_FUSED = 1;
constexpr uint32_t STATS_TABLE_CAP = 256 * 1024;
constexpr uint32_t ROW_CHUNK_L1_BUDGET = 320 * 1024;
constexpr uint32_t BATCH_CHUNKS = 4;
constexpr uint32_t INC_EVERY = 8;
constexpr uint32_t BALANCE_MIN_TILES = 64;
constexpr uint32_t SCATTER_RUNS = 4;
constexpr uint32_t READ_DEPTH = 4;  // min(READ_DEPTH, BATCH_CHUNKS, 15)
constexpr uint32_t SPLIT_RUNS = 0;
constexpr uint32_t OWN_SRC_DST = 1;
constexpr uint32_t OWN_BANDS = 1;  // => own_bands is always 1
constexpr uint32_t EAGER_PUSH = 1;
constexpr bool XG_PRE = false;  // => xg_pre is always false
constexpr uint32_t OWN_BANDS_MAX = 16;
constexpr uint32_t STAGE_L1_BUDGET = 128 * 1024;
constexpr uint32_t L1I_BANK_WALK_MAX_NB = 16;
constexpr uint32_t STAGE_MAX_LANES = 4;
constexpr uint32_t STAGE_MAX_NB = 16;
constexpr uint32_t CB_ALIGN = 64;
// global semaphores per port set: ready, bulk, stats, OWN_BANDS_MAX band counters, stage counter
constexpr uint32_t NUM_GLOBAL_SEMS = 3 + OWN_BANDS_MAX + 1;

enum class Dir : uint8_t { FWD = 0, BWD = 1 };

struct Geometry {
    uint32_t W_local = 0, M = 0, num_images = 0, R = 0, Wlt = 0, G = 0, W = 0, Wt = 0, w_valid_last = 0;
    bool aligned() const;
};
Geometry derive_geometry(const ttnn::Shape& shape, uint32_t G);

struct Split {
    uint32_t k = 1, n_groups = 1, rows_hi = 1, big = 1, cw_nom = 1;
    uint32_t last_crt = 0;  // 0 = by rows_hi / big
    uint32_t last_rows() const;
    uint32_t n_active() const;
    uint32_t member_width(uint32_t m, uint32_t Wlt) const;
};
uint32_t stats_k_cap(uint32_t R, uint32_t G, uint32_t nstats);
Split select_split(uint32_t R, uint32_t Wlt, uint32_t Nw, uint32_t G, uint32_t k_cap);
// HEIGHT / WIDTH sharded input; throws std::runtime_error like the Python RuntimeError. Fills the worker cores in
// shard order (row-major or column-major by the shard orientation).
Split select_sharded_split(const Geometry& geo, const Tensor& input, std::vector<CoreCoord>& shard_order);
// Every core of a shard grid in shard-assignment order.
std::vector<CoreCoord> ordered_shard_cores(const CoreRangeSet& grid, bool row_major);

// Host replica of agrn_port_common.hpp for_each_packet (counts only).
struct Walk {
    uint32_t nb = 0, rows_eff = 0, cols_eff = 0, row_stride = 0, slot_stride = 0, run_pages = 1, scatter_runs = 1;
    uint32_t shard_w = 0;
    uint32_t split_runs = 0;
    uint32_t chunks(uint32_t q, uint32_t part, uint32_t first_bank, uint32_t stride) const;
};

// ---- mesh plan (Python build_plan): per mesh, axis, ring, links, excluded cores ----
struct ChipPlan {
    uint32_t p = 0;
    std::optional<ttnn::MeshCoordinate> next, prev;
    std::array<std::vector<uint32_t>, 2> links;   // [Dir] forwarding link indices (first num_links)
    std::array<std::vector<CoreCoord>, 2> ports;  // [Dir][link] logical port core
};
struct MeshPlan {
    uint32_t G = 0;
    bool ring = false;
    uint32_t num_links = 0;
    CoreCoord grid;
    std::map<ttnn::MeshCoordinate, ChipPlan> chips;
    std::vector<std::vector<ttnn::MeshCoordinate>> groups;
};
// min over every group hop of the forwarding links; nullopt when a needed hop has none (Python usable_links).
std::optional<uint32_t> usable_links(tt::tt_metal::distributed::MeshDevice& mesh, uint32_t cluster_axis, bool ring);
// Memoized on (mesh id, mesh shape, axis, ring, links, excluded cores). Port placement as the Python: per link the
// free worker with the fewest NoC1 hops to the link's Ethernet core (ties: logical row, then column), fallback the
// worker nearest physical column 0 below the Ethernet row.
const MeshPlan& build_mesh_plan(
    tt::tt_metal::distributed::MeshDevice& mesh,
    uint32_t cluster_axis,
    bool ring,
    uint32_t num_links,
    const std::vector<CoreCoord>& excluded);

// Python _schedule (fabric_all_gather): (slot, part) entries per direction, own slot first.
using Schedule = std::vector<std::pair<uint32_t, uint32_t>>;
std::pair<Schedule, Schedule> schedule(uint32_t p, uint32_t G, bool ring, bool balance);

// ---- per-chip own-entry staging plan (Python stage_plan) ----
struct StageLane {
    uint32_t n_st = 0, tot = 0;
};
struct StagePlan {
    std::vector<StageLane> lanes;
    std::array<uint32_t, STAGE_MAX_NB> unit_off{};
    uint32_t max_pages = 0, nch_max = 0;
};

// ---- the whole program plan (everything create_descriptor sizes from) ----
struct ProgramPlan {
    Geometry geo;
    Split split;
    std::vector<CoreCoord> shard_order;  // sharded input: workers in worker-index order
    bool x_sharded = false, pre_combine = false, is_ru = false, mask_tail = false, xg_pre = false;
    bool out_sharded = false, out_in_place = false, balance = false, stage_on = false;
    uint32_t nstats = 1, kt = 1, P = 1, gather_tiles = 1, gt_pc = 0, gt_max = 1, rpo = 0, n_owners = 0, nfinal = 1;
    uint32_t crt_max = 1, cw_max = 1, row_chunk = 1, slot_bytes = 0, table_bytes = 0;
    uint32_t nb = 0, run_pages = 1, stats_payload = 0, out_page = 0, src_max = 1, own_bands = 1;
    uint32_t fixed_cb_bytes = 0, stage_budget = 0, stage_max_pages = 0;
    Walk walk;
    std::vector<uint32_t> band_need, band_row_end;
    std::map<ttnn::MeshCoordinate, std::optional<StagePlan>> stage_plans;  // by chip
};

// Free L1 for static CBs on this device: [base, lowest allocated L1 buffer) (Python _l1_cb_room).
std::pair<uint32_t, uint32_t> l1_cb_room(tt::tt_metal::distributed::MeshDevice& mesh);

// Solves the plan; memoized on its integer inputs (attrs, tensor specs, the mesh plan identity and the free L1),
// so compute_program_hash, the validators and create_descriptor share one solve. Throws UnsupportedAxisValue when
// the fixed CBs do not fit below the lowest L1 buffer (Python _check_l1_fit).
const ProgramPlan& solve_program_plan(
    const AllGatherRmsNormParams& attrs, const AllGatherRmsNormInputs& tensors, const MeshPlan& mesh_plan);

// The R_U staging tensor spec a call needs (Python staging_buffer): DRAM interleaved, TILE, output dtype.
tt::tt_metal::TensorSpec staging_spec(const Geometry& geo, tt::tt_metal::DataType dtype);

}  // namespace ttnn::operations::all_gather_rms_norm::planner
