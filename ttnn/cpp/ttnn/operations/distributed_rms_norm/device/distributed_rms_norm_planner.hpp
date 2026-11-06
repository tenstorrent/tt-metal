// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/reflection.hpp>

#include "distributed_rms_norm_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"

// The planner of the Python descriptor builder: groups / links / exchange schedule (per-mesh state, memoized by
// mesh id + shape), the local regime (R1 / R3-i / R1-HS / R3-s) and blocking against the L1 budget, and the
// exchange landing layout. Pure integers in, a const& plan out; a hit never re-solves.
namespace ttnn::operations::distributed_rms_norm::detail {

// ---- knobs the source fixes (collapsed to their only value) ----
constexpr uint32_t W_CHUNK_TILES = 32;
constexpr uint32_t STREAM_DEPTH = 2;
constexpr uint32_t SCALE_BLOCK_TILES = 8;
constexpr uint32_t MIN_BLOCKS_PER_CORE = 1;
constexpr uint32_t NUM_WAVES = 2;
constexpr bool PORT_OWN_AHEAD = false;
constexpr bool READ_WAVE_GATE = true;
constexpr uint32_t R3_ROWS_PER_GROUP = 2;
constexpr uint32_t R3_MAX_GROUP_CORES = 24;
constexpr uint32_t R3_MIN_SLICE_TILES = 4;
constexpr uint32_t R3_OCCUPANCY_GAIN = 2;
constexpr uint32_t R3_MIN_GROUP_CORES = 4;
constexpr uint64_t ESW_MIN_BYTES = 2'200'000;  // early small wave: x bytes per chip
constexpr uint32_t ESW_CLASSES = 3;
constexpr double ESW_READ_GBPS = 300.0;
constexpr double AICLK_GHZ = 1.35;
constexpr uint32_t LANDING_L1_RESERVE = 64 * 1024;
constexpr uint32_t PORT_LANDING_MAX_BYTES = 512 * 1024;
constexpr uint32_t LANDING_RING_WAVES = 4;
constexpr uint32_t LANDING_MIN_WORDS = 4096;  // landing allocation bucket: powers of two from here
constexpr uint32_t OUT_NOC_SPLIT_MIN_BYTES = 512 * 1024;
constexpr uint32_t BH_DRAM_SPLIT_X = 9;  // "alternate": odd banks on NOC0 (Blackhole, DRAM-interleaved output)
constexpr uint32_t FIN_COMPACT_MAX_G = 16;
constexpr uint32_t SEG_ALIGN = 64;
constexpr uint32_t PLAN_ATTEMPTS = 4;  // re-plans against the landing a plan needs

// ---- per-mesh exchange state ----
struct DestHeader {
    tt::tt_metal::distributed::MeshCoordinate coord;
    uint32_t mcast_code = 0;  // 0 unicast; (eth dir + 1) << 8 | hop range
    uint32_t hops = 0;
    bool operator==(const DestHeader&) const = default;
};
struct SendItem {
    uint32_t slot = 0;
    uint32_t header = 0;
    uint32_t wait = 0;  // 0 own, j + 1 relay of entry j
};
struct ReadyOut {
    tt::tt_metal::distributed::MeshCoordinate source;
    uint32_t source_dir = 0;
    uint32_t header = 0;
};
// One chip's schedule (exchange_plan's dict); arrays indexed by dir::FWD / dir::BWD.
struct ChipSchedule {
    uint32_t position = 0;  // p
    std::optional<tt::tt_metal::distributed::MeshCoordinate> next, prev;
    std::array<std::vector<SendItem>, 2> sends;
    std::array<std::vector<DestHeader>, 2> dests;
    std::array<std::vector<uint32_t>, 2> entries;
    std::array<std::vector<ReadyOut>, 2> ready_out;
    std::array<uint32_t, 2> n_wait{};
};
struct MeshExchange {
    std::map<tt::tt_metal::distributed::MeshCoordinate, ChipSchedule> chips;
    std::map<tt::tt_metal::distributed::MeshCoordinate, tt::tt_fabric::FabricNodeId> nodes;
};

// usable_links of the groups along cluster_axis (memoized by mesh id, shape, axis, ring).
uint32_t usable_links(tt::tt_metal::distributed::MeshDevice& mesh, uint32_t cluster_axis, bool ring);
// The schedule (memoized by mesh id, shape, axis, ring, mode; computed on the mesh's first call).
const MeshExchange& mesh_exchange(
    tt::tt_metal::distributed::MeshDevice& mesh, uint32_t cluster_axis, bool ring, ExchangeMode mode);
// True on a 2D fabric config (routing by destination node).
bool is_2d_fabric();
// Whether the active fabric config wraps cluster_axis (Ring validation).
bool fabric_wraps_axis(uint32_t cluster_axis);

// Persistent global semaphores over the whole grid: [0] ready, [1..G] one arrival counter per slot. Grow-only per
// mesh (ttsl::Indestructible), synchronized once at creation.
struct ExchangeSemaphores {
    uint32_t ready_addr = 0;
    std::vector<uint32_t> arrival_addrs;  // by slot q
};
const ExchangeSemaphores& exchange_semaphores(tt::tt_metal::distributed::MeshDevice& mesh, uint32_t group_size);

// ---- local plan ----
enum class Regime : uint8_t { R1, R3_INTERLEAVED, R1_HEIGHT_SHARDED, R3_WIDTH_SHARDED };

struct CoreWork {
    tt::tt_metal::CoreCoord core;
    uint32_t start_tile_row = 0;
    uint32_t rows = 0;
    uint32_t w0 = 0;
    uint32_t V = 0;
    uint32_t index = 0;  // 0 = root
    tt::tt_metal::CoreCoord root;
};

struct Geometry {
    bool has_gamma = false;
    uint32_t W = 0, Wt = 0, S = 0, C = 0, num_w_chunks = 0, Wp = 0, WS = 0;
    uint32_t T_in = 0, T_out = 0, T_g = 0, es = 0, es_g = 0;
    bool gamma_is_tile = false;
    bool interm_fp32 = false;  // interm format FLOAT32 vs BFLOAT16
    uint32_t T_i = 0;
    bool sharded = false, combine = false;
    uint32_t G_w = 0;
    bool masked = false;
    uint32_t G = 0;
    bool gamma_resident = false;
    bool out_native = false;
    bool fin_compact() const { return G <= FIN_COMPACT_MAX_G; }
    uint32_t fin_tpr() const { return fin_compact() ? 1 : G; }
};

// Everything the solve reads, integers only (equal inputs -> the same plan). Built per call by plan_inputs.
struct PlanInputs {
    std::vector<uint32_t> shape;  // logical input shape
    tt::tt_metal::DataType input_dtype = tt::tt_metal::DataType::BFLOAT16;
    std::optional<tt::tt_metal::DataType> gamma_dtype;
    bool gamma_is_tile = false;
    tt::tt_metal::TensorMemoryLayout input_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    bool input_l1 = false;
    std::array<uint32_t, 2> shard_shape{};
    bool shard_row_major = true;
    std::vector<std::array<uint32_t, 4>> input_shard_rects;   // x0, y0, x1, y1
    std::vector<std::array<uint32_t, 4>> output_shard_rects;  // busy for port placement
    bool out_native = false;
    bool out_dram_interleaved = false;
    bool blackhole = false;
    bool fp32_acc = false;
    uint32_t group_size = 0;
    uint32_t num_links = 0;
    uint32_t grid_x = 0, grid_y = 0;
    uint32_t l1_budget = 0;  // bytes below the lowest L1 buffer, the call's exchange landing not counted (device state)
    bool operator==(const PlanInputs&) const = default;
};

struct Plan {
    Regime regime = Regime::R1;
    std::vector<CoreWork> work;
    Geometry g;
    uint32_t block_rows = 0;  // R
    uint32_t x_depth = 1;
    uint32_t rows_max = 0;
    // ports, by (dir, link): ports[link][dir]
    std::vector<std::array<tt::tt_metal::CoreCoord, 2>> ports;
    // exchange layout
    uint32_t wave_rows = 0;        // Rw
    uint32_t waves_per_block = 0;  // nwpb
    uint32_t num_waves = 0;
    uint32_t landing_waves = 0;  // K
    uint32_t seg_stride = 0;
    uint32_t landing_words = 0;                   // words the layout uses
    uint32_t landing_alloc_words = 0;             // its allocation bucket (the scratch tensor's row)
    std::vector<uint32_t> esw_offsets;            // per root, plan roots order
    std::vector<uint64_t> rows_before;            // num_waves + 1
    std::vector<std::vector<uint32_t>> seg_wave;  // [link][wave] bytes a port sends
    std::array<uint32_t, 3> out_split{};          // split_x, noc0 banks west, east; zeros = off
    // core sets (merged rectangles)
    tt::tt_metal::CoreRangeSet compute_cores, port_cores, all_cores;
    std::vector<const CoreWork*> roots() const;
};

// Resolves the per-call inputs, reading the L1 budget from the device and the tensors (input, gamma, output,
// scratch) and the persistent semaphores.
PlanInputs plan_inputs(const DistributedRmsNormParams& attrs, const DistributedRmsNormInputs& t, const Tensor& output);
// Memoized solve; throws RegimeRefusal where the Python raised NotImplementedError.
const Plan& solve_plan(const PlanInputs& inputs);
// What the plan fixes in the program (hashed by compute_program_hash beside the attributes and tensor specs).
ttsl::hash::hash_t plan_hash(const Plan& plan);

// One allocator-owned compute-core circular buffer.
struct CbEntry {
    uint32_t index = 0;
    tt::DataFormat format = tt::DataFormat::Invalid;
    uint32_t page_size = 0;
    uint32_t pages = 0;
};
// The compute-core CB inventory (cb_inventory of the Python planner), in its order: the single source of the L1 fit
// (summed by the blocking search) and of the program's CB descriptors. Shard-backed CBs are not listed. The formats
// are those of the x, output and gamma CBs; the fit passes any (formats do not change the bytes).
std::vector<CbEntry> cb_inventory(
    const Geometry& g,
    uint32_t block_rows,
    uint32_t x_depth,
    tt::DataFormat x_format,
    tt::DataFormat output_format,
    tt::DataFormat gamma_format);
// Bytes per port core the landing occupies (its allocation bucket).
uint32_t landing_bytes_for(uint32_t landing_words);
// The landing tensor spec: L1 HEIGHT_SHARDED uint32 ROW_MAJOR over the whole grid, (grid.x * grid.y, words).
tt::tt_metal::TensorSpec landing_spec(tt::tt_metal::distributed::MeshDevice& mesh, uint32_t landing_alloc_words);

}  // namespace ttnn::operations::distributed_rms_norm::detail
