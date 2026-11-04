// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device parameters of the matmul config model: everything that is a property of the chip (worker grid, physical NoC
// placement of the workers and DRAM endpoints, clock, NoC / DRAM bandwidth) plus the constants fitted on that chip and
// the L1 per-core compute cost tables. A new device is a new DeviceParams entry; nothing else in the model is device
// specific.
//
// Entries: Blackhole p150 (the 11x10 worker grid the model was trained on) and Blackhole Galaxy (12x10; the p150 entry
// plus the 12th Tensix column; the model has no 12-column training rows). The numbers are exactly the ones the model
// was trained with; do not edit them without retraining.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace ttnn::operations::matmul::model {

enum class Arch : std::uint8_t { kBlackholeP150, kBlackholeGalaxy };

enum class Fidelity : std::uint8_t { kLoFi, kHiFi2, kHiFi3, kHiFi4 };

// Math phases per tile product at a fidelity (LoFi 1 ... HiFi4 4)
std::uint32_t fidelity_phases(Fidelity f);

// Constants fitted on the device. Units: "cycles" are device
// clock cycles, rates are fractions of the spec rate unless stated.
struct FittedConstants {
    double dram_eff{};    // achievable fraction of spec DRAM read bandwidth
    double noc_eff{};     // achievable fraction of the per-core NoC read/write rate
    double l1_frac{};     // chip interleaved-L1 read bandwidth as a fraction of the DRAM spec
    double lat_dram{};    // cycles: one batch of DRAM page reads, issue to barrier
    double lat_l1{};      // cycles: one batch of interleaved-L1 page reads
    double launch_us{};   // us: program launch + kernel prologue/epilogue
    double tile_rt{};     // cycles: MultiCore factory tile read (unused: MultiCore is not modeled)
    double link_eff{};    // achievable fraction of a NoC link's rate
    double bank_frac{};   // DRAM bandwidth one bank serves, fraction of the chip's
    double burst_KB{};    // KB per reader per barrier at which interleaved-L1 bandwidth halves
    double rl_init{};     // cycles per reloaded subblock (spill/reload)
    double u2d{};         // cycles per reloaded fp32 tile (unpack-to-dest)
    double lat_mcast{};   // cycles: mcast sender handshake per K block
    double ack_rx{};      // cycles per mcast receiver ack
    double pack_Bpc{};    // packer bytes per cycle
    double call{};        // cycles per matmul_block call
    double lat_write{};   // cycles: one subblock of output page writes + barrier
    double sfpu_tile{};   // cycles per output tile of SFPU activation
    double issue{};       // cycles per page read issued by one reader core
    double mc_eff0{};     // multicast rate / link rate at zero receivers
    double mc_eff_rx{};   // drop of that fraction per receiver
    double noc_eff_cc{};  // one core's DRAM read rate with concurrent readers, fraction of noc_bytes_per_cycle
    double reuse_link{};  // fraction of the lockstep link load Reuse's unsynchronised cores put on a link
    double dram_w_eff{};  // achievable fraction of spec DRAM bandwidth for writes
    double wf_reuse{};    // Reuse output write rate relative to 2D
    double pack_Bpc32{};  // packer bytes per cycle for fp32 partials
    double wf_in0{};      // 1D in0-mcast output write rate relative to 2D
    double wf_in1{};      // 1D in1-mcast output write rate relative to 2D
    double lat_shard{};   // cycles per K step: in0 mcast sender handoff for sharded in0
    double pad_elem{};    // cycles per element the in0 reader zero-fills in a partial K tile
};

// L1 compute model: per (subblock h, w, fidelity) measured compute costs, in cycles. Points tables are indexed by
// in0_block_w (kw) and interpolated (see matmul_model_features.cpp, l1_cycles).
struct L1ShapeCosts {
    static constexpr std::size_t kMaxPoints = 8;
    std::uint32_t h{}, w{};
    Fidelity fidelity{};
    double calls{};  // cycles per matmul_block call
    double kref{};   // measured whole-block reference at kw = nb = S = 1
    bool has_f1{};
    double f1{};  // first-reload extra, one subblock per block
    bool has_f2_kw1{};
    double f2_kw1{};  // first-reload extra, several subblocks, measured at kw = 1
    bool has_f2_kw4{};
    double f2_kw4{};  // ... at kw = 4
    std::uint32_t n_xp{};
    std::array<std::uint32_t, kMaxPoints> xp_kw{};
    std::array<double, kMaxPoints> xp{};  // reload after the partials' arrival
    std::uint32_t n_xr{};
    std::array<std::uint32_t, kMaxPoints> xr_kw{};
    std::array<double, kMaxPoints> xr{};  // reload with the partials ready
};

struct L1CostTables {
    double overhead_per_subblock{};  // o_c, cycles
    double pack_p0{};                // pack(hw) = pack_p0 + pack_p1 * hw, cycles
    double pack_p1{};
    double block_overhead{};  // o_blk, cycles
    std::span<const L1ShapeCosts> shapes;
};

struct DeviceParams {
    static constexpr std::size_t kMaxCols = 16, kMaxRows = 16, kMaxChannels = 12, kEndpoints = 3, kMaxBanks = 16;
    std::string_view name;

    // Worker grid the model places configs on (configs larger than this cannot be scored); the interleaved-L1 page
    // spread also covers this whole grid.
    std::uint32_t worker_grid_x{}, worker_grid_y{};
    // NoC torus size (physical coordinates) and the physical column / row of each logical worker column / row.
    std::uint32_t noc_size_x{}, noc_size_y{};
    std::array<std::uint32_t, kMaxCols> worker_cols{};
    std::array<std::uint32_t, kMaxRows> worker_rows{};
    // DRAM channels: physical (x, y) of each channel's endpoints; banks: (channel, endpoint used on NOC0, on NOC1).
    std::uint32_t num_channels{};
    std::array<std::array<std::array<std::uint32_t, 2>, kEndpoints>, kMaxChannels> channel_endpoints{};
    std::uint32_t num_banks{};  // interleaved DRAM banks (page i lives in bank i mod num_banks)
    std::array<std::array<std::uint32_t, 3>, kMaxBanks> banks{};

    double clock_hz{};             // Tensix clock
    double noc_bytes_per_cycle{};  // per-core NoC rate and per-link rate, bytes / cycle
    double dram_GBs{};             // spec DRAM bandwidth, GB/s (1e9 bytes)
    FittedConstants c;             // fitted on this device
    L1CostTables l1;               // L1 per-core compute model tables
};

// The parameter table of a device. Throws std::invalid_argument for an arch without an entry.
const DeviceParams& device_params(Arch arch);

// The Blackhole entry for a worker grid: p150 up to 11x10, Galaxy for 12 columns; std::nullopt when none covers it.
std::optional<Arch> blackhole_arch_for_grid(std::uint32_t grid_x, std::uint32_t grid_y);

}  // namespace ttnn::operations::matmul::model
