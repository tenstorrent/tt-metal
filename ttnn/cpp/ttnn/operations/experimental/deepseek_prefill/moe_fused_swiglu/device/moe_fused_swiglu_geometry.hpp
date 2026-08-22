// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::geometry {

inline constexpr uint32_t TILE = 32;
inline constexpr uint32_t M_BLOCK = 8;
inline constexpr uint32_t DEST_LIMIT = 8;
inline constexpr uint32_t OUT_SUBBLOCK_H_GU = 1;
inline constexpr uint32_t OUT_SUBBLOCK_H_DN_MAX = 4;
inline constexpr uint32_t ELTWISE_BLK = 8;
inline constexpr uint32_t DEPTH_W = 2;
inline constexpr uint32_t DEPTH_X = 2;
inline constexpr uint32_t DEPTH_H = 3;
inline constexpr uint32_t DEPTH_OUT = 2;
inline constexpr uint32_t XSTICK_ROWS = 1;
inline constexpr uint32_t WD_AHEAD = 1;
inline constexpr bool WD_MROW_ROUNDS = true;
inline constexpr bool WD_MGROUPS = false;
inline constexpr uint32_t WD_MGROUP_MIN_BLOCKS = 4;
inline constexpr bool W_RESIDENT = true;
inline constexpr bool WD_RESIDENT = true;
inline constexpr uint32_t GU_CHUNKS = 3;
inline constexpr bool XPRIO = true;
inline constexpr uint32_t HACK_AHEAD = 2;
inline constexpr uint32_t H_ROUND_NOC1_MASK = 0;
inline constexpr bool SCATTER_ONE_SIGNAL = true;
inline constexpr uint32_t WD_SPLIT = 3;
inline constexpr bool H_MCAST_POSTED = false;
inline constexpr uint32_t MAILBOX_MAGIC = 0xC0FFEE01;
inline constexpr uint32_t MAILBOX_WORDS = 16;
inline constexpr uint32_t L1_CB_RESERVE = 70656;
inline constexpr uint32_t NOC_MAX_TRANSACTION_ID = 15;

inline constexpr uint32_t CB_X_IN = 0;
inline constexpr uint32_t CB_X_TILES = 1;
inline constexpr uint32_t CB_X_STAGE = 2;
inline constexpr uint32_t CB_W_GATE = 3;
inline constexpr uint32_t CB_W_UP = 4;
inline constexpr uint32_t CB_W_DOWN = 5;
inline constexpr uint32_t CB_H = 6;
inline constexpr uint32_t CB_IDX_SCRATCH = 7;
inline constexpr uint32_t CB_COUNTS_SCRATCH = 8;
inline constexpr uint32_t CB_GATHER_GATE = 9;
inline constexpr uint32_t CB_GATHER_UP = 10;
inline constexpr uint32_t CB_SLICE_GATE = 11;
inline constexpr uint32_t CB_SLICE_UP = 12;
inline constexpr uint32_t CB_H_SLICE = 13;
inline constexpr uint32_t CB_OUT_TILES = 14;
inline constexpr uint32_t CB_GATE_ACC = 15;
inline constexpr uint32_t CB_UP_ACC = 16;
inline constexpr uint32_t CB_GATE_SILU = 17;
inline constexpr uint32_t CB_H_LOCAL = 18;
inline constexpr uint32_t CB_OUT_INTERM = 19;
// Two independent FIFO views over CB_X_STAGE's 64-byte backing. Reader pushes
// both once after publishing the runtime-count mailbox; compute and writer
// consume one view each. CB_X_STAGE itself remains exclusively the per-row
// compute->reader tilization completion channel. No extra L1 allocation.
inline constexpr uint32_t CB_MAILBOX_WRITER = 20;
inline constexpr uint32_t CB_MAILBOX_COMPUTE = 21;

inline constexpr uint32_t SEM_X_BASE = 0;
inline constexpr uint32_t SEM_H_BASE = 2;
inline constexpr uint32_t SEM_GO = 4;
inline constexpr uint32_t SEM_DATA = 5;
inline constexpr uint32_t SEM_HSLICE = 6;
inline constexpr uint32_t SEM_XSTAGED = 7;
inline constexpr uint32_t SEM_H_RDY_BASE = 8;
inline constexpr uint32_t SEM_H_FREE = 11;
inline constexpr uint32_t SEM_WDSPLIT = 12;
inline constexpr uint32_t SEM_PHASE_FREE = 13;
inline constexpr uint32_t SEM_HROW_FREE = 14;
inline constexpr uint32_t SEM_COUNT = 15;
inline constexpr uint32_t NUM_DEVICE_SEMAPHORES = 16;

enum class FormatKey : uint8_t { Bfp8, Bf16, Weight, Out, U32, XIn };

struct CbView {
    uint32_t index;
    uint32_t pages;
    uint32_t page_size;
    FormatKey format;
};

struct CbAllocation {
    uint32_t total_size;
    std::vector<CbView> views;
};

struct ScatterPlan {
    uint32_t slice_pages = 0;
    uint32_t gather_pages = 0;
    std::vector<uint32_t> sizes;
};

class Blocking {
public:
    Blocking(
        uint32_t hgroups,
        uint32_t kgroups,
        uint32_t emb,
        uint32_t hidden,
        uint32_t m_t_max,
        uint32_t w_tile,
        uint32_t bfp8_tile,
        uint32_t bf16_tile,
        uint32_t x_stick,
        uint32_t l1_budget,
        uint32_t out_tile,
        bool enable_phase_alias,
        bool x_is_rm);

    std::vector<CbView> cb_layout(bool x_is_rm, uint32_t out_tile, uint32_t idx_page, uint32_t counts_page) const;
    std::vector<CbAllocation> cb_allocations(
        bool x_is_rm, uint32_t out_tile, uint32_t idx_page, uint32_t counts_page, bool enable_phase_alias) const;
    uint64_t l1_bytes(bool x_is_rm, uint32_t out_tile, bool enable_phase_alias) const;
    bool phase_cb_alias(uint32_t out_tile) const;
    uint32_t phase_cb_alias_pages(uint32_t out_tile) const;
    std::string describe() const;

    uint32_t hgroups;
    uint32_t kgroups;
    uint32_t num_cores;
    uint32_t emb;
    uint32_t hidden;
    uint32_t emb_t;
    uint32_t hid_t;
    uint32_t m_t_max;
    uint32_t m_eff_min;
    std::vector<uint32_t> kr_sizes;
    std::vector<uint32_t> kr_starts;
    uint32_t kr_pad;
    uint32_t gu_chunks_target;
    uint32_t hn_pad;
    uint32_t gu_chunks;
    uint32_t gu_chunk_w;
    uint32_t hn_block;
    uint32_t gu_in1_subblocks;
    bool balanced_hn;
    std::vector<uint32_t> hn_sizes;
    std::vector<uint32_t> hn_starts;
    bool wd_mrow_rounds;
    std::vector<uint32_t> ec_sizes;
    std::vector<uint32_t> ec_starts;
    uint32_t ec_max;
    uint32_t mgroup_rows;
    uint32_t mgroup_cores;
    std::vector<uint32_t> ec_group_sizes;
    std::vector<uint32_t> ec_group_starts;
    uint32_t ec_group_max;
    bool wd_mgroups;
    uint32_t wd_ec_max;
    uint32_t out_subblock_h_dn;
    uint32_t gather_pages;
    uint32_t slice_pages;
    uint32_t max_m_blocks;
    uint32_t depth_x;
    uint32_t depth_w;
    uint32_t wd_ahead;
    uint32_t depth_h;
    uint32_t hack_ahead;
    uint32_t w_tile;
    uint32_t bfp8_tile;
    uint32_t bf16_tile;
    uint32_t x_stick;
    uint32_t out_tile;
    bool enable_phase_alias;
    bool x_is_rm;
    uint32_t l1_budget;
    bool wd_resident;
    uint32_t depth_wd;
    bool wd_packed;
    uint32_t wd_split;

private:
    struct HnChoice {
        uint32_t hn_pad;
        uint32_t chunks;
        ScatterPlan plan;
        bool balanced;
    };

    HnChoice choose_hn_pad() const;
    bool depth_wd_legal(uint32_t depth) const;
    uint32_t min_depth_wd() const;
    uint32_t next_smaller_depth_wd(uint32_t depth) const;
};

uint32_t nd_shard_n_tiles(const Tensor& tensor);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::geometry
