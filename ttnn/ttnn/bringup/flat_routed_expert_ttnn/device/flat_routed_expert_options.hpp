// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace ttnn::operations::bringup::flat_routed_expert {

// The layout / program choices a caller makes for the flat routed expert: typed, part of the plan cache key and of
// every program hash, passed from Python (FlatRoutedExpert(layout=...), presets in flat_layouts.py). Every default is
// the plan's own choice - what an unset environment gave before these were options (ENV_CLEANUP.md). -1: unset.
struct FlatLayoutOptions {
    bool drisc = true;              // DRAM-core (DRISC) gate/up weight readers where the device runs DRISC firmware
    bool drisc_dual = true;         // DRISC slots may also be served by their Tensix readers (dual, chosen per call)
    bool drisc_in_overlap = false;  // DRISC inside the combine overlaps (the programs' plans have cmb_rt set)
    // pure DRISC: every slot's gate/up from the DRAM cores, no Tensix weight reads (tiles in switch mode); -1: pure
    // for plans without reader down columns, dual / hybrid for the others; 0 / 1 force it (preset DRISC_ONLY)
    int32_t drisc_pure = -1;
    int32_t max_down = -1;         // down cores at most (set at all: no automatic down-core trim; 0 or more: no cap)
    int32_t x_helpers = -1;        // x helper relays per rectangle, 1..4 (-1: 2 with M-groups, else 1)
    bool half_width = false;       // 16-column gate/up units (half-width gate / up tiles), with hg_units a core
    int32_t hg_units = -1;         // 16-column units per gate/up core (set at all: no automatic units layout)
    int32_t row0 = -1, row1 = -1;  // plan on grid rows [row0, row1] only (-1: every row)
    // x delivery (program factory): -1 unicast snakes where the plan allows (one primary per rectangle, full
    // rectangles, no x_once / X2 / XFWD), else the multicast; 0 multicast, 1 tree, 2 column chains, 3 snakes
    int32_t x_path = -1;
    int32_t x_snakes = -1;       // snakes per rectangle (-1: 1)
    int32_t x_pieces = -1;       // cut-through pieces per x block (-1: 4, 2 or 1 - the largest 16 B aligned)
    int32_t x_fwd_ncrisc = -1;   // the forwarding on the gate/up NCRISC (-1: on for column chains / snakes)
    int32_t x_fwd_noc = -1;      // the forwarding NoC (-1: NOC0 with the NCRISC forwarder, else the receivers')
    int32_t x_fwd_depth = -1;    // pieces in flight per forwarder (-1: 3 with the NCRISC forwarder, else 6)
    int32_t x_relay_depth = -1;  // relay head-write batches in flight, 1..8 (-1: 4)
    bool x_once = false;         // x read from DRAM once: rectangle 0's primary multicasts into both rectangles
    int32_t dw_batch = -1;       // down-weight blocks per read barrier (-1: DW_BATCH)
    uint32_t pin_ratio = 8;      // pin only an expert this many times the next one's rows (0: always pin)
    // one subgrid (one M group of gate/up cores) also for small I (It <= 32 tiles takes 2 or 3 subgrids by default,
    // which the overlap-era program factory no longer builds): with 16-column units, e.g. a split shared expert's slice
    bool single_subgrid = false;

    // every field, for the plan cache key and the program hashes
    std::vector<uint32_t> key() const;
    // one line for the logs
    std::string str() const;
    bool operator==(const FlatLayoutOptions&) const = default;
};

// Perf / debug probes of the flat routed expert. Never set by model code. Read from the environment in this one place
// (FlatProbes::from_env, at each plan / program build as before; not in the program hash); every value keeps its
// MIMO_FL_* / FLAT_* name and is parsed where it is used (cstr / on / u32). Results of the probes: the worklogs.
struct FlatProbes {
    std::optional<std::string> flat_cmb_log;        // FLAT_CMB_LOG (debug output)
    std::optional<std::string> flat_wp_keylog;      // FLAT_WP_KEYLOG (debug output)
    std::optional<std::string> flat_wp_log;         // FLAT_WP_LOG (debug output)
    std::optional<std::string> cmb_early;           // MIMO_FL_CMB_EARLY
    std::optional<std::string> cols;                // MIMO_FL_COLS
    std::optional<std::string> dch;                 // MIMO_FL_DCH
    std::optional<std::string> dn_noc0;             // MIMO_FL_DN_NOC0
    std::optional<std::string> dn_noread;           // MIMO_FL_DN_NOREAD
    std::optional<std::string> dring;               // MIMO_FL_DRING
    std::optional<std::string> drisc_alt_burst;     // MIMO_FL_DRISC_ALT_BURST
    std::optional<std::string> drisc_alt_dma_max;   // MIMO_FL_DRISC_ALT_DMA_MAX
    std::optional<std::string> drisc_alt_maxcnt;    // MIMO_FL_DRISC_ALT_MAXCNT
    std::optional<std::string> drisc_assign;        // MIMO_FL_DRISC_ASSIGN
    std::optional<std::string> drisc_burst;         // MIMO_FL_DRISC_BURST
    std::optional<std::string> drisc_dbg;           // MIMO_FL_DRISC_DBG (debug output)
    std::optional<std::string> drisc_dma_max;       // MIMO_FL_DRISC_DMA_MAX
    std::optional<std::string> drisc_drain;         // MIMO_FL_DRISC_DRAIN
    std::optional<std::string> drisc_free_sent;     // MIMO_FL_DRISC_FREE_SENT
    std::optional<std::string> drisc_job1_noc;      // MIMO_FL_DRISC_JOB1_NOC
    std::optional<std::string> drisc_maxcnt;        // MIMO_FL_DRISC_MAXCNT
    std::optional<std::string> drisc_mpfe;          // MIMO_FL_DRISC_MPFE
    std::optional<std::string> drisc_noc1;          // MIMO_FL_DRISC_NOC1
    std::optional<std::string> drisc_prof;          // MIMO_FL_DRISC_PROF (debug output)
    std::optional<std::string> drisc_rd_down;       // MIMO_FL_DRISC_RD_DOWN
    std::optional<std::string> drisc_rd_swap;       // MIMO_FL_DRISC_RD_SWAP
    std::optional<std::string> drisc_slots;         // MIMO_FL_DRISC_SLOTS
    std::optional<std::string> drisc_stream_split;  // MIMO_FL_DRISC_STREAM_SPLIT
    std::optional<std::string> drisc_tiles;         // MIMO_FL_DRISC_TILES
    std::optional<std::string> drisc_ub;            // MIMO_FL_DRISC_UB
    std::optional<std::string> dump_rt;             // MIMO_FL_DUMP_RT (debug output)
    std::optional<std::string> dw_rot;              // MIMO_FL_DW_ROT
    std::optional<std::string> dyn_pull;            // MIMO_FL_DYN_PULL
    std::optional<std::string> dyn_zone;            // MIMO_FL_DYN_ZONE (debug output)
    std::optional<std::string> dz;                  // MIMO_FL_DZ (debug output)
    std::optional<std::string> fidelity;            // MIMO_FL_FIDELITY
    std::optional<std::string> fwd_vc;              // MIMO_FL_FWD_VC
    std::optional<std::string> gu_bf16;             // MIMO_FL_GU_BF16
    std::optional<std::string> gu_full_sync;        // MIMO_FL_GU_FULL_SYNC
    std::optional<std::string> gu_noread;           // MIMO_FL_GU_NOREAD
    std::optional<std::string> gu_nreg;             // MIMO_FL_GU_NREG
    std::optional<std::string> gu_split;            // MIMO_FL_GU_SPLIT
    std::optional<std::string> hbuf;                // MIMO_FL_HBUF
    std::optional<std::string> hg_auto;             // MIMO_FL_HG_AUTO
    std::optional<std::string> hg_drisc;            // MIMO_FL_HG_DRISC
    std::optional<std::string> hpieces;             // MIMO_FL_HPIECES
    std::optional<std::string> hu;                  // MIMO_FL_HU
    std::optional<std::string> h_nodata;            // MIMO_FL_H_NODATA
    std::optional<std::string> h_vc;                // MIMO_FL_H_VC
    std::optional<std::string> link_depth;          // MIMO_FL_LINK_DEPTH
    std::optional<std::string> mt;                  // MIMO_FL_MT
    std::optional<std::string> no_act;              // MIMO_FL_NO_ACT
    std::optional<std::string> no_math;             // MIMO_FL_NO_MATH
    std::optional<std::string> no_shape;            // MIMO_FL_NO_SHAPE
    std::optional<std::string> pcd_r;               // MIMO_FL_PCD_R
    std::optional<std::string> rdn_hi;              // MIMO_FL_RDN_HI
    std::optional<std::string> rdown;               // MIMO_FL_RDOWN
    std::optional<std::string> rd_samecol;          // MIMO_FL_RD_SAMECOL
    std::optional<std::string> rm_chunks;           // MIMO_FL_RM_CHUNKS
    std::optional<std::string> sb_slots;            // MIMO_FL_SB_SLOTS
    std::optional<std::string> se_dbg;              // MIMO_FL_SE_DBG (debug output)
    std::optional<std::string> skip_cols;           // MIMO_FL_SKIP_COLS
    std::optional<std::string> tz_skip;             // MIMO_FL_TZ_SKIP
    std::optional<std::string> uneven;              // MIMO_FL_UNEVEN
    std::optional<std::string> waitz;               // MIMO_FL_WAITZ (debug output)
    std::optional<std::string> w_nofwd;             // MIMO_FL_W_NOFWD
    std::optional<std::string> w_noread;            // MIMO_FL_W_NOREAD
    std::optional<std::string> x2;                  // MIMO_FL_X2
    std::optional<std::string> xdown;               // MIMO_FL_XDOWN
    std::optional<std::string> xfwd;                // MIMO_FL_XFWD
    std::optional<std::string> xhelp_all;           // MIMO_FL_XHELP_ALL
    std::optional<std::string> xhelp_noc0;          // MIMO_FL_XHELP_NOC0
    std::optional<std::string> xmc_nodata;          // MIMO_FL_XMC_NODATA
    std::optional<std::string> xprim_only;          // MIMO_FL_XPRIM_ONLY
    std::optional<std::string> xrd_batch;           // MIMO_FL_XRD_BATCH
    std::optional<std::string> xrd_big;             // MIMO_FL_XRD_BIG
    std::optional<std::string> xrd_dyn;             // MIMO_FL_XRD_DYN
    std::optional<std::string> xrd_noc;             // MIMO_FL_XRD_NOC
    std::optional<std::string> xrd_skip;            // MIMO_FL_XRD_SKIP
    std::optional<std::string> xsolo;               // MIMO_FL_XSOLO
    std::optional<std::string> xsolo_ids;           // MIMO_FL_XSOLO_IDS
    std::optional<std::string> xsolo_k;             // MIMO_FL_XSOLO_K
    std::optional<std::string> xsolo_noidx;         // MIMO_FL_XSOLO_NOIDX
    std::optional<std::string> xsolo_seq;           // MIMO_FL_XSOLO_SEQ
    std::optional<std::string> x_resident;          // MIMO_FL_X_RESIDENT
    std::optional<std::string> x_slots_max;         // MIMO_FL_X_SLOTS_MAX
    std::optional<std::string> yrm_nowrite;         // MIMO_FL_YRM_NOWRITE
    std::optional<std::string> yrm_packtile;        // MIMO_FL_YRM_PACKTILE
    std::optional<std::string> yrm_self;            // MIMO_FL_YRM_SELF
    std::optional<std::string> yrm_vc;              // MIMO_FL_YRM_VC
    std::optional<std::string> y_nc;                // MIMO_FL_Y_NC
    std::optional<std::string> zones;               // MIMO_FL_ZONES (debug output)
    static FlatProbes from_env();
};

// a probe's raw value (nullptr: unset), as std::getenv gave it
inline const char* cstr(const std::optional<std::string>& v) { return v ? v->c_str() : nullptr; }
// a probe set to a non-zero number
inline bool on(const std::optional<std::string>& v) { return v && std::atoi(v->c_str()) != 0; }
// a probe's number, or dflt when unset
inline uint32_t u32(const std::optional<std::string>& v, uint32_t dflt) {
    return v ? static_cast<uint32_t>(std::atoi(v->c_str())) : dflt;
}

}  // namespace ttnn::operations::bringup::flat_routed_expert
