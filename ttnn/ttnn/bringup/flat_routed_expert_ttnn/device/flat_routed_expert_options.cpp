// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_options.hpp"

#include <fmt/format.h>

namespace ttnn::operations::bringup::flat_routed_expert {

std::vector<uint32_t> FlatLayoutOptions::key() const {
    const auto u = [](int32_t v) { return static_cast<uint32_t>(v); };
    return {drisc,        drisc_dual,     drisc_in_overlap, u(drisc_pure), u(max_down), u(x_helpers), half_width,
            u(hg_units),  u(row0),        u(row1),          u(x_path),     u(x_snakes), u(x_pieces),  u(x_fwd_ncrisc),
            u(x_fwd_noc), u(x_fwd_depth), u(x_relay_depth), x_once,        u(dw_batch), pin_ratio,    single_subgrid};
}

std::string FlatLayoutOptions::str() const {
    return fmt::format(
        "drisc {} dual {} in_overlap {} pure {} max_down {} x_helpers {} half_width {} hg_units {} rows {}..{} x_path "
        "{} "
        "snakes {} pieces {} fwd_ncrisc {} fwd_noc {} fwd_depth {} relay_depth {} x_once {} dw_batch {} pin_ratio {} "
        "single_subgrid {}",
        drisc,
        drisc_dual,
        drisc_in_overlap,
        drisc_pure,
        max_down,
        x_helpers,
        half_width,
        hg_units,
        row0,
        row1,
        x_path,
        x_snakes,
        x_pieces,
        x_fwd_ncrisc,
        x_fwd_noc,
        x_fwd_depth,
        x_relay_depth,
        x_once,
        dw_batch,
        pin_ratio,
        single_subgrid);
}

FlatProbes FlatProbes::from_env() {
    const auto read = [](const char* name) -> std::optional<std::string> {
        const char* v = std::getenv(name);
        return v != nullptr ? std::optional<std::string>(v) : std::nullopt;
    };
    FlatProbes p;
    p.flat_cmb_log = read("FLAT_CMB_LOG");
    p.flat_wp_keylog = read("FLAT_WP_KEYLOG");
    p.flat_wp_log = read("FLAT_WP_LOG");
    p.cmb_early = read("MIMO_FL_CMB_EARLY");
    p.cols = read("MIMO_FL_COLS");
    p.dch = read("MIMO_FL_DCH");
    p.dn_noc0 = read("MIMO_FL_DN_NOC0");
    p.dn_noread = read("MIMO_FL_DN_NOREAD");
    p.dring = read("MIMO_FL_DRING");
    p.drisc_alt_burst = read("MIMO_FL_DRISC_ALT_BURST");
    p.drisc_alt_dma_max = read("MIMO_FL_DRISC_ALT_DMA_MAX");
    p.drisc_alt_maxcnt = read("MIMO_FL_DRISC_ALT_MAXCNT");
    p.drisc_assign = read("MIMO_FL_DRISC_ASSIGN");
    p.drisc_burst = read("MIMO_FL_DRISC_BURST");
    p.drisc_dbg = read("MIMO_FL_DRISC_DBG");
    p.drisc_dma_max = read("MIMO_FL_DRISC_DMA_MAX");
    p.drisc_drain = read("MIMO_FL_DRISC_DRAIN");
    p.drisc_free_sent = read("MIMO_FL_DRISC_FREE_SENT");
    p.drisc_job1_noc = read("MIMO_FL_DRISC_JOB1_NOC");
    p.drisc_maxcnt = read("MIMO_FL_DRISC_MAXCNT");
    p.drisc_mpfe = read("MIMO_FL_DRISC_MPFE");
    p.drisc_noc1 = read("MIMO_FL_DRISC_NOC1");
    p.drisc_prof = read("MIMO_FL_DRISC_PROF");
    p.drisc_rd_down = read("MIMO_FL_DRISC_RD_DOWN");
    p.drisc_rd_swap = read("MIMO_FL_DRISC_RD_SWAP");
    p.drisc_slots = read("MIMO_FL_DRISC_SLOTS");
    p.drisc_stream_split = read("MIMO_FL_DRISC_STREAM_SPLIT");
    p.drisc_tiles = read("MIMO_FL_DRISC_TILES");
    p.drisc_ub = read("MIMO_FL_DRISC_UB");
    p.dump_rt = read("MIMO_FL_DUMP_RT");
    p.dw_rot = read("MIMO_FL_DW_ROT");
    p.dyn_pull = read("MIMO_FL_DYN_PULL");
    p.dyn_zone = read("MIMO_FL_DYN_ZONE");
    p.dz = read("MIMO_FL_DZ");
    p.fidelity = read("MIMO_FL_FIDELITY");
    p.fwd_vc = read("MIMO_FL_FWD_VC");
    p.gu_bf16 = read("MIMO_FL_GU_BF16");
    p.gu_full_sync = read("MIMO_FL_GU_FULL_SYNC");
    p.gu_noread = read("MIMO_FL_GU_NOREAD");
    p.gu_nreg = read("MIMO_FL_GU_NREG");
    p.gu_split = read("MIMO_FL_GU_SPLIT");
    p.hbuf = read("MIMO_FL_HBUF");
    p.hg_auto = read("MIMO_FL_HG_AUTO");
    p.hg_drisc = read("MIMO_FL_HG_DRISC");
    p.hpieces = read("MIMO_FL_HPIECES");
    p.hu = read("MIMO_FL_HU");
    p.h_nodata = read("MIMO_FL_H_NODATA");
    p.h_vc = read("MIMO_FL_H_VC");
    p.link_depth = read("MIMO_FL_LINK_DEPTH");
    p.mt = read("MIMO_FL_MT");
    p.no_act = read("MIMO_FL_NO_ACT");
    p.no_math = read("MIMO_FL_NO_MATH");
    p.no_shape = read("MIMO_FL_NO_SHAPE");
    p.pcd_r = read("MIMO_FL_PCD_R");
    p.rdn_hi = read("MIMO_FL_RDN_HI");
    p.rdown = read("MIMO_FL_RDOWN");
    p.rd_samecol = read("MIMO_FL_RD_SAMECOL");
    p.rm_chunks = read("MIMO_FL_RM_CHUNKS");
    p.sb_slots = read("MIMO_FL_SB_SLOTS");
    p.se_dbg = read("MIMO_FL_SE_DBG");
    p.skip_cols = read("MIMO_FL_SKIP_COLS");
    p.tz_skip = read("MIMO_FL_TZ_SKIP");
    p.uneven = read("MIMO_FL_UNEVEN");
    p.waitz = read("MIMO_FL_WAITZ");
    p.w_nofwd = read("MIMO_FL_W_NOFWD");
    p.w_noread = read("MIMO_FL_W_NOREAD");
    p.x2 = read("MIMO_FL_X2");
    p.xdown = read("MIMO_FL_XDOWN");
    p.xfwd = read("MIMO_FL_XFWD");
    p.xhelp_all = read("MIMO_FL_XHELP_ALL");
    p.xhelp_noc0 = read("MIMO_FL_XHELP_NOC0");
    p.xmc_nodata = read("MIMO_FL_XMC_NODATA");
    p.xprim_only = read("MIMO_FL_XPRIM_ONLY");
    p.xrd_batch = read("MIMO_FL_XRD_BATCH");
    p.xrd_big = read("MIMO_FL_XRD_BIG");
    p.xrd_dyn = read("MIMO_FL_XRD_DYN");
    p.xrd_noc = read("MIMO_FL_XRD_NOC");
    p.xrd_skip = read("MIMO_FL_XRD_SKIP");
    p.xsolo = read("MIMO_FL_XSOLO");
    p.xsolo_ids = read("MIMO_FL_XSOLO_IDS");
    p.xsolo_k = read("MIMO_FL_XSOLO_K");
    p.xsolo_noidx = read("MIMO_FL_XSOLO_NOIDX");
    p.xsolo_seq = read("MIMO_FL_XSOLO_SEQ");
    p.x_resident = read("MIMO_FL_X_RESIDENT");
    p.x_slots_max = read("MIMO_FL_X_SLOTS_MAX");
    p.yrm_nowrite = read("MIMO_FL_YRM_NOWRITE");
    p.yrm_packtile = read("MIMO_FL_YRM_PACKTILE");
    p.yrm_self = read("MIMO_FL_YRM_SELF");
    p.yrm_vc = read("MIMO_FL_YRM_VC");
    p.y_nc = read("MIMO_FL_Y_NC");
    p.zones = read("MIMO_FL_ZONES");
    return p;
}

}  // namespace ttnn::operations::bringup::flat_routed_expert
