// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — the compile-time argument ORDER, once, for all three kernels.
//
// These lists define the positional contract consumed by the standard C++ program factory in
// `device/moe_fused_swiglu_program_factory.cpp` and the Python descriptor both use these lists.
//
// This exists because the indices used to be written out by hand. `TA_BASE = 35` and
// `CT_XMCAST = 47` were literals that had to track the length of a scalar block on the host, an
// inserted argument silently shifted every accessor in three kernels, and the workaround was to
// smuggle ~20 later parameters through `-D` defines instead. `COUNT` below is that length,
// computed rather than counted.
//
// Deliberately include-free so the COMPUTE translation unit can pull it in too.

#pragma once

#include <cstdint>

// clang-format off
#define MOE_READER_CT_ARGS(X) \
    X(INPUT_FORMAT) X(M_T_MAX) X(LOCAL_EXPERT_ID) \
    X(EMB_T) X(HID_T) X(KR_PAD) X(HN_PAD) X(EC_MAX) X(WD_EC_MAX) X(EC_GROUP_MAX) X(M_BLOCK) X(HGROUPS) X(KGROUPS) X(NUM_CORES) \
    X(SEM_GO) X(SEM_DATA) X(SEM_HSLICE) X(SEM_XSTAGED) X(SEM_H_RDY_BASE) X(SEM_H_FREE) X(SEM_WDSPLIT) X(SEM_HROW_FREE) \
    X(SEM_PHASE_FREE) X(PHASE_CB_ALIAS) X(H_ROUND_NOC1_MASK) X(SCATTER_ONE_SIGNAL) \
    X(X_PAGE) X(X_SLICE) X(COUNTS_PAGE) X(IDX_PAGE) X(W_TILE_BYTES) X(BFP8_TILE) X(MAILBOX_MAGIC) \
    X(WD_AHEAD) X(M_EFF_MIN) X(W_RESIDENT) X(WD_RESIDENT) X(WD_MROW_ROUNDS) X(WD_MGROUPS) X(MGROUP_ROWS) X(WD_MGROUP_MIN_BLOCKS) X(GU_CHUNKS) X(XPRIO) X(HACK_AHEAD) \
    X(DEPTH_H) X(DEPTH_X) X(WD_SPLIT) X(WG_SHARD_W) X(WD_SHARD_W) X(GATHER_PAGES) \
    X(NEED_START) X(READ_X_AT_OFFSET) X(START_PAGE) \
    X(CB_X_IN) X(CB_X_TILES) X(CB_X_STAGE) X(CB_W_GATE) X(CB_W_DOWN) X(CB_H) X(CB_H_LOCAL) \
    X(CB_IDX_SCRATCH) X(CB_COUNTS_SCRATCH) X(CB_GATHER_GATE) X(CB_GATHER_UP) X(CB_UP_ACC) X(CB_MAILBOX_COMPUTE) X(CB_MAILBOX_WRITER)

#define MOE_WRITER_CT_ARGS(X) \
    X(EMB_T) X(HID_T) X(KR_PAD) X(HN_PAD) X(EC_MAX) X(WD_EC_MAX) X(EC_GROUP_MAX) X(M_BLOCK) X(HGROUPS) X(KGROUPS) X(NUM_CORES) \
    X(SEM_GO) X(SEM_DATA) X(SEM_HSLICE) X(SEM_XSTAGED) X(SEM_H_RDY_BASE) X(SEM_H_FREE) X(SEM_WDSPLIT) X(SEM_PHASE_FREE) X(SEM_HROW_FREE) X(PHASE_CB_ALIAS) \
    X(W_TILE_BYTES) X(BFP8_TILE) X(OUT_TILE_BYTES) X(MAILBOX_MAGIC) X(M_EFF_MIN) X(W_RESIDENT) X(WD_RESIDENT) \
    X(GU_CHUNKS) X(XPRIO) X(WD_MROW_ROUNDS) X(WD_MGROUPS) X(MGROUP_ROWS) X(WD_MGROUP_MIN_BLOCKS) X(DEPTH_H) X(H_ROUND_NOC1_MASK) X(SCATTER_ONE_SIGNAL) X(WD_SPLIT) X(WG_SHARD_W) X(WD_SHARD_W) X(GATHER_PAGES) X(PHASE_ALIAS_PAGES) \
    X(DIRECT_WRITE) X(OUT_M_T) \
    X(CB_W_UP) X(CB_W_DOWN) X(CB_OUT_TILES) X(CB_GATE_ACC) X(CB_UP_ACC) \
    X(CB_GATHER_GATE) X(CB_GATHER_UP) X(CB_H_SLICE) X(CB_H_LOCAL) X(CB_H) X(CB_MAILBOX_WRITER)

#define MOE_COMPUTE_CT_ARGS(X) \
    X(M_BLOCK) X(KR_PAD) X(HN_PAD) X(EC_MAX) X(WD_EC_MAX) X(EC_GROUP_MAX) X(HGROUPS) X(KGROUPS) X(HID_T) X(INPUT_FORMAT) \
    X(OUT_SUBBLOCK_H_GU) X(OUT_SUBBLOCK_H_DN) X(OUT_SUBBLOCK_H_DN_MAX) X(MAILBOX_MAGIC) X(M_EFF_MIN) X(DEPTH_X) X(HN_BLOCK) X(WD_RESIDENT) X(WD_MROW_ROUNDS) X(WD_MGROUPS) X(MGROUP_ROWS) X(WD_MGROUP_MIN_BLOCKS) \
    X(GU_CHUNKS) X(ELTWISE_BLK) X(DEST_LIMIT) X(GATHER_PAGES) \
    X(CB_X_IN) X(CB_X_TILES) X(CB_X_STAGE) X(CB_MAILBOX_COMPUTE) X(CB_W_GATE) X(CB_W_UP) X(CB_W_DOWN) \
    X(CB_GATE_ACC) X(CB_UP_ACC) X(CB_GATE_SILU) X(CB_H_LOCAL) X(CB_H) \
    X(CB_OUT_INTERM) X(CB_OUT_TILES) \
    X(CB_GATHER_GATE) X(CB_GATHER_UP) X(CB_SLICE_GATE) X(CB_SLICE_UP) X(CB_H_SLICE)
// clang-format on

// `CT(NAME)` reads the argument by name; `Ct::COUNT` is the length of the scalar block, i.e. the
// offset the mcast / TensorAccessor blocks start at.
#define MOE_DECLARE_CT_ENUM(LIST) enum class Ct : uint32_t { LIST(MOE_CT_ENUMERATOR) COUNT }
#define MOE_CT_ENUMERATOR(name) name,
#define CT(name) get_compile_time_arg_val(static_cast<uint32_t>(Ct::name))
#define CT_COUNT static_cast<uint32_t>(Ct::COUNT)
