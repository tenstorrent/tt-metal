// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Constants of the WH SDPA core loop (wh_sdpa_core.h), from tt-lab-private h3-wh src/wh_proxy.h and
// src/wh_sdpa_bench.h. One 64-query block of a head of 128 (or 64) against 64-key blocks; operand chunks of 1024 BF16.
#pragma once

#include <stdint.h>

// TRISC index of this compile (the core loop is one source for all three threads).
#if defined(UCK_CHLKC_UNPACK)
#define TT_TRISC_INDEX 0
#elif defined(UCK_CHLKC_MATH)
#define TT_TRISC_INDEX 1
#elif defined(UCK_CHLKC_PACK)
#define TT_TRISC_INDEX 2
#endif

// A head's softmax constants (scale 1/sqrt(dim)), FP32 bits, and its geometry: d slices of 16 (QK's fills) and O^T's
// halves of 64 d (PV's fills a key slice).
struct WhSdpaHead {
    uint32_t dim, slices, halves;
    uint32_t exp_a;       // 128/ln2 scale: S units to 2^-7 log2 units
    uint32_t log2_scale;  // scale/ln2
    uint32_t margin;      // m rises only past the block maximum less this: 8 / log2_scale in BF16, negated
};
inline constexpr WhSdpaHead whp_sdpa_head128 = {128u, 8u, 2u, 0x418293EEu, 0x3E0293EEu, 0xC27B0000u};
inline constexpr WhSdpaHead whp_sdpa_head64 = {64u, 4u, 1u, 0x41B8AA3Bu, 0x3E38AA3Bu, 0xC2310000u};
// Chunks of 1024 a key block takes: K's d slices, then V^T's 4 key slices of every half.
constexpr uint32_t whp_sdpa_block_chunks(const WhSdpaHead& head) { return head.slices + 4u * head.halves; }
// 65535 less the exp's offset (Schraudolph), rounded: the exp rounds 65535 - t to UINT16, whose saturation is the
// clamp at t = 0.
constexpr uint32_t whp_sdpa_exp_c = 0x47408497u;
constexpr uint32_t whp_sdpa_m_init = 0xFF7F0000u;  // the lowest finite BF16
constexpr uint32_t whp_sdpa_x_min = 0xC2FE0000u;   // -127: alpha's log2 floor
constexpr uint32_t whp_sdpa_magic = 0x4B40007Fu;   // 1.5 2^23 + 127: rounds alpha's log2 and biases its exponent
constexpr uint32_t whp_sdpa_poly[3] = {0x3F317AFDu, 0x3E7805FFu, 0x3D6150AEu};  // 2^f on [-0.5, 0.5]
constexpr uint32_t whp_sdpa_recip_seed = 0x7EF311C3u;

// A face of 64 rows by 16, or a chunk of four 16 x 16 faces, BF16.
#define WH_SDPA_FACE_BYTES 2048u
#define WH_SDPA_Q_BYTES (8u * WH_SDPA_FACE_BYTES)
#define WH_SDPA_P_BYTES (8u * WH_SDPA_FACE_BYTES)
#define WH_SDPA_KV_BYTES (16u * WH_SDPA_FACE_BYTES)
// 256 Dst rows of 16 BF16: one O^T d half, or one S^T buffer.
#define WH_SDPA_REGION_BYTES 8192u
