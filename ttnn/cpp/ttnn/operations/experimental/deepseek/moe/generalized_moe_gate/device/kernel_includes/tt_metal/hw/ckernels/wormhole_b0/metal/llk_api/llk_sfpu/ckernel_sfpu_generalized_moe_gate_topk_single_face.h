// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "../../../../../../../tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_generalized_moe_gate_topk_single_face.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_sum_top2() {
    _generalized_moe_gate_sum_top2<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_sort_top4_groups() {
    _generalized_moe_gate_sort_top4_groups<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_top8(uint32_t eps, uint32_t scale) {
    _generalized_moe_gate_top8<APPROXIMATION_MODE, is_fp32_dest_acc_en>(eps, scale);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_top8_ungrouped(uint32_t eps, uint32_t scale) {
    _generalized_moe_gate_top8_ungrouped<APPROXIMATION_MODE, is_fp32_dest_acc_en>(eps, scale);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_probe_lanemap() {
    _gmg_probe_lanemap();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_shift_hi_groups() {
    _gmg_shift_hi_groups();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_probe_offsets() {
    _gmg_probe_offsets();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_rotate_groups_by4() {
    _gmg_rotate_groups_by4();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_topk_init() {
    _init_generalized_moe_gate_topk<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
