// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    using namespace tt::constants;
    auto ignore_index = static_cast<int32_t>(get_arg(args::ignore_index));
    auto num_tiles_per_core = get_arg(args::num_tiles_per_core);
    auto start_id = get_arg(args::start_id);
    auto C = get_arg(args::C);
    auto W = get_arg(args::W);
    auto element_size = get_arg(args::element_size);

    uint32_t target_element_size = 4;  // sizeof(int32)

    const auto addrg_input = TensorAccessor(tensor::input);
    const auto addrg_target = TensorAccessor(tensor::target);
#if defined(WEIGHT)
    const auto addrg_weight = TensorAccessor(tensor::weight);
#endif

    constexpr uint32_t onetile = 1;

#if defined(DIVISOR)
    const auto addrg_divisor = TensorAccessor(tensor::divisor);

    DataflowBuffer dfb_divisor_obj(dfb::divisor);
    read_tile(dfb_divisor_obj, addrg_divisor, 0);
#endif

    uint32_t Wf = (W + FACE_WIDTH - 1) / FACE_WIDTH;
    uint32_t Wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;
    uint32_t Ct = (C + TILE_HEIGHT - 1) / TILE_HEIGHT;

    DataflowBuffer dfb_input_obj(dfb::input);
    DataflowBuffer dfb_target_obj(dfb::target);
    DataflowBuffer dfb_tmp_input_obj(dfb::tmp_input);
#if defined(WEIGHT)
    DataflowBuffer dfb_weight_obj(dfb::weight);
    DataflowBuffer dfb_tmp_weight_obj(dfb::tmp_weight);
#endif

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        uint32_t n = i / Wf;
        uint32_t w = i % Wf * FACE_WIDTH;
        auto nt = n / TILE_HEIGHT;
        auto wt = w / TILE_WIDTH;

        uint32_t traget_noc_id = nt * Wt + wt;
        uint32_t target_offset;
        get_noc_offset(n, w, target_element_size, target_offset);
        read_tile(
            dfb_target_obj,
            addrg_target,
            traget_noc_id,
            NOC_MINIMUM_READ_SIZE / element_size * target_element_size,
            target_offset);

#if defined(WEIGHT)
        dfb_tmp_weight_obj.reserve_back(onetile);
        CoreLocalMem<volatile FP32_DEST_ACC_FTYPE> tmp_weight_l1_ptr(dfb_tmp_weight_obj.get_write_ptr());
#endif

        dfb_tmp_input_obj.reserve_back(onetile);
        dfb_target_obj.wait_front(onetile);

        CoreLocalMem<volatile FP32_DEST_ACC_FTYPE> tmp_input_l1_ptr(dfb_tmp_input_obj.get_write_ptr());
        CoreLocalMem<volatile int32_t> target_l1_ptr(dfb_target_obj.get_read_ptr());

        uint32_t idx_max = std::min(w + FACE_WIDTH, W);
        for (uint32_t idx = 0; idx < idx_max; idx++) {
            int32_t target_val = target_l1_ptr[idx];

            if (target_val != ignore_index && (0 <= target_val && target_val < static_cast<int32_t>(C))) {
                uint32_t noc_id = (n * Ct * Wt) + (target_val / TILE_HEIGHT) * Wt + (w / TILE_WIDTH);
                uint32_t tilized_idx = get_tilized_idx(target_val, w + idx);
                read_value(dfb_input_obj, addrg_input, noc_id, tilized_idx);

                dfb_input_obj.wait_front(onetile);
                CoreLocalMem<volatile uint16_t> input_l1_ptr(dfb_input_obj.get_read_ptr());

                tmp_input_l1_ptr[idx] = fp32_dest_acc_cast(input_l1_ptr[tilized_idx]);

                dfb_input_obj.pop_front(onetile);
#if defined(WEIGHT)
                {
                    uint32_t noc_id = target_val / TILE_WIDTH;
                    uint32_t tilized_idx = get_tilized_idx(0, target_val);
                    read_value(dfb_weight_obj, addrg_weight, noc_id, tilized_idx);

                    dfb_weight_obj.wait_front(onetile);
                    CoreLocalMem<volatile uint16_t> weight_l1_ptr(dfb_weight_obj.get_read_ptr());

                    tmp_weight_l1_ptr[idx] = fp32_dest_acc_cast(weight_l1_ptr[tilized_idx]);
                    dfb_weight_obj.pop_front(onetile);
                }
#endif
            } else {
                tmp_input_l1_ptr[idx] = fp32_dest_acc_cast(0.0f);
            }
        }
        dfb_tmp_input_obj.push_back(onetile);
#if defined(WEIGHT)
        dfb_tmp_weight_obj.push_back(onetile);
#endif
        dfb_target_obj.pop_front(onetile);
    }
}
