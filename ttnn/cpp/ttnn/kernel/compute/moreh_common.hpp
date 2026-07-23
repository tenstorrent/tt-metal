/*
 * SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/mask.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"

namespace ckernel {

ALWI void pack_tile_with_dt(uint32_t ifrom_dst, DataflowBuffer icb) {
#if defined FP32_DEST_ACC_EN
    pack_reconfig_data_format(icb.get_id());
#endif
    pack_tile(ifrom_dst, icb.get_id());
}

ALWI void copy_tile_init_with_dt(DataflowBuffer icb, uint32_t transpose = 0) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format_srca(icb.get_id());
#endif
    copy_init(icb.get_id(), transpose);
}

ALWI void add_tiles_init_with_dt(DataflowBuffer icb0, DataflowBuffer icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    add_tiles_init(icb0.get_id(), icb1.get_id());
}

ALWI void add_bcast_rows_init_short_with_dt(DataflowBuffer icb0, DataflowBuffer icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    add_bcast_rows_init_short(icb0.get_id(), icb1.get_id());
}

ALWI void add_bcast_cols_init_short_with_dt(DataflowBuffer icb0, DataflowBuffer icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    add_bcast_cols_init_short(icb0.get_id(), icb1.get_id());
}

ALWI void add_bcast_scalar_init_short_with_dt(DataflowBuffer icb0, DataflowBuffer icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    add_bcast_scalar_init_short(icb0.get_id(), icb1.get_id());
}

ALWI void sub_tiles_init_with_dt(DataflowBuffer icb0, DataflowBuffer icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    sub_tiles_init(icb0.get_id(), icb1.get_id());
}

ALWI void sub_bcast_rows_init_short_with_dt(DataflowBuffer icb0, DataflowBuffer icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWSUB, BroadcastType::ROW, MathFidelity::LoFi>(
        icb0.get_id(), icb1.get_id())));
    UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(icb0.get_id(), icb1.get_id())));
}

ALWI void sub_bcast_cols_init_short_with_dt(DataflowBuffer icb0, DataflowBuffer icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    sub_bcast_cols_init_short(icb0.get_id(), icb1.get_id());
}

ALWI void sub_tiles_bcast_scalar_init_short_with_dt(DataflowBuffer icb0, DataflowBuffer icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    sub_tiles_bcast_scalar_init_short(icb0.get_id(), icb1.get_id());
}

ALWI void mul_tiles_init_with_dt(DataflowBuffer icb0, DataflowBuffer icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    mul_tiles_init(icb0.get_id(), icb1.get_id());
}

ALWI void mul_bcast_rows_init_short_with_dt(DataflowBuffer icb0, DataflowBuffer icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    mul_bcast_rows_init_short(icb0.get_id(), icb1.get_id());
}

ALWI void mul_bcast_cols_init_short_with_dt(DataflowBuffer icb0, DataflowBuffer icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    mul_bcast_cols_init_short(icb0.get_id(), icb1.get_id());
}

ALWI void mul_tiles_bcast_scalar_init_short_with_dt(DataflowBuffer icb0, DataflowBuffer icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    mul_tiles_bcast_scalar_init_short(icb0.get_id(), icb1.get_id());
}

class ArgFetcher {
private:
    int arg_idx = 0;

public:
    template <typename T>
    T get_next_arg_val() {
        return get_arg_val<T>(arg_idx++);
    }
};

ALWI void mul_tiles_to_cb(
    DataflowBuffer icb0,
    DataflowBuffer icb1,
    DataflowBuffer ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);
    icb0.wait_front(itile0 + 1);
    icb1.wait_front(itile1 + 1);

    tile_regs_acquire();
    mul_tiles_init_with_dt(icb0, icb1);
    mul_tiles(icb0.get_id(), icb1.get_id(), itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        icb0.pop_front(pop0);
    }
    if (pop1) {
        icb1.pop_front(pop1);
    }

    ocb.push_back(onetile);
}

ALWI void mul_tiles_and_negative_to_cb(
    DataflowBuffer icb0,
    DataflowBuffer icb1,
    DataflowBuffer ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);
    icb0.wait_front(itile0 + 1);
    icb1.wait_front(itile1 + 1);

    tile_regs_acquire();
    mul_tiles_init_with_dt(icb0, icb1);
    mul_tiles(icb0.get_id(), icb1.get_id(), itile0, itile1, dst0);

    negative_tile_init();
    negative_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        icb0.pop_front(pop0);
    }
    if (pop1) {
        icb1.pop_front(pop1);
    }

    ocb.push_back(onetile);
}

ALWI void mul_tiles_and_mask_tile_to_cb(
    DataflowBuffer icb0,
    DataflowBuffer icb1,
    DataflowBuffer maskcb,
    DataflowBuffer ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t mtile = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1,
    uint32_t popm = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);
    icb0.wait_front(itile0 + 1);
    icb1.wait_front(itile1 + 1);
    maskcb.wait_front(mtile + 1);

    tile_regs_acquire();
    mul_tiles_init_with_dt(icb0, icb1);
    mul_tiles(icb0.get_id(), icb1.get_id(), itile0, itile1, dst0);

    constexpr int dst_mask = 1;
    copy_tile_init_with_dt(maskcb);
    copy_tile(maskcb.get_id(), mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst0, dst_mask);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        icb0.pop_front(pop0);
    }
    if (pop1) {
        icb1.pop_front(pop1);
    }
    if (popm) {
        maskcb.pop_front(popm);
    }

    ocb.push_back(onetile);
}

ALWI void mul_tiles_log_to_cb(
    DataflowBuffer icb0,
    DataflowBuffer icb1,
    DataflowBuffer ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);
    icb0.wait_front(itile0 + 1);
    icb1.wait_front(itile1 + 1);

    tile_regs_acquire();
    mul_tiles_init_with_dt(icb0, icb1);
    mul_tiles(icb0.get_id(), icb1.get_id(), itile0, itile1, dst0);

    log_tile_init();
    log_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        icb0.pop_front(pop0);
    }
    if (pop1) {
        icb1.pop_front(pop1);
    }

    ocb.push_back(onetile);
}

ALWI void mul_tiles_bcast_rows_to_cb(
    DataflowBuffer icb0,
    DataflowBuffer icb1,
    DataflowBuffer ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);

    icb0.wait_front(itile0 + 1);
    icb1.wait_front(itile1 + 1);

    tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    mul_bcast_rows_init_short(icb0.get_id(), icb1.get_id());
    mul_tiles_bcast_rows(icb0.get_id(), icb1.get_id(), itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        icb0.pop_front(pop0);
    }
    if (pop1) {
        icb1.pop_front(pop1);
    }

    ocb.push_back(onetile);
}

ALWI void mul_tiles_bcast_rows_log_to_cb(
    DataflowBuffer icb0,
    DataflowBuffer icb1,
    DataflowBuffer ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);

    icb0.wait_front(itile0 + 1);
    icb1.wait_front(itile1 + 1);

    tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    mul_bcast_rows_init_short(icb0.get_id(), icb1.get_id());
    mul_tiles_bcast_rows(icb0.get_id(), icb1.get_id(), itile0, itile1, dst0);

    log_tile_init();
    log_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        icb0.pop_front(pop0);
    }
    if (pop1) {
        icb1.pop_front(pop1);
    }

    ocb.push_back(onetile);
}

ALWI void mul_tiles_bcast_cols_to_cb(
    DataflowBuffer icb0,
    DataflowBuffer icb1,
    DataflowBuffer ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);

    icb0.wait_front(itile0 + 1);
    icb1.wait_front(itile1 + 1);

    tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    mul_bcast_cols_init_short(icb0.get_id(), icb1.get_id());
    mul_tiles_bcast_cols(icb0.get_id(), icb1.get_id(), itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        icb0.pop_front(pop0);
    }
    if (pop1) {
        icb1.pop_front(pop1);
    }

    ocb.push_back(onetile);
}

ALWI void mul_tiles_bcast_cols_log_to_cb(
    DataflowBuffer icb0,
    DataflowBuffer icb1,
    DataflowBuffer ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);

    icb0.wait_front(itile0 + 1);
    icb1.wait_front(itile1 + 1);

    tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    mul_bcast_cols_init_short(icb0.get_id(), icb1.get_id());
    mul_tiles_bcast_cols(icb0.get_id(), icb1.get_id(), itile0, itile1, dst0);

    log_tile_init();
    log_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        icb0.pop_front(pop0);
    }
    if (pop1) {
        icb1.pop_front(pop1);
    }

    ocb.push_back(onetile);
}

ALWI void copy_tile_to_cb(DataflowBuffer icb, DataflowBuffer ocb, uint32_t itile = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);
    icb.wait_front(itile + 1);

    tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    copy_tile(icb.get_id(), itile, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop) {
        icb.pop_front(pop);
    }
    ocb.push_back(onetile);
}

ALWI void sign_tile_to_cb(DataflowBuffer icb, DataflowBuffer ocb, uint32_t itile = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);
    icb.wait_front(itile + 1);

    tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    copy_tile(icb.get_id(), itile, dst0);

    sign_tile_init();
    sign_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop) {
        icb.pop_front(pop);
    }
    ocb.push_back(onetile);
}

ALWI void add_tiles_to_cb(
    DataflowBuffer icb0,
    DataflowBuffer icb1,
    DataflowBuffer ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);
    icb0.wait_front(itile0 + 1);
    icb1.wait_front(itile1 + 1);

    tile_regs_acquire();
    add_tiles_init_with_dt(icb0, icb1);
    add_tiles(icb0.get_id(), icb1.get_id(), itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        icb0.pop_front(pop0);
    }
    if (pop1) {
        icb1.pop_front(pop1);
    }

    ocb.push_back(onetile);
}

ALWI void mask_tile_to_cb(
    DataflowBuffer icb,
    DataflowBuffer maskcb,
    DataflowBuffer ocb,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    constexpr int dst_mask = 1;

    ocb.reserve_back(onetile);
    icb.wait_front(itile + 1);
    maskcb.wait_front(mtile + 1);

    tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    copy_tile(icb.get_id(), itile, dst0);

    copy_tile_init_with_dt(maskcb);
    copy_tile(maskcb.get_id(), mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst0, dst_mask);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop) {
        icb.pop_front(pop);
    }
    if (popm) {
        maskcb.pop_front(popm);
    }

    ocb.push_back(onetile);
}

ALWI void sub_tiles_bcast_cols_to_cb(
    DataflowBuffer icb0,
    DataflowBuffer icb1,
    DataflowBuffer ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);

    icb0.wait_front(itile0 + 1);
    icb1.wait_front(itile1 + 1);

    tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    sub_bcast_cols_init_short(icb0.get_id(), icb1.get_id());
    sub_tiles_bcast<BroadcastType::COL>(icb0.get_id(), icb1.get_id(), itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        icb0.pop_front(pop0);
    }
    if (pop1) {
        icb1.pop_front(pop1);
    }

    ocb.push_back(onetile);
}

ALWI void sub_tiles_bcast_rows_to_cb(
    DataflowBuffer icb0,
    DataflowBuffer icb1,
    DataflowBuffer ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);

    icb0.wait_front(itile0 + 1);
    icb1.wait_front(itile1 + 1);

    tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    // sub_bcast_rows_init_short();
    {
        MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWSUB, BroadcastType::ROW, MathFidelity::LoFi>(
            icb0.get_id(), icb1.get_id())));
        UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(icb0.get_id(), icb1.get_id())));
    }
    sub_tiles_bcast<BroadcastType::ROW>(icb0.get_id(), icb1.get_id(), itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        icb0.pop_front(pop0);
    }
    if (pop1) {
        icb1.pop_front(pop1);
    }

    ocb.push_back(onetile);
}

ALWI void sub_tiles_to_cb(
    DataflowBuffer icb0,
    DataflowBuffer icb1,
    DataflowBuffer ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);
    icb0.wait_front(itile0 + 1);
    icb1.wait_front(itile1 + 1);

    tile_regs_acquire();
    sub_tiles_init_with_dt(icb0, icb1);
    sub_tiles(icb0.get_id(), icb1.get_id(), itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        icb0.pop_front(pop0);
    }
    if (pop1) {
        icb1.pop_front(pop1);
    }

    ocb.push_back(onetile);
}

ALWI void exp_tile_to_cb(
    DataflowBuffer icb, DataflowBuffer ocb, uint32_t itile = 0, uint32_t dst = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;

    ocb.reserve_back(onetile);
    icb.wait_front(itile + 1);

    tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    copy_tile(icb.get_id(), itile, dst);

    exp_tile_init();
    exp_tile(dst);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst, ocb);
    tile_regs_release();

    if (pop) {
        icb.pop_front(pop);
    }
    ocb.push_back(onetile);
}

ALWI void rexp_tile_to_cb(
    DataflowBuffer icb, DataflowBuffer ocb, uint32_t itile = 0, uint32_t dst = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;

    ocb.reserve_back(onetile);
    icb.wait_front(itile + 1);

    tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    copy_tile(icb.get_id(), itile, dst);

    negative_tile_init();
    negative_tile(dst);

    exp_tile_init();
    exp_tile(dst);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst, ocb);
    tile_regs_release();

    if (pop) {
        icb.pop_front(pop);
    }
    ocb.push_back(onetile);
}

ALWI void exp_tile_and_mask_tile_to_cb(
    DataflowBuffer icb,
    DataflowBuffer maskcb,
    DataflowBuffer ocb,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1,
    uint32_t dst = 0,
    uint32_t dst_mask = 1) {
    constexpr uint32_t onetile = 1;

    ocb.reserve_back(onetile);
    icb.wait_front(itile + 1);
    maskcb.wait_front(mtile + 1);

    tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    copy_tile(icb.get_id(), itile, dst);

    if (pop) {
        icb.pop_front(pop);
    }

    exp_tile_init();
    exp_tile(dst);

    copy_tile_init_with_dt(maskcb);
    copy_tile(maskcb.get_id(), mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst, dst_mask);
    tile_regs_commit();

    if (popm) {
        maskcb.pop_front(popm);
    }

    tile_regs_wait();
    pack_tile_with_dt(dst, ocb);
    tile_regs_release();

    ocb.push_back(onetile);
}

ALWI void rexp_tile_and_mask_tile_to_cb(
    DataflowBuffer icb,
    DataflowBuffer maskcb,
    DataflowBuffer ocb,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1,
    uint32_t dst = 0,
    uint32_t dst_mask = 1) {
    constexpr uint32_t onetile = 1;

    ocb.reserve_back(onetile);
    icb.wait_front(itile + 1);
    maskcb.wait_front(mtile + 1);

    tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    copy_tile(icb.get_id(), itile, dst);

    if (pop) {
        icb.pop_front(pop);
    }

    negative_tile_init();
    negative_tile(dst);

    exp_tile_init();
    exp_tile(dst);

    copy_tile_init_with_dt(maskcb);
    copy_tile(maskcb.get_id(), mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst, dst_mask);
    tile_regs_commit();

    if (popm) {
        maskcb.pop_front(popm);
    }

    tile_regs_wait();
    pack_tile_with_dt(dst, ocb);
    tile_regs_release();

    ocb.push_back(onetile);
}

ALWI void recip_tile_to_cb(DataflowBuffer icb, DataflowBuffer ocb, uint32_t itile = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);
    icb.wait_front(itile + 1);

    tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    copy_tile(icb.get_id(), itile, dst0);

    recip_tile_init();
    recip_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop) {
        icb.pop_front(pop);
    }
    ocb.push_back(onetile);
}

ALWI void log_tile_to_cb(DataflowBuffer icb, DataflowBuffer ocb, uint32_t itile = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    ocb.reserve_back(onetile);
    icb.wait_front(itile + 1);

    tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    copy_tile(icb.get_id(), itile, dst0);

    log_tile_init();
    log_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop) {
        icb.pop_front(pop);
    }
    ocb.push_back(onetile);
}

// TODO(seunghwan100): If p is 2 and decimal is 0, we can use sqrt_tile.
ALWI void power_tile_to_cb(
    DataflowBuffer cb_x,
    DataflowBuffer cb_xpow,
    DataflowBuffer cb_logx,
    DataflowBuffer cb_decimal,
    DataflowBuffer cb_exp_lxmd,
    DataflowBuffer cb_correct_xpow,
    uint32_t p,
    bool p_is_negative) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    // x^p
    tile_regs_acquire();
    cb_x.wait_front(onetile);
    cb_xpow.reserve_back(onetile);

    copy_tile_init_with_dt(cb_x);
    copy_tile(cb_x.get_id(), 0, dst0);

    power_iterative_tile_init();
    power_iterative_tile(dst0, p);

    if (p_is_negative) {
        recip_tile_init();
        recip_tile(dst0);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_xpow);
    tile_regs_release();

    cb_xpow.push_back(onetile);
    // We don't pop cb_x here.

    // log(x)
    tile_regs_acquire();
    cb_logx.reserve_back(onetile);

    copy_tile_init_with_dt(cb_x);
    copy_tile(cb_x.get_id(), 0, dst0);

    log_tile_init();
    log_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_logx);
    tile_regs_release();

    cb_x.pop_front(onetile);
    cb_logx.push_back(onetile);

    // exp(log(x) * decimal)
    tile_regs_acquire();
    cb_logx.wait_front(onetile);
    cb_exp_lxmd.reserve_back(onetile);

    mul_tiles_init_with_dt(cb_logx, cb_decimal);
    mul_tiles(cb_logx.get_id(), cb_decimal.get_id(), 0, 0, dst0);

    exp_tile_init();
    exp_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_exp_lxmd);
    tile_regs_release();

    cb_logx.pop_front(onetile);
    cb_exp_lxmd.push_back(onetile);

    // x^p * exp(log(x) * decimal)(==(x + decimal)^p)
    tile_regs_acquire();
    cb_xpow.wait_front(onetile);
    cb_exp_lxmd.wait_front(onetile);
    cb_correct_xpow.reserve_back(onetile);

    mul_tiles_init_with_dt(cb_xpow, cb_exp_lxmd);
    mul_tiles(cb_xpow.get_id(), cb_exp_lxmd.get_id(), 0, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_correct_xpow);
    tile_regs_release();

    cb_xpow.pop_front(onetile);
    cb_exp_lxmd.pop_front(onetile);
    cb_correct_xpow.push_back(onetile);
}

ALWI void power_tile_with_abs_x_to_cb(
    DataflowBuffer cb_x,
    DataflowBuffer cb_xpow,
    DataflowBuffer cb_logx,
    DataflowBuffer cb_decimal,
    DataflowBuffer cb_exp_lxmd,
    DataflowBuffer cb_correct_xpow,
    uint32_t p,
    bool p_is_negative) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    // x^p
    tile_regs_acquire();
    cb_x.wait_front(onetile);
    cb_xpow.reserve_back(onetile);

    copy_tile_init_with_dt(cb_x);
    copy_tile(cb_x.get_id(), 0, dst0);

    abs_tile_init();
    abs_tile(dst0);

    power_iterative_tile_init();
    power_iterative_tile(dst0, p);

    if (p_is_negative) {
        recip_tile_init();
        recip_tile(dst0);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_xpow);
    tile_regs_release();

    cb_xpow.push_back(onetile);
    // We don't pop cb_x here.

    // log(x)
    tile_regs_acquire();
    cb_logx.reserve_back(onetile);

    copy_tile_init_with_dt(cb_x);
    copy_tile(cb_x.get_id(), 0, dst0);

    abs_tile_init();
    abs_tile(dst0);

    log_tile_init();
    log_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_logx);
    tile_regs_release();

    cb_x.pop_front(onetile);
    cb_logx.push_back(onetile);

    // exp(log(x) * decimal)
    tile_regs_acquire();
    cb_logx.wait_front(onetile);
    cb_exp_lxmd.reserve_back(onetile);

    mul_tiles_init_with_dt(cb_logx, cb_decimal);
    mul_tiles(cb_logx.get_id(), cb_decimal.get_id(), 0, 0, dst0);

    exp_tile_init();
    exp_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_exp_lxmd);
    tile_regs_release();

    cb_logx.pop_front(onetile);
    cb_exp_lxmd.push_back(onetile);

    // x^p * exp(log(x) * decimal)(==(x + decimal)^p)
    tile_regs_acquire();
    cb_xpow.wait_front(onetile);
    cb_exp_lxmd.wait_front(onetile);
    cb_correct_xpow.reserve_back(onetile);

    mul_tiles_init_with_dt(cb_xpow, cb_exp_lxmd);
    mul_tiles(cb_xpow.get_id(), cb_exp_lxmd.get_id(), 0, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_correct_xpow);
    tile_regs_release();

    cb_xpow.pop_front(onetile);
    cb_exp_lxmd.pop_front(onetile);
    cb_correct_xpow.push_back(onetile);
}

ALWI void power_and_recip_tile_to_cb(
    DataflowBuffer cb_x,
    DataflowBuffer cb_xpow,
    DataflowBuffer cb_logx,
    DataflowBuffer cb_decimal,
    DataflowBuffer cb_exp_lxmd,
    DataflowBuffer cb_recip_xpow,
    uint32_t p,
    bool p_is_negative) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    // x^p
    cb_x.wait_front(onetile);
    cb_xpow.reserve_back(onetile);

    tile_regs_acquire();
    copy_tile_init_with_dt(cb_x);
    copy_tile(cb_x.get_id(), 0, dst0);

    power_iterative_tile_init();
    power_iterative_tile(dst0, p);

    if (p_is_negative) {
        recip_tile_init();
        recip_tile(dst0);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_xpow);
    tile_regs_release();

    cb_xpow.push_back(onetile);
    // We don't pop cb_x here.

    // log(x)
    cb_logx.reserve_back(onetile);

    tile_regs_acquire();
    copy_tile_init_with_dt(cb_x);
    copy_tile(cb_x.get_id(), 0, dst0);

    log_tile_init();
    log_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_logx);
    tile_regs_release();

    cb_x.pop_front(onetile);
    cb_logx.push_back(onetile);

    // exp(log(x) * decimal)
    cb_logx.wait_front(onetile);
    cb_exp_lxmd.reserve_back(onetile);

    tile_regs_acquire();
    mul_tiles_init_with_dt(cb_logx, cb_decimal);
    mul_tiles(cb_logx.get_id(), cb_decimal.get_id(), 0, 0, dst0);

    exp_tile_init();
    exp_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_exp_lxmd);
    tile_regs_release();

    cb_logx.pop_front(onetile);
    cb_exp_lxmd.push_back(onetile);

    // 1 / (x^p * exp(log(x) * decimal))(==1 / (x + decimal)^p)
    cb_xpow.wait_front(onetile);
    cb_exp_lxmd.wait_front(onetile);
    cb_recip_xpow.reserve_back(onetile);

    tile_regs_acquire();
    mul_tiles_init_with_dt(cb_xpow, cb_exp_lxmd);
    mul_tiles(cb_xpow.get_id(), cb_exp_lxmd.get_id(), 0, 0, dst0);

    recip_tile_init();
    recip_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_recip_xpow);
    tile_regs_release();

    cb_xpow.pop_front(onetile);
    cb_exp_lxmd.pop_front(onetile);
    cb_recip_xpow.push_back(onetile);
}

ALWI void copy_tile_to_dst(DataflowBuffer icb, uint32_t itile = 0, uint32_t dst = 0, bool cb_wait_and_pop = true) {
    constexpr uint32_t onetile = 1;
    if (cb_wait_and_pop) {
        icb.wait_front(onetile);
    }
    reconfig_data_format_srca(icb.get_id());
    copy_init(icb.get_id());
    copy_tile(icb.get_id(), itile, dst);
    if (cb_wait_and_pop) {
        icb.pop_front(onetile);
    }
}

ALWI void pack_tile_from_dst(DataflowBuffer ocb, uint32_t dst = 0) {
    constexpr uint32_t onetile = 1;
    ocb.reserve_back(onetile);
    pack_reconfig_data_format(ocb.get_id());
    pack_tile(dst, ocb.get_id());
    ocb.push_back(onetile);
}

}  // namespace ckernel
