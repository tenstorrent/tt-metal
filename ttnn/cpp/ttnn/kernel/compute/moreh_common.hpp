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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckernel {

namespace ckl = ::compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
inline constexpr auto moreh_data_format_reconfig = ckl::DataFormatReconfig::Enabled;
#else
inline constexpr auto moreh_data_format_reconfig = ckl::DataFormatReconfig::Disabled;
#endif

template <const auto& Dfb>
inline constexpr auto moreh_input = ckl::input(
    Dfb,
    ckl::WaitPolicy::None,
    ckl::PopPolicy::None,
    ckl::InputTileMapping::Scalar,
    moreh_data_format_reconfig,
    ckl::TileAddressing::Offset);

template <const auto& Dfb>
inline constexpr auto moreh_output =
    ckl::output(Dfb, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, moreh_data_format_reconfig);

// The DataflowBuffer type the chain elements built from Dfb's spec take. A token puts the same access
// patterns on its input and output specs, so this one alias serves both; a uint32_t id gives UNKNOWN
// patterns. On WH/BH it is plain DataflowBuffer.
template <const auto& Dfb>
using moreh_dfb_t =
    DataflowBufferFor<moreh_input<Dfb>.producer_access_pattern, moreh_input<Dfb>.consumer_access_pattern>;

template <typename IDFB>
ALWI void pack_tile_with_dt(uint32_t ifrom_dst, IDFB icb) {
#if defined FP32_DEST_ACC_EN
    pack_reconfig_data_format(icb.get_id());
#endif
    pack_tile(ifrom_dst, icb.get_id());
}

template <typename IDFB>
ALWI void copy_tile_init_with_dt(IDFB icb, uint32_t transpose = 0) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format_srca(icb.get_id());
#endif
    copy_init(icb.get_id(), transpose);
}

template <typename DFB0, typename DFB1>
ALWI void add_tiles_init_with_dt(DFB0 icb0, DFB1 icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    add_init(icb0.get_id(), icb1.get_id());
}

template <typename DFB0, typename DFB1>
ALWI void add_bcast_rows_init_with_dt(DFB0 icb0, DFB1 icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    add_bcast_rows_init(icb0.get_id(), icb1.get_id());
}

template <typename DFB0, typename DFB1>
ALWI void add_bcast_cols_init_with_dt(DFB0 icb0, DFB1 icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    add_bcast_cols_init(icb0.get_id(), icb1.get_id());
}

template <typename DFB0, typename DFB1>
ALWI void add_bcast_scalar_init_with_dt(DFB0 icb0, DFB1 icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    add_bcast_scalar_init(icb0.get_id(), icb1.get_id());
}

template <typename DFB0, typename DFB1>
ALWI void sub_tiles_init_with_dt(DFB0 icb0, DFB1 icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    sub_init(icb0.get_id(), icb1.get_id());
}

template <typename DFB0, typename DFB1>
ALWI void sub_bcast_rows_init_with_dt(DFB0 icb0, DFB1 icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWSUB, BroadcastType::ROW, MathFidelity::LoFi>(
        icb0.get_id(), icb1.get_id())));
    UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(icb0.get_id(), icb1.get_id())));
}

template <typename DFB0, typename DFB1>
ALWI void sub_bcast_cols_init_with_dt(DFB0 icb0, DFB1 icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    sub_bcast_cols_init(icb0.get_id(), icb1.get_id());
}

template <typename DFB0, typename DFB1>
ALWI void sub_bcast_scalar_init_with_dt(DFB0 icb0, DFB1 icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    sub_bcast_scalar_init(icb0.get_id(), icb1.get_id());
}

template <typename DFB0, typename DFB1>
ALWI void mul_tiles_init_with_dt(DFB0 icb0, DFB1 icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    mul_init(icb0.get_id(), icb1.get_id());
}

template <typename DFB0, typename DFB1>
ALWI void mul_bcast_rows_init_with_dt(DFB0 icb0, DFB1 icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    mul_bcast_rows_init(icb0.get_id(), icb1.get_id());
}

template <typename DFB0, typename DFB1>
ALWI void mul_bcast_cols_init_with_dt(DFB0 icb0, DFB1 icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    mul_bcast_cols_init(icb0.get_id(), icb1.get_id());
}

template <typename DFB0, typename DFB1>
ALWI void mul_bcast_scalar_init_with_dt(DFB0 icb0, DFB1 icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0.get_id(), icb1.get_id());
#endif
    mul_bcast_scalar_init(icb0.get_id(), icb1.get_id());
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

template <const auto& Dfb0, const auto& Dfb1, const auto& DfbOut>
ALWI void mul_tiles_to_dfb(
    moreh_dfb_t<Dfb0>& dfb0,
    moreh_dfb_t<Dfb1>& dfb1,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    dfb0.wait_front(itile0 + 1);
    dfb1.wait_front(itile1 + 1);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, moreh_input<Dfb0>, moreh_input<Dfb1>>{dfb0, dfb1, itile0, itile1},
        ckl::PackTile<moreh_output<DfbOut>>{dfb_out});

    if (pop0) {
        dfb0.pop_front(pop0);
    }
    if (pop1) {
        dfb1.pop_front(pop1);
    }
}

template <const auto& Dfb0, const auto& Dfb1, const auto& DfbOut>
ALWI void mul_tiles_and_negative_to_dfb(
    moreh_dfb_t<Dfb0>& dfb0,
    moreh_dfb_t<Dfb1>& dfb1,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    dfb0.wait_front(itile0 + 1);
    dfb1.wait_front(itile1 + 1);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, moreh_input<Dfb0>, moreh_input<Dfb1>>{dfb0, dfb1, itile0, itile1},
        ckl::Negative<>{},
        ckl::PackTile<moreh_output<DfbOut>>{dfb_out});

    if (pop0) {
        dfb0.pop_front(pop0);
    }
    if (pop1) {
        dfb1.pop_front(pop1);
    }
}

template <const auto& Dfb0, const auto& Dfb1, const auto& DfbMask, const auto& DfbOut>
ALWI void mul_tiles_and_mask_tile_to_dfb(
    moreh_dfb_t<Dfb0>& dfb0,
    moreh_dfb_t<Dfb1>& dfb1,
    moreh_dfb_t<DfbMask>& dfb_mask,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t mtile = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1,
    uint32_t popm = 1) {
    dfb0.wait_front(itile0 + 1);
    dfb1.wait_front(itile1 + 1);
    dfb_mask.wait_front(mtile + 1);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, moreh_input<Dfb0>, moreh_input<Dfb1>>{dfb0, dfb1, itile0, itile1},
        ckl::CopyTile<moreh_input<DfbMask>, ckl::Dst::D1>{dfb_mask, mtile},
        ckl::Mask<>{},
        ckl::PackTile<moreh_output<DfbOut>>{dfb_out});

    if (pop0) {
        dfb0.pop_front(pop0);
    }
    if (pop1) {
        dfb1.pop_front(pop1);
    }
    if (popm) {
        dfb_mask.pop_front(popm);
    }
}

template <ckl::BroadcastDim Bcast, const auto& Dfb0, const auto& Dfb1, const auto& DfbOut>
ALWI void mul_tiles_bcast_to_dfb(
    moreh_dfb_t<Dfb0>& dfb0,
    moreh_dfb_t<Dfb1>& dfb1,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    dfb0.wait_front(itile0 + 1);
    dfb1.wait_front(itile1 + 1);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, moreh_input<Dfb0>, ckl::input(moreh_input<Dfb1>, Bcast)>{
            dfb0, dfb1, itile0, itile1},
        ckl::PackTile<moreh_output<DfbOut>>{dfb_out});

    if (pop0) {
        dfb0.pop_front(pop0);
    }
    if (pop1) {
        dfb1.pop_front(pop1);
    }
}

template <const auto& Dfb0, const auto& Dfb1, const auto& DfbOut>
ALWI void mul_tiles_bcast_rows_to_dfb(
    moreh_dfb_t<Dfb0>& dfb0,
    moreh_dfb_t<Dfb1>& dfb1,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    mul_tiles_bcast_to_dfb<ckl::BroadcastDim::Row, Dfb0, Dfb1, DfbOut>(dfb0, dfb1, dfb_out, itile0, itile1, pop0, pop1);
}

template <const auto& Dfb0, const auto& Dfb1, const auto& DfbOut>
ALWI void mul_tiles_bcast_cols_to_dfb(
    moreh_dfb_t<Dfb0>& dfb0,
    moreh_dfb_t<Dfb1>& dfb1,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    mul_tiles_bcast_to_dfb<ckl::BroadcastDim::Col, Dfb0, Dfb1, DfbOut>(dfb0, dfb1, dfb_out, itile0, itile1, pop0, pop1);
}

template <const auto& DfbIn, const auto& DfbOut>
ALWI void copy_tile_to_dfb(
    moreh_dfb_t<DfbIn>& dfb_in, moreh_dfb_t<DfbOut>& dfb_out, uint32_t itile = 0, uint32_t pop = 1) {
    dfb_in.wait_front(itile + 1);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::CopyTile<moreh_input<DfbIn>>{dfb_in, itile},
        ckl::PackTile<moreh_output<DfbOut>>{dfb_out});

    if (pop) {
        dfb_in.pop_front(pop);
    }
}

#ifndef ARCH_QUASAR  // compute_kernel_lib Abs, Sign, Log and PowerIterative are not ported to Quasar
template <const auto& DfbIn, const auto& DfbOut>
ALWI void sign_tile_to_dfb(
    moreh_dfb_t<DfbIn>& dfb_in, moreh_dfb_t<DfbOut>& dfb_out, uint32_t itile = 0, uint32_t pop = 1) {
    dfb_in.wait_front(itile + 1);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::CopyTile<moreh_input<DfbIn>>{dfb_in, itile},
        ckl::Sign<>{},
        ckl::PackTile<moreh_output<DfbOut>>{dfb_out});

    if (pop) {
        dfb_in.pop_front(pop);
    }
}
#endif  // !ARCH_QUASAR

template <const auto& Dfb0, const auto& Dfb1, const auto& DfbOut>
ALWI void add_tiles_to_dfb(
    moreh_dfb_t<Dfb0>& dfb0,
    moreh_dfb_t<Dfb1>& dfb1,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    dfb0.wait_front(itile0 + 1);
    dfb1.wait_front(itile1 + 1);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::BinaryFpu<ckl::BinaryFpuOp::Add, moreh_input<Dfb0>, moreh_input<Dfb1>>{dfb0, dfb1, itile0, itile1},
        ckl::PackTile<moreh_output<DfbOut>>{dfb_out});

    if (pop0) {
        dfb0.pop_front(pop0);
    }
    if (pop1) {
        dfb1.pop_front(pop1);
    }
}

template <const auto& DfbIn, const auto& DfbMask, const auto& DfbOut>
ALWI void mask_tile_to_dfb(
    moreh_dfb_t<DfbIn>& dfb_in,
    moreh_dfb_t<DfbMask>& dfb_mask,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1) {
    dfb_in.wait_front(itile + 1);
    dfb_mask.wait_front(mtile + 1);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::CopyTile<moreh_input<DfbIn>>{dfb_in, itile},
        ckl::CopyTile<moreh_input<DfbMask>, ckl::Dst::D1>{dfb_mask, mtile},
        ckl::Mask<>{},
        ckl::PackTile<moreh_output<DfbOut>>{dfb_out});

    if (pop) {
        dfb_in.pop_front(pop);
    }
    if (popm) {
        dfb_mask.pop_front(popm);
    }
}

// Same contract as mask_tile_to_dfb, but the masked lanes are written as +inf instead of 0. A MIN
// pool must treat padding as +inf, otherwise a zeroed pad wins whenever every real lane of the row
// is positive (see the softmin shift in moreh_softmax_{h,w}(_large).cpp). Mirrors the MINUS_INF
// handling of the moreh_norm ord_other kernels; ckl::MaskPosInf is backed by mask_posinf_tile.
template <const auto& DfbIn, const auto& DfbMask, const auto& DfbOut>
ALWI void mask_posinf_tile_to_dfb(
    moreh_dfb_t<DfbIn>& dfb_in,
    moreh_dfb_t<DfbMask>& dfb_mask,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1) {
    dfb_in.wait_front(itile + 1);
    dfb_mask.wait_front(mtile + 1);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::CopyTile<moreh_input<DfbIn>>{dfb_in, itile},
        ckl::CopyTile<moreh_input<DfbMask>, ckl::Dst::D1>{dfb_mask, mtile},
        ckl::MaskPosInf<>{},
        ckl::PackTile<moreh_output<DfbOut>>{dfb_out});

    if (pop) {
        dfb_in.pop_front(pop);
    }
    if (popm) {
        dfb_mask.pop_front(popm);
    }
}

template <ckl::BroadcastDim Bcast, const auto& Dfb0, const auto& Dfb1, const auto& DfbOut>
ALWI void sub_tiles_bcast_to_dfb(
    moreh_dfb_t<Dfb0>& dfb0,
    moreh_dfb_t<Dfb1>& dfb1,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    dfb0.wait_front(itile0 + 1);
    dfb1.wait_front(itile1 + 1);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::BinaryFpu<ckl::BinaryFpuOp::Sub, moreh_input<Dfb0>, ckl::input(moreh_input<Dfb1>, Bcast)>{
            dfb0, dfb1, itile0, itile1},
        ckl::PackTile<moreh_output<DfbOut>>{dfb_out});

    if (pop0) {
        dfb0.pop_front(pop0);
    }
    if (pop1) {
        dfb1.pop_front(pop1);
    }
}

template <const auto& Dfb0, const auto& Dfb1, const auto& DfbOut>
ALWI void sub_tiles_bcast_cols_to_dfb(
    moreh_dfb_t<Dfb0>& dfb0,
    moreh_dfb_t<Dfb1>& dfb1,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    sub_tiles_bcast_to_dfb<ckl::BroadcastDim::Col, Dfb0, Dfb1, DfbOut>(dfb0, dfb1, dfb_out, itile0, itile1, pop0, pop1);
}

template <const auto& Dfb0, const auto& Dfb1, const auto& DfbOut>
ALWI void sub_tiles_bcast_rows_to_dfb(
    moreh_dfb_t<Dfb0>& dfb0,
    moreh_dfb_t<Dfb1>& dfb1,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    sub_tiles_bcast_to_dfb<ckl::BroadcastDim::Row, Dfb0, Dfb1, DfbOut>(dfb0, dfb1, dfb_out, itile0, itile1, pop0, pop1);
}

template <const auto& Dfb0, const auto& Dfb1, const auto& DfbOut>
ALWI void sub_tiles_to_dfb(
    moreh_dfb_t<Dfb0>& dfb0,
    moreh_dfb_t<Dfb1>& dfb1,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    sub_tiles_bcast_to_dfb<ckl::BroadcastDim::None, Dfb0, Dfb1, DfbOut>(
        dfb0, dfb1, dfb_out, itile0, itile1, pop0, pop1);
}

template <bool Negative, const auto& DfbIn, const auto& DfbOut>
ALWI void exp_tile_to_dfb_impl(
    moreh_dfb_t<DfbIn>& dfb_in, moreh_dfb_t<DfbOut>& dfb_out, uint32_t itile = 0, uint32_t pop = 1) {
    dfb_in.wait_front(itile + 1);

    if constexpr (Negative) {
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<moreh_input<DfbIn>>{dfb_in, itile},
            ckl::Negative<>{},
            ckl::Exp<>{},
            ckl::PackTile<moreh_output<DfbOut>>{dfb_out});
    } else {
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<moreh_input<DfbIn>>{dfb_in, itile},
            ckl::Exp<>{},
            ckl::PackTile<moreh_output<DfbOut>>{dfb_out});
    }

    if (pop) {
        dfb_in.pop_front(pop);
    }
}

template <const auto& DfbIn, const auto& DfbOut>
ALWI void exp_tile_to_dfb(
    moreh_dfb_t<DfbIn>& dfb_in, moreh_dfb_t<DfbOut>& dfb_out, uint32_t itile = 0, uint32_t pop = 1) {
    exp_tile_to_dfb_impl<false, DfbIn, DfbOut>(dfb_in, dfb_out, itile, pop);
}

template <const auto& DfbIn, const auto& DfbOut>
ALWI void rexp_tile_to_dfb(
    moreh_dfb_t<DfbIn>& dfb_in, moreh_dfb_t<DfbOut>& dfb_out, uint32_t itile = 0, uint32_t pop = 1) {
    exp_tile_to_dfb_impl<true, DfbIn, DfbOut>(dfb_in, dfb_out, itile, pop);
}

template <bool Negative, const auto& DfbIn, const auto& DfbMask, const auto& DfbOut>
ALWI void exp_tile_and_mask_tile_to_dfb_impl(
    moreh_dfb_t<DfbIn>& dfb_in,
    moreh_dfb_t<DfbMask>& dfb_mask,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1) {
    dfb_in.wait_front(itile + 1);
    dfb_mask.wait_front(mtile + 1);

    if constexpr (Negative) {
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<moreh_input<DfbIn>>{dfb_in, itile},
            ckl::Negative<>{},
            ckl::Exp<>{},
            ckl::CopyTile<moreh_input<DfbMask>, ckl::Dst::D1>{dfb_mask, mtile},
            ckl::Mask<>{},
            ckl::PackTile<moreh_output<DfbOut>>{dfb_out});
    } else {
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<moreh_input<DfbIn>>{dfb_in, itile},
            ckl::Exp<>{},
            ckl::CopyTile<moreh_input<DfbMask>, ckl::Dst::D1>{dfb_mask, mtile},
            ckl::Mask<>{},
            ckl::PackTile<moreh_output<DfbOut>>{dfb_out});
    }

    if (pop) {
        dfb_in.pop_front(pop);
    }
    if (popm) {
        dfb_mask.pop_front(popm);
    }
}

template <const auto& DfbIn, const auto& DfbMask, const auto& DfbOut>
ALWI void exp_tile_and_mask_tile_to_dfb(
    moreh_dfb_t<DfbIn>& dfb_in,
    moreh_dfb_t<DfbMask>& dfb_mask,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1) {
    exp_tile_and_mask_tile_to_dfb_impl<false, DfbIn, DfbMask, DfbOut>(
        dfb_in, dfb_mask, dfb_out, itile, mtile, pop, popm);
}

template <const auto& DfbIn, const auto& DfbMask, const auto& DfbOut>
ALWI void rexp_tile_and_mask_tile_to_dfb(
    moreh_dfb_t<DfbIn>& dfb_in,
    moreh_dfb_t<DfbMask>& dfb_mask,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1) {
    exp_tile_and_mask_tile_to_dfb_impl<true, DfbIn, DfbMask, DfbOut>(
        dfb_in, dfb_mask, dfb_out, itile, mtile, pop, popm);
}

template <const auto& DfbIn, const auto& DfbOut>
ALWI void recip_tile_to_dfb(
    moreh_dfb_t<DfbIn>& dfb_in, moreh_dfb_t<DfbOut>& dfb_out, uint32_t itile = 0, uint32_t pop = 1) {
    dfb_in.wait_front(itile + 1);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::CopyTile<moreh_input<DfbIn>>{dfb_in, itile},
        ckl::Recip<>{},
        ckl::PackTile<moreh_output<DfbOut>>{dfb_out});

    if (pop) {
        dfb_in.pop_front(pop);
    }
}

#ifndef ARCH_QUASAR  // compute_kernel_lib Abs, Sign, Log and PowerIterative are not ported to Quasar
template <const auto& DfbIn, const auto& DfbOut>
ALWI void log_tile_to_dfb(
    moreh_dfb_t<DfbIn>& dfb_in, moreh_dfb_t<DfbOut>& dfb_out, uint32_t itile = 0, uint32_t pop = 1) {
    dfb_in.wait_front(itile + 1);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::CopyTile<moreh_input<DfbIn>>{dfb_in, itile},
        ckl::Log<>{},
        ckl::PackTile<moreh_output<DfbOut>>{dfb_out});

    if (pop) {
        dfb_in.pop_front(pop);
    }
}
#endif  // !ARCH_QUASAR

#ifndef ARCH_QUASAR  // compute_kernel_lib Abs, Sign, Log and PowerIterative are not ported to Quasar
template <
    bool AbsX,
    bool RecipFinal,
    const auto& DfbX,
    const auto& DfbXpow,
    const auto& DfbLogX,
    const auto& DfbDecimal,
    const auto& DfbExpLogXMulDecimal,
    const auto& DfbOut>
ALWI void power_tile_to_dfb_impl(
    moreh_dfb_t<DfbX>& dfb_x,
    moreh_dfb_t<DfbXpow>& dfb_xpow,
    moreh_dfb_t<DfbLogX>& dfb_logx,
    moreh_dfb_t<DfbDecimal>& dfb_decimal,
    moreh_dfb_t<DfbExpLogXMulDecimal>& dfb_exp_logx_mul_decimal,
    moreh_dfb_t<DfbOut>& dfb_out,
    uint32_t p,
    bool p_is_negative) {
    // x^p
    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::CopyTile<ckl::input(DfbX, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, moreh_data_format_reconfig)>{
            dfb_x},
        ckl::Optional<AbsX, ckl::Abs<>>{},
        ckl::PowerIterative<>{p},
        ckl::runtime_if(p_is_negative, ckl::Recip<>{}),
        ckl::PackTile<moreh_output<DfbXpow>>{dfb_xpow});

    // log(x)
    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::CopyTile<ckl::input(DfbX, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, moreh_data_format_reconfig)>{
            dfb_x},
        ckl::Optional<AbsX, ckl::Abs<>>{},
        ckl::Log<>{},
        ckl::PackTile<moreh_output<DfbLogX>>{dfb_logx});

    // exp(log(x) * decimal)
    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Mul,
            ckl::input(DfbLogX, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, moreh_data_format_reconfig),
            ckl::input(DfbDecimal, ckl::WaitPolicy::None, ckl::PopPolicy::None, moreh_data_format_reconfig)>{
            dfb_logx, dfb_decimal},
        ckl::Exp<>{},
        ckl::PackTile<moreh_output<DfbExpLogXMulDecimal>>{dfb_exp_logx_mul_decimal});

    // x^p * exp(log(x) * decimal), optionally followed by reciprocal.
    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Mul,
            ckl::input(DfbXpow, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, moreh_data_format_reconfig),
            ckl::input(
                DfbExpLogXMulDecimal, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, moreh_data_format_reconfig)>{
            dfb_xpow, dfb_exp_logx_mul_decimal},
        ckl::Optional<RecipFinal, ckl::Recip<>>{},
        ckl::PackTile<moreh_output<DfbOut>>{dfb_out});
}
#endif  // !ARCH_QUASAR

#ifndef ARCH_QUASAR  // compute_kernel_lib Abs, Sign, Log and PowerIterative are not ported to Quasar
template <
    const auto& DfbX,
    const auto& DfbXpow,
    const auto& DfbLogX,
    const auto& DfbDecimal,
    const auto& DfbExpLogXMulDecimal,
    const auto& DfbCorrectXpow>
ALWI void power_tile_to_dfb(
    moreh_dfb_t<DfbX>& dfb_x,
    moreh_dfb_t<DfbXpow>& dfb_xpow,
    moreh_dfb_t<DfbLogX>& dfb_logx,
    moreh_dfb_t<DfbDecimal>& dfb_decimal,
    moreh_dfb_t<DfbExpLogXMulDecimal>& dfb_exp_logx_mul_decimal,
    moreh_dfb_t<DfbCorrectXpow>& dfb_correct_xpow,
    uint32_t p,
    bool p_is_negative) {
    power_tile_to_dfb_impl<false, false, DfbX, DfbXpow, DfbLogX, DfbDecimal, DfbExpLogXMulDecimal, DfbCorrectXpow>(
        dfb_x, dfb_xpow, dfb_logx, dfb_decimal, dfb_exp_logx_mul_decimal, dfb_correct_xpow, p, p_is_negative);
}
#endif  // !ARCH_QUASAR

#ifndef ARCH_QUASAR  // compute_kernel_lib Abs, Sign, Log and PowerIterative are not ported to Quasar
template <
    const auto& DfbX,
    const auto& DfbXpow,
    const auto& DfbLogX,
    const auto& DfbDecimal,
    const auto& DfbExpLogXMulDecimal,
    const auto& DfbCorrectXpow>
ALWI void power_tile_with_abs_x_to_dfb(
    moreh_dfb_t<DfbX>& dfb_x,
    moreh_dfb_t<DfbXpow>& dfb_xpow,
    moreh_dfb_t<DfbLogX>& dfb_logx,
    moreh_dfb_t<DfbDecimal>& dfb_decimal,
    moreh_dfb_t<DfbExpLogXMulDecimal>& dfb_exp_logx_mul_decimal,
    moreh_dfb_t<DfbCorrectXpow>& dfb_correct_xpow,
    uint32_t p,
    bool p_is_negative) {
    power_tile_to_dfb_impl<true, false, DfbX, DfbXpow, DfbLogX, DfbDecimal, DfbExpLogXMulDecimal, DfbCorrectXpow>(
        dfb_x, dfb_xpow, dfb_logx, dfb_decimal, dfb_exp_logx_mul_decimal, dfb_correct_xpow, p, p_is_negative);
}
#endif  // !ARCH_QUASAR

#ifndef ARCH_QUASAR  // compute_kernel_lib Abs, Sign, Log and PowerIterative are not ported to Quasar
template <
    const auto& DfbX,
    const auto& DfbXpow,
    const auto& DfbLogX,
    const auto& DfbDecimal,
    const auto& DfbExpLogXMulDecimal,
    const auto& DfbRecipXpow>
ALWI void power_and_recip_tile_to_dfb(
    moreh_dfb_t<DfbX>& dfb_x,
    moreh_dfb_t<DfbXpow>& dfb_xpow,
    moreh_dfb_t<DfbLogX>& dfb_logx,
    moreh_dfb_t<DfbDecimal>& dfb_decimal,
    moreh_dfb_t<DfbExpLogXMulDecimal>& dfb_exp_logx_mul_decimal,
    moreh_dfb_t<DfbRecipXpow>& dfb_recip_xpow,
    uint32_t p,
    bool p_is_negative) {
    power_tile_to_dfb_impl<false, true, DfbX, DfbXpow, DfbLogX, DfbDecimal, DfbExpLogXMulDecimal, DfbRecipXpow>(
        dfb_x, dfb_xpow, dfb_logx, dfb_decimal, dfb_exp_logx_mul_decimal, dfb_recip_xpow, p, p_is_negative);
}
#endif  // !ARCH_QUASAR

template <typename DFB0, typename DFB1, typename ODFB>
ALWI void mul_tiles_to_cb(
    DFB0 icb0, DFB1 icb1, ODFB ocb, uint32_t itile0 = 0, uint32_t itile1 = 0, uint32_t pop0 = 1, uint32_t pop1 = 1) {
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

template <typename IDFB, typename ODFB>
ALWI void copy_tile_to_cb(IDFB icb, ODFB ocb, uint32_t itile = 0, uint32_t pop = 1) {
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

template <typename DFB0, typename DFB1, typename ODFB>
ALWI void add_tiles_to_cb(
    DFB0 icb0, DFB1 icb1, ODFB ocb, uint32_t itile0 = 0, uint32_t itile1 = 0, uint32_t pop0 = 1, uint32_t pop1 = 1) {
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

template <typename DFB0, typename DFB1, typename ODFB>
ALWI void sub_tiles_to_cb(
    DFB0 icb0, DFB1 icb1, ODFB ocb, uint32_t itile0 = 0, uint32_t itile1 = 0, uint32_t pop0 = 1, uint32_t pop1 = 1) {
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

// TODO(seunghwan100): If p is 2 and decimal is 0, we can use sqrt_tile.
template <
    typename DFBX,
    typename DFBXPow,
    typename DFBLogX,
    typename DFBDecimal,
    typename DFBExpLxmd,
    typename DFBCorrectXPow>
ALWI void power_tile_to_cb(
    DFBX cb_x,
    DFBXPow cb_xpow,
    DFBLogX cb_logx,
    DFBDecimal cb_decimal,
    DFBExpLxmd cb_exp_lxmd,
    DFBCorrectXPow cb_correct_xpow,
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

template <typename IDFB>
ALWI void copy_tile_to_dst(IDFB icb, uint32_t itile = 0, uint32_t dst = 0, bool cb_wait_and_pop = true) {
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

template <typename ODFB>
ALWI void pack_tile_from_dst(ODFB ocb, uint32_t dst = 0) {
    constexpr uint32_t onetile = 1;
    ocb.reserve_back(onetile);
    pack_reconfig_data_format(ocb.get_id());
    pack_tile(dst, ocb.get_id());
    ocb.push_back(onetile);
}

}  // namespace ckernel
