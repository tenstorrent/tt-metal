// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

namespace ckernel
{
using namespace unpacker;

inline void _llk_unpack_AB_scalar_block_init_(const ckernel::TensorShape& tensor_shape_a, const ckernel::TensorShape& tensor_shape_b)
{
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    TT_SETADCXX(p_setadc::UNP0, tensor_shape_a.total_tensor_size() - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP1, tensor_shape_b.total_tensor_size() - 1, 0x0);
}

inline void _llk_unpack_AB_scalar_block_(const std::uint32_t address_a, const std::uint32_t address_b, const std::uint32_t block_size)
{
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    constexpr std::uint8_t kNoAddressIncrement = 0;
    TTI_UNPACR(SrcB, kNoAddressIncrement, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    for (std::uint32_t i = 0; i < block_size; ++i)
    {
        TTI_UNPACR(SrcA, kNoAddressIncrement, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_INCADCZW(p_setadc::UNP_A, 0, 0, 1, 0);
    }

    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

} // namespace ckernel
