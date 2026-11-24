// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_mul_reduce_scalar.h. Same instruction sequence and
// access kinds as production, expressed through the HAL.

#pragma once

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"

using namespace ckernel;

/*************************************************************************
 * LLK MUL REDUCE SCALAR UNPACK - Unpacker operations for fused mul+reduce
 *************************************************************************/

/**
 * @brief Switch UNPACK state for mul_reduce_scalar reduce phase
 *
 * Prepares for the reduce phase where the math thread reuses destination
 * registers as source operands. Resets UNPACK counters, signals context switch,
 * and sets DVALID flags for srcA and srcB.
 *
 * Must be called after multiply phase and before reduce phase.
 */
inline void _llk_unpack_mul_reduce_scalar_switch_to_reduce_()
{
    namespace adc = hal::counters::adc;

    // TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
    // semaphore_post(semaphore::UNPACK_SYNC) (MMIO)
    hal::sync::semaphore::post<hal::sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();
    // Stall_Clr_Cntrl=1: wait on Unpackers[i].SrcBank -- the bank actually cleared and published --
    // not MatrixUnit.Src?Bank, which is a different bank once double-buffering reaches steady state.
    // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the zero-source + set-data-valid flavor
    // (UNP_ZEROSRC, SET_DVALID, Stall_Clr_Cntrl) has no HAL descriptor, so the production encoding is kept.
    TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    // t6_semaphore_get(semaphore::UNPACK_SYNC)
    hal::sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();
}
