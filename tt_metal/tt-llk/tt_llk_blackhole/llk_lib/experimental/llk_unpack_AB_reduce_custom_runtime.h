// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_reduce_custom_runtime.h. Same instruction
// sequence and access kinds as production, expressed through the HAL with the helpers of
// llk_unpack_AB_reduce_custom.h (hal_port::reduce_block_max).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "experimental/llk_unpack_AB_reduce_custom.h"
#include "hal/mop.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * Configures MOP (Macro Operation) for block-based reduce_max_row unpacking operations.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native reduce unpacking LLK MOP configuration.
 * Use the standard _llk_unpack_AB_mop_config_ for general-purpose block reduction operations.
 */

inline void _llk_unpack_AB_reduce_block_max_row_mop_config_runtime_(std::uint32_t block_ct_dim, bool respect_trigger = false)
{
    const std::uint32_t outerloop = respect_trigger ? (block_ct_dim / 2) : block_ct_dim;
    hal_port::reduce_block_max::program_mop(outerloop); // innerloop = 1
}

/**
 * Initializes unpacker configuration for block-based reduce_max_row operations.
 * Sets up tile dimensions and saves unpacker state that will be modified during operation.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile is num_faces faces (4 for a 32x32 tile, 2 for a 16x32 tiny tile)
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * @param tensor_shape Shape of the operand tile (4 faces for 32x32, 2 faces for a 16x32 tiny tile).
 *
 * This function should NOT be used as a substitute for native reduce unpacking LLK initialization.
 * Use the standard _llk_unpack_AB_reduce_init_ for general-purpose reduction operations.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_unpack_AB_reduce_block_max_row_init_runtime_(std::uint32_t block_ct_dim, bool respect_trigger, const ckernel::TensorShape tensor_shape)
{
    hal_port::reduce_block_max::configure(tensor_shape);

    _llk_unpack_AB_reduce_block_max_row_mop_config_runtime_(block_ct_dim, respect_trigger); // Unpack operand and scaler
}

/**
 * Performs unpacking for block-based reduce_max_row operation across multiple tiles (runtime version).
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native _llk_unpack_AB_ LLK.
 * Use the standard _llk_unpack_AB_ in a loop for general-purpose block reduction operations.
 */
inline void _llk_unpack_AB_reduce_block_max_row_runtime_(
    const std::uint32_t address_a, const std::uint32_t address_b, bool respect_trigger = false, bool overlap_first_half = false)
{
    namespace rbm = hal_port::reduce_block_max;
    using Mop     = hal::mop::Runner<hal::mop::MopTemplate::Template1>;

    rbm::begin(address_a, address_b);

    if (respect_trigger)
    {
        // Barrier for the split reduce. run()#1 reads cols [0,N/2), run()#2 reads [N/2,N)
        // (MOP outerloop = block_ct_dim/2; Z continues across the two run()s). Overlap path: run()#1
        // waits on the early first-half token so it overlaps the second-half pack; else on FPU_SFPU.
        // run()#2 always FPU_SFPU. Literal branch — SEMWAIT encodes the sem index as immediate.
        if (overlap_first_half)
        {
            rbm::wait_while_zero<rbm::UNPACK_MATH_DONE_SEM>();
        }
        else
        {
            rbm::wait_while_zero<rbm::FPU_SFPU_SEM>();
        }
        Mop::run();
        rbm::wait_while_zero<rbm::FPU_SFPU_SEM>();
        Mop::run();
    }
    else
    {
        Mop::run();
    }

    rbm::end();
}

/**
 * Uninitializes block-based reduce_max_row unpacker operation (runtime version).
 * Restores the unpacker state that was saved during initialization.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native reduce unpacking cleanup.
 * Standard _llk_unpack_AB_reduce_init_ operations typically don't require explicit cleanup.
 */
inline void _llk_unpack_AB_reduce_block_max_row_uninit_runtime_(bool respect_trigger = false, bool overlap_first_half = false)
{
    namespace rbm = hal_port::reduce_block_max;

    rbm::restore();

    if (respect_trigger)
    {
        // t6_semaphore_get(semaphore::FPU_SFPU)
        hal::sync::semaphore::get<rbm::FPU_SFPU_SEM>();
        if (overlap_first_half)
        {
            // Balance the first-half-token post; conditional so the non-overlap path leaves it untouched.
            hal::sync::semaphore::get<rbm::UNPACK_MATH_DONE_SEM>();
        }
    }
}
