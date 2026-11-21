// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm reader. Hart p of R feeds NEOs p, p+R, ... (C/R of them) through STRIDED rings, one entry each
// in strict rotation: its k-th entry of a ring lands with NEO p + (k mod C/R) * R.
//
// Per kernel, for every NEO this hart serves:
//   mean_scaler, var_scaler  reduce scalers 1/(2W) and 1/W (explicit sync, written by the hart)
//   epsilon                  epsilon at element [0, 0], zeros elsewhere (explicit sync; read by a scalar broadcast)
//   affine_stream            the Wt tiles of `weight`, then the Wt tiles of `bias` (implicit sync)
// then, round by round, NEO t's tile-row (Wt tiles of `x`, implicit sync) into input_block.
//
// NEO t's row in round k is local row k*C + t when it exists, else NEO 0's row of that round (every NEO
// takes the same number of entries; the writer stores the recomputed row again with the same bytes).
//
// Protocols (target page, ring protocols): implicit sync for every ring a NoC read fills (one tagged read per
// entry, no reserve/push); the three constant rings are listed in this kernel's
// disable_dfb_implicit_sync_for and use reserve/push. Their tiles are zeroed by the NoC engine; the write
// transaction id is set to the user id 0 first, so those untagged writes cannot inherit a ring's tag. The read
// tag is handed back at the end.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "layer_norm_rows.hpp"

void kernel_main() {
    const uint32_t num_rows = get_arg(args::num_rows);
    const uint32_t row_start = get_arg(args::row_start);
    const uint32_t epsilon_word = get_arg(args::epsilon_word);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t group = C / R;  // NEOs this hart serves
    constexpr float mean_scaler_value = __builtin_bit_cast(float, static_cast<uint32_t>(get_arg(args::mean_scaler_bits)));
    constexpr float var_scaler_value = __builtin_bit_cast(float, static_cast<uint32_t>(get_arg(args::var_scaler_bits)));

    const uint32_t p = get_my_thread_id();
    const uint32_t rounds = layer_norm_rows::rounds(num_rows, C);

    DataflowBuffer input_block(dfb::input_block);
    DataflowBuffer affine_stream(dfb::affine_stream);
    DataflowBuffer mean_scaler(dfb::mean_scaler);
    DataflowBuffer var_scaler(dfb::var_scaler);
    DataflowBuffer epsilon(dfb::epsilon);
    Noc noc;
    const auto acc_x = TensorAccessor(tensor::x);
    const auto acc_weight = TensorAccessor(tensor::weight);
    const auto acc_bias = TensorAccessor(tensor::bias);

    // ---- constants, one entry per NEO served ----
    noc_async_write_set_trid(0, noc.get_noc_id());
    for (uint32_t j = 0; j < group; ++j) {
        dataflow_kernel_lib::prepare_reduce_scaler<dfb::mean_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            mean_scaler, mean_scaler_value);
        dataflow_kernel_lib::prepare_reduce_scaler<dfb::var_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            var_scaler, var_scaler_value);

        // epsilon: zero the tile over the NoC, then write element [0, 0] (the only element a scalar broadcast
        // reads) from the hart and flush it out of the DM cache before the compute side may read it.
        epsilon.reserve_back(1);
        const uint32_t eps_addr = epsilon.get_write_ptr();
        noc.async_write_zeros(epsilon, epsilon.get_tile_size());
        noc.write_zeros_l1_barrier();
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eps_addr)[0] = epsilon_word;
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
        flush_l2_cache_range(eps_addr, epsilon.get_tile_size());
#endif
        epsilon.push_back(1);
    }

    // ---- gamma, then beta: Wt tiles each, to every NEO served (the compute prologue expands them) ----
    for (uint32_t w = 0; w < Wt; ++w) {
        for (uint32_t j = 0; j < group; ++j) {
            noc.async_read<NocOptions::TXN_ID>(acc_weight, affine_stream, {.page_id = w}, {});
        }
    }
    for (uint32_t w = 0; w < Wt; ++w) {
        for (uint32_t j = 0; j < group; ++j) {
            noc.async_read<NocOptions::TXN_ID>(acc_bias, affine_stream, {.page_id = w}, {});
        }
    }

    // ---- the input, one tile-row per NEO per round ----
    for (uint32_t round = 0; round < rounds; ++round) {
        for (uint32_t w = 0; w < Wt; ++w) {
            for (uint32_t j = 0; j < group; ++j) {
                noc.async_read<NocOptions::TXN_ID>(
                    acc_x, input_block, {.page_id = layer_norm_rows::row_of(row_start, num_rows, C, p + j * R, round) * Wt + w}, {});
            }
        }
    }
    // Hand the hart back with the user id in its read tag register: the register is sticky.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
