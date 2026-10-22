// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar batch_norm reader: R DM harts feed C NEOs through STRIDED rings.
//
// Ring geometry (host): every reader -> compute ring has R producers and C consumers, so producer hart p
// owns entries p, p+R, ... and its k-th push lands with consumer (p + (k mod C/R) * R). Hart p therefore
// feeds NEOs p, p+R, ... in strict rotation, one tile each, on every ring independently.
//
// The unit of work is one (n, c) plane of HtWt tiles, numbered n * channels + c. Every NEO runs
// rounds = ceil(num_planes / C) rounds; NEO t's round-k plane is plane_start + k*C + t when that is a real
// plane of this core, else NEO 0's plane of the same round (a duplicate the writer stores again).
//
// Per plane (channel c = plane % channels), in the order the compute consumes:
//   mean <- page c, var <- page c, weight <- page c, bias <- page c   (one tile each; the [1, C, 1, 1]
//   parameters tile-pad to one tile per channel with the value at (0, 0))
//   in <- pages plane * HtWt + 0 .. HtWt-1
//
// The streamed rings use DFB implicit sync (noc.async_read<TXN_ID>); eps is RISC-filled and pushed
// explicitly, so the host opts it out of implicit sync.
//
// Compile-time args: HtWt, channels, compute_threads   Runtime args: plane_start, num_planes, eps
// Defines: FUSE_WEIGHT, FUSE_BIAS

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"

void kernel_main() {
    const uint32_t plane_start = get_arg(args::plane_start);
    const uint32_t num_planes = get_arg(args::num_planes);
    const uint32_t eps_bits = get_arg(args::eps);
    constexpr uint32_t HtWt = get_arg(args::HtWt);
    constexpr uint32_t channels = get_arg(args::channels);
    constexpr uint32_t C = get_arg(args::compute_threads);

    const uint32_t p = get_my_thread_id();
    const uint32_t R = get_num_threads();
    const uint32_t group = C / R;  // NEOs fed by this hart: p, p+R, ...
    const uint32_t rounds = (num_planes + C - 1) / C;

    Noc noc;
    const auto in_acc = TensorAccessor(tensor::input);
    const auto mean_acc = TensorAccessor(tensor::mean);
    const auto var_acc = TensorAccessor(tensor::var);
    DataflowBuffer in(dfb::in);
    DataflowBuffer eps(dfb::eps);
    DataflowBuffer mean(dfb::mean);
    DataflowBuffer var(dfb::var);
#ifdef FUSE_WEIGHT
    const auto weight_acc = TensorAccessor(tensor::weight);
    DataflowBuffer weight(dfb::weight);
#endif
#ifdef FUSE_BIAS
    const auto bias_acc = TensorAccessor(tensor::bias);
    DataflowBuffer bias(dfb::bias);
#endif

    // Plane of NEO t in round k: its own plane when real, else NEO 0's plane of that round.
    auto plane_of = [&](uint32_t t, uint32_t k) __attribute__((always_inline)) {
        const uint32_t local = k * C + t;
        return plane_start + (local < num_planes ? local : k * C);
    };

    // One eps scalar tile per NEO in this hart's group (explicit sync, pushed in rotation order).
    for (uint32_t j = 0; j < group; ++j) {
        generate_bcast_unary_scalar(eps, eps_bits);
    }

    for (uint32_t k = 0; k < rounds; ++k) {
        // Per-channel parameters, one tile per plane, rotating over the group.
        for (uint32_t j = 0; j < group; ++j) {
            const uint32_t c = plane_of(p + j * R, k) % channels;
            noc.async_read<NocOptions::TXN_ID>(mean_acc, mean, {.page_id = c}, {});
            noc.async_read<NocOptions::TXN_ID>(var_acc, var, {.page_id = c}, {});
#ifdef FUSE_WEIGHT
            noc.async_read<NocOptions::TXN_ID>(weight_acc, weight, {.page_id = c}, {});
#endif
#ifdef FUSE_BIAS
            noc.async_read<NocOptions::TXN_ID>(bias_acc, bias, {.page_id = c}, {});
#endif
        }
        // The plane's tiles, rotating over the group per tile.
        for (uint32_t t = 0; t < HtWt; ++t) {
            for (uint32_t j = 0; j < group; ++j) {
                noc.async_read<NocOptions::TXN_ID>(in_acc, in, {.page_id = plane_of(p + j * R, k) * HtWt + t}, {});
            }
        }
    }

    // No finish() calls: each ~DataflowBuffer() (reverse declaration order) tops up a partial ISR batch
    // and drains the ring. The host sizes the windows so no ring ever ends on a partial batch.
}
