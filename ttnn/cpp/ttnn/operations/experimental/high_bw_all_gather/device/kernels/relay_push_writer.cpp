// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PROTOTYPE one-hop relay writer (see relay_push_reader.cpp). One sender per (link, direction); every packet it
// sends goes to the direct torus neighbour in its direction, as one contiguous bank-lane write.
//
// Protocol (caller-owned semaphores, all left at zero):
//   start: each link-0 sender increments ready on its neighbour's sync core (worker 0); the sync core waits for
//          its four neighbours, then releases its local senders. A neighbour's staging region and output are
//          therefore never written while it still runs an earlier op.
//   rounds: after each round's data, a sender increments done on every worker core of its neighbour, on the same
//          connection as the data. A core's done thus reaches links * 4 per round; the reader waits on it before
//          forwarding staged packets, and the writer finally waits for all R rounds and subtracts them.

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#include <array>
#include <cstdint>

using namespace ttnn::operations::ccl::common;

namespace {
constexpr uint32_t kData = 0, kMeta = 1, kHeaders = 4;
constexpr uint32_t kMaxWorkers = 8;
constexpr uint32_t kBatch = 4;
constexpr uint32_t kRoundEnd = 0, kLocal = 3, kTerminate = 4;
constexpr uint32_t kNeighbours = 4;

void send_sem_inc(
    tt::tt_fabric::WorkerToFabricEdmSender& conn, volatile PACKET_HEADER_TYPE* hdr, uint64_t sem_noc_addr) {
    hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, 1, true});
    conn.wait_for_empty_write_slot();
    conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
}
}  // namespace

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t workers = get_compile_time_arg_val(1);
    constexpr uint32_t packet_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t per_round = get_compile_time_arg_val(3);
    constexpr auto output_args = TensorAccessorArgs<4>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t ready_sem = get_common_arg_val<uint32_t>(1);
    const uint32_t done_sem = get_common_arg_val<uint32_t>(2);

    size_t arg = 0;
    const uint32_t worker = get_arg_val<uint32_t>(arg++);
    const uint32_t link = get_arg_val<uint32_t>(arg++);
    const uint32_t nbr_chip = get_arg_val<uint32_t>(arg++);
    const uint32_t nbr_mesh = get_arg_val<uint32_t>(arg++);
    std::array<uint32_t, kMaxWorkers> core_x{}, core_y{};
    for (uint32_t w = 0; w < kMaxWorkers; ++w) {
        core_x[w] = get_arg_val<uint32_t>(arg++);
        core_y[w] = get_arg_val<uint32_t>(arg++);
    }

    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
    conn.open();

    const uint32_t headers_l1 = get_write_ptr(kHeaders);
    for (uint32_t h = 0; h < 1 + kBatch; ++h) {
        auto* hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(headers_l1 + h * sizeof(PACKET_HEADER_TYPE));
        fabric_set_unicast_route((HybridMeshPacketHeader*)hdr, nbr_chip, nbr_mesh);
    }
    auto* sem_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(headers_l1);
    const auto output = TensorAccessor(output_args, output_addr, page_bytes);
    auto* ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem);
    auto* done_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem);

    // Start fence with the four neighbours.
    if (link == 0) {
        send_sem_inc(conn, sem_hdr, get_noc_addr(core_x[0], core_y[0], ready_sem));
    }
    if (worker == 0) {
        noc_semaphore_wait_min(ready_ptr, kNeighbours);
        noc_semaphore_inc(get_noc_addr(ready_sem), uint32_t{0} - kNeighbours);
        for (uint32_t w = 1; w < workers; ++w) {
            noc_semaphore_inc(get_noc_addr(core_x[w], core_y[w], ready_sem), 1);
        }
        noc_async_atomic_barrier();
    } else {
        noc_semaphore_wait_min(ready_ptr, 1);
        noc_semaphore_inc(get_noc_addr(ready_sem), uint32_t{0} - 1u);
        noc_async_atomic_barrier();
    }

    uint32_t pending = 0;
    auto retire = [&]() {
        if (pending) {
            noc_async_writes_flushed();
            cb_pop_front(kMeta, pending);
            cb_pop_front(kData, pending);
            pending = 0;
        }
    };
    uint32_t rounds = 0;
    while (true) {
        cb_wait_front(kMeta, pending + 1);
        cb_wait_front(kData, pending + 1);
        invalidate_l1_cache();
        const auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(kMeta) + pending * 16);
        const uint32_t kind = meta[0], target = meta[1], n = meta[2];
        if (kind == kRoundEnd || kind == kTerminate) {
            retire();
            cb_pop_front(kMeta, 1);
            cb_pop_front(kData, 1);
            if (kind == kTerminate) {
                rounds = n;
                break;
            }
            for (uint32_t w = 0; w < workers; ++w) {
                send_sem_inc(conn, sem_hdr, get_noc_addr(core_x[w], core_y[w], done_sem));
            }
            continue;
        }
        const uint32_t data_l1 = get_read_ptr(kData) + pending * packet_bytes;
        const uint32_t bytes = n * page_bytes;
        if (kind == kLocal) {
            noc_async_write(data_l1, output.get_noc_addr(target), bytes);
        } else {
            auto* hdr =
                reinterpret_cast<volatile PACKET_HEADER_TYPE*>(headers_l1 + (1 + pending) * sizeof(PACKET_HEADER_TYPE));
            tt::tt_fabric::linear::to_noc_unicast_write(bytes, hdr, target, output);
            perform_payload_send<false, false>(conn, data_l1, bytes, hdr);
        }
        if (++pending == kBatch) {
            retire();
        }
    }

    // Every neighbour sender has finished every round, so all rows for this chip have landed.
    noc_async_write_barrier();
    const uint32_t expected = per_round * rounds;
    noc_semaphore_wait_min(done_ptr, expected);
    noc_semaphore_inc(get_noc_addr(done_sem), uint32_t{0} - expected);
    noc_async_atomic_barrier();
    conn.close();
}
