// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>

// The fused all-reduce on one fabric sender core, on a single DM processor so it also fits the router
// cores, whose other one runs the router: all_reduce_async's worker_reader.cpp and worker_writer.cpp in
// one. The slots it forwards are written by this same program, so it first waits for each of its link's
// output cores to bump sem_ar_ready once, then reads this chip's slot of all of them before sending.

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_headers_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_src_id = get_compile_time_arg_val(2);
constexpr uint32_t cb_scratch_id = get_compile_time_arg_val(3);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(4);
constexpr uint32_t page_size = get_compile_time_arg_val(5);
constexpr uint32_t sem_ar_ready_id = get_compile_time_arg_val(6);
constexpr ccl_routing_utils::line_multicast_route_info_t forward_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<7>();
constexpr ccl_routing_utils::line_multicast_route_info_t backward_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<7 + ccl_routing_utils::num_line_multicast_args>();

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t pages_per_core = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    const size_t out_ready_sem_bank_addr =
        get_common_arg_val<uint32_t>(ttnn::ccl::AllReduceSemaphoreCommonArgs::semaphore);

    // Only start opening the connection here; open_finish's barrier is deferred until there is data.
    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    Noc noc;
    CircularBuffer cb_headers(cb_headers_id);
    CircularBuffer cb_src(cb_src_id);

    cb_headers.reserve_back(1);
    auto packet_header_buffer_addr_forward = cb_headers.get_write_ptr();
    cb_headers.push_back(1);
    cb_headers.reserve_back(1);
    auto packet_header_buffer_addr_backward = cb_headers.get_write_ptr();
    cb_headers.push_back(1);
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_forward, forward_multicast_route_info);
    ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_backward, backward_multicast_route_info);

    // The scratch sits at one address on the output cores and here (cb_scratch is bound to it on both).
    const uint32_t core_bytes = pages_per_core * page_size;
    const uint32_t slot_addr = get_write_ptr(cb_scratch_id) + my_chip_id * core_bytes;

    Semaphore<> ar_ready(sem_ar_ready_id);
    ar_ready.wait(num_cores);
    ar_ready.set(0);

    cb_src.reserve_back(num_cores * pages_per_core);
    UnicastEndpoint src_ep;
    for (uint32_t core = 0; core < num_cores; ++core) {
        noc.async_read(
            src_ep,
            cb_src,
            core_bytes,
            {.noc_x = core_noc_x[core], .noc_y = core_noc_y[core], .addr = slot_addr},
            {.offset_bytes = core * core_bytes});
    }
    noc.async_read_barrier();

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open_finish();
    }

    // Each output core's slot to the same address on every other chip; its last packet also bumps that
    // core's out_ready there.
    size_t l1_read_addr = cb_src.get_write_ptr();
    for (uint32_t core = 0; core < num_cores; ++core) {
        const uint64_t sema_noc_addr = safe_get_noc_addr(core_noc_x[core], core_noc_y[core], out_ready_sem_bank_addr);
        for (uint32_t page = 0; page < pages_per_core; page += packet_size_in_pages) {
            const uint32_t n = std::min(pages_per_core - page, packet_size_in_pages);
            const uint64_t dest_noc_addr =
                safe_get_noc_addr(core_noc_x[core], core_noc_y[core], slot_addr + page * page_size);
            if (page + n >= pages_per_core) {
                fused_write_atomic_and_advance_local_read_address_for_fabric_write(
                    dest_noc_addr,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection,
                    l1_read_addr,
                    n * page_size,
                    sema_noc_addr,
                    static_cast<uint32_t>(1),
                    false);
                noc.async_writes_flushed();
            } else {
                write_and_advance_local_read_address_for_fabric_write(
                    dest_noc_addr, pkt_hdr_forward, pkt_hdr_backward, fabric_connection, l1_read_addr, n * page_size);
            }
        }
    }

    // This chip's own increment of each output core's out_ready.
    // Device 2.0 migration: legacy primitive retained: out_ready_sem_bank_addr is the address of a GlobalSemaphore.
    // Semaphore<> binds to per-program ids via get_semaphore<>(id), so it cannot wrap a GlobalSemaphore.
    for (uint32_t core = 0; core < num_cores; ++core) {
        noc_semaphore_inc(safe_get_noc_addr(core_noc_x[core], core_noc_y[core], out_ready_sem_bank_addr), 1);
    }
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_start();
        fabric_connection.close_finish();
    }
    noc.async_write_barrier();
}
