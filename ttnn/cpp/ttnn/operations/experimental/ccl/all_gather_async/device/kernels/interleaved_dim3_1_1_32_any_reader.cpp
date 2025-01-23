// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This should ideally be merged with `ccl_send_reader` when we are able to support compile time args
//       that don't require macros to function

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/buffer_constants.hpp>
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"

#include "cpp/ttnn/operations/ccl/common/kernels/command_processor.hpp"

#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"

#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/io_descriptors.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "cpp/ttnn/tensor/enum_types.hpp"
#include <cstdint>
#include <utility>

using arg_idx_t = uint16_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint16_t my_chip_id = get_compile_time_arg_val(0);
constexpr TensorMemoryLayout tensor0_layout = static_cast<TensorMemoryLayout>(get_compile_time_arg_val(1));
constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(2));
constexpr Layout tensor0_page_layout = static_cast<Layout>(get_compile_time_arg_val(3));
constexpr uint32_t cb0_id = get_compile_time_arg_val(4);
constexpr uint32_t num_pages_read_total = get_compile_time_arg_val(5);
constexpr uint32_t num_workers = get_compile_time_arg_val(6);

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    const uint16_t packet_size_in_pages = get_arg_val<uint32_t>(arg_idx++);
    uint16_t tensor0_page_size = get_arg_val<uint32_t>(arg_idx++);

    // print every compile and runtime arg in uint32_t
    DPRINT << "ct args: \n";
    DPRINT << "my_chip_id: " << (uint32_t)my_chip_id << "\n";
    DPRINT << "tensor0_layout: " << (uint32_t)tensor0_layout << "\n";
    DPRINT << "buffer0_type: " << (uint32_t)buffer0_type << "\n";
    DPRINT << "tensor0_page_layout: " << (uint32_t)tensor0_page_layout << "\n";
    DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << "num_pages_read_total: " << (uint32_t)num_pages_read_total << "\n";
    DPRINT << "num_workers: " << (uint32_t)num_workers << "\n";

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << "packet_size_in_pages: " << (uint32_t)packet_size_in_pages << "\n";
    DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";

    // interleaved addrgen
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    auto tensor0_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    DPRINT << "tensor -> CB: " << (uint32_t)cb0_id << "\n";
    DPRINT << "packet size in pages: " << (uint32_t)packet_size_in_pages << "\n";

    uint32_t tile_id = 0;
    for (uint32_t i = 0; i < num_pages_read_total / packet_size_in_pages + 1; i++) {
        DPRINT << "i: " << i << "\n";
        cb_reserve_back(cb0_id, packet_size_in_pages);
        const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
        uint32_t l1_write_addr = l1_write_addr_base;

        uint16_t num_pages_to_read = (i == num_pages_read_total / packet_size_in_pages)
                                         ? num_pages_read_total % packet_size_in_pages
                                         : packet_size_in_pages;
        for (uint16_t j = 0; j < num_pages_to_read; j++) {
            noc_async_read_tile(tile_id, tensor0_addrgen, l1_write_addr);
            l1_write_addr += tensor0_page_size;
            tile_id++;
        }
        DPRINT << "tile_id: " << tile_id << "\n";

        noc_async_read_barrier();
        cb_push_back(cb0_id, packet_size_in_pages);
    }

    DPRINT << "DONE \n";
}
