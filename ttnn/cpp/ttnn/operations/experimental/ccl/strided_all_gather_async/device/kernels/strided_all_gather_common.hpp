// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

FORCE_INLINE uint32_t
get_next_tile_input(uint32_t local_tile_index, uint32_t input_start_tile_index, uint32_t ag_parallel_factor) {
    // Imagine the input is already permuted (this has nothing to do with our all gather, it's just ordering the output
    // of all gather such it will be ideal for matmul) We split up the work evenly amongst the all gather cores.
    // Probably the best way is just to round robin through the input amongst the various all gather cores.  Ignore
    // direction since you send the same thing forward and backward.  For now just send the whole thing in that order,
    // we can add finer grain fidelity to correspond to the syncs for matmul.  Right now just sync once when we reach
    // the end of the buffer.
    return input_start_tile_index + local_tile_index * ag_parallel_factor;
}

FORCE_INLINE uint32_t get_next_tile_output(
    uint32_t local_tile_index,
    uint32_t input_start_tile_index,
    uint32_t ag_parallel_factor,
    uint32_t input_tensor_Wt,
    uint32_t output_tensor_Wt,
    uint32_t device_index) {
    uint32_t input_tile_index = input_start_tile_index + local_tile_index * ag_parallel_factor;
    uint32_t input_row = input_tile_index / input_tensor_Wt;
    uint32_t input_col = input_tile_index % input_tensor_Wt;
    return input_row * output_tensor_Wt + device_index * input_tensor_Wt +
           input_col;  // TODO should pass device_index*input_tensor_Wt to prevent recalculating them
}

FORCE_INLINE uint32_t
get_sender_id(uint32_t direction, uint32_t my_chip_id, uint32_t slices_received, uint32_t ring_size) {
    int32_t sender_chip_id;
    if (direction == 1) {
        sender_chip_id = my_chip_id + slices_received + 1;
        return (sender_chip_id >= (int)ring_size) ? sender_chip_id - ring_size : sender_chip_id;
    } else {
        sender_chip_id = my_chip_id - (slices_received + 1);
        return (sender_chip_id < 0) ? ring_size + sender_chip_id : sender_chip_id;
    }
}
