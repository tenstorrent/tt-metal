// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace ttnn::operations::ccl::common;

void print_tile_rows(
    uint32_t cb_idx,
    uint32_t tile_idx,
    bool untilize = false,
    uint16_t start_row = 0,
    uint16_t end_row = 32,
    uint8_t start_col = 0,
    uint8_t end_col = 32) {
    DPRINT << "cb_idx: " << cb_idx << " tile_idx: " << tile_idx << ENDL();
    DPRINT << "======" << ENDL();
    for (uint16_t r = start_row; r < end_row; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_idx,
                      tile_idx,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)start_col,
                          .w1 = (uint8_t)end_col,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

namespace detail {

inline void dispatch_input_local_device_flushed(
    uint32_t input_token_read_addr, uint64_t output_token_write_addr, uint32_t output_page_size) {
    noc_async_write(input_token_read_addr, output_token_write_addr, output_page_size);
    noc_async_writes_flushed();
}

void zero_buffer_async(uint32_t write_addr, int bytes) {
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        uint32_t curr_bytes = std::min(bytes, MEM_ZEROS_SIZE);
        noc_async_read(zeros_noc_addr, write_addr, curr_bytes);
        write_addr += curr_bytes;
        bytes -= curr_bytes;
    }
}

void zero_buffer_barrier() { noc_async_read_barrier(); }

inline Polarity simple_reverse_polarity(Polarity polarity) {
    return polarity == Polarity::POSITIVE ? Polarity::NEGATIVE : Polarity::POSITIVE;
}

template <tt::tt_fabric::Topology Topology, uint32_t MeshRows, uint32_t MeshCols>
uint32_t get_simple_route(uint32_t linearized_src_mesh_coord, uint32_t linearized_dest_mesh_coord, Polarity polarity) {
    auto [src_row, src_col] = get_mesh_coords<MeshRows, MeshCols>(linearized_src_mesh_coord);
    auto [dest_row, dest_col] = get_mesh_coords<MeshRows, MeshCols>(linearized_dest_mesh_coord);
    // default_polary is for ties in a ring
    // if default is positive, then for a E-W tie, we go East, and for a N-S tie, we go South
    // if default is negative, then for a E-W tie, we go West, and for a N-S tie, we go North
    if (src_row == dest_row) {
        if (polarity == Polarity::POSITIVE) {
            return src_col < dest_col ? eth_chan_directions::EAST : eth_chan_directions::WEST;
        } else {
            return src_col <= dest_col ? eth_chan_directions::WEST : eth_chan_directions::EAST;
        }
    } else {
        if (polarity == Polarity::POSITIVE) {
            return src_row < dest_row ? eth_chan_directions::SOUTH : eth_chan_directions::NORTH;
        } else {
            return src_row <= dest_row ? eth_chan_directions::NORTH : eth_chan_directions::SOUTH;
        }
    }
}

// Get multicast direction based on axis and polarity
// COLS axis (vertical) → SOUTH (positive), NORTH (negative)
// ROWS axis (horizontal) → EAST (positive), WEST (negative)
template <ReplicateGroup Axis, Polarity P>
constexpr uint32_t get_multicast_direction() {
    if constexpr (Axis == ReplicateGroup::ROWS) {
        if constexpr (P == Polarity::POSITIVE) {
            return eth_chan_directions::SOUTH;
        } else {
            return eth_chan_directions::NORTH;
        }
    } else {
        if constexpr (P == Polarity::POSITIVE) {
            return eth_chan_directions::EAST;
        } else {
            return eth_chan_directions::WEST;
        }
    }
}

// Fabric multicast metadata write helper - handles both unicast (1 page) and scatter (2 pages) cases
template <uint32_t PageSize, uint32_t PositiveDistance, uint32_t NegativeDistance, typename AddrGenT>
FORCE_INLINE void fabric_multicast_metadata_write(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* fabric_connection_east,
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* fabric_connection_west,
    volatile PACKET_HEADER_TYPE* unicast_header_pos,
    volatile PACKET_HEADER_TYPE* unicast_header_neg,
    volatile PACKET_HEADER_TYPE* scatter_header_pos,
    volatile PACKET_HEADER_TYPE* scatter_header_neg,
    const AddrGenT& addr_gen,
    uint32_t src_addr,
    uint32_t dest_page_id,
    uint32_t pages_to_write) {
    // Compute addresses once
    const uint64_t local_noc_addr_0 = addr_gen.get_noc_addr(dest_page_id, 0);
    const uint64_t fabric_noc_addr_0 = linear::addrgen_detail::get_noc_address(addr_gen, dest_page_id, 0);

    if (pages_to_write == 1) {
        noc_async_write(src_addr, local_noc_addr_0, PageSize);

        const auto cmd_header = tt::tt_fabric::NocUnicastCommandHeader{fabric_noc_addr_0};

        tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_write(
            fabric_connection_east,
            unicast_header_pos,
            src_addr,
            static_cast<uint16_t>(PageSize),
            cmd_header,
            static_cast<uint8_t>(1),
            static_cast<uint8_t>(PositiveDistance));

        tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_write(
            fabric_connection_west,
            unicast_header_neg,
            src_addr,
            static_cast<uint16_t>(PageSize),
            cmd_header,
            static_cast<uint8_t>(1),
            static_cast<uint8_t>(NegativeDistance));
    } else {
        const uint64_t local_noc_addr_1 = addr_gen.get_noc_addr(dest_page_id + 1, 0);
        const uint64_t fabric_noc_addr_1 = linear::addrgen_detail::get_noc_address(addr_gen, dest_page_id + 1, 0);

        noc_async_write(src_addr, local_noc_addr_0, PageSize);
        noc_async_write(src_addr + PageSize, local_noc_addr_1, PageSize);

        // For scatter with 2 equal-sized chunks: first chunk size is explicit, last is implicit
        // Total payload = 2 * PageSize, with chunk_size[0] = PageSize
        constexpr uint16_t total_payload_size = 2 * PageSize;
        const auto scatter_cmd_header = tt::tt_fabric::NocUnicastScatterCommandHeader{
            {fabric_noc_addr_0, fabric_noc_addr_1}, {static_cast<uint16_t>(PageSize)}};

        tt::tt_fabric::linear::experimental::fabric_multicast_noc_scatter_write(
            fabric_connection_east,
            scatter_header_pos,
            src_addr,
            total_payload_size,
            scatter_cmd_header,
            static_cast<uint8_t>(1),
            static_cast<uint8_t>(PositiveDistance));

        tt::tt_fabric::linear::experimental::fabric_multicast_noc_scatter_write(
            fabric_connection_west,
            scatter_header_neg,
            src_addr,
            total_payload_size,
            scatter_cmd_header,
            static_cast<uint8_t>(1),
            static_cast<uint8_t>(NegativeDistance));
    }
    noc_async_writes_flushed();
}

// Bidirectional fabric multicast atomic increment - sends to both positive and negative directions
template <uint32_t PositiveDistance, uint32_t NegativeDistance>
FORCE_INLINE void fabric_multicast_bidirectional_atomic_inc(
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* fabric_connection_pos,
    tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender* fabric_connection_neg,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint64_t semaphore_noc_addr) {
    const auto cmd_header = tt::tt_fabric::NocUnicastAtomicIncCommandHeader{semaphore_noc_addr, 1, true};

    tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_atomic_inc(
        fabric_connection_pos,
        packet_header,
        cmd_header,
        static_cast<uint8_t>(1),
        static_cast<uint8_t>(PositiveDistance));

    tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_atomic_inc(
        fabric_connection_neg,
        packet_header,
        cmd_header,
        static_cast<uint8_t>(1),
        static_cast<uint8_t>(NegativeDistance));
}

template <typename DataType>
FORCE_INLINE DataType* tile_row_offset(DataType* indices_address, uint32_t row) {
    constexpr uint32_t num_face_width = 2;
    constexpr uint32_t num_face_height = 2;
    constexpr uint32_t FaceWidth = 16;
    constexpr uint32_t FaceHeight = 16;
    constexpr uint32_t TileHeight = 32;
    constexpr uint32_t TileWidth = 32;
    uint32_t offset = 0;
    uint32_t local_row = row;
    if (row >= FaceHeight) {
        offset += num_face_width * FaceHeight * FaceWidth;  // if it was generic, multiply by row/FaceHeight
        local_row -= FaceHeight;
    }
    offset += local_row * FaceWidth;
    return (DataType*)(indices_address + offset);
}

template <typename DataType>
FORCE_INLINE DataType* tile_col_offset(DataType* indices_address, uint32_t col) {
    constexpr uint32_t FaceWidth = 16;
    constexpr uint32_t FaceHeight = 16;
    uint32_t offset = 0;
    uint32_t local_col = col;
    if (col >= FaceWidth) {
        offset += FaceHeight * FaceWidth;  // if it was generic, multiply by col/FaceWidth
        local_col -= FaceWidth;
    }
    offset += local_col;
    return (DataType*)(indices_address + offset);
}

template <typename DataType>
FORCE_INLINE DataType* tile_offset(DataType* indices_address, uint32_t full_buffer_row) {
    constexpr uint32_t num_face_width = 2;
    constexpr uint32_t num_face_height = 2;
    constexpr uint32_t FaceWidth = 16;
    constexpr uint32_t FaceHeight = 16;
    constexpr uint32_t TileHeight = 32;
    constexpr uint32_t TileWidth = 32;
    constexpr uint32_t TileSize = TileHeight * TileWidth;
    return (DataType*)(indices_address + (full_buffer_row / TileHeight) * TileSize);
}

template <bool reuse_index, uint32_t indices_tensor_cb_id, uint32_t tile_height = 32, typename DataType = uint16_t>
FORCE_INLINE DataType* wait_indices(DataType* base_indices_addr, DataType*& tile_base, uint32_t local_token) {
    if constexpr (reuse_index) {
        // if re-using, the full buffer is already in the CB, so we need to offset into the relevant tile
        DataType* token_indices = tile_offset(base_indices_addr, local_token);
        // then we need to offset into the relevant row within the tile
        return tile_row_offset(token_indices, local_token % tile_height);
    } else {
        if (local_token % tile_height == 0) {
            // at tile boundary, wait for the next tile and update tile_base
            cb_wait_front(indices_tensor_cb_id, 1);
            tile_base = (DataType*)(get_read_ptr(indices_tensor_cb_id));
            return tile_base;
        } else {
            // still within the same tile, offset into the relevant row from tile_base
            return tile_row_offset(tile_base, local_token % tile_height);
        }
    }
}

template <bool reuse_index, uint32_t indices_tensor_cb_id, uint32_t tile_height = 32>
FORCE_INLINE void pop_indices(uint32_t local_token) {
    if constexpr (!reuse_index) {
        if ((local_token + 1) % tile_height == 0) {
            cb_pop_front(indices_tensor_cb_id, 1);
        }
    }
}

}  // namespace detail

using namespace ttnn::operations::ccl::common;

void kernel_main() {
    constexpr uint32_t input_tensor_cb_id = get_named_compile_time_arg_val("input_tensor_cb_id");
    constexpr uint32_t indices_tensor_cb_id = get_named_compile_time_arg_val("indices_tensor_cb_id");
    constexpr uint32_t mapping_tensor_cb_id = get_named_compile_time_arg_val("mapping_tensor_cb_id");
    constexpr uint32_t packet_header_cb_id = get_named_compile_time_arg_val("packet_header_cb_id");
    constexpr uint32_t send_preparation_buffer_cb_id = get_named_compile_time_arg_val("send_preparation_buffer_id");
    constexpr uint32_t scores_tensor_cb_id = get_named_compile_time_arg_val("scores_tensor_cb_id");

    constexpr uint32_t input_pages = get_named_compile_time_arg_val("input_pages");
    constexpr uint32_t indices_pages = get_named_compile_time_arg_val("indices_pages");
    constexpr uint32_t mapping_pages = get_named_compile_time_arg_val("mapping_pages");
    constexpr uint32_t output_pages = get_named_compile_time_arg_val("output_pages");
    constexpr uint32_t metadata_pages = get_named_compile_time_arg_val("metadata_pages");

    constexpr uint32_t input_page_size = get_named_compile_time_arg_val("input_page_size");
    constexpr uint32_t indices_page_size = get_named_compile_time_arg_val("indices_page_size");
    constexpr uint32_t mapping_page_size = get_named_compile_time_arg_val("mapping_page_size");
    constexpr uint32_t output_page_size = get_named_compile_time_arg_val("output_page_size");
    constexpr uint32_t metadata_page_size = get_named_compile_time_arg_val("metadata_page_size");

    constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    constexpr uint32_t hidden_size = get_named_compile_time_arg_val("hidden_size");
    constexpr uint32_t batch_size = get_named_compile_time_arg_val("batch_size");
    constexpr uint32_t selected_experts_k = get_named_compile_time_arg_val("selected_experts_k");
    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t tokens_per_device = get_named_compile_time_arg_val("tokens_per_device");

    constexpr uint32_t num_links = get_named_compile_time_arg_val("num_links");
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_named_compile_time_arg_val("topology");

    constexpr uint32_t src_mesh_id = get_named_compile_time_arg_val("src_mesh_id");
    constexpr uint32_t src_chip_id = get_named_compile_time_arg_val("src_chip_id");
    constexpr uint32_t mesh_rows = get_named_compile_time_arg_val("mesh_rows");
    constexpr uint32_t mesh_cols = get_named_compile_time_arg_val("mesh_cols");
    constexpr uint32_t aligned_input_page_size = get_named_compile_time_arg_val("aligned_input_page_size");
    constexpr uint32_t aligned_indices_page_size = get_named_compile_time_arg_val("aligned_indices_page_size");
    constexpr uint32_t aligned_mapping_page_size = get_named_compile_time_arg_val("aligned_mapping_page_size");
    constexpr uint32_t aligned_output_page_size = get_named_compile_time_arg_val("aligned_output_page_size");
    constexpr uint32_t aligned_metadata_page_size = get_named_compile_time_arg_val("aligned_metadata_page_size");

    constexpr uint32_t fabric_max_packet_size = get_named_compile_time_arg_val("fabric_max_packet_size");
    constexpr uint32_t l1_alignment = get_named_compile_time_arg_val("l1_alignment");
    constexpr uint32_t dram_alignment = get_named_compile_time_arg_val("dram_alignment");
    constexpr uint32_t metadata_alignment = get_named_compile_time_arg_val("metadata_alignment");
    constexpr uint32_t scores_alignment = get_named_compile_time_arg_val("scores_alignment");
    constexpr uint32_t output_alignment = get_named_compile_time_arg_val("output_alignment");
    constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
    constexpr uint32_t cluster_axis = get_named_compile_time_arg_val("cluster_axis");
    constexpr uint32_t max_indices_pages_per_packet = get_named_compile_time_arg_val("max_indices_pages_per_packet");
    constexpr uint32_t num_connections = get_named_compile_time_arg_val("num_connections");

    // Synchronization with tilizer cores - wait for E-D buffer to be zeroed before sending init semaphore
    constexpr uint32_t num_tilizer_cores = get_named_compile_time_arg_val("num_tilizer_cores");
    constexpr uint32_t ed_buffer_ready_semaphore_id = get_named_compile_time_arg_val("ed_buffer_ready_semaphore_id");

    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto input_scores_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();
    constexpr auto gathered_scores_args = TensorAccessorArgs<input_scores_args.next_compile_time_args_offset()>();

    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);          // 0
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);        // 1
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);        // 2
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);         // 3
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);       // 4
    uint32_t global_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);      // 5
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);        // 6
    uint32_t input_scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);   // 7
    uint32_t gathered_scores_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);  // 8
    uint32_t subtoken_offset = get_arg_val<uint32_t>(rt_args_idx++);               // 9
    uint32_t subtoken_size = get_arg_val<uint32_t>(rt_args_idx++);                 // 10
    uint32_t indices_start = get_arg_val<uint32_t>(rt_args_idx++);                 // 11
    uint32_t indices_end = get_arg_val<uint32_t>(rt_args_idx++);                   // 12

    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;

    constexpr uint32_t num_directions = 4;
    constexpr std::array<bool, num_directions> directions = DIRECTIONS;

    std::array<tt::tt_fabric::WorkerToFabricEdmSender, num_directions> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_args_idx);

    uint32_t send_preparation_buffer_address = get_write_ptr(send_preparation_buffer_cb_id);
    detail::zero_buffer_async(send_preparation_buffer_address, tokens_per_device * num_devices * sizeof(uint8_t));

    constexpr ReplicateGroup axis = ReplicateGroup(cluster_axis);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t col = linearized_mesh_coord % mesh_cols;

    constexpr uint32_t dispatch_index = axis == ReplicateGroup::COLS ? row : col;
    // Based on cluster axis, we only need to dispatch to the devices that are along the axis
    // If ReplicateGroup is COLs/AXIS is 1, then we dispatch alonw the ROW, and vice versa
    // For ReplicateGroup COLs/AXIS is 1, the device_begin_idx is the start of the row, and the device_end_idx is the
    // end of the row For ReplicateGroup ROWs/AXIS is 0, the device_begin_idx is the start of the column, and the
    // device_end_idx is the end of the column
    constexpr uint32_t device_begin_idx = axis == ReplicateGroup::COLS ? col : row * mesh_cols;
    constexpr uint32_t device_end_idx =
        (axis == ReplicateGroup::COLS)
            ? (col + mesh_rows * mesh_cols)   // last is col+(mesh_rows-1)*mesh_cols; add one stride
            : (row * mesh_cols + mesh_cols);  // last is row*mesh_cols+(mesh_cols-1); add one
    constexpr uint32_t device_stride = axis == ReplicateGroup::COLS ? mesh_cols : 1;

    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address, output_page_size);
    const auto gathered_scores_addr_gen =
        TensorAccessor(gathered_scores_args, gathered_scores_tensor_address, indices_page_size);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, metadata_page_size);

    uint32_t packet_header_buffer_address = get_read_ptr(packet_header_cb_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* metadata_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));
    auto* mcast_noc_unicast_packet_header_pos =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + 2 * sizeof(PACKET_HEADER_TYPE));
    auto* mcast_noc_unicast_packet_header_neg =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + 3 * sizeof(PACKET_HEADER_TYPE));
    auto* mcast_noc_scatter_packet_header_pos =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + 4 * sizeof(PACKET_HEADER_TYPE));
    auto* mcast_noc_scatter_packet_header_neg =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + 5 * sizeof(PACKET_HEADER_TYPE));

    detail::zero_buffer_barrier();
    open_direction_connections_barrier(directions, fabric_connections);

    // Wait for tilizer cores to finish zeroing E-D buffer before sending init semaphore
    // This ensures remote devices don't receive init signal before their E-D buffer is ready
    // Tilizers signal by multicast writing 1 to this semaphore, so we wait for value 1
    uint32_t ed_ready_sem_addr = get_semaphore(ed_buffer_ready_semaphore_id);
    noc_semaphore_wait((uint32_t*)ed_ready_sem_addr, 1);
    noc_semaphore_set((uint32_t*)ed_ready_sem_addr, 0);  // Reset for next iteration

    // Send initialization semaphore to configured targets for synchronization
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
    constexpr uint32_t positive_distance = dispatch_index % 2 == 0 ? (dispatch_devices - 1) / 2 : dispatch_devices / 2;
    constexpr uint32_t negative_distance = (dispatch_devices - 1) - positive_distance;
    detail::fabric_multicast_bidirectional_atomic_inc<positive_distance, negative_distance>(
        &fabric_connections[eth_chan_directions::EAST],
        &fabric_connections[eth_chan_directions::WEST],
        metadata_packet_header,
        init_noc_semaphore_addr);

    // Wait for all devices to complete initialization synchronization
    bool needs_barrier = false;
    noc_semaphore_wait((uint32_t*)init_semaphore_address, dispatch_devices - 1);
    noc_semaphore_set((uint32_t*)init_semaphore_address, 0);

    // Write out the indices tensor
    constexpr bool reuse_index = true;
    uint32_t base_indices_addr = reuse_index ? get_read_ptr(indices_tensor_cb_id) : 0;
    constexpr uint32_t base_page = dispatch_index * indices_pages;
    for (uint32_t indices_page = indices_start; indices_page < indices_end;
         indices_page += max_indices_pages_per_packet) {
        uint32_t pages_left = indices_end - indices_page;
        uint32_t pages_to_write = std::min(max_indices_pages_per_packet, pages_left);

        cb_wait_front(indices_tensor_cb_id, max_indices_pages_per_packet);

        uint32_t indices_addr = get_read_ptr(indices_tensor_cb_id);
        // fabric write the indices tensor to metadata tensor
        detail::fabric_multicast_metadata_write<metadata_page_size, positive_distance, negative_distance>(
            &fabric_connections[eth_chan_directions::EAST],
            &fabric_connections[eth_chan_directions::WEST],
            mcast_noc_unicast_packet_header_pos,
            mcast_noc_unicast_packet_header_neg,
            mcast_noc_scatter_packet_header_pos,
            mcast_noc_scatter_packet_header_neg,
            metadata_addr_gen,
            indices_addr,
            base_page + indices_page,
            pages_to_write);
        cb_pop_front(indices_tensor_cb_id, max_indices_pages_per_packet);
    }

    for (uint32_t input_scores_page = indices_start; input_scores_page < indices_end;
         input_scores_page += max_indices_pages_per_packet) {
        uint32_t pages_left = indices_end - input_scores_page;
        uint32_t pages_to_write = std::min(max_indices_pages_per_packet, pages_left);
        cb_wait_front(scores_tensor_cb_id, max_indices_pages_per_packet);
        uint32_t base_input_scores_addr = get_read_ptr(scores_tensor_cb_id);
        // fabric write the scores tensor to gathered scores tensor
        detail::fabric_multicast_metadata_write<indices_page_size, positive_distance, negative_distance>(
            &fabric_connections[eth_chan_directions::EAST],
            &fabric_connections[eth_chan_directions::WEST],
            mcast_noc_unicast_packet_header_pos,
            mcast_noc_unicast_packet_header_neg,
            mcast_noc_scatter_packet_header_pos,
            mcast_noc_scatter_packet_header_neg,
            gathered_scores_addr_gen,
            base_input_scores_addr,
            base_page + input_scores_page,
            pages_to_write);
        cb_pop_front(scores_tensor_cb_id, max_indices_pages_per_packet);
    }

    constexpr uint32_t tile_height = 32;
    uint16_t* base_indices_ptr = (uint16_t*)base_indices_addr;
    uint16_t* tile_base = nullptr;  // tracks current tile base for non-reuse case
    cb_wait_front(mapping_tensor_cb_id, mapping_pages);
    uint16_t* devices_for_experts = (uint16_t*)get_read_ptr(mapping_tensor_cb_id);
    uint8_t* send_preparation_buffer = (uint8_t*)send_preparation_buffer_address;
    for (uint32_t local_token = 0; local_token < tokens_per_device; local_token++) {
        // global_token is the global token index for the current token
        // we need the global token index to write to the output buffer – each global token that could potentially be
        // sent has a unique output buffer address to ensure that it is not overwritten by another token
        uint32_t global_token = (local_token + (tokens_per_device * dispatch_index));

        uint16_t* token_indices = detail::wait_indices<reuse_index, indices_tensor_cb_id, tile_height>(
            base_indices_ptr, tile_base, local_token);
        cb_wait_front(input_tensor_cb_id, 1);
        // The reader already reads from input+subtoken_offset and writes to CB at position 0
        // So we read from CB position 0, not CB+subtoken_offset
        uint32_t input_token_read_addr = get_read_ptr(input_tensor_cb_id);

        for (uint32_t k = 0; k < selected_experts_k; k++) {
            // get the expert that is chosen for the current token
            uint16_t expert_chosen = detail::tile_col_offset(token_indices, k)[0];
            uint16_t d = devices_for_experts[expert_chosen];

            if (send_preparation_buffer[(local_token * num_devices) + d] == 0) {
                if (d == linearized_mesh_coord) {
                    // if the expert lives on the current device, we dispatch the input token to it
                    uint64_t output_token_write_addr = get_noc_addr(global_token, output_addr_gen) + subtoken_offset;
                    detail::dispatch_input_local_device_flushed(
                        input_token_read_addr, output_token_write_addr, subtoken_size);
                    needs_barrier = true;
                } else if (is_configured_target<linearized_mesh_coord, mesh_rows, mesh_cols, axis>(d)) {
                    // if the expert lives on a remote device, we dispatch the input token to it
                    // if axis is specified then we only send to the devices that are along the axis
                    // if axis is not specified then we send to all devices
                    fabric_send_chip_unicast_noc_unicast_1d<
                        linearized_mesh_coord,
                        topology,
                        mesh_rows,
                        mesh_cols,
                        fabric_max_packet_size>(
                        output_addr_gen,
                        fabric_connections,
                        unicast_packet_header,
                        d,
                        input_token_read_addr,
                        global_token,
                        (int)subtoken_size,
                        output_alignment,
                        subtoken_offset);
                }
                send_preparation_buffer[(local_token * num_devices) + d] = 1;
            }
        }
        cb_pop_front(input_tensor_cb_id, 1);
        detail::pop_indices<reuse_index, indices_tensor_cb_id, tile_height>(local_token);
    }
    if (needs_barrier) {
        noc_async_write_barrier();
    }

    cb_pop_front(mapping_tensor_cb_id, mapping_pages);

    const uint64_t global_noc_semaphore_addr = get_noc_addr(global_semaphore_address);
    detail::fabric_multicast_bidirectional_atomic_inc<positive_distance, negative_distance>(
        &fabric_connections[eth_chan_directions::EAST],
        &fabric_connections[eth_chan_directions::WEST],
        metadata_packet_header,
        global_noc_semaphore_addr);
    noc_async_write_barrier();
    noc_semaphore_wait((uint32_t*)global_semaphore_address, dispatch_devices - 1);
    noc_semaphore_set((uint32_t*)global_semaphore_address, 0);

    close_direction_connections(directions, fabric_connections);
}
