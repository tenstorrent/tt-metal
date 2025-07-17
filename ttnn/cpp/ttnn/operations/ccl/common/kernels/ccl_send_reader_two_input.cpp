// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This should ideally be merged with `ccl_send_reader` when we are able to support compile time args
//       that don't require macros to function

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"

#include "ttnn/operations/ccl/common/kernels/command_processor.hpp"

#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/io_descriptors.hpp"
#include "api/ttnn/tensor/enum_types.hpp"
#include <cstdint>
#include <utility>

using arg_idx_t = uint16_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

struct no_addrgen {};
static constexpr size_t num_packet_headers_storable = 8;
constexpr uint16_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);

#ifdef NO_TENSOR_MODE
constexpr TensorMemoryLayout tensor0_layout = TensorMemoryLayout::INTERLEAVED;
constexpr BufferType buffer0_type = BufferType::DRAM;
constexpr Layout tensor0_page_layout = Layout::TILE;
constexpr uint32_t cb0_id = tt::CB::c_in0;
constexpr TensorMemoryLayout tensor1_layout = TensorMemoryLayout::INTERLEAVED;
constexpr BufferType buffer1_type = BufferType::DRAM;
constexpr Layout tensor1_page_layout = Layout::TILE;
constexpr uint32_t cb1_id = tt::CB::c_in1;
#else
constexpr TensorMemoryLayout tensor0_layout = static_cast<TensorMemoryLayout>(get_compile_time_arg_val(2));
constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(3));
constexpr Layout tensor0_page_layout = static_cast<Layout>(get_compile_time_arg_val(4));
constexpr uint32_t cb0_id = get_compile_time_arg_val(5);
#ifndef SINGLE_TENSOR
constexpr TensorMemoryLayout tensor1_layout = static_cast<TensorMemoryLayout>(get_compile_time_arg_val(6));
constexpr BufferType buffer1_type = static_cast<BufferType>(get_compile_time_arg_val(7));
constexpr Layout tensor1_page_layout = static_cast<Layout>(get_compile_time_arg_val(8));
constexpr uint32_t cb1_id = get_compile_time_arg_val(9);
#endif
#endif

#ifdef TENSOR0_SHARDED_MEM_LAYOUT
#ifdef SINGLE_TENSOR
// SINGLE INPUT MODE - SHARDED
    using Tensor0ShardInfo = ShardedInfo<
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12)>;
#else
// TWO INPUT MODE
    using Tensor0ShardInfo = ShardedInfo<
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15),
        get_compile_time_arg_val(16)>;

#endif
constexpr Tensor0ShardInfo test_object {};
static_assert(test_object.number_of_cores > 0, "Misconfigured sharded addrgen fields for tensor0. Field \"number_of_cores\" was resolved to 0 but it must not be 0.");
static_assert(test_object.page_size_jump > 0, "Misconfigured sharded addrgen fields for tensor0. Field \"page_size_jump\" was resolved to 0 but it must not be 0.");
static_assert(test_object.pages_per_tensor_row > 0, "Misconfigured sharded addrgen fields for tensor0. Field \"pages_per_tensor_row\" was resolved to 0 but it must not be 0.");
#else
using Tensor0ShardInfo = ShardedInfo<0,0,0,0,0,0,0>;
#endif

#ifndef SINGLE_TENSOR
#if defined(TENSOR1_SHARDED_MEM_LAYOUT)
#if defined(TENSOR0_SHARDED_MEM_LAYOUT)
  using Tensor1ShardInfo = ShardedInfo<
        get_compile_time_arg_val(17),
        get_compile_time_arg_val(18),
        get_compile_time_arg_val(19),
        get_compile_time_arg_val(20),
        get_compile_time_arg_val(21),
        get_compile_time_arg_val(22),
        get_compile_time_arg_val(23)>;
#else
// Then we are only consuming ct args for second operand and we resume from operation 8
    using Tensor1ShardInfo = ShardedInfo<
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15),
        get_compile_time_arg_val(16)>;
#endif

constexpr Tensor1ShardInfo test_object2 {};
static_assert(test_object2.number_of_cores > 0, "Misconfigured sharded addrgen fields for tensor1. Field \"number_of_cores\" was resolved to 0 but it must not be 0.");
static_assert(test_object2.page_size_jump > 0, "Misconfigured sharded addrgen fields for tensor1. Field \"page_size_jump\" was resolved to 0 but it must not be 0.");
static_assert(test_object2.pages_per_tensor_row > 0, "Misconfigured sharded addrgen fields for tensor1. Field \"pages_per_tensor_row\" was resolved to 0 but it must not be 0.");
#else
using Tensor1ShardInfo = ShardedInfo<0,0,0,0,0,0,0>;
#endif
#endif

template <
    tt::tt_metal::TensorMemoryLayout tensor_layout,
    tt::tt_metal::BufferType buffer_type,
    tt::tt_metal::Layout page_layout,
    typename ShardingInfoType>
auto build_source_address_generator(
    std::size_t& arg_idx,
    address_t tensor_address,
    std::size_t page_size,
    uint32_t cb_id_in) {
    constexpr bool is_sharded = is_sharded_tensor_layout(tensor_layout);
    constexpr bool is_interleaved = tensor_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    constexpr bool is_tile_page_layout = page_layout == tt::tt_metal::Layout::TILE;
    constexpr bool is_row_major_layout = page_layout == tt::tt_metal::Layout::ROW_MAJOR;
    static_assert(
        is_sharded || is_interleaved,
        "Only sharded and interleaved tensor layouts are supported but the unified address generator. A tensor layout "
        "not matching TensorMemoryLayout::WIDTH_SHARDED, TensorMemoryLayout::HEIGHT_SHARDED, "
        "TensorMemoryLayout::BLOCK_SHARDED, or TensorMemoryLayout::INTERLEAVED was specified.");
    bool addrgen_enabled = get_arg_val<uint32_t>(arg_idx++) != 0;
    if constexpr (tensor_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        if constexpr (is_row_major_layout) {
            InterleavedAddrGen<buffer_type ==BufferType::DRAM > ret_val = {
                .bank_base_address = tensor_address, .page_size = page_size};
            return ret_val;
        } else {
            InterleavedAddrGenFast<buffer_type ==BufferType::DRAM> ret_val = {
                .bank_base_address = tensor_address, .page_size = page_size, .data_format = get_dataformat(cb_id_in)};
            return ret_val;
        }
    } else if constexpr (
        tensor_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED ||
        tensor_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
        tensor_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
        const auto [mapping_table, rt_increment] = experimental::shard_addr_gen_utils::get_shard_map<ShardingInfoType>(get_arg_addr(arg_idx));
        if (addrgen_enabled)
        {
            arg_idx += rt_increment;
        }
        experimental::ShardedAddrGen<ShardingInfoType> ret_val = {
            .bank_base_address = tensor_address, .shard_array=mapping_table};
        return ret_val;
    } else {
        ASSERT(false);
        InterleavedAddrGen<buffer_type ==BufferType::DRAM > ret_val = {
                .bank_base_address = tensor_address, .page_size = page_size};
        return ret_val;
    }
}

// TODO: rename to tensor IO command context
struct wrapped_worker_slice_read_context {
    uint32_t curr_tile_id = 0;
    uint32_t offset_into_worker_slice = 0;
};
struct inline_value_context {
    uint32_t value = 0;
    uint32_t wrap = 0;
};
struct remote_sem_change_context {
    uint32_t value = 0;
};
using remote_sem_wait_context = remote_sem_change_context;
using remote_atomic_inc_context = remote_sem_change_context;

struct noc_transfer_burst_context {
    uint32_t bank_base_address = 0;
    uint16_t num_transfers_total = 0;
    uint16_t current_noc_transfer = 0;
};
union cmd_specific_context {
    wrapped_worker_slice_read_context wrapped_worker_slice_read_ctx;
    noc_transfer_burst_context noc_transfer_burst_ctx;

    // sem wait and atomic inc
    inline_value_context inline_value_ctx;
    cmd_specific_context() {}
};

template <typename Addrgen>
struct command_context_t;

template <typename Addrgen>
void update_ccl_command(
    arg_idx_t& arg_idx, command_context_t<Addrgen>& cmd_ctx, const ttnn::ccl::cmd::CclCommandHeader& cmd_header);

template <typename Addrgen>
struct command_context_t final {
    command_context_t(
        FabricConnectionManager& fabric_connection,
        Addrgen& addrgen,
        uint16_t num_commands,
        arg_idx_t start_arg_idx,
        uint8_t cb_id,
        uint16_t page_size,
        uint8_t packet_size_in_pages,
        size_t packet_header_buffer_addr,
        uint8_t stream_id) :
        fabric_connection(fabric_connection),
        tensor_addrgen(addrgen),
        cmd_specific_ctx(),
        packet_header_buffer_addr(packet_header_buffer_addr),
        num_commands(num_commands),
        arg_idx(start_arg_idx),
        command_idx(0),
        page_size(page_size),
        cb_id(cb_id),
        packet_size_in_pages(packet_size_in_pages),
        stream_id(stream_id) {
        ASSERT(num_commands == 0 || arg_idx > 4);
    }
    FabricConnectionManager& fabric_connection;
    ttnn::ccl::cmd::CclCommandTensor command_tensor;
    ttnn::ccl::cmd::CclCommandHeader current_cmd_header;
    // TODO: optimize packing
    address_info_t src_addr_info;
    address_info_t dest_addr_info;
    core_descriptor_info_t core_desc_info;
    Addrgen& tensor_addrgen;
    cmd_specific_context cmd_specific_ctx;
    size_t packet_header_buffer_addr = 0;

    uint16_t num_commands = 0;
    arg_idx_t arg_idx = 0;
    uint16_t command_idx = 0;

    uint16_t page_size = 0;
    ttnn::ccl::cmd::CclCommandAddrType src_addr_type = ttnn::ccl::cmd::CclCommandAddrType::NONE;
    ttnn::ccl::cmd::CclCommandAddrType dest_addr_type = ttnn::ccl::cmd::CclCommandAddrType::NONE;
    ttnn::ccl::cmd::CclCommandCoreDescriptorType core_desc_type = ttnn::ccl::cmd::CclCommandCoreDescriptorType::ADDRGEN;
    uint8_t cb_id = 0;
    uint8_t packet_size_in_pages = 0;
    uint8_t stream_id;

    bool populated = false;

    bool command_requires_fabric() const {
        return current_cmd_header.dest_type != ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY;
    }

    FORCE_INLINE bool is_complete() const { return command_idx >= num_commands; }

    FORCE_INLINE void complete_current_command() {
        command_idx++;
        populated = false;
    }

    FORCE_INLINE bool current_command_active() const { return populated; }

    void fetch_next_command() {
        populated = true;

        this->current_cmd_header = ttnn::ccl::cmd::CclCommandHeader::from_uint32(get_arg_val<uint32_t>(arg_idx++));
#ifdef DEBUG_PRINT_ENABLED
        DPRINT << "CMD (code=" << (uint32_t)current_cmd_header.code
               << ", args=" << (uint32_t)current_cmd_header.arg_count << ", idx=" << (uint32_t)(arg_idx - 1) << "\n";
#endif
        update_ccl_command(arg_idx, *this, current_cmd_header);
        switch (current_cmd_header.code) {
            case ttnn::ccl::cmd::CclCommandCode::STREAM_CB_TO_TENSOR:
            case ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_EDM: {
#ifndef NO_TENSOR_MODE
                const shape_t worker_start_offset_global = v2::worker_wrapped_offset_to_coord(
                    command_tensor.tensor_slice_shape, command_tensor.worker_start_offset_in_slice);
                const shape_t global_offset = command_tensor.tensor_slice_offset + worker_start_offset_global;

                const size_t curr_tile_id = get_flat_index_from_shape(command_tensor.tensor_shape, global_offset);
                cmd_specific_ctx.wrapped_worker_slice_read_ctx = wrapped_worker_slice_read_context{curr_tile_id};
#endif
            } break;
            case ttnn::ccl::cmd::CclCommandCode::WAIT_VALUE:
            case ttnn::ccl::cmd::CclCommandCode::ATOMIC_INC:
            case ttnn::ccl::cmd::CclCommandCode::NOC_READ_BURST:
            case ttnn::ccl::cmd::CclCommandCode::NOC_WRITE_BURST:
            case ttnn::ccl::cmd::CclCommandCode::RAW_INLINE_WRITE_BYTES: break;
            default: ASSERT(false);
        }
    }
};

template <typename Addrgen>
void update_ccl_command(
    arg_idx_t& arg_idx, command_context_t<Addrgen>& cmd_ctx, const ttnn::ccl::cmd::CclCommandHeader& cmd_header) {
    using namespace ttnn::ccl::cmd;

    arg_idx_t arg_idx_old = arg_idx;
    for (arg_idx_t i = 0; i < cmd_header.arg_count; i++) {
        // Note that we choose to reinterpret our pointers as volatile so that in the future we can add streaming
        // of additional commands from some backing memory (e.g. dram or L1), potentially by another core, without
        // having to track down this code and add volatile casts later (which would be a potentially tricky bug to
        // root cause).
        const CclCommandArgHeader command_arg_header =
            CclCommandArgHeader::from_uint32(get_arg_val<uint32_t>(arg_idx++));
        const CclCommandArgCode command_arg_code = command_arg_header.code;
        auto& cmd_tensor = cmd_ctx.command_tensor;
        switch (command_arg_code) {
            case CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES:
                CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::unpack(
                    reinterpret_cast<volatile uint32_t*>(get_arg_addr(arg_idx)), cmd_tensor.tensor_shape);
                arg_idx += CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::size_in_words();
                break;
            case CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES:
                CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::unpack(
                    reinterpret_cast<volatile uint32_t*>(get_arg_addr(arg_idx)), cmd_tensor.tensor_slice_shape);
                arg_idx += CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_SHAPE_IN_PAGES>::size_in_words();
                break;
            case CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES:
                CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::unpack(
                    reinterpret_cast<volatile uint32_t*>(get_arg_addr(arg_idx)), cmd_tensor.tensor_slice_offset);
                arg_idx += CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::size_in_words();
                break;
            case CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES:
                CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::unpack(
                    reinterpret_cast<volatile uint32_t*>(get_arg_addr(arg_idx)),
                    cmd_tensor.worker_start_offset_in_slice);
                arg_idx += CclCommandArg<CclCommandArgCode::SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES>::size_in_words();
                break;
            case CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE:
                CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::unpack(
                    reinterpret_cast<volatile uint32_t*>(get_arg_addr(arg_idx)), cmd_tensor.worker_pages_per_slice);
                arg_idx += CclCommandArg<CclCommandArgCode::SET_WORKER_PAGES_PER_SLICE>::size_in_words();
                break;
            case CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES:
                CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::unpack(
                    reinterpret_cast<volatile uint32_t*>(get_arg_addr(arg_idx)), cmd_tensor);
                arg_idx += CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::size_in_words();
                break;

            case CclCommandArgCode::SET_TARGET_VALUE:
            case CclCommandArgCode::SET_ATOMIC_INC_VALUE: {
                bool val_inline = static_cast<bool>(command_arg_header.inline_value0);
                ASSERT(val_inline);
                cmd_ctx.cmd_specific_ctx.inline_value_ctx = inline_value_context{};
                cmd_ctx.cmd_specific_ctx.inline_value_ctx.value = command_arg_header.inline_value1;
            } break;

            case CclCommandArgCode::SET_ADDRESS_INFO: {
                const auto src_dest_type = static_cast<SRC_DEST_TYPE>(command_arg_header.inline_value0);
                const auto addr_type =
                    static_cast<ttnn::ccl::cmd::CclCommandAddrType>(command_arg_header.inline_value1);
                auto& addr_info = src_dest_type == SRC_DEST_TYPE::SRC ? cmd_ctx.src_addr_info : cmd_ctx.dest_addr_info;
                auto& cmd_ctx_addr_type =
                    src_dest_type == SRC_DEST_TYPE::SRC ? cmd_ctx.src_addr_type : cmd_ctx.dest_addr_type;
                cmd_ctx_addr_type = addr_type;
                switch (addr_type) {
                    case ttnn::ccl::cmd::CclCommandAddrType::CIRCULAR_BUFFER_ID:
                        cmd_ctx.cb_id = command_arg_header.inline_value2;
                        break;
                    case ttnn::ccl::cmd::CclCommandAddrType::ABSOLUTE_ADDRESS:
                    case ttnn::ccl::cmd::CclCommandAddrType::RELATIVE_ADDRESS:
                        addr_info.address = get_arg_val<uint32_t>(arg_idx++);
                        break;
                    case ttnn::ccl::cmd::CclCommandAddrType::SEMAPHORE_ID:
                        addr_info.address = get_semaphore(command_arg_header.inline_value2);
                        break;
                    case ttnn::ccl::cmd::CclCommandAddrType::NONE: break;
                    default: ASSERT(false); break;
                };
            } break;

            case CclCommandArgCode::SET_CORE_DESCRIPTOR_INFO: {
                cmd_ctx.core_desc_type =
                    static_cast<ttnn::ccl::cmd::CclCommandCoreDescriptorType>(command_arg_header.inline_value0);
                switch (cmd_ctx.core_desc_type) {
                    case ttnn::ccl::cmd::CclCommandCoreDescriptorType::NONE:
                    case ttnn::ccl::cmd::CclCommandCoreDescriptorType::ADDRGEN:
                    case ttnn::ccl::cmd::CclCommandCoreDescriptorType::LOCAL: break;
                    case ttnn::ccl::cmd::CclCommandCoreDescriptorType::NOC_XY:
                        cmd_ctx.core_desc_info.core_desc_args.noc_unicast =
                            ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY{
                                command_arg_header.inline_value1, command_arg_header.inline_value2};
                        break;
                    case ttnn::ccl::cmd::CclCommandCoreDescriptorType::RECTANGLE:
                        cmd_ctx.core_desc_info.core_desc_args.noc_multicast =
                            ttnn::ccl::cmd::CclCommandCoreDescriptorTypeMcast::from_uint32(
                                get_arg_val<uint32_t>(arg_idx++));
                        break;
                    default: ASSERT(false);
                };

            } break;

            case CclCommandArgCode::SET_NOC_TRANSFER_BURST_START_INFO:
                cmd_ctx.cmd_specific_ctx.noc_transfer_burst_ctx.num_transfers_total = command_arg_header.inline_value0;
                cmd_ctx.cmd_specific_ctx.noc_transfer_burst_ctx.bank_base_address = get_arg_val<uint32_t>(arg_idx++);
                cmd_ctx.cmd_specific_ctx.noc_transfer_burst_ctx.current_noc_transfer = 0;
                break;

            default: {
                ASSERT(false);
            }
        };
    }
}

template <typename Addrgen>
void try_advance_inline_write_or_atomic_inc(command_context_t<Addrgen>& cmd_ctx) {
    const size_t value = cmd_ctx.cmd_specific_ctx.inline_value_ctx.value;
    const size_t dest_bank_addr = cmd_ctx.dest_addr_info.address;
    bool is_remote_atomic_inc_over_fabric = cmd_ctx.command_requires_fabric();

    // noc mcast atomic inc not supported yet
    ASSERT(
        cmd_ctx.core_desc_type == ttnn::ccl::cmd::CclCommandCoreDescriptorType::NOC_XY ||
        cmd_ctx.core_desc_type == ttnn::ccl::cmd::CclCommandCoreDescriptorType::LOCAL);
    const uint8_t dest_noc0_x = cmd_ctx.core_desc_type == ttnn::ccl::cmd::CclCommandCoreDescriptorType::LOCAL
                                    ? my_x[0]
                                    : cmd_ctx.core_desc_info.core_desc_args.noc_unicast.x;
    const uint8_t dest_noc0_y = cmd_ctx.core_desc_type == ttnn::ccl::cmd::CclCommandCoreDescriptorType::LOCAL
                                    ? my_y[0]
                                    : cmd_ctx.core_desc_info.core_desc_args.noc_unicast.y;

    bool write_local = !is_remote_atomic_inc_over_fabric;
    if (is_remote_atomic_inc_over_fabric) {
        ASSERT(cmd_ctx.core_desc_type == ttnn::ccl::cmd::CclCommandCoreDescriptorType::NOC_XY);

        ASSERT(cmd_ctx.packet_header_buffer_addr != 0);
        auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE *>(cmd_ctx.packet_header_buffer_addr);

        uint64_t dest_noc_addr_for_pkt = safe_get_noc_addr(dest_noc0_x, dest_noc0_y, dest_bank_addr, 0);
        if (cmd_ctx.current_cmd_header.code == ttnn::ccl::cmd::CclCommandCode::ATOMIC_INC) {
            pkt_hdr->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{dest_noc_addr_for_pkt, static_cast<uint16_t>(value), 32});
        } else {
            pkt_hdr->to_noc_unicast_inline_write(
                tt::tt_fabric::NocUnicastInlineWriteCommandHeader{dest_noc_addr_for_pkt, static_cast<uint16_t>(value)});
        }

        switch (cmd_ctx.current_cmd_header.dest_type) {
            case ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST: {
                pkt_hdr->to_chip_unicast(cmd_ctx.current_cmd_header.get_unicast_dest_args().distance_in_hops);

                auto& fabric_connection = cmd_ctx.current_cmd_header.get_unicast_dest_args().is_forward_direction
                                              ? cmd_ctx.fabric_connection.get_forward_connection()
                                              : cmd_ctx.fabric_connection.get_backward_connection();
                fabric_connection.wait_for_empty_write_slot();
                fabric_connection.send_payload_flush_blocking_from_address(
                    cmd_ctx.packet_header_buffer_addr, sizeof(PACKET_HEADER_TYPE));
            } break;
            case ttnn::ccl::cmd::CclCommandDestType::CHIP_MULTICAST: {
                write_local = true;
                const auto& mcast_args = cmd_ctx.current_cmd_header.get_multicast_dest_args();
                if (cmd_ctx.fabric_connection.has_forward_connection()) {
                    pkt_hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                        1, static_cast<uint8_t>(mcast_args.num_targets_forward_direction)});
                    cmd_ctx.fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                    cmd_ctx.fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                        cmd_ctx.packet_header_buffer_addr, sizeof(PACKET_HEADER_TYPE));
                }

                // Write the mcast packet (backward)
                if (cmd_ctx.fabric_connection.has_backward_connection()) {
                    pkt_hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                        1, static_cast<uint8_t>(mcast_args.num_targets_backward_direction)});
                    cmd_ctx.fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                    cmd_ctx.fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
                        cmd_ctx.packet_header_buffer_addr, sizeof(PACKET_HEADER_TYPE));
                }

            } break;

            default: {
                ASSERT(false);
            } break;
        };

    }
    if (write_local) {
        uint64_t dest_noc_addr = get_noc_addr(dest_noc0_x, dest_noc0_y, dest_bank_addr);
        if (cmd_ctx.current_cmd_header.code == ttnn::ccl::cmd::CclCommandCode::ATOMIC_INC) {
            noc_semaphore_inc(dest_noc_addr, value);
        } else if (cmd_ctx.current_cmd_header.code == ttnn::ccl::cmd::CclCommandCode::RAW_INLINE_WRITE_BYTES) {
            noc_inline_dw_write(dest_noc_addr, value);
        }
    }
}

#ifndef NO_TENSOR_MODE
template <tt::tt_metal::TensorMemoryLayout TENSOR_LAYOUT, tt::tt_metal::Layout MEM_LAYOUT, typename Addrgen>
void try_advance_read_tensor_to_cb(command_context_t<Addrgen>& cmd_ctx) {
    if (!cb_pages_reservable_at_back(cmd_ctx.cb_id, cmd_ctx.packet_size_in_pages)) {
        return;
    }

    DPRINT << "tensor -> CB: " << (uint32_t)cmd_ctx.cb_id << "\n";

    wrapped_worker_slice_read_context& cmd_specific_ctx = cmd_ctx.cmd_specific_ctx.wrapped_worker_slice_read_ctx;
    const uint16_t max_pages_readable = std::min<size_t>(
        cmd_ctx.packet_size_in_pages,
        cmd_ctx.command_tensor.worker_pages_per_slice - cmd_specific_ctx.offset_into_worker_slice);

    uint16_t contig_pages_advanced = 1;
    cb_reserve_back(cmd_ctx.cb_id, cmd_ctx.packet_size_in_pages);
    const uint32_t l1_write_addr_base = get_write_ptr(cmd_ctx.cb_id);
    uint32_t l1_write_addr = l1_write_addr_base;

    for (uint16_t i = 0; i < max_pages_readable; i += contig_pages_advanced) {
        const auto [noc_addr, contig_pages_] = get_noc_addr_and_contiguous_pages<TENSOR_LAYOUT,MEM_LAYOUT>(
            cmd_specific_ctx.curr_tile_id,
            cmd_specific_ctx.offset_into_worker_slice,
            cmd_ctx.command_tensor.worker_start_offset_in_slice,
            cmd_ctx.tensor_addrgen,
            cmd_ctx.command_tensor.tensor_slice_shape);

        {
            contig_pages_advanced = std::min<uint16_t>(max_pages_readable, contig_pages_);
            contig_pages_advanced = std::min<uint16_t>(cmd_ctx.packet_size_in_pages - i, contig_pages_);
            ASSERT(contig_pages_advanced > 0);
            ASSERT(contig_pages_advanced <= cmd_ctx.packet_size_in_pages);
            noc_async_read(noc_addr, l1_write_addr, cmd_ctx.page_size * contig_pages_advanced);
        }
        l1_write_addr += cmd_ctx.page_size * contig_pages_advanced;

        bool done_worker_slice = ttnn::ccl::v2::advance_worker_global_page(
            cmd_specific_ctx.curr_tile_id,  // Updated internally
            cmd_specific_ctx.offset_into_worker_slice,
            cmd_ctx.command_tensor.worker_start_offset_in_slice,
            cmd_ctx.command_tensor.worker_pages_per_slice,
            cmd_ctx.command_tensor.tensor_slice_shape,
            cmd_ctx.command_tensor.tensor_slice_offset,
            cmd_ctx.command_tensor.tensor_shape,
            contig_pages_advanced);
    }

    noc_async_read_barrier();

    cb_push_back(cmd_ctx.cb_id, cmd_ctx.packet_size_in_pages);
}
#endif

void write_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    size_t packet_header_buffer_addr,
    const ttnn::ccl::cmd::CclCommandHeader& current_cmd_header,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    const size_t payload_l1_address = l1_read_addr;

    auto pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE *>(packet_header_buffer_addr);

    pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    switch (current_cmd_header.dest_type) {
        case ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST: {
            const auto& unicast_args = current_cmd_header.get_unicast_dest_args();
            auto& fabric_conn = unicast_args.is_forward_direction ? fabric_connection.get_forward_connection()
                                                                  : fabric_connection.get_backward_connection();

            pkt_hdr->to_chip_unicast(unicast_args.distance_in_hops);

            fabric_conn.wait_for_empty_write_slot();
            fabric_conn.send_payload_without_header_non_blocking_from_address(l1_read_addr, payload_size_bytes);
            fabric_conn.send_payload_flush_blocking_from_address((uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
        } break;
        case ttnn::ccl::cmd::CclCommandDestType::CHIP_MULTICAST: {
            const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
            uint64_t dest_noc_addr = safe_get_noc_addr(static_cast<uint8_t>(dest_noc_xy.x), static_cast<uint8_t>(dest_noc_xy.y), dest_addr);
            noc_async_write(
                payload_l1_address, dest_noc_addr, payload_size_bytes);
            const auto& mcast_args = current_cmd_header.get_multicast_dest_args();
            if (fabric_connection.has_forward_connection()) {
                pkt_hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                    1, static_cast<uint8_t>(mcast_args.num_targets_forward_direction)});
                fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
                    l1_read_addr, payload_size_bytes);
                fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                    (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
            }

            if (fabric_connection.has_backward_connection()) {
                pkt_hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                    1, static_cast<uint8_t>(mcast_args.num_targets_backward_direction)});
                fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
                    l1_read_addr, payload_size_bytes);
                fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                    (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
            }
        } break;
        default: {
            DPRINT << "default\n";
            ASSERT(false);
        } break;
    }

    l1_read_addr += payload_size_bytes;
}

FORCE_INLINE void write_payload_then_advance_read_address(
    uint64_t noc0_dest_noc_addr,
    size_t packet_header_buffer_addr,
    const ttnn::ccl::cmd::CclCommandHeader& current_cmd_header,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    size_t payload_size_bytes) {
    static_assert(
        is_power_of_2(sizeof(PACKET_HEADER_TYPE)),
        "sizeof(PACKET_HEADER_TYPE) is not a power of two which violates the below assertion");

    switch (current_cmd_header.dest_type) {
        case ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST: [[fallthrough]];
        case ttnn::ccl::cmd::CclCommandDestType::CHIP_MULTICAST:
            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                packet_header_buffer_addr,
                current_cmd_header,
                fabric_connection,
                l1_read_addr,
                payload_size_bytes);
            break;

        case ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY: {
            const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
            // Convert to our local noc_index based address
            noc_async_write(
                l1_read_addr, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);
            l1_read_addr += payload_size_bytes;
        } break;
    }
}

#ifndef NO_TENSOR_MODE
// Both this and try_advance_read_tensor_to_cb are very similar - particularly with the main
// packetization loop. We should look into refactoring this to reduce code size (and duplication)
// even if it results in a mild perf hit, it's probably worth while until generality is achieved.
// At the very least, we can have templatization of the main function that can specialize
// based on command type so we can avoid the perf overhead of the branching that would otherwise
// be required.
template <tt::tt_metal::TensorMemoryLayout TENSOR_LAYOUT, tt::tt_metal::Layout MEM_LAYOUT, typename Addrgen>
void try_advance_write_tensor_from_cb(command_context_t<Addrgen>& cmd_ctx) {
    if (!cb_pages_available_at_front(cmd_ctx.cb_id, cmd_ctx.packet_size_in_pages)) {
        return;
    }
    DPRINT << "CB -> tensor: " << (uint32_t)cmd_ctx.stream_id << "\n";

    wrapped_worker_slice_read_context& cmd_specific_ctx = cmd_ctx.cmd_specific_ctx.wrapped_worker_slice_read_ctx;
    const uint16_t max_pages_writable = std::min<size_t>(
        cmd_ctx.packet_size_in_pages,
        cmd_ctx.command_tensor.worker_pages_per_slice - cmd_specific_ctx.offset_into_worker_slice);
    ASSERT(cmd_ctx.command_tensor.worker_pages_per_slice >= cmd_specific_ctx.offset_into_worker_slice);

    cb_wait_front(cmd_ctx.cb_id, cmd_ctx.packet_size_in_pages);
    size_t l1_read_addr = get_read_ptr(cmd_ctx.cb_id);

    uint16_t contig_pages_advanced = 1;
    for (uint16_t i = 0; i < max_pages_writable; i += contig_pages_advanced) {
        // This needs to be cleaned up a little bit.
        // There's a basic usability issue here in that when/if the write is sent over the fabric,
        // then the fabric expects noc x/y coordinates to be provided as noc0 coordinates.
        // However, if we're writing locally, then we need to actually write using `noc_index` based coordinates.
        // This can lead to a discrepancy, so to stay consistent, we always generate noc0 based addresses here
        // so we can reliably translate to `noc_index` based addresses writing locally, inside the write function
        auto const [noc0_dest_noc_addr, contig_pages_] =
            get_noc_addr_and_contiguous_pages_for_fabric_write<TENSOR_LAYOUT,MEM_LAYOUT>(
                cmd_specific_ctx.curr_tile_id,
            cmd_specific_ctx.offset_into_worker_slice,
            cmd_ctx.command_tensor.worker_start_offset_in_slice,
            cmd_ctx.tensor_addrgen,
            cmd_ctx.command_tensor.tensor_slice_shape);
        contig_pages_advanced = std::min<uint16_t>(contig_pages_, max_pages_writable);
        contig_pages_advanced = std::min<uint16_t>(cmd_ctx.packet_size_in_pages - i, contig_pages_);

        write_payload_then_advance_read_address(
            noc0_dest_noc_addr,
            cmd_ctx.packet_header_buffer_addr,
            cmd_ctx.current_cmd_header,
            cmd_ctx.fabric_connection,
            l1_read_addr,
            contig_pages_advanced * cmd_ctx.page_size);

        auto done_worker_slice = ttnn::ccl::v2::advance_worker_global_page(
            cmd_specific_ctx.curr_tile_id,  // Updated internally
            cmd_specific_ctx.offset_into_worker_slice,
            cmd_ctx.command_tensor.worker_start_offset_in_slice,
            cmd_ctx.command_tensor.worker_pages_per_slice,
            cmd_ctx.command_tensor.tensor_slice_shape,
            cmd_ctx.command_tensor.tensor_slice_offset,
            cmd_ctx.command_tensor.tensor_shape,
            contig_pages_advanced);
    }
    noc_async_writes_flushed();

    cb_pop_front(cmd_ctx.cb_id, cmd_ctx.packet_size_in_pages);
}
#endif

static FORCE_INLINE ttnn::ccl::cmd::noc_transfer_info get_next_noc_transfer_in_burst(arg_idx_t& arg_idx) {
    auto noc_yx_in_16bits_each = get_arg_val<uint32_t>(arg_idx + 1);
    noc_grid_index_t noc_x = static_cast<noc_grid_index_t>(noc_yx_in_16bits_each & 0xFF);
    noc_grid_index_t noc_y = static_cast<noc_grid_index_t>((noc_yx_in_16bits_each >> 16) & 0xFF);

    uint32_t noc_transfer_size_bytes = get_arg_val<uint32_t>(arg_idx + 2);
    uint32_t bank_addr_offset = get_arg_val<uint32_t>(arg_idx);
    return {safe_get_noc_addr(noc_x, noc_y, bank_addr_offset), noc_transfer_size_bytes};
}

static FORCE_INLINE size_t get_args_consumed_by_noc_transfer_info_in_burst() { return 3; }

FORCE_INLINE static ttnn::ccl::cmd::noc_transfer_info advance_to_next_noc_transaction_in_burst(
    noc_transfer_burst_context& noc_burst_ctx, arg_idx_t& arg_idx) {
    const auto noc_transfer_info = get_next_noc_transfer_in_burst(arg_idx);
    arg_idx += get_args_consumed_by_noc_transfer_info_in_burst();

    noc_burst_ctx.current_noc_transfer++;
    return noc_transfer_info;
}

static void try_advance_noc_read_burst(
    noc_transfer_burst_context& noc_burst_ctx, uint32_t cb_id, uint32_t packet_size_in_pages, arg_idx_t& arg_idx) {
    if (!cb_pages_reservable_at_back(cb_id, packet_size_in_pages)) {
        return;
    }

    auto wrptr = get_write_ptr(cb_id);
    ttnn::ccl::cmd::noc_transfer_info transfer_info;
    size_t num_transfers_in_group = get_arg_val<uint32_t>(arg_idx++);
    for (size_t i = 0; i < num_transfers_in_group; i++) {
        auto transfer_info = advance_to_next_noc_transaction_in_burst(noc_burst_ctx, arg_idx);

        // Add the offset to the base address tp resolve the full address
        uint64_t src_noc_addr = noc_burst_ctx.bank_base_address + transfer_info.noc_addr;

        noc_async_read(src_noc_addr, wrptr, transfer_info.noc_transfer_size_bytes);
        wrptr += transfer_info.noc_transfer_size_bytes;
    }
    ASSERT(noc_burst_ctx.current_noc_transfer <= noc_burst_ctx.num_transfers_total);

    noc_async_read_barrier();
    cb_push_back(cb_id, packet_size_in_pages);
}

static void try_advance_noc_write_burst(
    FabricConnectionManager& fabric_connection,
    noc_transfer_burst_context& noc_burst_ctx,
    uint32_t cb_id,
    uint32_t packet_size_in_pages,
    size_t packet_header_buffer_addr,
    const ttnn::ccl::cmd::CclCommandHeader& current_cmd_header,
    arg_idx_t& arg_idx) {
    if (!cb_pages_available_at_front(cb_id, packet_size_in_pages)) {
        return;
    }
    size_t cb_rdptr = get_read_ptr(cb_id);
    size_t num_transfers_in_group = get_arg_val<uint32_t>(arg_idx++);
    for (size_t i = 0; i < num_transfers_in_group; i++) {
        auto transfer_info = advance_to_next_noc_transaction_in_burst(noc_burst_ctx, arg_idx);

        // Add the offset to the base address tp resolve the full address
        uint64_t dest_noc_addr = noc_burst_ctx.bank_base_address + transfer_info.noc_addr;
        // Import from reference kernel
        write_payload_then_advance_read_address(
            dest_noc_addr,
            packet_header_buffer_addr,
            current_cmd_header,
            fabric_connection,
            cb_rdptr,
            transfer_info.noc_transfer_size_bytes);
    }
    noc_async_writes_flushed();

    cb_pop_front(cb_id, packet_size_in_pages);
}

template <tt::tt_metal::TensorMemoryLayout TENSOR_LAYOUT, tt::tt_metal::Layout MEM_LAYOUT, typename Addrgen>
void try_advance(command_context_t<Addrgen>& cmd_ctx) {
    switch (cmd_ctx.current_cmd_header.code) {
        case ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_EDM:  // STREAM TENSOR TO CB
#ifndef NO_TENSOR_MODE
            try_advance_read_tensor_to_cb<TENSOR_LAYOUT, MEM_LAYOUT>(cmd_ctx);
#endif
            break;
        case ttnn::ccl::cmd::CclCommandCode::STREAM_CB_TO_TENSOR:
#ifndef NO_TENSOR_MODE
            try_advance_write_tensor_from_cb<TENSOR_LAYOUT, MEM_LAYOUT>(cmd_ctx);
#endif
            break;

        case ttnn::ccl::cmd::CclCommandCode::NOC_READ_BURST:
            try_advance_noc_read_burst(
                cmd_ctx.cmd_specific_ctx.noc_transfer_burst_ctx,
                cmd_ctx.cb_id,
                cmd_ctx.packet_size_in_pages,
                cmd_ctx.arg_idx);
            break;

        case ttnn::ccl::cmd::CclCommandCode::NOC_WRITE_BURST:
            try_advance_noc_write_burst(
                cmd_ctx.fabric_connection,
                cmd_ctx.cmd_specific_ctx.noc_transfer_burst_ctx,
                cmd_ctx.cb_id,
                cmd_ctx.packet_size_in_pages,
                cmd_ctx.packet_header_buffer_addr,
                cmd_ctx.current_cmd_header,
                cmd_ctx.arg_idx);
            break;

        case ttnn::ccl::cmd::CclCommandCode::ATOMIC_INC: [[fallthrough]];
        case ttnn::ccl::cmd::CclCommandCode::RAW_INLINE_WRITE_BYTES:
            try_advance_inline_write_or_atomic_inc(cmd_ctx);
            break;
        case ttnn::ccl::cmd::CclCommandCode::WAIT_VALUE:
            // Nothing to actively do to advance - just needs to wait for completion
            break;
        default: ASSERT(false); break;
    };

    // Advance to next command index
    switch (cmd_ctx.current_cmd_header.code) {
        case ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_EDM:  // STREAM TENSOR TO CB
        case ttnn::ccl::cmd::CclCommandCode::STREAM_CB_TO_TENSOR:
            if (cmd_ctx.cmd_specific_ctx.wrapped_worker_slice_read_ctx.offset_into_worker_slice >=
                cmd_ctx.command_tensor.worker_pages_per_slice) {
                DPRINT << "t_stream cmd cmpl\n";
                cmd_ctx.complete_current_command();
            }
            break;

        case ttnn::ccl::cmd::CclCommandCode::ATOMIC_INC: [[fallthrough]];
        case ttnn::ccl::cmd::CclCommandCode::RAW_INLINE_WRITE_BYTES:
            DPRINT << "at_inc cmd cmpl\n";
            cmd_ctx.complete_current_command();
            break;
        case ttnn::ccl::cmd::CclCommandCode::WAIT_VALUE:
            // Technically we are implementating semaphore wait as WAIT_MIN. FUTURE work to make separate commands
            if (*reinterpret_cast<volatile uint32_t*>(cmd_ctx.src_addr_info.address) >=
                cmd_ctx.cmd_specific_ctx.inline_value_ctx.value) {
                DPRINT << "Completing waitval command\n";
                cmd_ctx.complete_current_command();
                invalidate_l1_cache();
            }
            break;

        case ttnn::ccl::cmd::CclCommandCode::NOC_READ_BURST: [[fallthrough]];
        case ttnn::ccl::cmd::CclCommandCode::NOC_WRITE_BURST:
            if (cmd_ctx.cmd_specific_ctx.noc_transfer_burst_ctx.current_noc_transfer ==
                cmd_ctx.cmd_specific_ctx.noc_transfer_burst_ctx.num_transfers_total) {
                DPRINT << "noc_burst cmd cmpl\n";
                cmd_ctx.complete_current_command();
            }
            break;
        default: ASSERT(false); break;
    };
}

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
#ifndef NO_TENSOR_MODE
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
#ifndef SINGLE_TENSOR
    address_t tensor_address1 = get_arg_val<address_t>(arg_idx++);
#endif
#endif
    uint8_t num_commands0 = get_arg_val<address_t>(arg_idx++);
    arg_idx_t command0_start_offset = get_arg_val<address_t>(arg_idx++);

#ifndef SINGLE_INPUT_MODE
    uint8_t num_commands1 = get_arg_val<address_t>(arg_idx++);
    arg_idx_t command1_start_offset = get_arg_val<address_t>(arg_idx++);
#endif

    // Assuming whole page transmissions (which is the only mode we support at the moment)
    // -> however, wanted to call it out here to make it clear that we need to pull this
    //    out when we start enabling other modes
    const uint16_t packet_size_in_pages = get_arg_val<uint32_t>(arg_idx++);
    uint16_t tensor0_page_size =
#ifndef NO_TENSOR_MODE
        get_arg_val<uint32_t>(arg_idx++);
#else
        0;
#endif
    uint16_t tensor1_page_size =
#if !defined(NO_TENSOR_MODE) and !defined(SINGLE_TENSOR)
        get_arg_val<uint32_t>(arg_idx++);
#else
        0;
#endif

    auto tensor0_addrgen =
#ifndef NO_TENSOR_MODE
        build_source_address_generator
            <tensor0_layout, buffer0_type, tensor0_page_layout,Tensor0ShardInfo>
            (arg_idx, tensor_address0, tensor0_page_size, cb0_id);
#else
        no_addrgen{};
#endif

#if !defined(SINGLE_INPUT_MODE)
    auto tensor1_addrgen =
#if !defined(NO_TENSOR_MODE) && !defined(SINGLE_TENSOR)
        build_source_address_generator
            <tensor1_layout, buffer1_type, tensor1_page_layout, Tensor1ShardInfo>
            (arg_idx, tensor_address1, tensor1_page_size, cb1_id);
#else
        no_addrgen{};
#endif
#endif

    // TODO: move to common
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);

    cb_reserve_back(reserved_packet_header_cb_id, num_packet_headers_storable);
    auto packet_header_buffer_addr0 = get_write_ptr(reserved_packet_header_cb_id);
    auto packet_header_buffer_addr1 =
        packet_header_buffer_addr0 + (num_packet_headers_storable >> 2) * sizeof(PACKET_HEADER_TYPE);

    auto operand_0_cmd_ctx = command_context_t(
        fabric_connection,
        tensor0_addrgen,
        num_commands0,
        command0_start_offset,
        cb0_id,
        tensor0_page_size,
        packet_size_in_pages,
        packet_header_buffer_addr0,
        0);

    // enabling either of the writes will cause the issue
    static_assert(sizeof(command_context_t<decltype(tensor0_addrgen)>) <= 120, "command_context_t is too big");
    uint8_t stream_done_mask =
#ifndef SINGLE_INPUT_MODE
        (static_cast<uint8_t>(num_commands1 == 0) << 1) |
#endif
        static_cast<uint8_t>(num_commands0 == 0);
#ifndef SINGLE_INPUT_MODE
    const uint8_t finish_value = 0x3;
    static_assert(sizeof(command_context_t<decltype(tensor1_addrgen)>) <= 120, "command_context_t is too big");
    auto operand_1_cmd_ctx = command_context_t(
        fabric_connection,
        tensor1_addrgen,
        num_commands1,
        command1_start_offset,
        cb1_id,
        tensor1_page_size,
        packet_size_in_pages,
        packet_header_buffer_addr1,
        1);
#else
    const uint8_t finish_value = 0x1;
#endif

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }
    while (stream_done_mask != finish_value) {
        if ((stream_done_mask & 0x1) == 0) {
            if (!operand_0_cmd_ctx.current_command_active()) {
                DPRINT << "get_cmd0\n";
                operand_0_cmd_ctx.fetch_next_command();
            };
            try_advance<tensor0_layout, tensor0_page_layout>(operand_0_cmd_ctx);
        }
        stream_done_mask |= static_cast<uint8_t>(operand_0_cmd_ctx.is_complete());
#ifndef SINGLE_INPUT_MODE
        if ((stream_done_mask & 0x2) == 0) {
            if (!operand_1_cmd_ctx.current_command_active()) {
                DPRINT << "get_cmd1\n";
                operand_1_cmd_ctx.fetch_next_command();
            }
            try_advance<tensor1_layout, tensor1_page_layout>(operand_1_cmd_ctx);
        }
        stream_done_mask |= (static_cast<uint8_t>(operand_1_cmd_ctx.is_complete()) << 1);
#endif
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
    DPRINT << "DONE \n";
}
