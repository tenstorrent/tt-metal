// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This should ideally be merged with `ccl_send_reader` when we are able to support compile time args
//       that don't require macros to function

#include "dataflow_api.h"
#include <cstddef>
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"

#include "ttnn/operations/ccl/common/kernels/command_processor.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/io_descriptors.hpp"
#include <cstdint>
#include <utility>

using namespace tt::tt_metal;

using arg_idx_t = uint16_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

// #define TRACE_ENABLED

#ifdef TRACE_ENABLED
enum TraceType {
    NONE = 0,
    HEADER_ATOMIC_INC = 1,
    HEADER_INLINE_WRITE = 2,
    NOC_ATOMIC_INC = 3,
    NOC_INLINE_DW_WRITE = 4,
    RESERVE_CB = 5,
    NOC_ASYNC_READ = 6,
    PUSH_CB = 7,
    WAIT_CB = 8,
    NOC_UNICAST_WRITE = 9,
    POP_CB = 10,
    TRY_ADVANCE_WAIT_VALUE = 11,
};

enum TRACE_DEST_TYPE {
    DEST_TYPE_NONE = 0,
    SEND_HEADER_UNICAST_FORWARD,
    SEND_HEADER_UNICAST_BACKWARD,
    SEND_HEADER_MCAST_FORWARD,
    SEND_HEADER_MCAST_BACKWARD,
    SEND_HEADER_MCAST_BOTH,
    SEND_PAYLOAD_HEADER_MCAST_FORWARD,
    SEND_PAYLOAD_HEADER_MCAST_BACKWARD,
    SEND_PAYLOAD_HEADER_MCAST_BOTH,
};

struct TraceInfo {
    uint32_t pkt_hdr_addr;
    uint32_t pkt_hdr_backward_addr;
    uint32_t trace_id;
    TraceType trace_type;
    TRACE_DEST_TYPE trace_dest_type;
    uint32_t cb_id;
    uint32_t packet_size_in_pages;
    uint64_t noc_addr;
    size_t value;
    uint32_t l1_addr;
    uint32_t size;
};

struct TraceCtx {
    uint32_t trace_idx;
    volatile TraceInfo* trace_hdr;
    uint32_t base_pkt_addr;
    uint32_t base_pkt_backward_addr;
};
#endif

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
#ifdef TRACE_ENABLED
void try_advance_inline_write_or_atomic_inc(command_context_t<Addrgen>& cmd_ctx, volatile TraceCtx* reader_trace_ctx, volatile TraceCtx* writer_trace_ctx) {
#else
void try_advance_inline_write_or_atomic_inc(command_context_t<Addrgen>& cmd_ctx) {
#endif
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
            // DPRINT << "to_noc_unicast_atomic_inc trace_idx=" << (uint32_t)writer_trace_ctx->trace_idx << "\n";
#ifdef TRACE_ENABLED
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_id = writer_trace_ctx->trace_idx;
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_type = TraceType::HEADER_ATOMIC_INC;
            
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_addr = writer_trace_ctx->base_pkt_addr + writer_trace_ctx->trace_idx * sizeof(PACKET_HEADER_TYPE);
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_backward_addr = writer_trace_ctx->base_pkt_backward_addr + writer_trace_ctx->trace_idx * sizeof(PACKET_HEADER_TYPE);
            auto *p = reinterpret_cast<PACKET_HEADER_TYPE *>(writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_addr);
            auto *p_b = reinterpret_cast<PACKET_HEADER_TYPE *>(writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_backward_addr);
            p->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{dest_noc_addr_for_pkt, static_cast<uint16_t>(value), 32});
            p_b->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{dest_noc_addr_for_pkt, static_cast<uint16_t>(value), 32});
#endif
            pkt_hdr->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{dest_noc_addr_for_pkt, static_cast<uint16_t>(value), 32});
        } else {
            // DPRINT << "to_noc_unicast_inline_write trace_idx=" << (uint32_t)writer_trace_ctx->trace_idx << "\n";
#ifdef TRACE_ENABLED
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_id = writer_trace_ctx->trace_idx;
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_type = TraceType::HEADER_INLINE_WRITE;

            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_addr = writer_trace_ctx->base_pkt_addr + writer_trace_ctx->trace_idx * sizeof(PACKET_HEADER_TYPE);
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_backward_addr = writer_trace_ctx->base_pkt_backward_addr + writer_trace_ctx->trace_idx * sizeof(PACKET_HEADER_TYPE);
            auto *p = reinterpret_cast<PACKET_HEADER_TYPE *>(writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_addr);
            auto *p_b = reinterpret_cast<PACKET_HEADER_TYPE *>(writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_backward_addr);
            p->to_noc_unicast_inline_write(
                tt::tt_fabric::NocUnicastInlineWriteCommandHeader{dest_noc_addr_for_pkt, static_cast<uint16_t>(value)});
            p_b->to_noc_unicast_inline_write(
                tt::tt_fabric::NocUnicastInlineWriteCommandHeader{dest_noc_addr_for_pkt, static_cast<uint16_t>(value)});
#endif
            pkt_hdr->to_noc_unicast_inline_write(
                tt::tt_fabric::NocUnicastInlineWriteCommandHeader{dest_noc_addr_for_pkt, static_cast<uint16_t>(value)});
        }

        switch (cmd_ctx.current_cmd_header.dest_type) {
            case ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST: {
                // DPRINT << "Sending remote atomic unicast trace_idx=" << (uint32_t)writer_trace_ctx->trace_idx << "\n";
                fabric_set_unicast_route<false>(pkt_hdr, cmd_ctx.current_cmd_header.get_unicast_dest_args().distance_in_hops);

                auto& fabric_connection = cmd_ctx.current_cmd_header.get_unicast_dest_args().is_forward_direction
                                              ? cmd_ctx.fabric_connection.get_forward_connection()
                                              : cmd_ctx.fabric_connection.get_backward_connection();
                fabric_connection.wait_for_empty_write_slot();
                fabric_connection.send_payload_flush_blocking_from_address(
                    cmd_ctx.packet_header_buffer_addr, sizeof(PACKET_HEADER_TYPE));
#ifdef TRACE_ENABLED
                auto *p = reinterpret_cast<PACKET_HEADER_TYPE *>(writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_addr);
                auto *p_b = reinterpret_cast<PACKET_HEADER_TYPE *>(writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_backward_addr);

                fabric_set_unicast_route<false>(p, cmd_ctx.current_cmd_header.get_unicast_dest_args().distance_in_hops);
                fabric_set_unicast_route<false>(p_b, cmd_ctx.current_cmd_header.get_unicast_dest_args().distance_in_hops);
                writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_dest_type = cmd_ctx.current_cmd_header.get_unicast_dest_args().is_forward_direction ? TRACE_DEST_TYPE::SEND_HEADER_UNICAST_FORWARD : TRACE_DEST_TYPE::SEND_HEADER_UNICAST_BACKWARD;
#endif
            } break;
            case ttnn::ccl::cmd::CclCommandDestType::CHIP_MULTICAST: {
                // DPRINT << "Sending remote atomic multicast trace_idx=" << (uint32_t)writer_trace_ctx->trace_idx << "\n";
                write_local = true;
                const auto& mcast_args = cmd_ctx.current_cmd_header.get_multicast_dest_args();

#ifdef TRACE_ENABLED
                if (cmd_ctx.fabric_connection.has_forward_connection() && cmd_ctx.fabric_connection.has_backward_connection()) {
                    writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_dest_type = TRACE_DEST_TYPE::SEND_HEADER_MCAST_BOTH;
                } else if (cmd_ctx.fabric_connection.has_forward_connection()) {
                    writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_dest_type = TRACE_DEST_TYPE::SEND_HEADER_MCAST_FORWARD;
                } else if (cmd_ctx.fabric_connection.has_backward_connection()) {
                    writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_dest_type = TRACE_DEST_TYPE::SEND_HEADER_MCAST_BACKWARD;
                }
#endif

                if (cmd_ctx.fabric_connection.has_forward_connection()) {
                    pkt_hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                        1, static_cast<uint8_t>(mcast_args.num_targets_forward_direction)});
#ifdef TRACE_ENABLED
                    auto *p = reinterpret_cast<PACKET_HEADER_TYPE *>(writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_addr);
                    p->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                        1, static_cast<uint8_t>(mcast_args.num_targets_forward_direction)});
#endif
                    {
                        cmd_ctx.fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                    }
                    cmd_ctx.fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                        cmd_ctx.packet_header_buffer_addr, sizeof(PACKET_HEADER_TYPE));
                }

                // Write the mcast packet (backward)
                if (cmd_ctx.fabric_connection.has_backward_connection()) {
                    pkt_hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                        1, static_cast<uint8_t>(mcast_args.num_targets_backward_direction)});
#ifdef TRACE_ENABLED
                    auto *p_b = reinterpret_cast<PACKET_HEADER_TYPE *>(writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_backward_addr);
                    p_b->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                        1, static_cast<uint8_t>(mcast_args.num_targets_backward_direction)});
#endif
                    {
                        cmd_ctx.fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                    }
                    cmd_ctx.fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
                        cmd_ctx.packet_header_buffer_addr, sizeof(PACKET_HEADER_TYPE));
                }

            } break;

            default: {
                ASSERT(false);
            } break;
        };
#ifdef TRACE_ENABLED
        writer_trace_ctx->trace_idx++;
#endif
    }

    if (write_local) {
        uint64_t dest_noc_addr = get_noc_addr(dest_noc0_x, dest_noc0_y, dest_bank_addr);
        if (cmd_ctx.current_cmd_header.code == ttnn::ccl::cmd::CclCommandCode::ATOMIC_INC) {
            // DPRINT << "noc_semaphore_inc trace_idx=" << (uint32_t)writer_trace_ctx->trace_idx << "\n";
            noc_semaphore_inc(dest_noc_addr, value);
#ifdef TRACE_ENABLED
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_id = writer_trace_ctx->trace_idx;
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_type = TraceType::NOC_ATOMIC_INC;
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].noc_addr = dest_noc_addr;
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].value = value;
            writer_trace_ctx->trace_idx++;
#endif
        } else if (cmd_ctx.current_cmd_header.code == ttnn::ccl::cmd::CclCommandCode::RAW_INLINE_WRITE_BYTES) {
            // DPRINT << "noc_inline_dw_write trace_idx=" << (uint32_t)writer_trace_ctx->trace_idx << "\n";
            noc_inline_dw_write(dest_noc_addr, value);
#ifdef TRACE_ENABLED
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_id = writer_trace_ctx->trace_idx;
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_type = TraceType::NOC_INLINE_DW_WRITE;
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].noc_addr = dest_noc_addr;
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].value = value;
            writer_trace_ctx->trace_idx++;
#endif
        }
    }
}

template <tt::tt_metal::TensorMemoryLayout TENSOR_LAYOUT, tt::tt_metal::Layout MEM_LAYOUT, typename AddrGen>
FORCE_INLINE std::pair<uint64_t, size_t> get_noc_addr_and_contiguous_pages(
    uint32_t curr_page_idx,
    const uint32_t offset_into_worker_slice,
    const ttnn::ccl::Shape4D<uint32_t>& offset_worker_slice,
    const AddrGen& address_generator,
    const ttnn::ccl::Shape4D<uint32_t>& tensor_slice_shape,
    uint8_t noc_id = noc_index) {
        constexpr uint32_t offset = 0;
        std::pair<uint64_t, size_t> ret_val =
            get_contiguous_noc_addr(curr_page_idx,address_generator,offset,noc_id);
        uint32_t flattened_offset_worker_slice = ttnn::ccl::v2::flattened_index(tensor_slice_shape, offset_worker_slice);
        uint32_t contig_until_edge_of_tensor_slice = tensor_slice_shape.x - ((flattened_offset_worker_slice + offset_into_worker_slice) % tensor_slice_shape.x);
        size_t contig_pages = std::min<int32_t>(ret_val.second, contig_until_edge_of_tensor_slice);
        return {ret_val.first, contig_pages};
}

template <tt::tt_metal::TensorMemoryLayout TENSOR_LAYOUT, tt::tt_metal::Layout MEM_LAYOUT, typename AddrGen>
FORCE_INLINE std::pair<uint64_t, uint16_t> get_noc_addr_and_contiguous_pages_for_fabric_write(
    uint32_t curr_page_idx,
    const uint32_t offset_into_worker_slice,
    const ttnn::ccl::Shape4D<uint32_t>& offset_worker_slice,
    const AddrGen& address_generator,
    const ttnn::ccl::Shape4D<uint32_t>& tensor_slice_shape) {
    return get_noc_addr_and_contiguous_pages<TENSOR_LAYOUT, MEM_LAYOUT, AddrGen>(
        curr_page_idx, offset_into_worker_slice, offset_worker_slice, address_generator, tensor_slice_shape, 0);
}

#ifndef NO_TENSOR_MODE
template <tt::tt_metal::TensorMemoryLayout TENSOR_LAYOUT, tt::tt_metal::Layout MEM_LAYOUT, typename Addrgen>
#ifdef TRACE_ENABLED
void try_advance_read_tensor_to_cb(command_context_t<Addrgen>& cmd_ctx, volatile TraceCtx* reader_trace_ctx, volatile TraceCtx* writer_trace_ctx) {
#else
void try_advance_read_tensor_to_cb(command_context_t<Addrgen>& cmd_ctx) {
#endif
    if (!cb_pages_reservable_at_back(cmd_ctx.cb_id, cmd_ctx.packet_size_in_pages)) {
        // DPRINT << "Not enough space in CB to read tensor to CB\n";
        return;
    }

    wrapped_worker_slice_read_context& cmd_specific_ctx = cmd_ctx.cmd_specific_ctx.wrapped_worker_slice_read_ctx;
    const uint16_t max_pages_readable = std::min<size_t>(
        cmd_ctx.packet_size_in_pages,
        cmd_ctx.command_tensor.worker_pages_per_slice - cmd_specific_ctx.offset_into_worker_slice);

    uint16_t contig_pages_advanced = 1;
    cb_reserve_back(cmd_ctx.cb_id, cmd_ctx.packet_size_in_pages);
#ifdef TRACE_ENABLED
    reader_trace_ctx->trace_hdr[reader_trace_ctx->trace_idx].trace_id = reader_trace_ctx->trace_idx;
    reader_trace_ctx->trace_hdr[reader_trace_ctx->trace_idx].trace_type = TraceType::RESERVE_CB;
    reader_trace_ctx->trace_hdr[reader_trace_ctx->trace_idx].cb_id = cmd_ctx.cb_id;
    reader_trace_ctx->trace_hdr[reader_trace_ctx->trace_idx].packet_size_in_pages = cmd_ctx.packet_size_in_pages;
    reader_trace_ctx->trace_idx++;
#endif

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
            // DPRINT << "noc_async_read trace_idx=" << (uint32_t)reader_trace_ctx->trace_idx <<  "l1_write_addr :" << (uint32_t)l1_write_addr << "\n";
            noc_async_read(noc_addr, l1_write_addr, cmd_ctx.page_size * contig_pages_advanced);
#ifdef TRACE_ENABLED
            reader_trace_ctx->trace_hdr[reader_trace_ctx->trace_idx].trace_id = reader_trace_ctx->trace_idx;
            reader_trace_ctx->trace_hdr[reader_trace_ctx->trace_idx].trace_type = TraceType::NOC_ASYNC_READ;
            reader_trace_ctx->trace_hdr[reader_trace_ctx->trace_idx].noc_addr = noc_addr;
            reader_trace_ctx->trace_hdr[reader_trace_ctx->trace_idx].l1_addr = l1_write_addr;
            reader_trace_ctx->trace_hdr[reader_trace_ctx->trace_idx].size = cmd_ctx.page_size * contig_pages_advanced;
            reader_trace_ctx->trace_idx++;
#endif
            // DPRINT << "noc_async_read trace_idx= end" << (uint32_t)reader_trace_ctx->trace_idx << "\n";
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
#ifdef TRACE_ENABLED
    reader_trace_ctx->trace_hdr[reader_trace_ctx->trace_idx].trace_id = reader_trace_ctx->trace_idx;
    reader_trace_ctx->trace_hdr[reader_trace_ctx->trace_idx].trace_type = TraceType::PUSH_CB;
    reader_trace_ctx->trace_hdr[reader_trace_ctx->trace_idx].cb_id = cmd_ctx.cb_id;
    reader_trace_ctx->trace_hdr[reader_trace_ctx->trace_idx].packet_size_in_pages = cmd_ctx.packet_size_in_pages;
    reader_trace_ctx->trace_idx++;
#endif
    noc_async_read_barrier();
    cb_push_back(cmd_ctx.cb_id, cmd_ctx.packet_size_in_pages);
}
#endif
namespace command_processor {

#ifdef TRACE_ENABLED
void write_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    size_t packet_header_buffer_addr,
    const ttnn::ccl::cmd::CclCommandHeader& current_cmd_header,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes,
    volatile TraceCtx* reader_trace_ctx, volatile TraceCtx* writer_trace_ctx) {
#else
void write_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    size_t packet_header_buffer_addr,
    const ttnn::ccl::cmd::CclCommandHeader& current_cmd_header,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
#endif
    const size_t payload_l1_address = l1_read_addr;

    auto pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE *>(packet_header_buffer_addr);

    pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

#ifdef TRACE_ENABLED
    writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_id = writer_trace_ctx->trace_idx;
    writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_type = TraceType::NOC_UNICAST_WRITE;
    writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_addr = writer_trace_ctx->base_pkt_addr + writer_trace_ctx->trace_idx * sizeof(PACKET_HEADER_TYPE);
    writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_backward_addr = writer_trace_ctx->base_pkt_backward_addr + writer_trace_ctx->trace_idx * sizeof(PACKET_HEADER_TYPE);
    auto *p = reinterpret_cast<volatile PACKET_HEADER_TYPE *>(writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_addr);
    auto *p_b = reinterpret_cast<volatile PACKET_HEADER_TYPE *>(writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_backward_addr);
    p->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
    p_b->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
#endif

    switch (current_cmd_header.dest_type) {
        case ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST: {
            const auto& unicast_args = current_cmd_header.get_unicast_dest_args();
            auto& fabric_conn = unicast_args.is_forward_direction ? fabric_connection.get_forward_connection()
                                                                  : fabric_connection.get_backward_connection();

            fabric_set_unicast_route<false>(pkt_hdr, unicast_args.distance_in_hops);

            fabric_conn.wait_for_empty_write_slot();
            fabric_conn.send_payload_without_header_non_blocking_from_address(l1_read_addr, payload_size_bytes);
            fabric_conn.send_payload_flush_blocking_from_address((uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
        } break;
        case ttnn::ccl::cmd::CclCommandDestType::CHIP_MULTICAST: {
            const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
            uint64_t dest_noc_addr = safe_get_noc_addr(static_cast<uint8_t>(dest_noc_xy.x), static_cast<uint8_t>(dest_noc_xy.y), dest_addr);
            // DPRINT << "noc_async_write trace_idx=" << (uint32_t)writer_trace_ctx->trace_idx << "\n";
            noc_async_write(
                payload_l1_address, dest_noc_addr, payload_size_bytes);
#ifdef TRACE_ENABLED
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].noc_addr = dest_noc_addr;
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].l1_addr = payload_l1_address;
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].size = payload_size_bytes;
#endif
            const auto& mcast_args = current_cmd_header.get_multicast_dest_args();

            // DPRINT << "payload_l1_address:" << (uint32_t)payload_l1_address << " l1_read_addr:" << (uint32_t)l1_read_addr << "\n";
#ifdef TRACE_ENABLED
            if (fabric_connection.has_forward_connection() && fabric_connection.has_backward_connection()) {
                writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_dest_type = TRACE_DEST_TYPE::SEND_PAYLOAD_HEADER_MCAST_BOTH;
            } else if (fabric_connection.has_forward_connection()) {
                writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_dest_type = TRACE_DEST_TYPE::SEND_PAYLOAD_HEADER_MCAST_FORWARD;
            } else if (fabric_connection.has_backward_connection()) {
                writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_dest_type = TRACE_DEST_TYPE::SEND_PAYLOAD_HEADER_MCAST_BACKWARD;
            }
#endif
            if (fabric_connection.has_forward_connection()) {
// #ifndef TRACE_ENABLED
//                 DeviceZoneScopedN("SendPayloadFwd");
// #endif
                pkt_hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                    1, static_cast<uint8_t>(mcast_args.num_targets_forward_direction)});
#ifdef TRACE_ENABLED
                auto *p = reinterpret_cast<volatile PACKET_HEADER_TYPE *>(writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_addr);
                p->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                    1, static_cast<uint8_t>(mcast_args.num_targets_forward_direction)});
#endif
                {
// #ifndef TRACE_ENABLED
                    DeviceZoneScopedN("WaitSlot");
// #endif
                    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                }
                fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
                    l1_read_addr, payload_size_bytes);
                fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                    (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
            }

            if (fabric_connection.has_backward_connection()) {
// #ifndef TRACE_ENABLED
//                 DeviceZoneScopedN("SendPayloadBwd");
// #endif
                pkt_hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                    1, static_cast<uint8_t>(mcast_args.num_targets_backward_direction)});
#ifdef TRACE_ENABLED
                auto *p_b = reinterpret_cast<volatile PACKET_HEADER_TYPE *>(writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].pkt_hdr_backward_addr);
                p_b->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                    1, static_cast<uint8_t>(mcast_args.num_targets_backward_direction)});
#endif
                {
// #ifndef TRACE_ENABLED
                    DeviceZoneScopedN("WaitSlotBwd");
// #endif
                    fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                }
                fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
                    l1_read_addr, payload_size_bytes);
                fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                    (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
            }
        } break;
        default: {
            ASSERT(false);
        } break;
    }
#ifdef TRACE_ENABLED
    writer_trace_ctx->trace_idx++;
#endif

    l1_read_addr += payload_size_bytes;
}
}

#ifdef TRACE_ENABLED
FORCE_INLINE void write_payload_then_advance_read_address(
    uint64_t noc0_dest_noc_addr,
    size_t packet_header_buffer_addr,
    const ttnn::ccl::cmd::CclCommandHeader& current_cmd_header,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    size_t payload_size_bytes,
    volatile TraceCtx* reader_trace_ctx, volatile TraceCtx* writer_trace_ctx) {
#else
FORCE_INLINE void write_payload_then_advance_read_address(
    uint64_t noc0_dest_noc_addr,
    size_t packet_header_buffer_addr,
    const ttnn::ccl::cmd::CclCommandHeader& current_cmd_header,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    size_t payload_size_bytes) {
#endif
    static_assert(
        is_power_of_2(sizeof(PACKET_HEADER_TYPE)),
        "sizeof(PACKET_HEADER_TYPE) is not a power of two which violates the below assertion");

    switch (current_cmd_header.dest_type) {
        case ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST: [[fallthrough]];
        case ttnn::ccl::cmd::CclCommandDestType::CHIP_MULTICAST:
#ifdef TRACE_ENABLED
            command_processor::write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                packet_header_buffer_addr,
                current_cmd_header,
                fabric_connection,
                l1_read_addr,
                payload_size_bytes, reader_trace_ctx, writer_trace_ctx);
#else
            command_processor::write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                packet_header_buffer_addr,
                current_cmd_header,
                fabric_connection,
                l1_read_addr,
                payload_size_bytes);
#endif
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
#ifdef TRACE_ENABLED
void try_advance_write_tensor_from_cb(command_context_t<Addrgen>& cmd_ctx, volatile TraceCtx* reader_trace_ctx, volatile TraceCtx* writer_trace_ctx) {
#else
void try_advance_write_tensor_from_cb(command_context_t<Addrgen>& cmd_ctx) {
#endif
    if (!cb_pages_available_at_front(cmd_ctx.cb_id, cmd_ctx.packet_size_in_pages)) {
        return;
    }

    wrapped_worker_slice_read_context& cmd_specific_ctx = cmd_ctx.cmd_specific_ctx.wrapped_worker_slice_read_ctx;
    const uint16_t max_pages_writable = std::min<size_t>(
        cmd_ctx.packet_size_in_pages,
        cmd_ctx.command_tensor.worker_pages_per_slice - cmd_specific_ctx.offset_into_worker_slice);
    ASSERT(cmd_ctx.command_tensor.worker_pages_per_slice >= cmd_specific_ctx.offset_into_worker_slice);

    {
        // DPRINT << "cb_wait_front trace_idx=" << (uint32_t)writer_trace_ctx->trace_idx << "\n";
        cb_wait_front(cmd_ctx.cb_id, cmd_ctx.packet_size_in_pages);
#ifdef TRACE_ENABLED
        writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_id = writer_trace_ctx->trace_idx;
        writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_type = TraceType::WAIT_CB;
        writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].cb_id = cmd_ctx.cb_id;
        writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].packet_size_in_pages = cmd_ctx.packet_size_in_pages;
        writer_trace_ctx->trace_idx++;
#endif
    }
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
        DPRINT << "contig_pages_advanced:" << (uint32_t)contig_pages_advanced << "\n";

#ifdef TRACE_ENABLED
        write_payload_then_advance_read_address(
            noc0_dest_noc_addr,
            cmd_ctx.packet_header_buffer_addr,
            cmd_ctx.current_cmd_header,
            cmd_ctx.fabric_connection,
            l1_read_addr,
            contig_pages_advanced * cmd_ctx.page_size,
            reader_trace_ctx,
            writer_trace_ctx);
#else
        write_payload_then_advance_read_address(
            noc0_dest_noc_addr,
            cmd_ctx.packet_header_buffer_addr,
            cmd_ctx.current_cmd_header,
            cmd_ctx.fabric_connection,
            l1_read_addr,
            contig_pages_advanced * cmd_ctx.page_size);
#endif

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
#ifdef TRACE_ENABLED
    writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_id = writer_trace_ctx->trace_idx;
    writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_type = TraceType::POP_CB;
    writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].cb_id = cmd_ctx.cb_id;
    writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].packet_size_in_pages = cmd_ctx.packet_size_in_pages;
    writer_trace_ctx->trace_idx++;
#endif
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

#ifdef TRACE_ENABLED
static void try_advance_noc_write_burst(
    FabricConnectionManager& fabric_connection,
    noc_transfer_burst_context& noc_burst_ctx,
    uint32_t cb_id,
    uint32_t packet_size_in_pages,
    size_t packet_header_buffer_addr,
    const ttnn::ccl::cmd::CclCommandHeader& current_cmd_header,
    arg_idx_t& arg_idx,
    volatile TraceCtx* reader_trace_ctx, volatile TraceCtx* writer_trace_ctx) {
#else
static void try_advance_noc_write_burst(
    FabricConnectionManager& fabric_connection,
    noc_transfer_burst_context& noc_burst_ctx,
    uint32_t cb_id,
    uint32_t packet_size_in_pages,
    size_t packet_header_buffer_addr,
    const ttnn::ccl::cmd::CclCommandHeader& current_cmd_header,
    arg_idx_t& arg_idx) {
#endif
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
#ifdef TRACE_ENABLED
        write_payload_then_advance_read_address(
            dest_noc_addr,
            packet_header_buffer_addr,
            current_cmd_header,
            fabric_connection,
            cb_rdptr,
            transfer_info.noc_transfer_size_bytes,
            reader_trace_ctx,
            writer_trace_ctx);
#else
        write_payload_then_advance_read_address(
            dest_noc_addr,
            packet_header_buffer_addr,
            current_cmd_header,
            fabric_connection,
            cb_rdptr,
            transfer_info.noc_transfer_size_bytes);
#endif
    }
    noc_async_writes_flushed();

    cb_pop_front(cb_id, packet_size_in_pages);
}

template <tt::tt_metal::TensorMemoryLayout TENSOR_LAYOUT, tt::tt_metal::Layout MEM_LAYOUT, typename Addrgen>
#ifdef TRACE_ENABLED
void try_advance(command_context_t<Addrgen>& cmd_ctx, volatile TraceCtx* reader_trace_ctx, volatile TraceCtx* writer_trace_ctx) {
#else
void try_advance(command_context_t<Addrgen>& cmd_ctx) {
#endif
// #ifndef TRACE_ENABLED
//     DeviceZoneScopedN("try_advance")
// #endif
    switch (cmd_ctx.current_cmd_header.code) {
        case ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_EDM:  // STREAM TENSOR TO CB
#ifndef NO_TENSOR_MODE
#ifdef TRACE_ENABLED
            try_advance_read_tensor_to_cb<TENSOR_LAYOUT, MEM_LAYOUT>(cmd_ctx, reader_trace_ctx, writer_trace_ctx);
#else
            try_advance_read_tensor_to_cb<TENSOR_LAYOUT, MEM_LAYOUT>(cmd_ctx);
#endif
#endif
            break;
        case ttnn::ccl::cmd::CclCommandCode::STREAM_CB_TO_TENSOR:
#ifndef NO_TENSOR_MODE
#ifdef TRACE_ENABLED
            try_advance_write_tensor_from_cb<TENSOR_LAYOUT, MEM_LAYOUT>(cmd_ctx, reader_trace_ctx, writer_trace_ctx);
#else
            try_advance_write_tensor_from_cb<TENSOR_LAYOUT, MEM_LAYOUT>(cmd_ctx);
#endif
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
#ifdef TRACE_ENABLED
            try_advance_noc_write_burst(
                cmd_ctx.fabric_connection,
                cmd_ctx.cmd_specific_ctx.noc_transfer_burst_ctx,
                cmd_ctx.cb_id,
                cmd_ctx.packet_size_in_pages,
                cmd_ctx.packet_header_buffer_addr,
                cmd_ctx.current_cmd_header,
                cmd_ctx.arg_idx,
                reader_trace_ctx,
                writer_trace_ctx);
#else
            try_advance_noc_write_burst(
                cmd_ctx.fabric_connection,
                cmd_ctx.cmd_specific_ctx.noc_transfer_burst_ctx,
                cmd_ctx.cb_id,
                cmd_ctx.packet_size_in_pages,
                cmd_ctx.packet_header_buffer_addr,
                cmd_ctx.current_cmd_header,
                cmd_ctx.arg_idx);
#endif
            break;

        case ttnn::ccl::cmd::CclCommandCode::ATOMIC_INC: [[fallthrough]];
        case ttnn::ccl::cmd::CclCommandCode::RAW_INLINE_WRITE_BYTES:
#ifdef TRACE_ENABLED
            try_advance_inline_write_or_atomic_inc(cmd_ctx, reader_trace_ctx, writer_trace_ctx);
#else
            try_advance_inline_write_or_atomic_inc(cmd_ctx);
#endif
            break;
        case ttnn::ccl::cmd::CclCommandCode::WAIT_VALUE:
            // DPRINT << "try_advance WAIT_VALUE arg_idx=" << (uint32_t)cmd_ctx.arg_idx << "\n";
#ifdef TRACE_ENABLED
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_id = writer_trace_ctx->trace_idx;
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].trace_type = TraceType::TRY_ADVANCE_WAIT_VALUE;
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].l1_addr = cmd_ctx.src_addr_info.address;
            writer_trace_ctx->trace_hdr[writer_trace_ctx->trace_idx].value = cmd_ctx.cmd_specific_ctx.inline_value_ctx.value;
#endif
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
                cmd_ctx.complete_current_command();
            }
            break;

        case ttnn::ccl::cmd::CclCommandCode::ATOMIC_INC: [[fallthrough]];
        case ttnn::ccl::cmd::CclCommandCode::RAW_INLINE_WRITE_BYTES:
            cmd_ctx.complete_current_command();
            break;
        case ttnn::ccl::cmd::CclCommandCode::WAIT_VALUE:
            // Technically we are implementating semaphore wait as WAIT_MIN. FUTURE work to make separate commands
            if (*reinterpret_cast<volatile uint32_t*>(cmd_ctx.src_addr_info.address) >=
                cmd_ctx.cmd_specific_ctx.inline_value_ctx.value) {
                // DeviceZoneScopedN("WAIT_VALUE");
                cmd_ctx.complete_current_command();
                invalidate_l1_cache();
#ifdef TRACE_ENABLED
                writer_trace_ctx->trace_idx++;
#endif
            }
            break;

        case ttnn::ccl::cmd::CclCommandCode::NOC_READ_BURST: [[fallthrough]];
        case ttnn::ccl::cmd::CclCommandCode::NOC_WRITE_BURST:
            if (cmd_ctx.cmd_specific_ctx.noc_transfer_burst_ctx.current_noc_transfer ==
                cmd_ctx.cmd_specific_ctx.noc_transfer_burst_ctx.num_transfers_total) {
                // DPRINT << "noc_burst cmd cmpl\n";
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
    const uint32_t trace_cb_idx1 = get_arg_val<address_t>(arg_idx++);
    const uint32_t trace_cb_idx2 = get_arg_val<address_t>(arg_idx++);
    const uint32_t trace_cb_idx3 = get_arg_val<address_t>(arg_idx++);
    const uint32_t trace_cb_idx4 = get_arg_val<address_t>(arg_idx++);
    const uint32_t trace_sync_sem_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t trace_core_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t trace_core_y = get_arg_val<uint32_t>(arg_idx++);

#ifdef TRACE_ENABLED
    const uint32_t trace_semaphore_addr = get_semaphore(trace_sync_sem_id);
    const uint64_t trace_semaphore_noc_addr = get_noc_addr(trace_core_x, trace_core_y, trace_semaphore_addr);

    noc_semaphore_set(reinterpret_cast<volatile uint32_t*>(trace_semaphore_addr), 0);

    cb_reserve_back(trace_cb_idx1, 1);
    cb_reserve_back(trace_cb_idx2, 1);
    cb_reserve_back(trace_cb_idx3, 1);
    cb_reserve_back(trace_cb_idx4, 1);
    auto l1_reader_address = get_write_ptr(trace_cb_idx1);
    auto l1_writer_address = get_write_ptr(trace_cb_idx2);
    auto l1_pkt_address = get_write_ptr(trace_cb_idx3);
    auto l1_pkt_bwd_address = get_write_ptr(trace_cb_idx4);
    
    volatile TraceCtx* reader_trace_ctx = reinterpret_cast<volatile TraceCtx*>(l1_reader_address);
    reader_trace_ctx->trace_idx = 0;
    reader_trace_ctx->trace_hdr = reinterpret_cast<volatile TraceInfo*>(
        l1_reader_address + sizeof(TraceCtx));

    volatile TraceCtx* writer_trace_ctx = reinterpret_cast<volatile TraceCtx*>(l1_writer_address);
    writer_trace_ctx->trace_idx = 0;
    writer_trace_ctx->trace_hdr = reinterpret_cast<volatile TraceInfo*>(
        l1_writer_address + sizeof(TraceCtx));
    writer_trace_ctx->base_pkt_addr = l1_pkt_address;
    writer_trace_ctx->base_pkt_backward_addr = l1_pkt_bwd_address;

    // DPRINT << "sizeof(TraceCtx) : " << (uint32_t)sizeof(TraceCtx) << "\n";
    // DPRINT << "sizeof(TraceInfo) : " << (uint32_t)sizeof(TraceInfo) << "\n";
    // DPRINT << "l1_reader_address: " << (uint32_t)l1_reader_address << "\n";
    // DPRINT << "l1_writer_address: " << (uint32_t)l1_writer_address << "\n";
    // DPRINT << "l1_pkt_address: " << (uint32_t)l1_pkt_address << "\n";
    // DPRINT << "l1_pkt_bwd_address: " << (uint32_t)l1_pkt_bwd_address << "\n";

#endif

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
        fabric_connection.open<true>();
    }

    {
        DeviceZoneScopedN("main_loop_no_trce")
        while (stream_done_mask != finish_value) {
            if ((stream_done_mask & 0x1) == 0) {
                if (!operand_0_cmd_ctx.current_command_active()) {
                    operand_0_cmd_ctx.fetch_next_command();
                };
#ifdef TRACE_ENABLED
                try_advance<tensor0_layout, tensor0_page_layout>(operand_0_cmd_ctx, reader_trace_ctx, writer_trace_ctx);
#else
                try_advance<tensor0_layout, tensor0_page_layout>(operand_0_cmd_ctx);
#endif
            }
            stream_done_mask |= static_cast<uint8_t>(operand_0_cmd_ctx.is_complete());
    #ifndef SINGLE_INPUT_MODE
            if ((stream_done_mask & 0x2) == 0) {
                if (!operand_1_cmd_ctx.current_command_active()) {
                    operand_1_cmd_ctx.fetch_next_command();
                }
#ifdef TRACE_ENABLED    
                try_advance<tensor1_layout, tensor1_page_layout>(operand_1_cmd_ctx, reader_trace_ctx, writer_trace_ctx);
#else
                try_advance<tensor1_layout, tensor1_page_layout>(operand_1_cmd_ctx);
#endif
            }
            stream_done_mask |= (static_cast<uint8_t>(operand_1_cmd_ctx.is_complete()) << 1);
    #endif
        }
        noc_async_write_barrier();
    }

#ifndef TRACE_ENABLED
        if (fabric_connection.is_logically_connected()) {
            fabric_connection.close();
        }
#endif

#ifdef TRACE_ENABLED
    {
#ifdef COMPILE_FOR_NCRISC
        {
            // DeviceZoneScopedN("wait_writer");
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(trace_semaphore_addr), 1);
        }

        if (fabric_connection.is_logically_connected()) {
            fabric_connection.open();
        }

        {
            DeviceZoneScopedN("main_loop_trace");
            uint32_t trace_size = sizeof(TraceCtx) + sizeof(TraceInfo) * reader_trace_ctx->trace_idx;
            uint32_t l1_addr = 0;
            // DPRINT << "READER trace_idx : " << (uint32_t)writer_trace_ctx->trace_idx << "trace_size : " << trace_size << "\n";
            for (uint32_t i = 0; i < reader_trace_ctx->trace_idx; i++) {
                // DeviceZoneScopedN("reader_loop");
                switch(reader_trace_ctx->trace_hdr[i].trace_type) {
                    case TraceType::NONE:
                    case TraceType::HEADER_ATOMIC_INC:
                    case TraceType::HEADER_INLINE_WRITE:
                    case TraceType::NOC_ATOMIC_INC:
                    case TraceType::NOC_INLINE_DW_WRITE:
                        break;
                    case TraceType::RESERVE_CB: {
                        while (!cb_pages_reservable_at_back(reader_trace_ctx->trace_hdr[i].cb_id, reader_trace_ctx->trace_hdr[i].packet_size_in_pages)) {
                        }
                        cb_reserve_back(reader_trace_ctx->trace_hdr[i].cb_id, reader_trace_ctx->trace_hdr[i].packet_size_in_pages);
                    } break;
                    case TraceType::NOC_ASYNC_READ: {
                        noc_async_read(reader_trace_ctx->trace_hdr[i].noc_addr, reader_trace_ctx->trace_hdr[i].l1_addr, reader_trace_ctx->trace_hdr[i].size);
                    } break;
                    case TraceType::PUSH_CB: {
                        noc_async_read_barrier();
                        cb_push_back(reader_trace_ctx->trace_hdr[i].cb_id, reader_trace_ctx->trace_hdr[i].packet_size_in_pages);
                    } break;
                    case TraceType::WAIT_CB:
                    case TraceType::NOC_UNICAST_WRITE:
                    case TraceType::POP_CB:
                    case TraceType::TRY_ADVANCE_WAIT_VALUE:
                        break;
                    default: break;
                }
            }
        }
#endif

#ifdef COMPILE_FOR_BRISC
        noc_semaphore_inc(trace_semaphore_noc_addr, 1);

        {
            DeviceZoneScopedN("main_loop_trace");
            uint32_t trace_size = sizeof(TraceCtx) + sizeof(TraceInfo) * writer_trace_ctx->trace_idx;
            uint32_t pkt_hdr_addr = 0;
            uint32_t pkt_hdr_backward_addr = 0;
            auto& fc = fabric_connection.get_forward_connection();
            auto& bc = fabric_connection.get_backward_connection();
            size_t l1_read_addr;

            auto* hdr = writer_trace_ctx->trace_hdr;
            const uint32_t n = writer_trace_ctx->trace_idx;

            // DPRINT << "WRITER trace_idx : " << (uint32_t)writer_trace_ctx->trace_idx << "trace_size : " << trace_size << "\n";
            for (uint32_t i = 0; i < n; ++i) {
                // DeviceZoneScopedN("writer_loop");
                const auto& h = hdr[i];

               if (h.trace_type == TraceType::HEADER_ATOMIC_INC ||
                        h.trace_type == TraceType::HEADER_INLINE_WRITE) {
                    pkt_hdr_addr = h.pkt_hdr_addr;
                    pkt_hdr_backward_addr = h.pkt_hdr_backward_addr;

                    auto send_header = [&](bool fwd, bool bwd) {
                        if (fwd) {
                            {
                                fc.wait_for_empty_write_slot();
                            }
                            fc.send_payload_flush_blocking_from_address(pkt_hdr_addr, sizeof(PACKET_HEADER_TYPE));
                        }
                        if (bwd) {
                            {   
                                bc.wait_for_empty_write_slot();
                            }
                            bc.send_payload_flush_blocking_from_address(pkt_hdr_backward_addr, sizeof(PACKET_HEADER_TYPE));
                        }
                    };

                    if (h.trace_dest_type == TRACE_DEST_TYPE::SEND_HEADER_UNICAST_FORWARD ||
                        h.trace_dest_type == TRACE_DEST_TYPE::SEND_HEADER_MCAST_FORWARD) {
                        send_header(true, false);
                    } else if (h.trace_dest_type == TRACE_DEST_TYPE::SEND_HEADER_UNICAST_BACKWARD ||
                            h.trace_dest_type == TRACE_DEST_TYPE::SEND_HEADER_MCAST_BACKWARD) {
                        send_header(false, true);
                    } else if (h.trace_dest_type == TRACE_DEST_TYPE::SEND_HEADER_MCAST_BOTH) {
                        send_header(true, true);
                    }
                }
                else if (h.trace_type == TraceType::NOC_ATOMIC_INC) {
                    noc_semaphore_inc(h.noc_addr, h.value);
                }
                else if (h.trace_type == TraceType::NOC_INLINE_DW_WRITE) {
                    noc_inline_dw_write(h.noc_addr, h.value);
                }
                else if (h.trace_type == TraceType::WAIT_CB) {
                    while (!cb_pages_available_at_front(h.cb_id, h.packet_size_in_pages)) {
                    }
                    cb_wait_front(h.cb_id, h.packet_size_in_pages);
                }
                else if (h.trace_type == TraceType::NOC_UNICAST_WRITE) {
                    pkt_hdr_addr = h.pkt_hdr_addr;
                    pkt_hdr_backward_addr = h.pkt_hdr_backward_addr;

                    auto send_payload_header = [&](bool fwd, bool bwd) {
                        noc_async_write(h.l1_addr, h.noc_addr, h.size);
                        if (fwd) {
                            // DeviceZoneScopedN("SendPayloadFwd");
                            {
                                // DeviceZoneScopedN("WaitSlot");
                                fc.wait_for_empty_write_slot();
                            }
                            fc.send_payload_without_header_non_blocking_from_address(h.l1_addr, h.size);
                        }
                        if (bwd) {
                            // DeviceZoneScopedN("SendPayloadBwd");
                            {
                                // DeviceZoneScopedN("WaitSlot");
                                bc.wait_for_empty_write_slot();
                            }
                            bc.send_payload_without_header_non_blocking_from_address(h.l1_addr, h.size);
                        }

                        if (fwd) {
                            fc.send_payload_flush_blocking_from_address(pkt_hdr_addr, sizeof(PACKET_HEADER_TYPE));
                        }
                        if (bwd) {
                            bc.send_payload_flush_blocking_from_address(pkt_hdr_backward_addr, sizeof(PACKET_HEADER_TYPE));
                        }
                    };

                    if (h.trace_dest_type == TRACE_DEST_TYPE::SEND_PAYLOAD_HEADER_MCAST_FORWARD) {
                        send_payload_header(true, false);
                    } else if (h.trace_dest_type == TRACE_DEST_TYPE::SEND_PAYLOAD_HEADER_MCAST_BACKWARD) {
                        send_payload_header(false, true);
                    } else if (h.trace_dest_type == TRACE_DEST_TYPE::SEND_PAYLOAD_HEADER_MCAST_BOTH) {
                        send_payload_header(true, true);
                    }
                }
                else if (h.trace_type == TraceType::POP_CB) {
                    noc_async_writes_flushed();
                    cb_pop_front(h.cb_id, h.packet_size_in_pages);
                }
                else if (h.trace_type == TraceType::TRY_ADVANCE_WAIT_VALUE) {
                    while (*reinterpret_cast<volatile uint32_t*>(h.l1_addr) < h.value) {
                    }
                }
            }
            noc_async_write_barrier();
        }

        if (fabric_connection.is_logically_connected()) {
            fabric_connection.close();
        }
#endif
    }
#endif
}