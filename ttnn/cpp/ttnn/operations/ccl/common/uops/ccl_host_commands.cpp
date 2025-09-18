// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/common/uops/ccl_host_commands.hpp"

#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt_stl/overloaded.hpp>

#include <variant>
namespace ttnn::ccl::cmd {

// This file defines commands that are resolved on a per worker level. This is the lowest level of
// command description (Intermediate Representation if you will) before being lowered directly to
// Ccl Command Process KernelCommands

namespace uops {

CclHostLowLevelWorkerCommand read_tensor_slice_to_cb_for_eventual_fabric_write(
    ttnn::ccl::v2::TensorSlice const& slice, size_t cb_id) {
    return CclHostLowLevelWorkerCommand{
        CclCommandCode::STREAM_TENSOR_TO_CB,
        slice,
        // At the moment, we don't support switching tensors from within a command stream
        // so we set none because we assume the command stream is fixed/assigned to a given tensor
        // based on order:
        // - Command stream 0: tensor 0
        // - Command stream 1: tensor 1
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),
        ttnn::ccl::cmd::CclCommandAddrType::CIRCULAR_BUFFER_ID,
        ttnn::ccl::cmd::CclCommandAddrCircularBufferId{cb_id},
        ttnn::ccl::cmd::CclCommandCoreDescriptorType::ADDRGEN,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeAddrgen(),
        ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST,
        // Hack to add packet header padding when doing reads
        ttnn::ccl::cmd::UnicastCommandDestArgs(0, true)};
};
CclHostLowLevelWorkerCommand read_tensor_slice_to_cb(ttnn::ccl::v2::TensorSlice const& slice, size_t cb_id) {
    return CclHostLowLevelWorkerCommand{
        CclCommandCode::STREAM_TENSOR_TO_CB,
        slice,
        // At the moment, we don't support switching tensors from within a command stream
        // so we set none because we assume the command stream is fixed/assigned to a given tensor
        // based on order:
        // - Command stream 0: tensor 0
        // - Command stream 1: tensor 1
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),
        ttnn::ccl::cmd::CclCommandAddrType::CIRCULAR_BUFFER_ID,
        ttnn::ccl::cmd::CclCommandAddrCircularBufferId{cb_id},
        ttnn::ccl::cmd::CclCommandCoreDescriptorType::ADDRGEN,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeAddrgen(),
        ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY,
        ttnn::ccl::cmd::LocalOnlyCommandDestArgs()};
};

CclHostLowLevelWorkerCommand local_write_cb_to_tensor_slice(ttnn::ccl::v2::TensorSlice const& slice, size_t cb_id) {
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::STREAM_CB_TO_TENSOR,
        ttnn::ccl::cmd::CclCommandArgs(slice),
        ttnn::ccl::cmd::CclCommandAddrType::CIRCULAR_BUFFER_ID,
        ttnn::ccl::cmd::CclCommandAddrCircularBufferId{cb_id},
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),
        ttnn::ccl::cmd::CclCommandCoreDescriptorType::ADDRGEN,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeAddrgen(),
        ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY,
        ttnn::ccl::cmd::LocalOnlyCommandDestArgs());
}

CclHostLowLevelWorkerCommand fabric_write_cb_to_tensor_slice(
    ttnn::ccl::v2::TensorSlice const& slice,
    size_t cb_id,
    std::variant<ttnn::ccl::cmd::UnicastCommandDestArgs, ttnn::ccl::cmd::MulticastCommandDestArgs> const& dest_args) {
    auto const dest_type = std::visit(
        tt::stl::overloaded{
            [](ttnn::ccl::cmd::UnicastCommandDestArgs const&) { return CclCommandDestType::CHIP_UNICAST; },
            [](ttnn::ccl::cmd::MulticastCommandDestArgs const&) { return CclCommandDestType::CHIP_MULTICAST; },
            [](auto&&) -> void {
                TT_THROW(
                    "ttnn::ccl::cmd::uops::fabric_write_cb_to_tensor_slice called with unsupported fabric dest_args "
                    "types. "
                    "Currently supported types are UnicastCommandDestArgs and MulticastCommandDestArgs");
            }},
        dest_args);
    auto dest_args_variant = std::visit(
        tt::stl::overloaded{
            [](ttnn::ccl::cmd::UnicastCommandDestArgs const& arg) -> ttnn::ccl::cmd::CclCommandDestArgs {
                return ttnn::ccl::cmd::UnicastCommandDestArgs(arg);
            },
            [](ttnn::ccl::cmd::MulticastCommandDestArgs const& arg) -> ttnn::ccl::cmd::CclCommandDestArgs {
                return ttnn::ccl::cmd::MulticastCommandDestArgs(arg);
            },
            [](auto&&) -> void {
                TT_THROW(
                    "ttnn::ccl::cmd::uops::fabric_write_cb_to_tensor_slice called with unsupported fabric dest_args "
                    "types. "
                    "Currently supported types are UnicastCommandDestArgs and MulticastCommandDestArgs");
            }},
        dest_args);

    return CclHostLowLevelWorkerCommand(
        CclCommandCode::STREAM_CB_TO_TENSOR,
        ttnn::ccl::cmd::CclCommandStreamTensorSlice(slice),
        // src
        ttnn::ccl::cmd::CclCommandAddrType::CIRCULAR_BUFFER_ID,
        ttnn::ccl::cmd::CclCommandAddrCircularBufferId{cb_id},

        // dest
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),

        ttnn::ccl::cmd::CclCommandCoreDescriptorType::ADDRGEN,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeAddrgen(),

        dest_type,
        dest_args_variant);
}

static ttnn::ccl::cmd::CclCommandAddrType get_semaphore_addr_type(semaphore_id_t const& semaphore_id) {
    return std::visit(
        tt::stl::overloaded{
            [](uint32_t) { return ttnn::ccl::cmd::CclCommandAddrType::SEMAPHORE_ID; },
            [](tt::tt_metal::GlobalSemaphore const*) { return ttnn::ccl::cmd::CclCommandAddrType::ABSOLUTE_ADDRESS; },
            [](auto&&) -> void {
                TT_THROW(
                    "ttnn::ccl::cmd::uops::get_semaphore_addr_type called with unsupported semaphore types. "
                    "Currently supported types are uint32_t (semaphore ID) and GlobalSemaphore");
            }},
        semaphore_id);
}
static ttnn::ccl::cmd::CclCommandAddrArgs get_semaphore_addr_val(semaphore_id_t const& semaphore_id) {
    using ttnn::ccl::cmd::CclCommandAddrArgs;
    return std::visit(
        tt::stl::overloaded{
            [](uint32_t id) -> CclCommandAddrArgs { return ttnn::ccl::cmd::CclCommandAddrSemaphoreId{id}; },
            [](tt::tt_metal::GlobalSemaphore const* semaphore) -> CclCommandAddrArgs {
                TT_FATAL(semaphore != nullptr, "Internal error: GlobalSemaphore pointer is null in call to get_semaphore_addr_val");
                return ttnn::ccl::cmd::CclCommandAddrAbsoluteAddress{semaphore->address()};
            },
            [](auto&&) -> void {
                TT_THROW(
                    "ttnn::ccl::cmd::uops::get_semaphore_addr_val called with unsupported semaphore types. "
                    "Currently supported types are uint32_t (semaphore ID) and GlobalSemaphore");
            }

        },
        semaphore_id);
}

CclHostLowLevelWorkerCommand local_semaphore_wait(semaphore_id_t const& semaphore_id, size_t value) {
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::WAIT_VALUE,
        ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::CclCommandWaitValue{value}),
        get_semaphore_addr_type(semaphore_id),
        get_semaphore_addr_val(semaphore_id),
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),
        ttnn::ccl::cmd::CclCommandCoreDescriptorType::LOCAL,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeAddrgen(),
        ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY,
        ttnn::ccl::cmd::LocalOnlyCommandDestArgs());
}

CclHostLowLevelWorkerCommand local_core_semaphore_set(semaphore_id_t const& semaphore_id, size_t value) {
    TT_FATAL(
        value < std::numeric_limits<uint32_t>::max(),
        "When invoking: local_core_inline_write. Raw inline writes currently are limited to values no larger than {} "
        "due to a command encoding limitation. Support for larger values is not yet added",
        std::numeric_limits<uint32_t>::max());
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::RAW_INLINE_WRITE_BYTES,
        ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::CclCommandInlineReadWrite{value}),
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone{},
        get_semaphore_addr_type(semaphore_id),
        get_semaphore_addr_val(semaphore_id),
        ttnn::ccl::cmd::CclCommandCoreDescriptorType::LOCAL,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeLocal(),
        ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY,
        ttnn::ccl::cmd::LocalOnlyCommandDestArgs());
}

CclHostLowLevelWorkerCommand local_core_semaphore_inc(semaphore_id_t const& semaphore_id, size_t value) {
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::ATOMIC_INC,
        ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::CclCommandAtomicInc{value}),
        // src
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),
        // dest
        get_semaphore_addr_type(semaphore_id),
        get_semaphore_addr_val(semaphore_id),
        ttnn::ccl::cmd::CclCommandCoreDescriptorType::LOCAL,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeLocal(),
        ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY,
        ttnn::ccl::cmd::LocalOnlyCommandDestArgs());
}

CclHostLowLevelWorkerCommand local_chip_noc_semaphore_inc(
    size_t dest_noc0_x,
    size_t dest_noc0_y,
    semaphore_id_t const& semaphore_id,
    // size_t semaphore_id,
    size_t value) {
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::ATOMIC_INC,
        ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::CclCommandAtomicInc{value}),
        // src
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),
        // dest
        get_semaphore_addr_type(semaphore_id),
        get_semaphore_addr_val(semaphore_id),
        ttnn::ccl::cmd::CclCommandCoreDescriptorType::NOC_XY,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY{dest_noc0_x, dest_noc0_y},
        ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY,
        ttnn::ccl::cmd::LocalOnlyCommandDestArgs());
}

static std::pair<CclCommandCoreDescriptorType, CclCommandCoreDescriptorArgs> optimize_mcast_core_desc_args(
    CclCommandCoreDescriptorTypeMcast const& noc_mcast_args) {
    bool is_really_a_unicast = noc_mcast_args.noc0_end_x == noc_mcast_args.noc0_start_x &&
                               noc_mcast_args.noc0_end_y == noc_mcast_args.noc0_start_y;
    auto core_desc_type =
        is_really_a_unicast ? CclCommandCoreDescriptorType::NOC_XY : CclCommandCoreDescriptorType::RECTANGLE;
    CclCommandCoreDescriptorArgs core_desc_args = is_really_a_unicast
                                                      ? CclCommandCoreDescriptorArgs{CclCommandCoreDescriptorTypeNocXY{
                                                            noc_mcast_args.noc0_start_x, noc_mcast_args.noc0_start_y}}
                                                      : CclCommandCoreDescriptorArgs{noc_mcast_args};
    return {core_desc_type, core_desc_args};
}

[[nodiscard]] CclHostLowLevelWorkerCommand fabric_unicast_semaphore_inc_mcast(
    semaphore_id_t const& semaphore_dest_args,
    CclCommandAtomicInc const& increment_args,
    CclCommandCoreDescriptorTypeMcast const& mcast_spec,
    UnicastCommandDestArgs const& unicast_args) {
    auto const [core_desc_type, core_desc_args] = optimize_mcast_core_desc_args(mcast_spec);
    TT_FATAL(
        core_desc_type != CclCommandCoreDescriptorType::RECTANGLE,
        "semaphore inc commands don't support noc multicast yet");
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::ATOMIC_INC,
        increment_args,
        // src
        CclCommandAddrType::NONE,
        CclCommandAddrNone(),
        // dest
        get_semaphore_addr_type(semaphore_dest_args),
        get_semaphore_addr_val(semaphore_dest_args),
        core_desc_type,
        core_desc_args,
        CclCommandDestType::CHIP_UNICAST,
        UnicastCommandDestArgs(unicast_args));
}

[[nodiscard]] CclHostLowLevelWorkerCommand local_chip_semaphore_inc_mcast(
    // CclCommandAddrSemaphoreId const& semaphore_dest_args,
    semaphore_id_t const& semaphore_dest_args,
    CclCommandAtomicInc const& increment_args,
    CclCommandCoreDescriptorTypeMcast const& mcast_spec) {
    auto const [core_desc_type, core_desc_args] = optimize_mcast_core_desc_args(mcast_spec);
    TT_FATAL(
        core_desc_type != CclCommandCoreDescriptorType::RECTANGLE,
        "semaphore inc commands don't support noc multicast yet");
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::ATOMIC_INC,
        increment_args,
        // src
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),
        // dest
        get_semaphore_addr_type(semaphore_dest_args),
        get_semaphore_addr_val(semaphore_dest_args),
        core_desc_type,
        core_desc_args,
        ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY,
        ttnn::ccl::cmd::LocalOnlyCommandDestArgs());
}

CclHostLowLevelWorkerCommand local_chip_noc_absolute_address_semaphore_inc(
    size_t dest_noc0_x, size_t dest_noc0_y, size_t bank_address, size_t value) {
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::ATOMIC_INC,
        ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::CclCommandAtomicInc{value}),

        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),

        ttnn::ccl::cmd::CclCommandAddrType::ABSOLUTE_ADDRESS,
        ttnn::ccl::cmd::CclCommandAddrAbsoluteAddress{bank_address},

        ttnn::ccl::cmd::CclCommandCoreDescriptorType::NOC_XY,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY{dest_noc0_x, dest_noc0_y},

        ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY,
        ttnn::ccl::cmd::LocalOnlyCommandDestArgs());
}

CclHostLowLevelWorkerCommand fabric_multicast_semaphore_inc(
    semaphore_id_t const& semaphore_dest_args,
    CclCommandAtomicInc const& increment_args,
    size_t dest_noc0_x,
    size_t dest_noc0_y,
    MulticastCommandDestArgs const& multicast_args) {
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::ATOMIC_INC,
        ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::CclCommandAtomicInc{increment_args}),

        // src
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),

        // dest
        get_semaphore_addr_type(semaphore_dest_args),
        get_semaphore_addr_val(semaphore_dest_args),

        ttnn::ccl::cmd::CclCommandCoreDescriptorType::NOC_XY,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY{dest_noc0_x, dest_noc0_y},

        ttnn::ccl::cmd::CclCommandDestType::CHIP_MULTICAST,
        ttnn::ccl::cmd::MulticastCommandDestArgs(multicast_args));
}

CclHostLowLevelWorkerCommand fabric_unicast_semaphore_inc(
    // CclCommandAddrSemaphoreId const& semaphore_dest_args,
    semaphore_id_t const& semaphore_dest_args,
    CclCommandAtomicInc const& increment_args,
    size_t dest_noc0_x,
    size_t dest_noc0_y,
    UnicastCommandDestArgs const& unicast_args) {
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::ATOMIC_INC,
        ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::CclCommandAtomicInc{increment_args}),

        // src
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),

        // dest
        get_semaphore_addr_type(semaphore_dest_args),
        get_semaphore_addr_val(semaphore_dest_args),

        ttnn::ccl::cmd::CclCommandCoreDescriptorType::NOC_XY,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY{dest_noc0_x, dest_noc0_y},

        ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST,
        ttnn::ccl::cmd::UnicastCommandDestArgs(unicast_args));
}

CclHostLowLevelWorkerCommand fabric_unicast_absolute_address_semaphore_inc(
    CclCommandAddrAbsoluteAddress const& address_dest_args,
    CclCommandAtomicInc const& increment_args,
    size_t dest_noc0_x,
    size_t dest_noc0_y,
    UnicastCommandDestArgs const& unicast_args) {
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::ATOMIC_INC,
        ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::CclCommandAtomicInc{increment_args}),

        // src
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),

        // dest
        ttnn::ccl::cmd::CclCommandAddrType::ABSOLUTE_ADDRESS,
        address_dest_args,

        ttnn::ccl::cmd::CclCommandCoreDescriptorType::NOC_XY,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY{dest_noc0_x, dest_noc0_y},

        ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST,
        ttnn::ccl::cmd::UnicastCommandDestArgs(unicast_args));
}


// Noc Read/Write commands
// Densely packs as many transfers as possible into a single packet
static std::vector<HostNocTransferBurstGrouping> densely_pack_noc_transfers(tt::stl::Span<noc_transfer_info> const& transfer_infos, size_t cb_size_bytes) {
    std::vector<HostNocTransferBurstGrouping> transfer_burst_groupings;

    size_t group_size_bytes = 0;
    transfer_burst_groupings.push_back({});
    for (size_t i = 0; i < transfer_infos.size(); i++) {
        group_size_bytes += transfer_infos[i].noc_transfer_size_bytes;
        bool create_new_group = group_size_bytes >= cb_size_bytes;
        if (create_new_group) {
            transfer_burst_groupings.push_back({});
            group_size_bytes = 0;
        }

        auto &group = transfer_burst_groupings.back();
        bool is_32B_aligned = (group_size_bytes & 0x1F) == 0;
        if (!is_32B_aligned) {
            group_size_bytes += 0x20 - (group_size_bytes & 0x1F);
        }

        group.num_transfers_per_packet++;
        group.transfer_infos.push_back(transfer_infos[i]);
    }

    return transfer_burst_groupings;
}

CclHostLowLevelWorkerCommand local_noc_read_burst_to_cb(
    CclCommandAddrAbsoluteAddress const& bank_base_address,
    tt::stl::Span<noc_transfer_info> const& transfer_infos,
    size_t cb_size_bytes,
    size_t cb_id
) {
    auto transfer_burst_groupings = densely_pack_noc_transfers(transfer_infos, cb_size_bytes);

    return CclHostLowLevelWorkerCommand(
        CclCommandCode::NOC_READ_BURST,
        ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::HostCclCommandNocTransferBurst{bank_base_address.absolute_address, transfer_infos.size(), transfer_burst_groupings}),
        ttnn::ccl::cmd::CclCommandAddrType::ABSOLUTE_ADDRESS,
        ttnn::ccl::cmd::CclCommandAddrAbsoluteAddress{bank_base_address},
        ttnn::ccl::cmd::CclCommandAddrType::CIRCULAR_BUFFER_ID,
        ttnn::ccl::cmd::CclCommandAddrCircularBufferId{cb_id}
    );
}

CclHostLowLevelWorkerCommand local_noc_write_burst_from_cb(
    CclCommandAddrAbsoluteAddress const& bank_base_address,
    tt::stl::Span<noc_transfer_info> const& transfer_infos,
    size_t cb_size_bytes,
    size_t cb_id
) {
    auto transfer_burst_groupings = densely_pack_noc_transfers(transfer_infos, cb_size_bytes);

    return CclHostLowLevelWorkerCommand(
        CclCommandCode::NOC_WRITE_BURST,
        ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::HostCclCommandNocTransferBurst{bank_base_address.absolute_address, transfer_infos.size(), transfer_burst_groupings}),
        ttnn::ccl::cmd::CclCommandAddrType::CIRCULAR_BUFFER_ID,
        ttnn::ccl::cmd::CclCommandAddrCircularBufferId{cb_id},
        ttnn::ccl::cmd::CclCommandAddrType::ABSOLUTE_ADDRESS,
        ttnn::ccl::cmd::CclCommandAddrAbsoluteAddress{bank_base_address}
    );
}

[[nodiscard]] CclHostLowLevelWorkerCommand fabric_unicast_noc_write_burst_from_cb(
    CclCommandAddrAbsoluteAddress const& bank_base_address,
    tt::stl::Span<noc_transfer_info> const& transfer_infos,
    size_t cb_size_bytes,
    size_t cb_id,
    UnicastCommandDestArgs const& unicast_args
) {
    auto transfer_burst_groupings = densely_pack_noc_transfers(transfer_infos, cb_size_bytes);

    return CclHostLowLevelWorkerCommand(
        CclCommandCode::NOC_WRITE_BURST,
        ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::HostCclCommandNocTransferBurst{bank_base_address.absolute_address, transfer_infos.size(), transfer_burst_groupings}),
        ttnn::ccl::cmd::CclCommandAddrType::CIRCULAR_BUFFER_ID,
        ttnn::ccl::cmd::CclCommandAddrCircularBufferId{cb_id},
        ttnn::ccl::cmd::CclCommandAddrType::ABSOLUTE_ADDRESS,
        ttnn::ccl::cmd::CclCommandAddrAbsoluteAddress{bank_base_address},
        ttnn::ccl::cmd::CclCommandCoreDescriptorType::NONE,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNone(),
        ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST,
        ttnn::ccl::cmd::UnicastCommandDestArgs(unicast_args)
    );
}

CclHostLowLevelWorkerCommand fabric_multicast_noc_write_burst_from_cb(
    CclCommandAddrAbsoluteAddress const& bank_base_address,
    tt::stl::Span<noc_transfer_info> const& transfer_infos,
    size_t cb_size_bytes,
    size_t cb_id,
    MulticastCommandDestArgs const& multicast_args
) {
    auto transfer_burst_groupings = densely_pack_noc_transfers(transfer_infos, cb_size_bytes);

    return CclHostLowLevelWorkerCommand(
        CclCommandCode::NOC_WRITE_BURST,
        ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::HostCclCommandNocTransferBurst{bank_base_address.absolute_address, transfer_infos.size(), transfer_burst_groupings}),
        ttnn::ccl::cmd::CclCommandAddrType::CIRCULAR_BUFFER_ID,
        ttnn::ccl::cmd::CclCommandAddrCircularBufferId{cb_id},
        ttnn::ccl::cmd::CclCommandAddrType::ABSOLUTE_ADDRESS,
        ttnn::ccl::cmd::CclCommandAddrAbsoluteAddress{bank_base_address},
        ttnn::ccl::cmd::CclCommandCoreDescriptorType::NONE,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNone(),
        ttnn::ccl::cmd::CclCommandDestType::CHIP_MULTICAST,
        ttnn::ccl::cmd::MulticastCommandDestArgs(multicast_args)
    );
}

}  // namespace uops

}  // namespace ttnn::ccl::cmd
