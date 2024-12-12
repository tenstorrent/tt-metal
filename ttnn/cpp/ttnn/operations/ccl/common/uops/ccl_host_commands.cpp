// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_host_commands.hpp"

#include <variant>
#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "tt_metal/impl/buffers/global_semaphore.hpp"

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
CclHostLowLevelWorkerCommand read_tensor_slice_to_cb(
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
        ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY,
        ttnn::ccl::cmd::LocalOnlyCommandDestArgs()};
};

CclHostLowLevelWorkerCommand local_write_cb_to_tensor_slice(
    ttnn::ccl::v2::TensorSlice const& slice, size_t cb_id) {
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
    std::variant<UnicastCommandDestArgs, MulticastCommandDestArgs> const& dest_args) {
    auto const dest_type = std::visit(
        [](auto const& arg) -> CclCommandDestType {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, UnicastCommandDestArgs>) {
                return CclCommandDestType::CHIP_UNICAST;
            } else if constexpr (std::is_same_v<T, MulticastCommandDestArgs>) {
                return CclCommandDestType::CHIP_MULTICAST;
            }
        },
        dest_args);
    auto dest_args_variant = std::visit(
        [](auto const& arg) -> ttnn::ccl::cmd::CclCommandDestArgs {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, UnicastCommandDestArgs>) {
                return ttnn::ccl::cmd::UnicastCommandDestArgs(arg);
            } else if constexpr (std::is_same_v<T, MulticastCommandDestArgs>) {
                return ttnn::ccl::cmd::MulticastCommandDestArgs(arg);
            }
        },
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
    return std::visit([](semaphore_id_t const& semaphore) -> ttnn::ccl::cmd::CclCommandAddrType {
        using T = std::decay_t<decltype(semaphore)>;
        if constexpr (std::is_same_v<T, uint32_t>) {
            return ttnn::ccl::cmd::CclCommandAddrType::SEMAPHORE_ID;
        } else if constexpr (std::is_same_v<T, GlobalSemaphore>) {
            return ttnn::ccl::cmd::CclCommandAddrType::ABSOLUTE_ADDRESS;
        } else {
            TT_THROW("ttnn::ccl::cmd::uops::get_semaphore_addr_type called with unsupported semaphore types. Currently supported types are uint32_t (semaphore ID) and GlobalSemaphore");
        }
    }, semaphore_id);
}
static ttnn::ccl::cmd::CclCommandAddrArgs get_semaphore_addr_val(semaphore_id_t const& semaphore_id) {
    return std::visit([](semaphore_id_t const& semaphore) -> ttnn::ccl::cmd::CclCommandAddrArgs {
        using T = std::decay_t<decltype(semaphore)>;
        if constexpr (std::is_same_v<T, uint32_t>) {
            return ttnn::ccl::cmd::CclCommandAddrSemaphoreId{std::get<uint32_t>(semaphore_id)};
        } else if constexpr (std::is_same_v<T, GlobalSemaphore>) {
            return ttnn::ccl::cmd::CclCommandAddrAbsoluteAddress{std::get<GlobalSemaphore>(semaphore_id).address()};
        } else {
            TT_THROW("ttnn::ccl::cmd::uops::get_semaphore_addr_type called with unsupported semaphore types. Currently supported types are uint32_t (semaphore ID) and GlobalSemaphore");
        }
    }, semaphore_id);
}

[[nodiscard]] CclHostLowLevelWorkerCommand local_semaphore_wait(semaphore_id_t const& semaphore_id, size_t value) {
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

// CclHostLowLevelWorkerCommand local_semaphore_wait(size_t semaphore_id, size_t value) {
//     return local_semaphore_wait(semaphore_id, value);
// }
// CclHostLowLevelWorkerCommand local_semaphore_wait(GlobalSemaphore const& semaphore_id, size_t value) {
//     return CclHostLowLevelWorkerCommand(
//         CclCommandCode::WAIT_VALUE,
//         ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::CclCommandWaitValue{value}),
//         ttnn::ccl::cmd::CclCommandAddrType::ABSOLUTE_ADDRESS,
//         ttnn::ccl::cmd::CclCommandAddrSemaphoreId{semaphore_id.address()},
//         ttnn::ccl::cmd::CclCommandAddrType::NONE,
//         ttnn::ccl::cmd::CclCommandAddrNone(),
//         ttnn::ccl::cmd::CclCommandCoreDescriptorType::LOCAL,
//         ttnn::ccl::cmd::CclCommandCoreDescriptorTypeAddrgen(),
//         ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY,
//         ttnn::ccl::cmd::LocalOnlyCommandDestArgs());
// }

// CclHostLowLevelWorkerCommand local_core_semaphore_inc(size_t semaphore_id, size_t value) {
CclHostLowLevelWorkerCommand local_core_semaphore_inc(semaphore_id_t const& semaphore_id, size_t value) {
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::ATOMIC_INC,
        ttnn::ccl::cmd::CclCommandArgs(ttnn::ccl::cmd::CclCommandAtomicInc{value}),
        // src
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),
        // dest
        get_semaphore_addr_type(semaphore_id), // ttnn::ccl::cmd::CclCommandAddrType::SEMAPHORE_ID,
        get_semaphore_addr_val(semaphore_id), // ttnn::ccl::cmd::CclCommandAddrSemaphoreId{semaphore_id},
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
        get_semaphore_addr_type(semaphore_id), // ttnn::ccl::cmd::CclCommandAddrType::SEMAPHORE_ID,
        get_semaphore_addr_val(semaphore_id), // ttnn::ccl::cmd::CclCommandAddrSemaphoreId{semaphore_id},
        ttnn::ccl::cmd::CclCommandCoreDescriptorType::NOC_XY,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY{dest_noc0_x, dest_noc0_y},
        ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY,
        ttnn::ccl::cmd::LocalOnlyCommandDestArgs());
}

[[nodiscard]] CclHostLowLevelWorkerCommand fabric_unicast_semaphore_inc_mcast(
    // CclCommandAddrSemaphoreId const& semaphore_dest_args,
    semaphore_id_t const& semaphore_dest_args,
    CclCommandAtomicInc const& increment_args,
    CclCommandCoreDescriptorTypeMcast const& mcast_spec,
    UnicastCommandDestArgs const& unicast_args) {
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::ATOMIC_INC,
        increment_args,
        // src
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),
        // dest
        get_semaphore_addr_type(semaphore_dest_args), // ttnn::ccl::cmd::CclCommandAddrType::SEMAPHORE_ID,
        get_semaphore_addr_val(semaphore_dest_args), // semaphore_dest_args,
        ttnn::ccl::cmd::CclCommandCoreDescriptorType::RECTANGLE,
        mcast_spec,
        ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST,
        ttnn::ccl::cmd::UnicastCommandDestArgs(unicast_args));
}

[[nodiscard]] CclHostLowLevelWorkerCommand local_chip_semaphore_inc_mcast(
    // CclCommandAddrSemaphoreId const& semaphore_dest_args,
    semaphore_id_t const& semaphore_dest_args,
    CclCommandAtomicInc const& increment_args,
    CclCommandCoreDescriptorTypeMcast const& mcast_spec) {
    return CclHostLowLevelWorkerCommand(
        CclCommandCode::ATOMIC_INC,
        increment_args,
        // src
        ttnn::ccl::cmd::CclCommandAddrType::NONE,
        ttnn::ccl::cmd::CclCommandAddrNone(),
        // dest
        get_semaphore_addr_type(semaphore_dest_args), // ttnn::ccl::cmd::CclCommandAddrType::SEMAPHORE_ID,
        get_semaphore_addr_val(semaphore_dest_args), // semaphore_dest_args,
        ttnn::ccl::cmd::CclCommandCoreDescriptorType::RECTANGLE,
        mcast_spec,
        ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY,
        ttnn::ccl::cmd::LocalOnlyCommandDestArgs());
}

CclHostLowLevelWorkerCommand local_chip_noc_absolute_address_semaphore_inc(
    size_t dest_noc0_x,
    size_t dest_noc0_y,
    size_t bank_address,
    size_t value) {
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
    CclCommandAddrSemaphoreId const& semaphore_dest_args,
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
        ttnn::ccl::cmd::CclCommandAddrType::SEMAPHORE_ID,
        semaphore_dest_args,

        ttnn::ccl::cmd::CclCommandCoreDescriptorType::NOC_XY,
        ttnn::ccl::cmd::CclCommandCoreDescriptorTypeNocXY{dest_noc0_x, dest_noc0_y},

        ttnn::ccl::cmd::CclCommandDestType::CHIP_MULTICAST,
        ttnn::ccl::cmd::MulticastCommandDestArgs(multicast_args));
}

CclHostLowLevelWorkerCommand fabric_unicast_semaphore_inc(
    CclCommandAddrSemaphoreId const& semaphore_dest_args,
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
        ttnn::ccl::cmd::CclCommandAddrType::SEMAPHORE_ID,
        semaphore_dest_args,

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

}  // namespace uops

}  // namespace ttnn::ccl::cmd
