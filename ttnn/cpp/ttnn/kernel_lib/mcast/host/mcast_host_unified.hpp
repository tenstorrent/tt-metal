// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp"

namespace ttnn::kernel_lib::host {

enum class McastCoreOrder { RowMajor, ColumnMajor };
enum class McastSenderPlacement { Uniform, Staggered };

struct McastUnifiedConfig {
    tt::tt_metal::NOC noc = tt::tt_metal::NOC::NOC_0;
    bool handshake = true;
    // Copied at construction. Null means all receivers; an empty set means no
    // acknowledgments. The current sender never acknowledges itself. Kernels
    // must independently obey this participation contract. Must be null when
    // handshaking is disabled; chain forwarding requires all receivers.
    const tt::tt_metal::CoreRangeSet* handshake_cores = nullptr;
    dataflow_kernel_lib::DataReadySignal data_ready = dataflow_kernel_lib::DataReadySignal::Flag;
    std::optional<uint32_t> base_sem_id;
    std::optional<std::vector<uint32_t>> sem_ids;
    dataflow_kernel_lib::TransferMode irregular_receiver_set_mode = dataflow_kernel_lib::TransferMode::Multicast;
};

struct McastFixedSenderConfig {
    // Uniform requires an in-group index. Staggered wraps index + group number.
    uint32_t sender_index = 0;
    McastSenderPlacement placement = McastSenderPlacement::Uniform;
};
struct McastRotatingSenderConfig {
    // Defaults to each receiver group. A separate grid inherits receiver order
    // and is split evenly across receiver groups.
    std::optional<tt::tt_metal::CoreRangeSet> sender_grid = std::nullopt;
};
struct McastExplicitSenderConfig {
    std::vector<std::vector<tt::tt_metal::CoreCoord>> senders_per_group;
};
using McastSenderConfig = std::variant<McastFixedSenderConfig, McastRotatingSenderConfig, McastExplicitSenderConfig>;

// A group is a set of receiver cores sharing the same data and the same sender,
// or the same ordered set of senders when the sender rotates.
// Mcast partitions the ordered receiver cores into consecutive groups of equal
// size, all using the same protocol. Construction prepares an owned family
// snapshot; the device is not consulted afterward. Legacy
// McastFamily remains available for custom partitions or unequal receiver populations.
class Mcast {
public:
    Mcast(
        tt::tt_metal::IDevice* device,
        const McastUnifiedConfig& config,
        const tt::tt_metal::CoreRangeSet& receivers,
        uint32_t receiver_group_size,
        const McastSenderConfig& sender_config = McastFixedSenderConfig{},
        McastCoreOrder receiver_order = McastCoreOrder::RowMajor);

    void attach(
        tt::tt_metal::ProgramDescriptor& descriptor,
        std::string_view prefix,
        std::span<const std::reference_wrapper<tt::tt_metal::KernelDescriptor>> kernels) const {
        family_.attach(descriptor, prefix, kernels);
    }
    void attach(
        tt::tt_metal::experimental::ProgramSpec& spec,
        tt::tt_metal::experimental::ProgramRunArgs& args,
        std::string_view prefix,
        std::span<const tt::tt_metal::experimental::KernelSpecName> kernels,
        std::span<const tt::tt_metal::experimental::SemaphoreSpecName> adopted_semaphores = {}) const {
        family_.attach(spec, args, prefix, kernels, adopted_semaphores);
    }
    void append_semaphores(tt::tt_metal::Program& program) { family_.append_semaphores(program); }
    template <typename Args>
    void append_compile_time_args_to(Args& destination) const {
        family_.append_compile_time_args_to(destination);
    }
    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        family_.append_runtime_args_to(destination, core);
    }
    McastArgumentOffsets append_kernel_args_to(
        std::vector<uint32_t>& compile_time_args,
        tt::tt_metal::KernelDescriptor::RuntimeArgs& runtime_args,
        const tt::tt_metal::CoreRangeSet& placement) const {
        return family_.append_kernel_args_to(compile_time_args, runtime_args, placement);
    }
    const tt::tt_metal::CoreRangeSet& participating_cores() const { return family_.participating_cores(); }
    tt::tt_metal::CoreRangeSet sender_only_cores() const { return family_.sender_only_cores(); }

private:
    McastFamily family_;
};

}  // namespace ttnn::kernel_lib::host
