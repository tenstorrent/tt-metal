// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_fabric_push_program_factory.hpp"

#include <algorithm>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t kStage = 32;   // pages staged in L1 a NoC read barrier
constexpr uint32_t kSlot = 2048;  // L1 bytes a staged page (the largest page: a tile)
constexpr uint32_t kClasses = 4;  // accessor argument sets the kernel is compiled for
const CoreCoord kCore{0, 0};

}  // namespace

ProgramDescriptor KvFabricPushProgramFactory::create_descriptor(
    const KvFabricPushParams& args,
    const KvFabricPushInputs& tensor_args,
    Tensor& /*output*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "kv_fabric_push: a program per mesh coordinate");
    const auto& coord = *mesh_dispatch_coordinate;
    auto* mesh = tensor_args.sources[0].device();
    const uint32_t linear = coord[0] * args.mesh_num_cols + coord[1];
    const auto& sources = tensor_args.sources;

    // A source's accessor takes everything but its config and page size from the common runtime args, so sources of
    // one memory layout and page size share the kernel's compile-time arguments (a class).
    std::vector<std::vector<uint32_t>> classes;
    std::vector<uint32_t> common;
    std::vector<std::variant<uint32_t, Buffer*>> rt;
    for (size_t s = 0; s < sources.size(); ++s) {
        Buffer* buffer = sources[s].buffer();
        const auto config =
            buffer->buffer_layout() == TensorMemoryLayout::INTERLEAVED
                ? tensor_accessor::ArgsConfig(tensor_accessor::ArgConfig::None)
                : tensor_accessor::ArgConfig::RuntimeRank | tensor_accessor::ArgConfig::RuntimeNumBanks |
                      tensor_accessor::ArgConfig::RuntimeTensorShape | tensor_accessor::ArgConfig::RuntimeShardShape |
                      tensor_accessor::ArgConfig::RuntimeBankCoords;
        const TensorAccessorArgs accessor(*buffer, config);
        const auto cta = accessor.get_compile_time_args();
        TT_FATAL(
            cta.size() == 2, "kv_fabric_push: source {} takes {} compile-time accessor args, not 2", s, cta.size());
        auto it = std::find(classes.begin(), classes.end(), cta);
        if (it == classes.end()) {
            TT_FATAL(classes.size() < kClasses, "kv_fabric_push: sources of more than {} layouts", kClasses);
            classes.push_back(cta);
            it = classes.end() - 1;
        }
        const uint32_t page = buffer->page_size();
        TT_FATAL(page <= kSlot, "kv_fabric_push: source {} pages of {} B, at most {} B", s, page, kSlot);
        rt.emplace_back(buffer);
        rt.emplace_back(page);
        rt.emplace_back(static_cast<uint32_t>(it - classes.begin()));
        rt.emplace_back(static_cast<uint32_t>(common.size()));
        rt.emplace_back(args.tilize[s]);
        const auto crta = accessor.get_common_runtime_args();
        common.insert(common.end(), crta.begin(), crta.end());
    }

    KernelDescriptor::CompileTimeArgs ct = {
        args.dst_banks,
        kStage,
        kSlot,
        static_cast<uint32_t>(sources.size()),
        args.max_dst_bytes,
        args.socket_config_addrs.at(linear),
        args.socket_page_size,
    };
    for (uint32_t c = 0; c < kClasses; ++c) {
        const auto& cta = classes[c < classes.size() ? c : 0];  // unused classes: any valid set
        ct.insert(ct.end(), cta.begin(), cta.end());
    }

    const CoreRangeSet cores(CoreRange(kCore, kCore));
    const uint32_t l1 = kStage * kSlot + args.socket_page_size + args.max_dst_bytes + 64;
    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = l1,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = tt::DataFormat::UInt8,
            .page_size = l1,
        }}},
    });

    KernelDescriptor writer;
    writer.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/kv_fabric_push/device/kernels/"
        "kv_fabric_push_writer.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = std::move(ct);
    writer.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
    };
    writer.common_runtime_args = std::move(common);
    desc.kernels.push_back(std::move(writer));
    tt::tt_metal::KernelHandle kernel = desc.kernels.size() - 1;

    // Every destination is reached through this mesh's exit chip: one connection, toward the first destination (the
    // host checks the others leave by the same direction); the kernel rewrites the route a destination.
    const auto src_node = mesh->get_fabric_node_id(coord);
    const tt::tt_fabric::FabricNodeId dst_node(tt::tt_fabric::MeshId{args.dst_mesh_id}, args.dst_chips.front());
    const auto links = tt::tt_fabric::get_forwarding_link_indices(*mesh, src_node, dst_node);
    TT_FATAL(!links.empty(), "kv_fabric_push: no fabric route from {} to {}", src_node, dst_node);
    std::vector<uint32_t> fabric_args;
    tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
        *mesh,
        src_node,
        std::vector<tt::tt_fabric::FabricNodeId>{dst_node},
        std::vector<uint32_t>{links.front()},
        desc,
        kernel,
        kCore,
        fabric_args);
    rt.insert(rt.end(), fabric_args.begin(), fabric_args.end());
    desc.kernels[kernel].emplace_runtime_args(kCore, rt);
    return desc;
}

}  // namespace ttnn::experimental::prim
