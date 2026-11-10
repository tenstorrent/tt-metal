// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_dispatch_2d_program_factory.hpp"
#include <tracy/Tracy.hpp>

#include <array>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt_stl/assert.hpp>

#include "kernels/sparse_dispatch_2d_layout.hpp"
#include "sparse_dispatch_2d_plan.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d {

namespace sparse_dispatch_2d_detail {

constexpr const char* KERNEL_DIR =
    "ttnn/cpp/ttnn/operations/experimental/moe_block/sparse_dispatch_2d/device/kernels/dataflow/";

// The mesh constants the kernels' common header reads (sparse_dispatch_2d_common.hpp), as defines.
tt::tt_metal::KernelDescriptor::Defines common_defines(const MeshTopology& topo, uint32_t S) {
    const auto span = span_subs(topo.R, topo.C, topo.wrap_x, topo.wrap_y);
    const auto list = [](const std::vector<uint8_t>& v) {
        std::vector<uint32_t> w(v.begin(), v.end());
        return fmt::format("{}", fmt::join(w, ","));
    };
    return {
        {"SPARSE_DISPATCH_R", std::to_string(topo.R)},
        {"SPARSE_DISPATCH_C", std::to_string(topo.C)},
        {"SPARSE_DISPATCH_S", std::to_string(S)},
        {"SPARSE_DISPATCH_GMAX", std::to_string(gmax(topo))},
        {"SPARSE_DISPATCH_RW", std::to_string(route_words(topo.num_chips()))},
        {"SPARSE_DISPATCH_WRAP_X", span.wrap_x ? "1" : "0"},
        {"SPARSE_DISPATCH_WRAP_Y", span.wrap_y ? "1" : "0"},
        {"SPARSE_DISPATCH_XRING", list(span.xring)},
        {"SPARSE_DISPATCH_YRING", list(span.yring)},
    };
}

// A CB of `size` bytes; its page is the whole CB unless `page_size` says otherwise.
tt::tt_metal::CBDescriptor cb(
    uint8_t index, uint32_t size, tt::DataFormat fmt, const CoreRangeSet& cores, uint32_t page_size = 0) {
    return tt::tt_metal::CBDescriptor{
        .total_size = size,
        .core_ranges = cores,
        .format_descriptors = {tt::tt_metal::CBFormatDescriptor{
            .buffer_index = index, .data_format = fmt, .page_size = page_size != 0 ? page_size : size}},
    };
}

// The buffers the runtime args address, by source (common/args_table.hpp; nullptr: absent). The arena's is last.
constexpr size_t kArenaSource = 3;
std::array<tt::tt_metal::Buffer*, 4> sources_of(const SparseDispatch2dInputs& t) {
    return {t.xbuf.buffer(), t.route.buffer(), t.sems.buffer(), t.arena ? t.arena->buffer() : nullptr};
}

// The sources' addresses (0: absent).
std::array<uint32_t, 4> addresses_of(const std::array<tt::tt_metal::Buffer*, 4>& sources) {
    std::array<uint32_t, 4> out{};
    for (size_t i = 0; i < sources.size(); ++i) {
        out[i] = sources[i] != nullptr ? static_cast<uint32_t>(sources[i]->address()) : 0u;
    }
    return out;
}

// Every chip's program descriptor.
tt::tt_metal::WorkloadDescriptor build_workload(const SparseDispatch2dParams& args, const SparseDispatch2dInputs& t) {
    namespace lay = sparse_dispatch_2d_layout;
    using tt::tt_metal::Buffer;
    using tt::tt_metal::KernelDescriptor;

    auto* mesh = t.xbuf.device();
    const auto& plan = args.ports->plan;
    const auto& port_cores = args.ports->cores;
    const auto& topo = plan.topo;
    const uint32_t N = topo.num_chips();
    const uint32_t RW = route_words(N);
    const uint32_t S = static_cast<uint32_t>(t.route.logical_shape()[-1]) / RW;
    Buffer* out_buf = t.xbuf.buffer();
    Buffer* plan_buf = t.route.buffer();
    Buffer* sems_buf = t.sems.buffer();
    const uint32_t row_bytes = out_buf->aligned_page_size();
    const uint32_t plan_bytes = plan_buf->aligned_page_size();
    const auto ml = manifest_layout(topo.R, topo.C, topo.wrap_x, topo.wrap_y, S);
    const uint32_t ring_offs = lay::pack_ring_offsets(ml.ring1_offset, ml.ring2_offset);
    const CbSizes sizes = cb_sizes(topo, S, row_bytes, plan_bytes, args.group, args.manifest_batch);
    Buffer* arena_buf = t.arena ? t.arena->buffer() : nullptr;
    const auto arena_off = sizes.arena_offsets();
    if (arena_buf != nullptr) {
        TT_FATAL(
            arena_buf->aligned_size_per_bank() >= sizes.arena_bytes(),
            "sparse_dispatch_2d: the arena holds {} B a core, the dispatch's CBs need {} B",
            arena_buf->aligned_size_per_bank(),
            sizes.arena_bytes());
    }
    // a CB at its arena offset (with an arena) or static (without)
    const auto place = [&](tt::tt_metal::CBDescriptor c, uint32_t which) {
        if (arena_buf != nullptr) {
            c.buffer = arena_buf;
            c.address_offset = arena_off[which];
        }
        return c;
    };
    const auto defines = common_defines(topo, S);
    const tt::DataFormat row_fmt = tt::tt_metal::datatype_to_dataformat_converter(t.xbuf.dtype());

    // compile-time args: the manifest CB, then the kernels' own (see the kernels)
    KernelDescriptor::CompileTimeArgs reader_ct = {
        CB_MANIFEST, CB_ROWS, CB_META, CB_PLAN, row_bytes, args.group, plan_bytes};
    tt::tt_metal::TensorAccessorArgs(out_buf).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(plan_buf).append_to(reader_ct);
    KernelDescriptor::CompileTimeArgs sender_ct = {
        CB_MANIFEST, CB_ROWS, CB_META, CB_STAGE, row_bytes, args.group, args.manifest_batch, args.pack};
    tt::tt_metal::TensorAccessorArgs(out_buf).append_to(sender_ct);

    const auto virt = [&](const tt::tt_metal::CoreCoord& c) { return mesh->worker_core_from_logical_core(c); };
    const auto node = [&](ChipCoord ch) { return mesh->get_fabric_node_id(mesh_coord(ch, plan.topo)); };

    // (every chip's port cores: the port kernels and their CBs go on the union, the same program on every chip (one
    // program for the mesh) - a core that is no port of a chip runs them idle)
    const CoreRangeSet port_set(port_cell_ranges(port_cores));
    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& [chip, ports] : plan.chips) {
        struct Port {
            Dir dir;
            uint32_t li;
            tt::tt_metal::CoreCoord core;
        };
        std::vector<Port> mine;
        std::set<std::pair<size_t, size_t>> taken;
        for (Dir d : kDirs) {
            const auto& p = ports[static_cast<size_t>(d)];
            if (!p) {
                continue;
            }
            for (uint32_t li = 0; li < p->links.size(); ++li) {
                const auto core = port_cores.at(PortKey{.chip = chip, .dir = d, .link_index = li});
                mine.push_back(Port{d, li, core});
                taken.insert({core.x, core.y});
            }
        }
        TT_FATAL(!mine.empty(), "sparse_dispatch_2d: chip ({}, {}) has no port", chip.r, chip.c);

        tt::tt_metal::ProgramDescriptor desc;
        // (the manifest first: at the CB base, the same address on every port core of every chip - remote senders
        // write into it at their own manifest's address)
        // (with an arena: the arena's first CbSizes::arena_bytes() a port core, the manifest at its base)
        desc.cbs.push_back(place(cb(CB_MANIFEST, sizes.manifest, tt::DataFormat::UInt32, port_set), 0));
        desc.cbs.push_back(place(cb(CB_ROWS, sizes.rows, row_fmt, port_set, row_bytes), 1));
        desc.cbs.push_back(place(cb(CB_META, sizes.meta, tt::DataFormat::UInt32, port_set, META_PAGE), 2));
        desc.cbs.push_back(place(cb(CB_PLAN, sizes.plan, tt::DataFormat::UInt32, port_set), 3));
        desc.cbs.push_back(place(cb(CB_STAGE, sizes.stage, tt::DataFormat::UInt32, port_set), 4));
        // (the port kernels' blocks of values, read from the DRAM args table at the kernels' start)
        desc.cbs.push_back(place(cb(CB_ARGS, sizes.args, tt::DataFormat::UInt32, port_set), 5));

        KernelDescriptor reader;
        reader.kernel_source = std::string(KERNEL_DIR) + "sparse_dispatch_2d_port_reader.cpp";
        reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader.core_ranges = port_set;
        reader.compile_time_args = reader_ct;
        reader.defines = defines;
        // (the sender on NOC1, the NoC the port cores are placed nearest their Ethernet core on; the reader on NOC0)
        reader.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::NOC_0};

        KernelDescriptor sender;
        sender.kernel_source = std::string(KERNEL_DIR) + "sparse_dispatch_2d_port_writer.cpp";
        sender.source_type = KernelDescriptor::SourceType::FILE_PATH;
        sender.core_ranges = port_set;
        sender.compile_time_args = sender_ct;
        sender.defines = defines;
        sender.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_1};

        const auto src_node = node(chip);
        for (const auto& [d, li, core] : mine) {
            const auto& port = *ports[static_cast<size_t>(d)];
            const auto w = wiring(plan, chip, d);

            // the reader's args, in lay::reader_arg order: out, plan, words, then the values from kRingOffsets on
            static_assert(
                lay::reader_arg::kWords == 2 && lay::reader_arg::kRingOffsets == 3 &&
                lay::reader_arg::kLink == lay::reader_arg::kCount - 1 &&
                lay::reader_arg::kLink - lay::reader_arg::kRingOffsets == 5);
            KernelDescriptor::RTArgList rd;
            rd.push_back(out_buf);
            rd.push_back(plan_buf);
            rd.push_back(sems_buf);
            rd.append({ring_offs, w.ring_mask, chip.r, chip.c, static_cast<uint32_t>(d), li});
            reader.emplace_runtime_args(core, rd);

            const ChipCoord peer = port.peer;
            const auto ready = virt(port_cores.at(PortKey{.chip = peer, .dir = opposite(d), .link_index = li}));
            const auto pn = node(peer);
            // the sender's args after out and words, in lay::sender_arg order from kReadyX
            std::vector<uint32_t> tail = {
                static_cast<uint32_t>(ready.x),
                static_cast<uint32_t>(ready.y),
                static_cast<uint32_t>(*pn.mesh_id),
                static_cast<uint32_t>(pn.chip_id),
                ring_offs};
            for (const auto& s : w.streams) {
                if (s.on) {
                    const auto v = virt(port_cores.at(PortKey{.chip = peer, .dir = s.target, .link_index = li}));
                    // (lay::sender_arg kStreamOn, kStreamX, kStreamY, kStreamRing)
                    tail.insert(tail.end(), {s.on, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), s.ring});
                } else {
                    tail.insert(tail.end(), {0u, 0u, 0u, 0u});
                }
            }
            tail.insert(tail.end(), {peer.r, peer.c, static_cast<uint32_t>(d)});
            // (out, sems, then the tail: where the kernel reads the fabric connection's args)
            TT_FATAL(
                lay::sender_arg::kReadyX + tail.size() == lay::sender_arg::kFabric,
                "sparse_dispatch_2d: the sender's fabric args start at {}, the kernel reads them at {}",
                lay::sender_arg::kReadyX + tail.size(),
                lay::sender_arg::kFabric);
            std::vector<uint32_t> fabric;
            tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                *mesh, src_node, pn, port.links[li], desc, core, fabric);
            tail.insert(tail.end(), fabric.begin(), fabric.end());

            KernelDescriptor::RTArgList sd;
            sd.push_back(out_buf);
            sd.push_back(sems_buf);
            sd.append(tail);
            sender.emplace_runtime_args(core, sd);
        }
        // (another chip's port core only: idle - sparse_dispatch::IDLE_PORT)
        for (const auto& [key, core] : port_cores) {
            if (!taken.insert({core.x, core.y}).second) {
                continue;
            }
            // (the reader's args from kRingOffsets on: zeros but the direction; the sender's from kReadyX on: the mark)
            std::vector<uint32_t> idle_reader(lay::reader_arg::kCount - lay::reader_arg::kRingOffsets, 0u);
            idle_reader[lay::reader_arg::kDir - lay::reader_arg::kRingOffsets] = lay::kIdlePort;
            KernelDescriptor::RTArgList rd;
            rd.push_back(out_buf);
            rd.push_back(plan_buf);
            rd.push_back(sems_buf);
            rd.append(idle_reader);
            reader.emplace_runtime_args(core, rd);
            KernelDescriptor::RTArgList sd;
            sd.push_back(out_buf);
            sd.push_back(sems_buf);
            sd.append({lay::kIdlePort});
            sender.emplace_runtime_args(core, sd);
        }
        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(sender));
        const ttnn::MeshCoordinate coord = mesh_coord(chip, plan.topo);
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(desc)});
    }
    return workload;
}

}  // namespace sparse_dispatch_2d_detail

SparseDispatch2dProgramFactory::cached_mesh_workload_t SparseDispatch2dProgramFactory::create_mesh_workload(
    const SparseDispatch2dParams& args,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/,
    const SparseDispatch2dInputs& t,
    Tensor& /*output*/) {
    using namespace sparse_dispatch_2d_detail;
    const auto sources = sources_of(t);
    auto& mesh = *t.xbuf.device();
    auto w = build_workload(args, t);
    // every core's values into the per-chip DRAM tables (the reader's block and the sender's in the args CB, which
    // sits on the arena when there is one): the chips' programs are then alike, one program for the mesh with nothing
    // patched on the device; the tensor addresses (the buffer bindings) become fixups
    ChipPrograms chips;
    for (auto& prog : w.programs) {
        TT_FATAL(prog.range.shape().mesh_size() == 1, "sparse_dispatch_2d: one program a chip before the args move");
        chips.emplace_back(prog.range.start_coord(), std::move(prog.descriptor));
    }
    // (the fabric connection's args in the block too: the sender opens it with build_from_eth_channel)
    std::vector<ArgsTableKernel> kernels = {
        ArgsTableKernel{.kernel = 0, .l1_offset = ARGS_READER_OFF, .bytes = ARGS_KERNEL_BYTES, .block_cb = CB_ARGS},
        ArgsTableKernel{.kernel = 1, .l1_offset = ARGS_SENDER_OFF, .bytes = ARGS_KERNEL_BYTES, .block_cb = CB_ARGS}};
    for (auto& kernel : kernels) {
        fixups_from_buffer_bindings(chips, kernel, sources);
    }
    std::string why;
    const auto table = runtime_args_to_dram(mesh, chips, kernels, addresses_of(sources), 0, why);
    TT_FATAL(table.tables != nullptr, "sparse_dispatch_2d: the port kernels' args to DRAM: {}", why);
    for (size_t i = 0; i < chips.size(); ++i) {
        w.programs[i].descriptor = std::move(chips[i].second);
    }
    std::string merge_why;
    auto descriptors = merge_chip_workload(std::move(w), "sparse_dispatch_2d", &merge_why);
    TT_FATAL(
        descriptors.programs.size() == 1,
        "sparse_dispatch_2d: the chips' programs still differ with their args in DRAM: {}",
        merge_why);
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    for (auto& [range, desc] : descriptors.programs) {
        shared_variables_t sv{
            .addresses = {
                .tables = table.tables,
                .table_kernels = table_kernels(table, kernels, 0, std::nullopt, 0),
                .args = {},
                .arena_cbs = arena_backed_cbs(desc, sources[kArenaSource])}};
        workload.add_program(range, tt::tt_metal::Program(desc));
        shared.emplace(range, std::move(sv));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void SparseDispatch2dProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached,
    const SparseDispatch2dParams& /*args*/,
    const SparseDispatch2dInputs& t,
    Tensor& /*output*/) {
    ZoneScopedN("sparse_dispatch_2d: override (hit)");
    using namespace sparse_dispatch_2d_detail;
    const auto sources = sources_of(t);
    const auto addresses = addresses_of(sources);
    for (auto& [range, sv] : cached.shared_variables) {
        auto& program = cached.workload.get_programs().at(range);
        write_program_addresses(program, sv.addresses, addresses, sources[kArenaSource]);
    }
}

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d
