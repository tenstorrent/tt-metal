// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "expert_slot_copy_program_factory.hpp"

#include <array>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy {

namespace expert_slot_copy_detail {

constexpr const char* KERNEL_DIR = "ttnn/ttnn/bringup/moe_block_main_ttnn/expert_slot_copy/device/kernels/dataflow/";
constexpr uint8_t CB_DATA = 0;
constexpr uint8_t CB_TAB = 1;
constexpr uint32_t ARG_ME = 6;  // (args 0..5: the buffers)
constexpr uint32_t ARG_JOBS = 8;
constexpr uint32_t RCV0 = ARG_JOBS + 3 * kMaxJobs;
constexpr uint32_t NARGS = RCV0 + 2 + kMaxLinks;

std::array<tt::tt_metal::Buffer*, 6> sources_of(const SlotCopyInputs& t) {
    return {t.src_gu.buffer(), t.src_d.buffer(), t.dst_gu.buffer(), t.dst_d.buffer(), t.runs.buffer(), t.sems.buffer()};
}

// A chip's port cores in (direction, link) order.
std::vector<std::pair<PortKey, tt::tt_metal::CoreCoord>> chip_cores(const SlotCopyPorts& ports, ChipCoord chip) {
    std::vector<std::pair<PortKey, tt::tt_metal::CoreCoord>> out;
    const auto& p = ports.plan.chips.at(chip);
    for (Dir d : kDirs) {
        if (!p[static_cast<size_t>(d)]) {
            continue;
        }
        for (uint32_t li = 0; li < p[static_cast<size_t>(d)]->links.size(); ++li) {
            const PortKey key{.chip = chip, .dir = d, .link_index = li};
            out.emplace_back(key, ports.cores.at(key));
        }
    }
    return out;
}

// Per chip, per port core (chip_cores' order): args ARG_ME .. NARGS - 1 (the job and receiver words) of this call.
std::map<ChipCoord, std::vector<std::vector<uint32_t>>> job_args(
    const SlotCopyParams& args, tt::tt_metal::distributed::MeshDevice* mesh) {
    const auto& ports = *args.ports;
    const auto& topo = ports.plan.topo;
    const auto virt = [&](ChipCoord ch, Dir d, uint32_t li) {
        const auto c =
            mesh->worker_core_from_logical_core(ports.cores.at(PortKey{.chip = ch, .dir = d, .link_index = li}));
        return static_cast<uint32_t>(c.x) | (static_cast<uint32_t>(c.y) << 16);
    };
    const auto node = [&](ChipCoord ch) {
        const auto n = mesh->get_fabric_node_id(ttnn::MeshCoordinate(ch.r, ch.c));
        return static_cast<uint32_t>(n.chip_id) | (static_cast<uint32_t>(*n.mesh_id) << 16);
    };
    std::map<ChipCoord, std::vector<std::vector<uint32_t>>> out;
    std::map<ChipCoord, std::map<PortKey, size_t>> index;
    for (const auto& [chip, p] : ports.plan.chips) {
        const auto cores = chip_cores(ports, chip);
        auto& v = out[chip];
        v.assign(cores.size(), std::vector<uint32_t>(NARGS - ARG_ME, 0u));
        for (size_t i = 0; i < cores.size(); ++i) {
            v[i][0] = topo.index(chip);
            index[chip][cores[i].first] = i;
        }
    }
    const auto add_job = [&](ChipCoord ch, size_t core, uint32_t w0, uint32_t w1, uint32_t w2) {
        auto& a = out.at(ch).at(core);
        const uint32_t j = a[ARG_JOBS - 1 - ARG_ME]++;
        TT_FATAL(j < kMaxJobs, "expert_slot_copy: more than {} jobs on one port core", kMaxJobs);
        a[ARG_JOBS - ARG_ME + 3 * j] = w0;
        a[ARG_JOBS - ARG_ME + 3 * j + 1] = w1;
        a[ARG_JOBS - ARG_ME + 3 * j + 2] = w2;
    };
    std::vector<bool> received(topo.num_chips(), false);
    for (const auto& jb : args.jobs) {
        TT_FATAL(jb.table < 256, "expert_slot_copy: run table {} (at most 256 tables)", jb.table);
        const ChipCoord a = topo.coord(jb.src), b = topo.coord(jb.dst);
        if (jb.src == jb.dst) {  // local: every port core of the chip takes a part of the chunks
            const auto n = static_cast<uint32_t>(out.at(a).size());
            for (uint32_t k = 0; k < n; ++k) {
                add_job(a, k, jb.table | (kLocal << 8) | (k << 16) | (n << 24), 0u, 0u);
            }
            continue;
        }
        TT_FATAL(!received[jb.dst], "expert_slot_copy: chip {} receives two copies in one call", jb.dst);
        received[jb.dst] = true;
        const Dir d = *first_hop(a, b, topo), back = *first_hop(b, a, topo);
        const auto& port = ports.plan.chips.at(a)[static_cast<size_t>(d)];
        TT_FATAL(port.has_value(), "expert_slot_copy: chip {} has no port towards chip {}", jb.src, jb.dst);
        const uint32_t L = std::min<uint32_t>(args.links, port->links.size());
        const uint32_t rcv = virt(b, back, 0);
        for (uint32_t li = 0; li < L; ++li) {
            add_job(
                a,
                index.at(a).at(PortKey{.chip = a, .dir = d, .link_index = li}),
                jb.table | (jb.dst << 8) | (li << 16) | (L << 24),
                node(b),
                rcv);
        }
        auto& r = out.at(b).at(index.at(b).at(PortKey{.chip = b, .dir = back, .link_index = 0}));
        r[RCV0 - ARG_ME] = L;
        r[RCV0 - ARG_ME + 1] = node(a);
        for (uint32_t li = 0; li < L; ++li) {
            r[RCV0 - ARG_ME + 2 + li] = virt(a, d, li);
        }
    }
    return out;
}

tt::tt_metal::CBDescriptor cb(uint8_t index, uint32_t page, uint32_t pages, const CoreRangeSet& cores) {
    return tt::tt_metal::CBDescriptor{
        .total_size = page * pages,
        .core_ranges = cores,
        .format_descriptors = {tt::tt_metal::CBFormatDescriptor{
            .buffer_index = index, .data_format = tt::DataFormat::UInt32, .page_size = page}},
    };
}

tt::tt_metal::WorkloadDescriptor build_workload(const SlotCopyParams& args, const SlotCopyInputs& t) {
    using tt::tt_metal::KernelDescriptor;
    auto* mesh = t.dst_gu.device();
    const auto& ports = *args.ports;
    const auto bufs = sources_of(t);
    const uint32_t tab_bytes = t.runs.buffer()->aligned_page_size();
    const auto calls = job_args(args, mesh);
    const auto node = [&](ChipCoord ch) { return mesh->get_fabric_node_id(ttnn::MeshCoordinate(ch.r, ch.c)); };
    const KernelDescriptor::Defines defines = {
        {"ESC_MAXJ", std::to_string(kMaxJobs)},
        {"ESC_MAXL", std::to_string(kMaxLinks)},
        {"ESC_CHUNK", std::to_string(args.chunk_bytes)},
        {"ESC_DEPTH", std::to_string(args.depth)},
        {"ESC_BATCH", std::to_string(args.batch)},
    };
    KernelDescriptor::CompileTimeArgs reader_ct = {CB_DATA, CB_TAB, tab_bytes};
    tt::tt_metal::TensorAccessorArgs(t.runs.buffer()).append_to(reader_ct);
    const KernelDescriptor::CompileTimeArgs writer_ct = {CB_DATA, CB_TAB};

    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& [chip, p] : ports.plan.chips) {
        const auto cores = chip_cores(ports, chip);
        TT_FATAL(!cores.empty(), "expert_slot_copy: chip ({}, {}) has no port", chip.r, chip.c);
        std::vector<CoreRange> ranges;
        for (const auto& [key, core] : cores) {
            ranges.emplace_back(core, core);
        }
        const CoreRangeSet set(ranges);
        tt::tt_metal::ProgramDescriptor desc;
        desc.cbs.push_back(cb(CB_DATA, args.chunk_bytes, args.depth, set));
        desc.cbs.push_back(cb(CB_TAB, tab_bytes, 1, set));

        KernelDescriptor reader;
        reader.kernel_source = std::string(KERNEL_DIR) + "slot_copy_reader.cpp";
        reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader.core_ranges = set;
        reader.compile_time_args = reader_ct;
        reader.defines = defines;
        reader.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::NOC_0};
        KernelDescriptor writer;
        writer.kernel_source = std::string(KERNEL_DIR) + "slot_copy_writer.cpp";
        writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer.core_ranges = set;
        writer.compile_time_args = writer_ct;
        writer.defines = defines;
        writer.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_1};

        const auto& per_core = calls.at(chip);
        for (size_t i = 0; i < cores.size(); ++i) {
            const auto& [key, core] = cores[i];
            KernelDescriptor::RTArgList rd;
            for (auto* b : bufs) {
                rd.push_back(b);
            }
            rd.append(per_core[i]);
            KernelDescriptor::RTArgList wr = rd;
            reader.emplace_runtime_args(core, rd);
            const auto& port = *p[static_cast<size_t>(key.dir)];
            std::vector<uint32_t> fabric;
            tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                node(chip), node(port.peer), port.links[key.link_index], desc, core, fabric);
            wr.append(fabric);
            writer.emplace_runtime_args(core, wr);
        }
        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(writer));
        const ttnn::MeshCoordinate coord(chip.r, chip.c);
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(desc)});
    }
    return workload;
}

}  // namespace expert_slot_copy_detail

SlotCopyProgramFactory::cached_mesh_workload_t SlotCopyProgramFactory::create_mesh_workload(
    const SlotCopyParams& args,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/,
    const SlotCopyInputs& t,
    Tensor& /*output*/) {
    using namespace expert_slot_copy_detail;
    const auto sources = sources_of(t);
    auto descriptors = build_workload(args, t);
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    for (auto& [range, desc] : descriptors.programs) {
        const auto coord = range.start_coord();
        std::vector<tt::tt_metal::CoreCoord> cores;
        for (const auto& [key, core] : chip_cores(*args.ports, ChipCoord{coord[0], coord[1]})) {
            cores.push_back(core);
        }
        shared_variables_t sv{.patches = take_address_patches(desc, sources), .cores = std::move(cores)};
        workload.add_program(range, tt::tt_metal::Program(desc));
        shared.emplace(range, std::move(sv));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void SlotCopyProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached, const SlotCopyParams& args, const SlotCopyInputs& t, Tensor& /*output*/) {
    using namespace expert_slot_copy_detail;
    const auto sources = sources_of(t);
    const auto calls = job_args(args, t.dst_gu.device());
    for (auto& [range, sv] : cached.shared_variables) {
        auto& program = cached.workload.get_programs().at(range);
        apply_address_patches(program, sv.patches, sources);
        const auto coord = range.start_coord();
        const auto& per_core = calls.at(ChipCoord{coord[0], coord[1]});
        for (size_t i = 0; i < sv.cores.size(); ++i) {
            for (uint32_t kernel : {0u, 1u}) {
                auto& rt = tt::tt_metal::GetRuntimeArgs(program, kernel, sv.cores[i]);
                for (size_t k = 0; k < per_core[i].size(); ++k) {
                    rt[ARG_ME + k] = per_core[i][k];
                }
            }
        }
    }
}

}  // namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy
