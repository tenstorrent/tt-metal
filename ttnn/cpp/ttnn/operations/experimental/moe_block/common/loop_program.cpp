// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "loop_program.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block {

namespace {

using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::DataMovementProcessor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::ReaderConfigDescriptor;
using tt::tt_metal::WriterConfigDescriptor;

// the kernel's RISCs (mb_loop.hpp's rid bits: 0 BRISC, 1 NCRISC, 2..4 TRISC 0..2)
uint32_t risc_bits(const KernelDescriptor& k) {
    return std::visit(
        [](const auto& c) -> uint32_t {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, ComputeConfigDescriptor>) {
                return 0x1C;
            } else if constexpr (std::is_same_v<T, DataMovementConfigDescriptor>) {
                return c.processor == DataMovementProcessor::RISCV_0 ? 1u : 2u;
            } else if constexpr (std::is_same_v<T, ReaderConfigDescriptor>) {
                return 2u;  // (a reader: RISCV_1)
            } else {
                return 1u;  // (a writer: RISCV_0)
            }
        },
        k.config);
}

uint32_t key(const CoreCoord& c) { return uint32_t(c.x) | (uint32_t(c.y) << 8); }

std::vector<CoreCoord> cores_of(const CoreRangeSet& s) {
    std::vector<CoreCoord> v;
    for (const auto& r : s.ranges()) {
        for (auto y = r.start_coord.y; y <= r.end_coord.y; ++y) {
            for (auto x = r.start_coord.x; x <= r.end_coord.x; ++x) {
                v.emplace_back(x, y);
            }
        }
    }
    return v;
}

// (index | last << 31, delta) entries of b vs a: the linear delta (b1 - b0) and the last loop's remainder
void deltas(
    const std::vector<uint32_t>& a0,
    const std::vector<uint32_t>& a1,
    const std::vector<uint32_t>& al,
    uint32_t loops,
    std::vector<uint32_t>& out) {
    for (size_t i = 0; i < a0.size(); ++i) {
        const uint32_t d = a1[i] - a0[i];
        const uint32_t r = al[i] - (a0[i] + (loops - 1) * d);
        if (d) {
            out.push_back(uint32_t(i));
            out.push_back(d);
        }
        if (r) {
            out.push_back(uint32_t(i) | 0x80000000u);
            out.push_back(r);
        }
    }
}

const KernelDescriptor::CoreRuntimeArgs& rt_of(const KernelDescriptor& k, const CoreCoord& c) {
    for (const auto& [cc, a] : k.runtime_args) {
        if (cc == c) {
            return a;
        }
    }
    static const KernelDescriptor::CoreRuntimeArgs none;
    return none;
}

}  // namespace

ProgramDescriptor merge_loop_programs(
    ProgramDescriptor p0, const ProgramDescriptor& p1, const ProgramDescriptor& plast, const LoopProgramSpec& spec) {
    const uint32_t K = spec.loops;
    TT_FATAL(K >= 2 && spec.words != 0, "token loops in one program: {} loops, loop words at {}", K, spec.words);
    TT_FATAL(
        p1.kernels.size() == p0.kernels.size() && plast.kernels.size() == p0.kernels.size() &&
            p1.semaphores.size() == p0.semaphores.size() && p1.cbs.size() == p0.cbs.size(),
        "token loops in one program: the loops' programs differ in their kernels, semaphores or CBs");
    for (size_t i = 0; i < p0.cbs.size(); ++i) {
        for (const auto* p : {&p1, &plast}) {
            const auto &a = p0.cbs[i], &b = p->cbs[i];
            TT_FATAL(
                a.total_size == b.total_size && a.address_offset == b.address_offset && a.core_ranges == b.core_ranges,
                "token loops in one program: the loops' CB {} differs",
                i);
        }
    }
    // (every core of the program: its RISCs, its semaphores)
    std::map<uint32_t, uint32_t> riscs, sems;
    std::map<uint32_t, CoreCoord> cores;
    for (const auto& k : p0.kernels) {
        for (const auto& c : cores_of(k.core_ranges)) {
            const uint32_t b = risc_bits(k);
            TT_FATAL(!(riscs[key(c)] & b), "token loops in one program: two kernels on one RISC of core {}", c.str());
            riscs[key(c)] |= b;
            cores[key(c)] = c;
        }
    }
    for (const auto& s : p0.semaphores) {
        TT_FATAL(s.initial_value == 0, "token loops in one program: semaphore {} starts at {}", s.id, s.initial_value);
        for (const auto& c : cores_of(s.core_ranges)) {
            sems[key(c)] |= 1u << s.id;
        }
    }
    auto lead_of = [&](uint32_t kc) {
        const uint32_t r = riscs.at(kc);
        TT_FATAL(r & 3, "token loops in one program: core {} runs no data movement kernel", cores.at(kc).str());
        return (r & 2) ? 2u : 1u;
    };
    // (the coordinator: the first core of the lead kernel with the fewest cores)
    int coord_kernel = -1;
    CoreCoord coord;
    size_t fewest = SIZE_MAX;
    for (size_t i = 0; i < p0.kernels.size(); ++i) {
        const auto& k = p0.kernels[i];
        const auto cs = cores_of(k.core_ranges);
        if (cs.empty() || !(risc_bits(k) & 3)) {
            continue;
        }
        bool lead = true;
        for (const auto& c : cs) {
            lead = lead && lead_of(key(c)) == risc_bits(k);
        }
        if (lead && cs.size() < fewest) {
            fewest = cs.size();
            coord_kernel = int(i);
            coord = cs[0];
        }
    }
    TT_FATAL(coord_kernel >= 0, "token loops in one program: no kernel leads every core it runs on");
    const CoreCoord cv = spec.virt(coord);
    const uint32_t coord_word = key(coord) | (uint32_t(cv.x) << 16) | (uint32_t(cv.y) << 24);
    for (size_t i = 0; i < p0.kernels.size(); ++i) {
        auto& k = p0.kernels[i];
        const auto& k1 = p1.kernels[i];
        const auto& kl = plast.kernels[i];
        TT_FATAL(
            k1.kernel_source == k.kernel_source && kl.kernel_source == k.kernel_source &&
                k1.compile_time_args == k.compile_time_args && kl.compile_time_args == k.compile_time_args &&
                k1.defines == k.defines && kl.defines == k.defines && k1.core_ranges == k.core_ranges &&
                k1.common_runtime_args.size() == k.common_runtime_args.size() &&
                kl.common_runtime_args.size() == k.common_runtime_args.size(),
            "token loops in one program: kernel {} ({}) differs between the loops but for its runtime args",
            i,
            k.kernel_source);
        const auto cs = cores_of(k.core_ranges);
        const uint32_t rb = risc_bits(k);
        const bool compute = rb & 0x1C;
        // (the cores this kernel leads: each its RISCs and the semaphores it resets)
        std::vector<uint32_t> roles;
        for (const auto& c : cs) {
            const uint32_t kc = key(c);
            if (!compute && lead_of(kc) == rb) {
                roles.push_back(kc);
                roles.push_back(riscs.at(kc));
                roles.push_back(sems.count(kc) ? sems.at(kc) : 0u);
            }
        }
        std::vector<uint32_t> table = {spec.words, 0u, 0u, coord_word, uint32_t(cores.size())};
        std::vector<uint32_t> cd;
        deltas(k.common_runtime_args, k1.common_runtime_args, kl.common_runtime_args, K, cd);
        table.push_back(uint32_t(cd.size() / 2));
        table.insert(table.end(), cd.begin(), cd.end());
        std::vector<uint32_t> sections;
        uint32_t np = 0;
        for (const auto& [c, a0] : k.runtime_args) {
            const auto &a1 = rt_of(k1, c), &al = rt_of(kl, c);
            TT_FATAL(
                a1.size() == a0.size() && al.size() == a0.size(),
                "token loops in one program: kernel {} ({}) core {}: the loops' arg counts differ",
                i,
                k.kernel_source,
                c.str());
            std::vector<uint32_t> d;
            deltas(a0, a1, al, K, d);
            if (!d.empty()) {
                sections.push_back(key(c));
                sections.push_back(uint32_t(d.size() / 2));
                sections.insert(sections.end(), d.begin(), d.end());
                ++np;
            }
        }
        TT_FATAL(
            !compute || (cd.empty() && np == 0),
            "token loops in one program: compute kernel {} ({}) has args that differ by the loop",
            i,
            k.kernel_source);
        table.push_back(np);
        table.insert(table.end(), sections.begin(), sections.end());
        // (the roles: one for every core when this kernel leads them all alike - high bit - else a core each)
        bool alike = roles.size() / 3 == cs.size();
        for (size_t j = 3; alike && j < roles.size(); j += 3) {
            alike = roles[j + 1] == roles[1] && roles[j + 2] == roles[2];
        }
        if (alike && !roles.empty()) {
            table.insert(table.end(), {0x80000000u, roles[1], roles[2]});
        } else {
            table.push_back(uint32_t(roles.size() / 3));
            table.insert(table.end(), roles.begin(), roles.end());
        }
        if (int(i) == coord_kernel) {  // (the cores it releases: every core of the program)
            table.push_back(uint32_t(cores.size()));
            for (const auto& [kc, c] : cores) {
                const CoreCoord v = spec.virt(c);
                table.push_back(uint32_t(v.x) | (uint32_t(v.y) << 16));
            }
        }
        const uint32_t lt = uint32_t(k.common_runtime_args.size());
        k.common_runtime_args.insert(k.common_runtime_args.end(), table.begin(), table.end());
        k.defines.emplace_back("MB_LOOPS", std::to_string(K));
        k.defines.emplace_back("MB_LT", std::to_string(lt));
        if (!roles.empty()) {  // (only a kernel that leads a core carries the barrier's lead and its resets)
            k.defines.emplace_back("MB_LOOP_LEAD", "1");
        }
        if (int(i) == coord_kernel) {  // (and only the coordinator's the release)
            k.defines.emplace_back("MB_LOOP_COORD", "1");
        }
    }
    return p0;
}

}  // namespace ttnn::operations::experimental::moe_block
