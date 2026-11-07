// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "address_patches.hpp"

#include <algorithm>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block {

namespace address_patches_detail {

uint16_t source_of(std::span<tt::tt_metal::Buffer* const> sources, const tt::tt_metal::Buffer* buffer) {
    const auto it = std::find(sources.begin(), sources.end(), buffer);
    TT_FATAL(it != sources.end(), "moe_block: a runtime-arg buffer binding is not one of the op's sources");
    return static_cast<uint16_t>(it - sources.begin());
}

}  // namespace address_patches_detail

AddressPatches take_address_patches(
    const tt::tt_metal::ProgramDescriptor& desc, std::span<tt::tt_metal::Buffer* const> sources) {
    using address_patches_detail::source_of;
    AddressPatches p;
    for (uint32_t i = 0; i < desc.cbs.size(); ++i) {
        const auto& cb = desc.cbs[i];
        if (cb.buffer != nullptr) {
            const uint16_t src = source_of(sources, cb.buffer);
            p.cbs.push_back({i, src, cb.address_offset});
            if (p.cb_src_addr.size() <= src) {
                p.cb_src_addr.resize(src + 1, 0);
            }
            p.cb_src_addr[src] = cb.buffer->address();
        }
    }
    for (uint32_t k = 0; k < desc.kernels.size(); ++k) {
        const auto& kd = desc.kernels[k];
        for (const auto& b : kd.buffer_bindings) {
            p.args.push_back({k, false, b.core, b.arg_idx, source_of(sources, b.buffer)});
        }
        for (const auto& b : kd.common_buffer_bindings) {
            p.args.push_back({k, true, {}, b.arg_idx, source_of(sources, b.buffer)});
        }
    }
    std::stable_sort(p.args.begin(), p.args.end(), [](const auto& a, const auto& b) { return a.src < b.src; });
    for (uint32_t i = 0; i < p.args.size();) {
        uint32_t j = i + 1;
        while (j < p.args.size() && p.args[j].src == p.args[i].src) {
            ++j;
        }
        p.groups.emplace_back(i, j);
        i = j;
    }
    return p;
}

void apply_address_patches(
    tt::tt_metal::Program& program, const AddressPatches& patches, std::span<tt::tt_metal::Buffer* const> sources) {
    const auto& args = patches.args;
    auto value = [&](const AddressPatches::Arg& a) -> uint32_t& {
        auto& data = a.common ? tt::tt_metal::GetCommonRuntimeArgs(program, a.kernel)
                              : tt::tt_metal::GetRuntimeArgs(program, a.kernel, a.core);
        return data[a.index];
    };
    for (const auto& [i, j] : patches.groups) {
        const uint16_t src = args[i].src;
        const auto* buffer = sources[src];
        TT_FATAL(buffer != nullptr, "moe_block: source {} is absent on a cache hit but was bound at build", src);
        const auto addr = static_cast<uint32_t>(buffer->address());
        if (value(args[i]) != addr) {
            for (size_t k = i; k < j; ++k) {
                value(args[k]) = addr;
            }
        }
    }
    if (patches.cbs.empty()) {
        return;
    }
    // (UpdateDynamicCircularBufferAddress marks the CB config changed: only for a source that moved)
    std::vector<bool> moved(patches.cb_src_addr.size(), false);
    for (uint16_t s = 0; s < patches.cb_src_addr.size(); ++s) {
        if (s < sources.size() && sources[s] != nullptr && sources[s]->address() != patches.cb_src_addr[s]) {
            moved[s] = true;
        }
    }
    const auto cbs = program.circular_buffers();
    for (const auto& cb : patches.cbs) {
        TT_FATAL(sources[cb.src] != nullptr, "moe_block: CB source {} is absent on a cache hit", cb.src);
        if (moved[cb.src]) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cbs[cb.index]->id(), *sources[cb.src], cb.offset);
        }
    }
    for (uint16_t s = 0; s < moved.size(); ++s) {
        if (moved[s]) {
            patches.cb_src_addr[s] = sources[s]->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::moe_block
