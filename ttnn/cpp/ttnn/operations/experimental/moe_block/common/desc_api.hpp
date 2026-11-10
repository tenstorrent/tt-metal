// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The host Program API's calls the flat expert and the combine build with, into a ProgramDescriptor instead of a
// Program (one program for the mesh merges descriptors - common/chip_program.hpp). A kernel's handle is its index in
// the descriptor, a CB's its index in desc.cbs: the Program built from the descriptor has the same kernel handles; its
// CB handles are program.circular_buffers()[index]->id() (cb_ids).
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block::desc_api {

using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;

inline KernelDescriptor::Defines defines_of(const std::map<std::string, std::string>& d) {
    return KernelDescriptor::Defines(d.begin(), d.end());
}
inline KernelDescriptor::NamedCompileTimeArgs named_of(const std::unordered_map<std::string, uint32_t>& d) {
    KernelDescriptor::NamedCompileTimeArgs v(d.begin(), d.end());
    std::sort(v.begin(), v.end());
    return v;
}

inline uint32_t CreateKernel(
    ProgramDescriptor& p,
    const std::string& file,
    const CoreRangeSet& cores,
    const tt::tt_metal::DataMovementConfig& c) {
    KernelDescriptor k;
    k.kernel_source = file;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = c.compile_args;
    k.named_compile_time_args = named_of(c.named_compile_args);
    k.defines = defines_of(c.defines);
    k.opt_level = c.opt_level;
    k.compiler_include_paths = c.compiler_include_paths;
    k.config =
        tt::tt_metal::DataMovementConfigDescriptor{.processor = c.processor, .noc = c.noc, .noc_mode = c.noc_mode};
    p.kernels.push_back(std::move(k));
    return static_cast<uint32_t>(p.kernels.size() - 1);
}

inline uint32_t CreateKernel(
    ProgramDescriptor& p, const std::string& file, const CoreRangeSet& cores, const tt::tt_metal::ComputeConfig& c) {
    KernelDescriptor k;
    k.kernel_source = file;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = c.compile_args;
    k.named_compile_time_args = named_of(c.named_compile_args);
    k.defines = defines_of(c.defines);
    k.opt_level = c.opt_level;
    k.compiler_include_paths = c.compiler_include_paths;
    k.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = c.math_fidelity,
        .fp32_dest_acc_en = c.fp32_dest_acc_en,
        .dst_full_sync_en = c.dst_full_sync_en,
        .unpack_to_dest_mode = c.unpack_to_dest_mode,
        .bfp8_pack_precise = c.bfp8_pack_precise,
        .math_approx_mode = c.math_approx_mode,
        .enable_trisc2_rvv = c.enable_trisc2_rvv};
    p.kernels.push_back(std::move(k));
    return static_cast<uint32_t>(p.kernels.size() - 1);
}

inline void SetRuntimeArgs(
    ProgramDescriptor& p, uint32_t k, const tt::tt_metal::CoreCoord& core, std::vector<uint32_t> v) {
    auto& rt = p.kernels[k].runtime_args;
    for (auto& [c, a] : rt) {
        if (c == core) {
            a = std::move(v);
            return;
        }
    }
    rt.emplace_back(core, std::move(v));
}

inline void SetCommonRuntimeArgs(ProgramDescriptor& p, uint32_t k, std::vector<uint32_t> v) {
    p.kernels[k].common_runtime_args = std::move(v);
}

// a semaphore on `cores` (the lowest id free on every one of them)
inline uint32_t CreateSemaphore(ProgramDescriptor& p, const CoreRangeSet& cores, uint32_t initial) {
    for (uint32_t id = 0;; ++id) {
        const bool taken = std::any_of(p.semaphores.begin(), p.semaphores.end(), [&](const auto& s) {
            return s.id == id && s.core_type == tt::CoreType::WORKER && s.core_ranges.intersects(cores);
        });
        if (!taken) {
            TT_FATAL(id < 16, "desc_api: no free semaphore id");
            p.semaphores.push_back(
                tt::tt_metal::SemaphoreDescriptor{.id = id, .core_ranges = cores, .initial_value = initial});
            return id;
        }
    }
}

inline uint32_t CreateCircularBuffer(
    ProgramDescriptor& p, const CoreRangeSet& cores, const tt::tt_metal::CircularBufferConfig& c) {
    TT_FATAL(c.remote_buffer_indices().empty(), "desc_api: remote CB indices");
    tt::tt_metal::CBDescriptor d{.total_size = c.total_size(), .core_ranges = cores};
    std::vector<uint8_t> idx(c.buffer_indices().begin(), c.buffer_indices().end());
    std::sort(idx.begin(), idx.end());
    for (const uint8_t i : idx) {
        TT_FATAL(!c.tiles()[i].has_value(), "desc_api: a CB's tile shape");
        d.format_descriptors.push_back(tt::tt_metal::CBFormatDescriptor{
            .buffer_index = i, .data_format = *c.data_formats()[i], .page_size = *c.page_sizes()[i]});
    }
    if (c.globally_allocated_address().has_value()) {
        TT_FATAL(c.shadow_global_buffer != nullptr, "desc_api: a CB at an address without its buffer");
        d.buffer = const_cast<tt::tt_metal::Buffer*>(c.shadow_global_buffer);
        d.address_offset = c.address_offset();
    }
    p.cbs.push_back(std::move(d));
    return static_cast<uint32_t>(p.cbs.size() - 1);
}

// a program built from the descriptor: its CB handles by descriptor index
inline std::vector<tt::tt_metal::CBHandle> cb_ids(tt::tt_metal::Program& program) {
    std::vector<tt::tt_metal::CBHandle> out;
    for (const auto& cb : program.circular_buffers()) {
        out.push_back(cb->id());
    }
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block::desc_api
