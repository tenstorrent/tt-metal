// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prepared_program.hpp"

#include <tt-metalium/mesh_buffer.hpp>

#include "device/prepared_program_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::generic {
// (generic_op's structural descriptor hash: defined in generic_op_device_operation.cpp, declared where used)
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor);
}  // namespace ttnn::operations::generic

namespace ttnn::operations::bringup::prepared_program {

int Prepared::program_of(const ttnn::MeshCoordinate& coord) const {
    for (size_t i = 0; i < desc.mesh_programs.size(); ++i) {
        if (desc.mesh_programs[i].first.contains(coord)) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

size_t Prepared::patch_count() const {
    size_t n = 0;
    for (size_t i = 0; i < rt.size(); ++i) {
        n += rt[i].size() + crt[i].size() + cbs[i].size();
    }
    return n;
}

std::shared_ptr<const Prepared> prepare(
    tt::tt_metal::experimental::MeshProgramDescriptor desc, const std::vector<Tensor>& io_tensors) {
    TT_FATAL(!io_tensors.empty(), "prepared_program: io_tensors must hold at least the output (last)");
    TT_FATAL(!desc.mesh_programs.empty(), "prepared_program: an empty MeshProgramDescriptor");
    auto p = std::make_shared<Prepared>();
    p->n_io = io_tensors.size();

    // the io tensors' addresses and buffers; the same tensor at two positions maps to its first; two distinct
    // tensors at one address cannot be told apart from a runtime arg
    std::vector<uint32_t> addr(io_tensors.size());
    std::vector<tt::tt_metal::Buffer*> buf(io_tensors.size());
    for (size_t j = 0; j < io_tensors.size(); ++j) {
        TT_FATAL(io_tensors[j].is_allocated(), "prepared_program: io tensor {} is not allocated", j);
        addr[j] = static_cast<uint32_t>(io_tensors[j].mesh_buffer().address());
        buf[j] = io_tensors[j].mesh_buffer().get_reference_buffer();
    }
    auto io_at = [&](uint32_t v, auto&& where) -> int {
        int hit = -1;
        for (size_t j = 0; j < addr.size(); ++j) {
            if (addr[j] != v) {
                continue;
            }
            if (hit < 0) {
                hit = static_cast<int>(j);
            } else {
                TT_FATAL(
                    buf[j] == buf[hit],
                    "prepared_program: io tensors {} and {} share address {:#x} ({}): a runtime arg holding it is "
                    "ambiguous",
                    hit,
                    j,
                    v,
                    where());
            }
        }
        return hit;
    };

    const size_t np = desc.mesh_programs.size();
    p->rt.resize(np);
    p->crt.resize(np);
    p->cbs.resize(np);
    for (size_t pi = 0; pi < np; ++pi) {
        const auto& pd = desc.mesh_programs[pi].second;
        for (uint32_t k = 0; k < pd.kernels.size(); ++k) {
            const auto& kd = pd.kernels[k];
            for (const auto& [core, args] : kd.runtime_args) {
                for (uint32_t i = 0; i < args.size(); ++i) {
                    const int j =
                        io_at(args[i], [&] { return fmt::format("kernel {} core {} arg {}", k, core.str(), i); });
                    if (j >= 0) {
                        p->rt[pi].push_back(RtPatch{
                            .kernel = k,
                            .x = static_cast<uint32_t>(core.x),
                            .y = static_cast<uint32_t>(core.y),
                            .arg = i,
                            .io = static_cast<uint32_t>(j)});
                    }
                }
            }
            for (uint32_t i = 0; i < kd.common_runtime_args.size(); ++i) {
                const int j =
                    io_at(kd.common_runtime_args[i], [&] { return fmt::format("kernel {} common arg {}", k, i); });
                if (j >= 0) {
                    p->crt[pi].push_back(CommonRtPatch{.kernel = k, .arg = i, .io = static_cast<uint32_t>(j)});
                }
            }
        }
        for (uint32_t c = 0; c < pd.cbs.size(); ++c) {
            const auto& cb = pd.cbs[c];
            TT_FATAL(cb.tensor == nullptr, "prepared_program: CB {} is bound by tensor: bind it by buffer", c);
            if (cb.buffer == nullptr) {
                continue;
            }
            for (size_t j = 0; j < buf.size(); ++j) {
                if (buf[j] == cb.buffer) {
                    p->cbs[pi].push_back(CbPatch{.cb = c, .io = static_cast<uint32_t>(j)});
                    break;
                }
            }
        }
    }

    // the program cache key: generic_op's structural hash of every (range, program), salted so a generic_op launch
    // of the same descriptor keeps its own entry - and everything a cache hit does not re-apply: a hit re-applies only
    // the patches, so two descriptors of one structure but other frozen state (a caller rebuilding it after something
    // moved) must not share a cached program. Per runtime arg slot: a patched slot by the io tensor it takes (tagged:
    // never equal to a frozen value), a frozen one by its value; per core its place and arg count (generic_op's hash
    // has neither); per CB its patch's io tensor or its frozen buffer's address, and its address offset. Equal keys
    // then mean the same slots patched from the same io tensors - what override_runtime_arguments' first-slot check
    // relies on (an io tensor's slots were all written together, by a program of the same key)
    ttsl::hash::hash_t h = ttsl::hash::hash_objects_with_default_seed(std::string_view("bringup.prepared_program"));
    ttsl::hash::hash_combine(h, p->n_io);
    auto slot = [&h](bool patched, uint32_t io_or_value) {
        ttsl::hash::hash_combine(h, patched);
        ttsl::hash::hash_combine(h, io_or_value);
    };
    for (size_t pi = 0; pi < np; ++pi) {
        const auto& [range, pd] = desc.mesh_programs[pi];
        ttsl::hash::hash_combine(h, range);
        ttsl::hash::hash_combine(h, ttnn::operations::generic::compute_program_descriptor_hash(pd));
        size_t r = 0;  // (p->rt[pi] is in kernel, core, arg order: the walk below meets its slots in turn)
        for (uint32_t k = 0; k < pd.kernels.size(); ++k) {
            ttsl::hash::hash_combine(h, pd.kernels[k].runtime_args.size());
            for (const auto& [core, args] : pd.kernels[k].runtime_args) {
                ttsl::hash::hash_combine(h, core.x);
                ttsl::hash::hash_combine(h, core.y);
                ttsl::hash::hash_combine(h, args.size());
                for (uint32_t i = 0; i < args.size(); ++i) {
                    const auto& pr = r < p->rt[pi].size() ? p->rt[pi][r] : RtPatch{.kernel = ~0u};
                    const bool patched = pr.kernel == k && pr.x == core.x && pr.y == core.y && pr.arg == i;
                    slot(patched, patched ? pr.io : args[i]);
                    r += patched;
                }
            }
            ttsl::hash::hash_combine(h, pd.kernels[k].common_runtime_args.size());
            for (uint32_t i = 0; i < pd.kernels[k].common_runtime_args.size(); ++i) {
                const CommonRtPatch* cp = nullptr;
                for (const auto& c : p->crt[pi]) {
                    if (c.kernel == k && c.arg == i) {
                        cp = &c;
                        break;
                    }
                }
                slot(cp != nullptr, cp != nullptr ? cp->io : pd.kernels[k].common_runtime_args[i]);
            }
        }
        TT_FATAL(r == p->rt[pi].size(), "prepared_program: patch walk out of step ({} of {})", r, p->rt[pi].size());
        for (uint32_t c = 0; c < pd.cbs.size(); ++c) {
            const auto& cb = pd.cbs[c];
            const CbPatch* bp = nullptr;
            for (const auto& b : p->cbs[pi]) {
                if (b.cb == c) {
                    bp = &b;
                    break;
                }
            }
            slot(bp != nullptr, bp != nullptr ? bp->io : (cb.buffer != nullptr ? cb.buffer->address() : 0u));
            ttsl::hash::hash_combine(h, cb.address_offset);
        }
    }
    p->hash = h;
    p->rt_of_io.assign(np, std::vector<std::vector<uint32_t>>(p->n_io));
    p->crt_of_io.assign(np, std::vector<std::vector<uint32_t>>(p->n_io));
    for (size_t pi = 0; pi < np; ++pi) {
        for (uint32_t r = 0; r < p->rt[pi].size(); ++r) {
            p->rt_of_io[pi][p->rt[pi][r].io].push_back(r);
        }
        for (uint32_t r = 0; r < p->crt[pi].size(); ++r) {
            p->crt_of_io[pi][p->crt[pi][r].io].push_back(r);
        }
    }
    p->desc = std::move(desc);
    return p;
}

Tensor run(const std::shared_ptr<const Prepared>& prepared, const std::vector<Tensor>& io_tensors) {
    TT_FATAL(prepared != nullptr, "prepared_program: no prepared program");
    TT_FATAL(!io_tensors.empty(), "prepared_program: io_tensors must hold at least the output (last)");
    using Op = PreparedProgramDeviceOperation;
    const Op::operation_attributes_t attributes{.prepared = prepared};
    const Op::tensor_args_t tensor_args{.io_tensors = io_tensors, .output_tensor = io_tensors.back()};
    return ttnn::device_operation::launch<Op>(attributes, tensor_args);
}

}  // namespace ttnn::operations::bringup::prepared_program
