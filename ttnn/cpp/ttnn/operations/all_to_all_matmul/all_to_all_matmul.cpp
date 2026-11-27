// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_matmul.hpp"

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/vocabulary/mesh_coord.hpp>
#include <tt_stl/indestructible.hpp>

#include "device/all_to_all_matmul_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn {

namespace {

using operations::all_to_all_matmul::AllToAllMatmulParams;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;

std::string coord_str(const MeshCoordinate& c) { return fmt::format("({}, {})", c[0], c[1]); }

// The live-cluster link check of the Python's _check_caller_errors: every hop of every group (both directions) has
// num_links usable links, and a Ring has its wrap. It reads only the mesh and the fabric, so it runs on the mesh's
// first call per (mesh id, shape, cluster_axis, topology, num_links) and later calls read the verdict (empty: ok),
// with the fabric's max payload read on that same first call.
struct LinkCheck {
    std::string verdict;
    uint32_t payload_bytes = 0;
};
using LinkCheckKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, tt::tt_fabric::Topology, int32_t>;

std::string check_links(MeshDevice* mesh, uint32_t cluster_axis, tt::tt_fabric::Topology topology, int32_t num_links) {
    const uint32_t rows = mesh->shape()[0];
    const uint32_t cols = mesh->shape()[1];
    std::vector<std::vector<MeshCoordinate>> groups;
    if (cluster_axis == 1) {
        for (uint32_t r = 0; r < rows; ++r) {
            auto& grp = groups.emplace_back();
            for (uint32_t c = 0; c < cols; ++c) {
                grp.emplace_back(r, c);
            }
        }
    } else {
        for (uint32_t c = 0; c < cols; ++c) {
            auto& grp = groups.emplace_back();
            for (uint32_t r = 0; r < rows; ++r) {
                grp.emplace_back(r, c);
            }
        }
    }
    const bool ring = topology == tt::tt_fabric::Topology::Ring;
    for (const auto& grp : groups) {
        const size_t G = grp.size();
        std::vector<std::pair<MeshCoordinate, MeshCoordinate>> hops;
        for (size_t i = 0; i + 1 < G; ++i) {
            hops.emplace_back(grp[i], grp[i + 1]);
        }
        if (ring && G > 2) {
            hops.emplace_back(grp.back(), grp.front());
        }
        for (const auto& [a, b] : hops) {
            for (const auto& [s, d] : {std::pair{a, b}, std::pair{b, a}}) {
                const auto links = tt::tt_fabric::get_forwarding_link_indices(
                    *mesh,
                    mesh->get_fabric_node_id(s), mesh->get_fabric_node_id(d));
                const bool wrap = (s == grp.back() && d == grp.front()) || (s == grp.front() && d == grp.back());
                if (links.empty() && ring && wrap) {
                    return fmt::format(
                        "all_to_all_matmul: Ring along cluster_axis={} has no wrap link {} -> {}",
                        cluster_axis,
                        coord_str(s),
                        coord_str(d));
                }
                if (std::cmp_less(links.size(), num_links)) {
                    return fmt::format(
                        "all_to_all_matmul: hop {} -> {} has {} usable link(s), {} requested",
                        coord_str(s),
                        coord_str(d),
                        links.size(),
                        num_links);
                }
            }
        }
    }
    return {};
}

const LinkCheck& link_check(
    MeshDevice* mesh, uint32_t cluster_axis, tt::tt_fabric::Topology topology, int32_t num_links) {
    static ttsl::Indestructible<std::map<LinkCheckKey, LinkCheck>> cache;
    static std::mutex mutex;
    const LinkCheckKey key{mesh->id(), mesh->shape()[0], mesh->shape()[1], cluster_axis, topology, num_links};
    const std::lock_guard lock(mutex);
    auto it = cache.get().find(key);
    if (it == cache.get().end()) {
        it = cache.get()
                 .emplace(
                     key,
                     LinkCheck{
                         .verdict = check_links(mesh, cluster_axis, topology, num_links),
                         .payload_bytes =
                             static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes())})
                 .first;
    }
    return it->second;
}

// Caller errors of the Python's _check_caller_errors, in its order: ValueError (std::invalid_argument), and the
// payload check's RuntimeError.
// qk_rope_head_dim != 0: q-stem mode (input (1, 1, S, Hl * (K + R))).
void check_caller_errors(
    const Tensor& input,
    const Tensor& weight,
    int32_t cluster_axis,
    tt::tt_fabric::Topology topology,
    int32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    int32_t qk_rope_head_dim = 0,
    const std::optional<MemoryConfig>& rope_memory_config = std::nullopt) {
    const auto& x = input.logical_shape();
    const auto& w = weight.logical_shape();
    const bool qstem = qk_rope_head_dim != 0;
    if (x.rank() != 4) {
        throw std::invalid_argument(fmt::format(
            "all_to_all_matmul: input must be rank 4 ({}), got {}", qstem ? "1, 1, S, Hl*(K+R)" : "B, Hl, S, K", x));
    }
    if (w.rank() != 4) {
        throw std::invalid_argument(fmt::format("all_to_all_matmul: weight must be rank 4 (1, H, K, N), got {}", w));
    }
    if (cluster_axis != 0 && cluster_axis != 1) {
        throw std::invalid_argument(
            fmt::format("all_to_all_matmul: cluster_axis must be 0 or 1, got {}", cluster_axis));
    }
    MeshDevice* mesh = input.device();
    const uint32_t G = mesh->shape()[cluster_axis];
    if (G < 2) {
        throw std::invalid_argument(
            fmt::format("all_to_all_matmul: cluster_axis={} has extent {}; nothing to exchange", cluster_axis, G));
    }
    const uint32_t H = w[1];
    const uint32_t Kw = w[2];
    const uint32_t N = w[3];
    if (w[0] != 1) {
        throw std::invalid_argument(fmt::format("all_to_all_matmul: weight leading dim must be 1, got {}", w));
    }
    if (qstem) {
        // the q-stem input's own shape here; its nope view (1, Hl, S, K) goes through the checks below as usual
        if (qk_rope_head_dim < 0 || qk_rope_head_dim % tt::constants::TILE_WIDTH != 0) {
            throw std::invalid_argument(fmt::format(
                "all_to_all_matmul: qk_rope_head_dim must be a positive multiple of 32, got {}", qk_rope_head_dim));
        }
        if (H % G != 0) {
            throw std::invalid_argument(
                fmt::format("all_to_all_matmul: weight has {} heads, not a multiple of G={}", H, G));
        }
        const uint64_t width = static_cast<uint64_t>(H / G) * (Kw + static_cast<uint32_t>(qk_rope_head_dim));
        if (x[0] != 1 || x[1] != 1 || x[3] != width) {
            throw std::invalid_argument(fmt::format(
                "all_to_all_matmul: q-stem input must be (1, 1, S, Hl*(K+R)) = (1, 1, S, {}*({}+{})), got {}",
                H / G,
                Kw,
                qk_rope_head_dim,
                x));
        }
        if (x[2] % (G * tt::constants::TILE_HEIGHT) != 0) {
            throw std::invalid_argument(
                fmt::format("all_to_all_matmul: q-stem mode needs S/G to be a multiple of 32 (S={}, G={})", x[2], G));
        }
        if (topology != tt::tt_fabric::Topology::Ring) {
            throw std::invalid_argument("all_to_all_matmul: q-stem mode needs topology Ring");
        }
        if (rope_memory_config.has_value() && rope_memory_config->memory_layout() != TensorMemoryLayout::INTERLEAVED) {
            throw std::invalid_argument("all_to_all_matmul: rope_memory_config must be interleaved");
        }
    } else if (rope_memory_config.has_value()) {
        throw std::invalid_argument("all_to_all_matmul: rope_memory_config needs qk_rope_head_dim");
    }
    const uint32_t Hl = qstem ? H / G : x[1];
    const uint32_t S = x[2];
    const uint32_t K = qstem ? Kw : x[3];
    if (H != G * Hl) {
        throw std::invalid_argument(
            fmt::format("all_to_all_matmul: weight has {} heads, expected G*Hl = {}*{} = {}", H, G, Hl, G * Hl));
    }
    if (Kw != K) {
        throw std::invalid_argument(fmt::format("all_to_all_matmul: weight K {} != input K {}", Kw, K));
    }
    if (S % G != 0) {
        throw std::invalid_argument(fmt::format("all_to_all_matmul: S={} not divisible by G={}", S, G));
    }
    if (K % tt::constants::TILE_WIDTH != 0 || N % tt::constants::TILE_WIDTH != 0) {
        throw std::invalid_argument(fmt::format("all_to_all_matmul: K ({}) and N ({}) must be multiples of 32", K, N));
    }
    if (memory_config.has_value() && memory_config->memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        throw std::invalid_argument("all_to_all_matmul: memory_config must be interleaved");
    }
    if (num_links < 1) {
        throw std::invalid_argument(fmt::format("all_to_all_matmul: num_links must be >= 1, got {}", num_links));
    }
    const LinkCheck& links = link_check(mesh, static_cast<uint32_t>(cluster_axis), topology, num_links);
    if (!links.verdict.empty()) {
        throw std::invalid_argument(links.verdict);
    }
    if (weight.layout() != Layout::TILE || weight.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        throw std::invalid_argument("all_to_all_matmul: weight must be TILE, interleaved");
    }
    if (qstem) {
        if (input.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED) {
            throw std::invalid_argument("all_to_all_matmul: q-stem input must be DRAM or L1 interleaved");
        }
    } else if (input.memory_config().buffer_type() != BufferType::DRAM) {
        throw std::invalid_argument("all_to_all_matmul: input must be DRAM interleaved");
    }
    const uint32_t payload = links.payload_bytes;
    // The Python takes 2048 B for a non-TILE input (refused right after by the support check).
    const uint32_t tile_bytes =
        input.layout() == Layout::TILE ? static_cast<uint32_t>(input.buffer()->page_size()) : 2048u;
    if (payload / tile_bytes < 1) {
        TT_THROW("all_to_all_matmul: fabric max payload {} B is smaller than one tile", payload);
    }
}

// Validates the call and resolves its attributes (the Python's validate() and run()'s defaults).
AllToAllMatmulParams resolve_params(
    const Tensor& input,
    const Tensor& weight,
    int32_t cluster_axis,
    tt::tt_fabric::Topology topology,
    int32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    int32_t qk_rope_head_dim = 0,
    const std::optional<MemoryConfig>& rope_memory_config = std::nullopt) {
    TT_FATAL(is_device_tensor(input), "all_to_all_matmul: input_tensor must be a device tensor");
    check_caller_errors(
        input, weight, cluster_axis, topology, num_links, memory_config, qk_rope_head_dim, rope_memory_config);
    AllToAllMatmulParams params{
        .cluster_axis = static_cast<uint32_t>(cluster_axis),
        .topology = topology,
        .num_links = static_cast<uint32_t>(num_links),
        .output_memory_config = memory_config.has_value() ? *memory_config : ttnn::DRAM_MEMORY_CONFIG,
    };
    if (qk_rope_head_dim > 0) {
        params.qk_rope_head_dim = static_cast<uint32_t>(qk_rope_head_dim);
        params.rope_memory_config = rope_memory_config.has_value() ? *rope_memory_config : ttnn::DRAM_MEMORY_CONFIG;
    }
    if (compute_kernel_config.has_value()) {
        params.math_fidelity = compute_kernel_config->math_fidelity;
        params.fp32_dest_acc_en = compute_kernel_config->fp32_dest_acc_en;
        params.math_approx_mode = compute_kernel_config->math_approx_mode;
    }
    operations::all_to_all_matmul::check_support(params, input, weight);
    return params;
}

// Plans, makes the per-mesh semaphores, and launches: [prefix] or [prefix, rope_seq].
std::vector<Tensor> launch(
    const Tensor& input_tensor,
    const Tensor& weight,
    const AllToAllMatmulParams& params,
    const std::optional<Tensor>& scratch_tensor) {
    namespace a2a = operations::all_to_all_matmul;
    MeshDevice* mesh = input_tensor.device();
    const a2a::MeshTopology& mesh_topo = a2a::mesh_topology(mesh, params.cluster_axis, params.num_links);
    const a2a::Plan& plan = a2a::solve_plan(mesh_topo, a2a::plan_inputs_of(params, mesh_topo, input_tensor, weight));
    // Per-mesh global semaphores first, before this call allocates any L1 tensor (scratch, output).
    a2a::mesh_resources(mesh, mesh_topo, plan);
    if (scratch_tensor.has_value()) {
        return ttnn::prim::all_to_all_matmul(input_tensor, weight, scratch_tensor, params);
    }
    const std::optional<Tensor> scratch =
        create_device_tensor(a2a::scratch_spec(mesh_topo, plan, input_tensor.dtype()), mesh);
    return ttnn::prim::all_to_all_matmul(input_tensor, weight, scratch, params);
}

}  // namespace

Tensor all_to_all_matmul(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    tt::tt_fabric::Topology topology,
    int32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    const std::optional<Tensor>& scratch_tensor) {
    const AllToAllMatmulParams params =
        resolve_params(input_tensor, weight, cluster_axis, topology, num_links, memory_config, compute_kernel_config);
    return launch(input_tensor, weight, params, scratch_tensor).at(0);
}

std::tuple<Tensor, Tensor> all_to_all_matmul_qstem(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t qk_rope_head_dim,
    tt::tt_fabric::Topology topology,
    int32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    const std::optional<Tensor>& scratch_tensor,
    const std::optional<MemoryConfig>& rope_memory_config) {
    if (qk_rope_head_dim <= 0) {
        throw std::invalid_argument(
            fmt::format("all_to_all_matmul: qk_rope_head_dim must be positive, got {}", qk_rope_head_dim));
    }
    const AllToAllMatmulParams params = resolve_params(
        input_tensor,
        weight,
        cluster_axis,
        topology,
        num_links,
        memory_config,
        compute_kernel_config,
        qk_rope_head_dim,
        rope_memory_config);
    std::vector<Tensor> outs = launch(input_tensor, weight, params, scratch_tensor);
    return {outs.at(0), outs.at(1)};
}

Tensor all_to_all_matmul_scratch_tensor(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    tt::tt_fabric::Topology topology,
    int32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    const std::optional<int32_t>& qk_rope_head_dim) {
    namespace a2a = operations::all_to_all_matmul;
    const int32_t rope = qk_rope_head_dim.value_or(0);
    if (qk_rope_head_dim.has_value() && rope <= 0) {
        throw std::invalid_argument(fmt::format("all_to_all_matmul: qk_rope_head_dim must be positive, got {}", rope));
    }
    const AllToAllMatmulParams params = resolve_params(
        input_tensor, weight, cluster_axis, topology, num_links, memory_config, compute_kernel_config, rope);
    MeshDevice* mesh = input_tensor.device();
    const a2a::MeshTopology& mesh_topo = a2a::mesh_topology(mesh, params.cluster_axis, params.num_links);
    const a2a::Plan& plan = a2a::solve_plan(mesh_topo, a2a::plan_inputs_of(params, mesh_topo, input_tensor, weight));
    a2a::mesh_resources(mesh, mesh_topo, plan);
    return create_device_tensor(a2a::scratch_spec(mesh_topo, plan, input_tensor.dtype()), mesh);
}

}  // namespace ttnn
