// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_rms_norm.hpp"

#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/mesh/mesh_device.hpp>
#include <tt_stl/indestructible.hpp>

#include "device/all_gather_rms_norm_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental {

namespace {

namespace agrn = ::ttnn::operations::experimental::ccl::all_gather_rms_norm;
using tt::tt_metal::distributed::MeshDevice;

constexpr uint32_t kDefaultMaxLinks = 2;  // num_links=None resolves to min(usable links, 2)

void check_weight(const char* name, const Tensor& w, uint32_t W) {
    const auto& s = w.logical_shape();
    std::vector<uint32_t> want;
    if (w.layout() == Layout::TILE) {
        want = {1, 1, 1, W};
    } else if (w.layout() == Layout::ROW_MAJOR) {
        want = {1, 1, tt::div_up(W, tt::constants::TILE_WIDTH), tt::constants::TILE_WIDTH};
    } else {
        throw std::invalid_argument(
            fmt::format("all_gather_rms_norm: {} layout {} is neither TILE nor ROW_MAJOR", name, w.layout()));
    }
    bool ok = s.rank() == want.size();
    for (uint32_t i = 0; ok && i < want.size(); ++i) {
        ok = s[static_cast<int32_t>(i)] == want[i];
    }
    if (!ok) {
        throw std::invalid_argument(fmt::format("all_gather_rms_norm: {} shape {} != {} for W={}", name, s, want, W));
    }
}

// Per-mesh: the usable links of the group hops (walks the fabric once per mesh / shape / axis / ring).
std::optional<uint32_t> usable_links_cached(MeshDevice& mesh, uint32_t cluster_axis, bool ring) {
    using Key = std::tuple<int, std::vector<uint32_t>, uint32_t, bool>;
    static ttsl::Indestructible<std::map<Key, std::optional<uint32_t>>> cache;
    const auto& shape = mesh.shape();
    std::vector<uint32_t> dims(shape.cbegin(), shape.cend());
    Key key{static_cast<int>(mesh.id()), std::move(dims), cluster_axis, ring};
    auto& map = cache.get();
    auto it = map.find(key);
    if (it == map.end()) {
        it = map.emplace(std::move(key), agrn::planner::usable_links(mesh, cluster_axis, ring)).first;
    }
    return it->second;
}

bool shares_memory(const Tensor& a, const Tensor& b) {
    const auto* ba = a.buffer();
    const auto* bb = b.buffer();
    if (ba->buffer_type() != bb->buffer_type()) {
        return false;
    }
    const uint64_t a0 = ba->address();  // smuggled-rta-ok: aliasing check of the call's tensors, never a kernel arg
    const uint64_t a1 = a0 + ba->aligned_size_per_bank();
    const uint64_t b0 = bb->address();  // smuggled-rta-ok: aliasing check of the call's tensors, never a kernel arg
    const uint64_t b1 = b0 + bb->aligned_size_per_bank();
    return a0 < b1 && b0 < a1;
}

}  // namespace

std::optional<Tensor> all_gather_rms_norm_scratch_tensor(const Tensor& input_tensor, uint32_t cluster_axis) {
    auto* mesh = input_tensor.device();
    TT_FATAL(mesh != nullptr && cluster_axis <= 1, "all_gather_rms_norm_scratch_tensor: bad input or cluster_axis");
    const auto geo = agrn::planner::derive_geometry(input_tensor.logical_shape(), mesh->shape()[cluster_axis]);
    if (geo.aligned()) {
        return std::nullopt;
    }
    return create_device_tensor(agrn::planner::staging_spec(geo, input_tensor.dtype()), mesh);
}

Tensor all_gather_rms_norm(
    const Tensor& input_tensor,
    int32_t cluster_axis,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    const std::string& norm_type,
    double epsilon,
    tt::tt_fabric::Topology topology,
    std::optional<int32_t> num_links,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor,
    const std::optional<Tensor>& scratch_tensor) {
    // ---- caller errors (Python ValueError), in the Python's order ----
    const auto& shape = input_tensor.logical_shape();
    if (shape.rank() < 2) {
        throw std::invalid_argument(fmt::format("all_gather_rms_norm: input rank must be >= 2, got {}", shape.rank()));
    }
    if (norm_type != "rms_norm" && norm_type != "layer_norm") {
        throw std::invalid_argument(fmt::format("all_gather_rms_norm: unknown norm_type '{}'", norm_type));
    }
    const bool is_ln = norm_type == "layer_norm";
    if (beta.has_value() && !gamma.has_value()) {
        throw std::invalid_argument("all_gather_rms_norm: beta requires gamma");
    }
    if (beta.has_value() && !is_ln) {
        throw std::invalid_argument("all_gather_rms_norm: beta is only valid with norm_type='layer_norm'");
    }
    // a Python int: a negative axis is refused here (ValueError), as the Python's `cluster_axis not in (0, 1)`
    if (cluster_axis < 0 || cluster_axis > 1) {
        throw std::invalid_argument(
            fmt::format("all_gather_rms_norm: cluster_axis must be 0 or 1, got {}", cluster_axis));
    }
    const auto axis = static_cast<uint32_t>(cluster_axis);
    if (!(epsilon >= 0.0) || std::isinf(epsilon)) {
        throw std::invalid_argument(
            fmt::format("all_gather_rms_norm: epsilon must be finite and >= 0, got {}", epsilon));
    }
    MeshDevice* mesh = input_tensor.device();
    TT_FATAL(mesh != nullptr, "all_gather_rms_norm: input_tensor must be on a mesh device");
    const uint32_t G = mesh->shape()[axis];
    if (G < 2) {
        throw std::invalid_argument(fmt::format("all_gather_rms_norm: cluster_axis {} has size {}", axis, G));
    }
    const uint32_t W = G * shape[-1];
    if (gamma.has_value()) {
        check_weight("gamma", *gamma, W);
    }
    if (beta.has_value()) {
        check_weight("beta", *beta, W);
        if (beta->dtype() != gamma->dtype() || beta->layout() != gamma->layout()) {
            throw std::invalid_argument("all_gather_rms_norm: beta must have gamma's dtype and layout");
        }
    }

    agrn::AllGatherRmsNormParams params{
        .cluster_axis = axis,
        .is_layer_norm = is_ln,
        .epsilon = epsilon,
        .topology = topology,
        .num_links = 1,  // resolved below, after the SUPPORTED check
        .math_fidelity = compute_kernel_config.has_value() ? compute_kernel_config->math_fidelity
                                                           : tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = compute_kernel_config.has_value() ? compute_kernel_config->fp32_dest_acc_en : true,
        .math_approx_mode = compute_kernel_config.has_value() ? compute_kernel_config->math_approx_mode : false,
        .output_memory_config =
            output_tensor.has_value() ? output_tensor->memory_config() : memory_config.value_or(DRAM_MEMORY_CONFIG),
    };
    const agrn::AllGatherRmsNormInputs inputs{
        .input = input_tensor, .gamma = gamma, .beta = beta, .output = output_tensor, .scratch = scratch_tensor};
    // SUPPORTED (num_links=None is resolved below), EXCLUSIONS, placement
    if (num_links.has_value()) {
        if (*num_links < 1 || *num_links > 2) {
            throw agrn::UnsupportedAxisValue(
                fmt::format("all_gather_rms_norm: num_links={} not in SUPPORTED [1, 2]", *num_links));
        }
        params.num_links = static_cast<uint32_t>(*num_links);
    }
    agrn::check_support(params, inputs);
    if (output_tensor.has_value()) {
        ttnn::Shape want = shape;
        want[-1] = W;
        if (output_tensor->logical_shape() != want || output_tensor->dtype() != input_tensor.dtype() ||
            output_tensor->layout() != input_tensor.layout()) {
            throw std::invalid_argument(fmt::format(
                "all_gather_rms_norm: output_tensor must be {} {} {}, got {} {} {}",
                want,
                input_tensor.dtype(),
                input_tensor.layout(),
                output_tensor->logical_shape(),
                output_tensor->dtype(),
                output_tensor->layout()));
        }
    }

    // ---- live fabric ----
    const bool ring = topology == tt::tt_fabric::Topology::Ring && G > 2;
    const std::optional<uint32_t> links = usable_links_cached(*mesh, axis, ring);
    if (!links.has_value() || *links == 0) {
        throw std::invalid_argument(fmt::format(
            "all_gather_rms_norm: topology {} along cluster_axis {} needs a link on every hop",
            static_cast<int>(topology),
            axis));
    }
    if (num_links.has_value() && params.num_links > *links) {
        throw std::invalid_argument(
            fmt::format("all_gather_rms_norm: num_links={}, usable links per hop: {}", *num_links, *links));
    }
    if (!num_links.has_value()) {
        params.num_links = std::min(*links, kDefaultMaxLinks);
    }

    // persistent per-mesh semaphores first, before this call allocates any L1 tensor
    const auto& mesh_plan = agrn::mesh_plan_of(params, input_tensor);
    (void)agrn::persistent_semaphores(*mesh, mesh_plan);

    // ---- R_U staging scratch ----
    std::optional<Tensor> scratch = scratch_tensor;
    const auto geo = agrn::planner::derive_geometry(shape, G);
    if (scratch.has_value()) {
        TT_FATAL(scratch->device() == mesh, "all_gather_rms_norm: scratch_tensor must be on the input's mesh");
        bool aliased = shares_memory(*scratch, input_tensor) ||
                       (output_tensor.has_value() && shares_memory(*scratch, *output_tensor));
        aliased = aliased || (gamma.has_value() && shares_memory(*scratch, *gamma)) ||
                  (beta.has_value() && shares_memory(*scratch, *beta));
        TT_FATAL(!aliased, "all_gather_rms_norm: scratch_tensor shares memory with an input or the output");
    } else if (!geo.aligned()) {
        scratch = create_device_tensor(agrn::planner::staging_spec(geo, input_tensor.dtype()), mesh);
    }
    return ttnn::prim::all_gather_rms_norm(input_tensor, gamma, beta, output_tensor, scratch, params);
}

}  // namespace ttnn::experimental
