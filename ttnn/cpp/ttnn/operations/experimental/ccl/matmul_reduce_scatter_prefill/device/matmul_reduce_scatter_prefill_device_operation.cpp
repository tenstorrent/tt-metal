// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_prefill_device_operation.hpp"

#include <algorithm>
#include <initializer_list>
#include <string_view>

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill {

using namespace tt::tt_metal;

namespace {

template <typename T>
void refuse_unless(bool supported, std::string_view axis, const T& value, std::string_view allowed) {
    TT_FATAL(supported, "matmul_reduce_scatter_prefill: {}={} is not supported (supported: {})", axis, value, allowed);
}

bool is_one_of(DataType dtype, std::initializer_list<DataType> allowed) {
    return std::find(allowed.begin(), allowed.end(), dtype) != allowed.end();
}

// DRAM or L1: the reader's TensorAccessor addresses either bank kind
void check_interleaved_tile(const Tensor& t, std::string_view what) {
    const MemoryConfig& mc = t.memory_config();
    TT_FATAL(
        (mc.buffer_type() == BufferType::DRAM || mc.buffer_type() == BufferType::L1) &&
            mc.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "matmul_reduce_scatter_prefill: {} must be DRAM or L1 interleaved",
        what);
    TT_FATAL(t.layout() == Layout::TILE, "matmul_reduce_scatter_prefill: {} must be TILE layout", what);
    // the planner and the kernels assume 32 x 32 tiles of 16 x 16 faces in the default order
    const Tile& tile = t.tensor_spec().tile();
    TT_FATAL(
        tile == Tile() && !tile.get_transpose_within_face() && !tile.get_transpose_of_faces(),
        "matmul_reduce_scatter_prefill: {} must use the default 32x32 tile (no transpose), got {}",
        what,
        tile);
}

// On `mesh` and stored on every device of it: the programs launch on the coordinates the tensors occupy, and the
// schedule waits on every peer of the group.
void check_on_full_mesh(const Tensor& t, const distributed::MeshDevice* mesh, std::string_view what) {
    TT_FATAL(t.device() == mesh, "matmul_reduce_scatter_prefill: {} must be on the input_tensor's mesh", what);
    TT_FATAL(
        t.device_storage().get_coords().size() == mesh->num_devices(),
        "matmul_reduce_scatter_prefill: {} must be stored on every device of the mesh ({} of {})",
        what,
        t.device_storage().get_coords().size(),
        mesh->num_devices());
}

// The same per-bank range of the same bank kind: an allocation, or a view of one (a view wraps the address in a buffer
// of its own, so comparing Buffer pointers misses it).
bool shares_memory(const Tensor& a, const Tensor& b) {
    const Buffer* x = a.buffer();
    const Buffer* y = b.buffer();
    return x->buffer_type() == y->buffer_type() && x->address() < y->address() + y->aligned_size_per_bank() &&
           y->address() < x->address() + x->aligned_size_per_bank();
}

// The output must not share a buffer with an operand: fin entries write output tiles while other cores still read
// A and W (and the resident operand is re-read across chunks), so no read-before-write order holds.
void validate_output(const MatmulReduceScatterPrefillParams& attrs, const MatmulReduceScatterPrefillInputs& t) {
    if (!t.output_tensor.has_value()) {
        return;
    }
    const Tensor& output = *t.output_tensor;
    check_on_full_mesh(output, t.input_tensor.device(), "output_tensor");
    const TensorSpec expected = MatmulReduceScatterPrefillDeviceOperation::compute_output_specs(attrs, t);
    TT_FATAL(
        output.logical_shape() == expected.logical_shape(),
        "matmul_reduce_scatter_prefill: output_tensor shape {} does not match {}",
        output.logical_shape(),
        expected.logical_shape());
    TT_FATAL(
        output.dtype() == expected.data_type(),
        "matmul_reduce_scatter_prefill: output_tensor dtype {} does not match {}",
        output.dtype(),
        expected.data_type());
    TT_FATAL(
        output.layout() == expected.layout(),
        "matmul_reduce_scatter_prefill: output_tensor layout {} does not match {}",
        output.layout(),
        expected.layout());
    TT_FATAL(
        output.memory_config() == expected.memory_config(),
        "matmul_reduce_scatter_prefill: output_tensor memory_config {} does not match {}",
        output.memory_config(),
        expected.memory_config());
    TT_FATAL(
        output.tensor_spec() == expected,
        "matmul_reduce_scatter_prefill: output_tensor spec {} does not match {}",
        output.tensor_spec(),
        expected);
    TT_FATAL(
        !shares_memory(output, t.input_tensor) && !shares_memory(output, t.weight),
        "matmul_reduce_scatter_prefill: output_tensor must not share a buffer with input_tensor or weight (in place is "
        "refused)");
}

}  // namespace

void check_support(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim,
    ttnn::ccl::Topology topology,
    const std::optional<int32_t>& num_links) {
    const auto& a_shape = input_tensor.logical_shape();
    const auto& w_shape = weight.logical_shape();
    TT_FATAL(a_shape.rank() >= 2, "matmul_reduce_scatter_prefill: input_tensor must have rank >= 2, got {}", a_shape);

    refuse_unless(
        is_one_of(input_tensor.dtype(), {DataType::BFLOAT16, DataType::BFLOAT8_B}),
        "dtype",
        input_tensor.dtype(),
        "BFLOAT16, BFLOAT8_B");
    refuse_unless(
        is_one_of(weight.dtype(), {DataType::BFLOAT16, DataType::BFLOAT8_B, DataType::BFLOAT4_B}),
        "weight_dtype",
        weight.dtype(),
        "BFLOAT16, BFLOAT8_B, BFLOAT4_B");
    const Layout layout = input_tensor.layout() != Layout::TILE ? input_tensor.layout() : weight.layout();
    refuse_unless(layout == Layout::TILE, "layout", layout, "TILE");
    const bool aligned = a_shape[-2] % tt::constants::TILE_HEIGHT == 0 &&
                         a_shape[-1] % tt::constants::TILE_WIDTH == 0 && w_shape[-1] % tt::constants::TILE_WIDTH == 0;
    refuse_unless(aligned, "alignment", aligned ? "tile_aligned" : "non_aligned", "tile_aligned");
    refuse_unless(cluster_axis == 0 || cluster_axis == 1, "cluster_axis", cluster_axis, "0, 1");
    refuse_unless(scatter_dim == -1 || scatter_dim == -2, "scatter_dim", scatter_dim, "-1, -2");
    refuse_unless(
        topology == ttnn::ccl::Topology::Linear || topology == ttnn::ccl::Topology::Ring,
        "topology",
        static_cast<int>(topology),
        "Linear, Ring");
    if (num_links.has_value()) {
        refuse_unless(*num_links == 1 || *num_links == 2, "num_links", *num_links, "1, 2");
    }
}

void check_operands(
    const Tensor& input_tensor, const Tensor& weight, const std::optional<MemoryConfig>& memory_config) {
    const auto& a_shape = input_tensor.logical_shape();
    const auto& w_shape = weight.logical_shape();
    bool leading_ones = true;
    for (int32_t i = 0; i + 2 < static_cast<int32_t>(a_shape.rank()); ++i) {
        leading_ones = leading_ones && a_shape[i] == 1;
    }
    TT_FATAL(
        a_shape.rank() >= 2 && a_shape.rank() <= 4 && leading_ones,
        "matmul_reduce_scatter_prefill: A must be rank 2-4 with leading dims 1, got {}",
        a_shape);
    TT_FATAL(w_shape.rank() == 2, "matmul_reduce_scatter_prefill: W must be (K, N), got {}", w_shape);
    TT_FATAL(a_shape[-1] == w_shape[0], "matmul_reduce_scatter_prefill: K mismatch, A {} vs W {}", a_shape, w_shape);
    check_interleaved_tile(input_tensor, "input_tensor");
    check_interleaved_tile(weight, "weight");
    TT_FATAL(
        !memory_config.has_value() || memory_config->memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "matmul_reduce_scatter_prefill: a sharded output memory_config is not supported");
}

uint32_t check_group(const Tensor& input_tensor, const Tensor& weight, int32_t cluster_axis, int32_t scatter_dim) {
    auto* mesh = input_tensor.device();
    TT_FATAL(mesh != nullptr, "matmul_reduce_scatter_prefill: input_tensor must be on a mesh device");
    check_on_full_mesh(input_tensor, mesh, "input_tensor");
    check_on_full_mesh(weight, mesh, "weight");
    const auto& mesh_shape = mesh->shape();
    TT_FATAL(mesh_shape.dims() == 2, "matmul_reduce_scatter_prefill: the mesh must be 2D, got {}", mesh_shape);
    const uint32_t G = mesh_shape[cluster_axis];
    TT_FATAL(G >= 2, "matmul_reduce_scatter_prefill: cluster_axis {} has {} device(s); need >= 2", cluster_axis, G);
    const uint32_t extent = scatter_dim == -2 ? input_tensor.logical_shape()[-2] : weight.logical_shape()[-1];
    const uint32_t extent_tile = scatter_dim == -2 ? tt::constants::TILE_HEIGHT : tt::constants::TILE_WIDTH;
    TT_FATAL(
        extent % (extent_tile * G) == 0,
        "matmul_reduce_scatter_prefill: scattered extent {} does not split into {} tile blocks",
        extent,
        G);
    return G;
}

MatmulReduceScatterPrefillDeviceOperation::program_factory_t
MatmulReduceScatterPrefillDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return MatmulReduceScatterPrefillProgramFactory{};
}

void MatmulReduceScatterPrefillDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    // The public entry checked these before resolving the plan; a direct prim caller has not.
    check_support(
        t.input_tensor,
        t.weight,
        static_cast<int32_t>(attrs.cluster_axis),
        attrs.scatter_dim,
        attrs.topology,
        std::nullopt);
    check_operands(t.input_tensor, t.weight, attrs.output_memory_config);
    check_group(t.input_tensor, t.weight, static_cast<int32_t>(attrs.cluster_axis), attrs.scatter_dim);
    const uint32_t staging_tiles = detail::staging_tiles(attrs.plan());
    const auto& staging_shard = t.staging.memory_config().shard_spec();
    TT_FATAL(
        t.staging.memory_config().buffer_type() == BufferType::L1 && staging_shard.has_value() &&
            staging_shard->shape[0] == staging_tiles * tt::constants::TILE_HEIGHT,
        "matmul_reduce_scatter_prefill: the staging tensor must be L1 sharded with {} tiles per core "
        "(allocate_staging)",
        staging_tiles);
    validate_output(attrs, t);
}

void MatmulReduceScatterPrefillDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    validate_output(attrs, t);
}

MatmulReduceScatterPrefillDeviceOperation::spec_return_value_t
MatmulReduceScatterPrefillDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    const uint32_t group = t.input_tensor.device()->shape()[attrs.cluster_axis];
    TT_FATAL(group > 0, "matmul_reduce_scatter_prefill: cluster_axis {} has no devices", attrs.cluster_axis);
    ttnn::Shape out_shape = t.input_tensor.logical_shape();
    const uint32_t n = t.weight.logical_shape()[-1];
    out_shape[-1] = n;
    if (attrs.scatter_dim == -2) {
        out_shape[-2] = out_shape[-2] / group;
    } else {
        out_shape[-1] = n / group;
    }
    return TensorSpec(
        out_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_memory_config));
}

std::vector<TensorTopology> MatmulReduceScatterPrefillDeviceOperation::compute_output_topologies(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    namespace common = ttnn::operations::ccl::common;
    // The matmul partial takes the union label of A and W, as a plain matmul would; the reduce-scatter consumes
    // cluster_axis, which becomes Shard{scatter_dim}. The ring rank is the coordinate along cluster_axis, so the
    // label's coordinates stand. Without an expressible label (nullopt, already reported) the partial's label stays.
    const TensorTopology& input_topology = t.input_tensor.tensor_topology();
    TensorTopology mm_topology = common::union_topology(
        {std::cref(input_topology), std::cref(t.weight.tensor_topology())}, input_topology.mesh_coords());
    std::optional<TensorTopology> output_topology = common::reduce_scatter_output_topology(
        mm_topology,
        attrs.cluster_axis,
        t.input_tensor.device()->shape(),
        static_cast<uint32_t>(t.input_tensor.logical_shape().rank()),
        attrs.scatter_dim);
    return {std::move(output_topology).value_or(std::move(mm_topology))};
}

MatmulReduceScatterPrefillDeviceOperation::tensor_return_value_t
MatmulReduceScatterPrefillDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (t.output_tensor.has_value()) {
        return *t.output_tensor;
    }
    return create_device_tensor(compute_output_specs(attrs, t), t.input_tensor.device());
}

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill

namespace ttnn::prim {

Tensor matmul_reduce_scatter_prefill(
    const Tensor& input_tensor,
    const Tensor& weight,
    const std::optional<Tensor>& output_tensor,
    const Tensor& staging,
    const ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill::MatmulReduceScatterPrefillParams&
        params) {
    using OperationType =
        ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill::MatmulReduceScatterPrefillDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{
            .input_tensor = input_tensor, .weight = weight, .output_tensor = output_tensor, .staging = staging});
}

}  // namespace ttnn::prim
