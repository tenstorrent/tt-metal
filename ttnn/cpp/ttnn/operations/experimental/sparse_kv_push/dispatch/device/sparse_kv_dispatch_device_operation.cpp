// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_dispatch_device_operation.hpp"

#include <tt-metalium/utils/math.hpp>
#include <cstdlib>
#include <string>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"
#include "ttnn/operations/experimental/sparse_kv_push/common/sparse_kv_push_common.hpp"
#include "ttnn/operations/experimental/sparse_kv_push/common/kernels/sparse_push_constants.hpp"
#include "sparse_kv_dispatch_layout.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

KvDispatchMesh kv_dispatch_mesh(tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t num_links) {
    const auto shape = mesh_device->shape();
    TT_FATAL(shape.dims() == 2, "sparse_kv_dispatch needs a 2D mesh, got {}", shape);
    KvDispatchMesh m{.rows = shape[0], .cols = shape[1]};
    // An axis is a ring when chip (0, 0)'s neighbour the minus way is the axis' last chip.
    const auto dirs = fabric_directions(mesh_device, tt::tt_metal::distributed::MeshCoordinate(0, 0), num_links);
    for (const auto& d : dirs) {
        if (m.cols >= 3 && d.neighbor[0] == 0 && d.neighbor[1] == m.cols - 1) {
            m.wrap_x = true;
        }
        if (m.rows >= 3 && d.neighbor[1] == 0 && d.neighbor[0] == m.rows - 1) {
            m.wrap_y = true;
        }
    }
    return m;
}

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

void check_semaphores(const SparseKvDispatchParams& p, tt::tt_metal::distributed::MeshDevice* mesh_device) {
    TT_FATAL(
        p.fence_semaphore && p.arrived_semaphore && p.credit_semaphore && p.inbox_semaphore,
        "sparse_kv_dispatch needs its four persistent semaphores");
    for (const auto* s : {&*p.fence_semaphore, &*p.arrived_semaphore, &*p.credit_semaphore, &*p.inbox_semaphore}) {
        TT_FATAL(s->device() == mesh_device, "sparse_kv_dispatch semaphores must be on the input's mesh device");
        const auto [cores, buffer_type] = s->attribute_values();
        TT_FATAL(cores.contains(p.core_grid), "sparse_kv_dispatch semaphores must cover its core grid {}", p.core_grid);
    }
    const uint32_t f = p.fence_semaphore->address(), ar = p.arrived_semaphore->address(),
                   c = p.credit_semaphore->address(), i = p.inbox_semaphore->address();
    TT_FATAL(f != ar && f != c && f != i && ar != c && ar != i && c != i, "sparse_kv_dispatch semaphores must differ");
}

void validate_runtime(const SparseKvDispatchParams& p, const SparseKvDispatchInputs& t) {
    auto* mesh_device = t.input_tensor.device();
    for (const Tensor* x : {&t.input_tensor, &t.output_tensor, &t.table, &t.inbox, &t.scratch}) {
        TT_FATAL(x->buffer() != nullptr && x->device() == mesh_device, "sparse_kv_dispatch tensors must be on device");
    }
    check_semaphores(p, mesh_device);
    if (t.batch_index_tensor.has_value()) {
        TT_FATAL(
            t.batch_index_tensor->buffer() != nullptr && t.batch_index_tensor->device() == mesh_device,
            "sparse_kv_dispatch batch_index_tensor must be allocated on the input's mesh device");
    }
    // the fused plan (every call: buffers, host chunk start, the table it writes)
    TT_FATAL(
        t.plan_indices.has_value() || (!p.plan_chunk_start.has_value() && !t.plan_chunk_start_tensor.has_value()),
        "sparse_kv_dispatch: plan_chunk_start / plan_chunk_start_tensor need plan_indices");
    if (t.plan_indices.has_value()) {
        TT_FATAL(
            t.plan_indices->buffer() != nullptr && t.plan_indices->device() == mesh_device,
            "sparse_kv_dispatch plan_indices must be allocated on the input's mesh device");
        if (t.plan_chunk_start_tensor.has_value()) {
            TT_FATAL(
                t.plan_chunk_start_tensor->buffer() != nullptr && t.plan_chunk_start_tensor->device() == mesh_device,
                "sparse_kv_dispatch plan_chunk_start_tensor must be allocated on the input's mesh device");
        }
        TT_FATAL(
            p.plan_chunk_start.value_or(0) < p.max_rows,
            "sparse_kv_dispatch plan_chunk_start {} is past max_rows {}",
            p.plan_chunk_start.value_or(0),
            p.max_rows);
        const auto layout = PlanTableLayout::make(p.max_rows, p.stripe_chunk);
        TT_FATAL(
            t.table.buffer()->size() >= uint64_t{layout.image_words} * 4,
            "sparse_kv_dispatch fused plan writes {} B of table, the table has {} B",
            uint64_t{layout.image_words} * 4,
            t.table.buffer()->size());
    }
    // the kernels address rows as pages: one page a row, row-major
    for (const Tensor* x : {&t.input_tensor, &t.output_tensor}) {
        TT_FATAL(
            x->layout() == Layout::ROW_MAJOR && x->buffer()->num_pages() == x->padded_shape()[0] * x->padded_shape()[2],
            "sparse_kv_dispatch input and output must be ROW_MAJOR with one page a row");
    }
    // arriving slices land in the inbox while this chip's senders still read the table: they must not overlap
    const auto* tb = t.table.buffer();
    const auto* ib = t.inbox.buffer();
    TT_FATAL(
        tb->buffer_type() != ib->buffer_type() ||  // (an L1 inbox and a DRAM table cannot overlap)
            tb->address() + tb->aligned_size_per_bank() <= ib->address() ||
            ib->address() + ib->aligned_size_per_bank() <= tb->address(),
        "sparse_kv_dispatch table and inbox must not overlap");
    const uint32_t cache_batch = t.input_tensor.padded_shape()[0];
    if (t.batch_index_tensor.has_value()) {
        TT_FATAL(
            p.batch_slot_layer_idx < p.batch_slot_num_layers && cache_batch % p.batch_slot_num_layers == 0,
            "sparse_kv_dispatch batch slot args out of range");
    } else {
        TT_FATAL(
            p.batch_index < cache_batch,
            "sparse_kv_dispatch batch_index {} >= cache batch {}",
            p.batch_index,
            cache_batch);
    }
}

void validate_config(const SparseKvDispatchParams& p, const SparseKvDispatchInputs& t) {
    if (p.active_from != 0) {
        const char* design = std::getenv("KVD_DESIGN");
        TT_FATAL(design != nullptr && std::string(design) == "3", "sparse_kv_dispatch active_from needs KVD_DESIGN=3");
        // the kernels read the plan header from page 0 in DRAM bank 0
        TT_FATAL(
            t.table.buffer() != nullptr && t.table.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM &&
                t.table.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "sparse_kv_dispatch active_from needs an interleaved DRAM table");
        TT_FATAL(
            p.stripe_chunk > 0 && p.active_from % p.stripe_chunk == 0 && p.active_from < p.max_rows,
            "sparse_kv_dispatch active_from {} must be a multiple of stripe_chunk {} below max_rows {}",
            p.active_from,
            p.stripe_chunk,
            p.max_rows);
    }
    auto* mesh_device = t.input_tensor.device();
    const uint32_t n = mesh_device->shape().mesh_size();
    validate_stripe_chunk(p.stripe_chunk, p.max_rows);
    TT_FATAL(n <= 32, "sparse_kv_dispatch masks hold at most 32 chips, the mesh has {}", n);
    TT_FATAL(
        p.max_rows % (n * p.stripe_chunk) == 0,
        "sparse_kv_dispatch max_rows must be whole slabs ({})",
        n * p.stripe_chunk);
    TT_FATAL(p.max_rows <= kvd::kMaxRows, "sparse_kv_dispatch supports at most {} rows", kvd::kMaxRows);
    TT_FATAL(p.num_links >= 1 && p.num_links <= 2, "sparse_kv_dispatch supports 1 or 2 links");
    TT_FATAL(p.planners >= 1 && p.planners <= 64, "sparse_kv_dispatch planners must be 1..64");
    TT_FATAL(
        !t.rank_table.has_value() || (!t.plan_indices.has_value() && p.active_from == 0),
        "sparse_kv_dispatch rank_table (compact landing) does not combine with the fused plan or active_from");
    TT_FATAL(p.part <= 2, "sparse_kv_dispatch part must be 0 (ALL), 1 (PREFIX) or 2 (REST), got {}", p.part);
    TT_FATAL(p.landing_slots >= 1, "sparse_kv_dispatch needs at least one landing slot");
    TT_FATAL(
        p.lanes_per_link >= 1 && p.lanes_per_link <= kvd::kMaxLanes / p.num_links,
        "sparse_kv_dispatch supports at most {} lanes (links x lanes per link)",
        kvd::kMaxLanes);
    const uint32_t lanes = p.num_links * p.lanes_per_link;
    const uint32_t mux_cores = p.lanes_per_link > 1 ? kvd::kDirections * p.num_links : 0;
    TT_FATAL(
        p.core_grid.num_cores() >= kvd::kDirections * lanes + mux_cores + p.planners,
        "sparse_kv_dispatch needs {} cores ({} ports + {} mux + {} planners), core_grid has {}",
        kvd::kDirections * lanes + mux_cores + p.planners,
        kvd::kDirections * lanes,
        mux_cores,
        p.planners,
        p.core_grid.num_cores());
    if (t.plan_indices.has_value()) {  // the fused plan: as sparse_push_plan's own checks
        const auto& idx = *t.plan_indices;
        TT_FATAL(idx.dtype() == DataType::UINT32, "sparse_kv_dispatch plan_indices must be UINT32");
        TT_FATAL(idx.layout() == Layout::ROW_MAJOR, "sparse_kv_dispatch plan_indices must be row major");
        TT_FATAL(
            idx.padded_shape().rank() == 4 && idx.padded_shape()[0] == 1 && idx.padded_shape()[1] == 1,
            "sparse_kv_dispatch plan_indices must be [1, 1, rows, topk]");
        TT_FATAL(
            idx.padded_shape()[3] % kPlanIdxAlign == 0,
            "sparse_kv_dispatch plan_indices topk must be a multiple of {}",
            kPlanIdxAlign);
        TT_FATAL(
            idx.memory_config().buffer_type() == BufferType::DRAM &&
                idx.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "sparse_kv_dispatch plan_indices must be interleaved DRAM");
        TT_FATAL(idx.device() == t.input_tensor.device(), "sparse_kv_dispatch plan_indices must be on the op's mesh");
        TT_FATAL(
            p.max_rows % ::sparse_push_plan::kChunkRows == 0,
            "sparse_kv_dispatch fused plan: max_rows must be whole {}-row chunks",
            ::sparse_push_plan::kChunkRows);
        TT_FATAL(
            !(p.plan_chunk_start.has_value() && t.plan_chunk_start_tensor.has_value()),
            "sparse_kv_dispatch: give plan_chunk_start or plan_chunk_start_tensor, not both");
        if (t.plan_chunk_start_tensor.has_value()) {
            TT_FATAL(
                t.plan_chunk_start_tensor->dtype() == DataType::UINT32 &&
                    t.plan_chunk_start_tensor->memory_config().buffer_type() == BufferType::DRAM &&
                    t.plan_chunk_start_tensor->logical_volume() == 1,
                "sparse_kv_dispatch plan_chunk_start_tensor must be a one-element UINT32 DRAM tensor");
        }
    }
    const auto routes = fabric_routes(mesh_device, p.num_links);
    TT_FATAL(routes.ok(), "sparse_kv_dispatch cannot route: {}", routes.problem);
    const auto mesh = kv_dispatch_mesh(mesh_device, p.num_links);
    TT_FATAL(mesh.cols >= 2 && mesh.rows >= 2, "sparse_kv_dispatch needs a 2D mesh of at least 2x2");
    // entry fields: hops are 3 bits (an axis of at most 8 chips), generations 4 bits
    TT_FATAL(
        mesh.cols <= 8 && mesh.rows <= 8 && mesh.gmax() <= 15, "sparse_kv_dispatch supports axes of up to 8 chips");
    // every chip's fabric directions must reach distinct neighbours (a 2-chip torus axis has two links to one chip,
    // which the slice exchange's first-hop ports cannot tell apart)
    for (const auto& c : tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape())) {
        const auto dirs = fabric_directions(mesh_device, c, p.num_links);
        for (size_t i = 0; i < dirs.size(); ++i) {
            for (size_t j = i + 1; j < dirs.size(); ++j) {
                TT_FATAL(
                    dirs[i].neighbor != dirs[j].neighbor,
                    "sparse_kv_dispatch: chip {} reaches one neighbour two ways",
                    c);
            }
        }
    }
    TT_FATAL(p.landing_slots <= 64, "sparse_kv_dispatch supports at most 64 landing slots");
    TT_FATAL(
        t.input_tensor.padded_shape()[2] >= p.max_rows / n,
        "sparse_kv_dispatch input needs at least {} rows per chip and slot",
        p.max_rows / n);

    TT_FATAL(
        ttnn::operations::ccl::common::tensor_dim_shard_factor(t.input_tensor, 2) == n,
        "sparse_kv_dispatch input must be sharded over its rows across every chip");
    const auto& out_shape = t.output_tensor.padded_shape();
    TT_FATAL(
        out_shape[0] == 1 && out_shape[1] == 1 &&
            (t.rank_table.has_value() || (out_shape[2] % n == 0 && out_shape[2] >= p.max_rows)),
        "sparse_kv_dispatch output must be [1, 1, rows >= max_rows, width] with rows a multiple of the chips, got {}",
        out_shape);
    TT_FATAL(
        t.output_tensor.buffer()->aligned_page_size() == t.input_tensor.buffer()->aligned_page_size(),
        "sparse_kv_dispatch output and input row pages differ");
    const uint32_t slot = inbox_slot_bytes(p.max_rows, p.stripe_chunk, n);
    TT_FATAL(
        t.inbox.buffer()->num_pages() == n && t.inbox.buffer()->aligned_page_size() == slot &&
            t.table.buffer()->num_pages() == n && t.table.buffer()->aligned_page_size() == slot,
        "sparse_kv_dispatch table and inbox must be {} pages of {} B (owner-major plan table)",
        n,
        slot);
    const auto scratch =
        KvDispatchScratch::make(p.max_rows, p.stripe_chunk, n, p.planners, mesh.gmax(), p.num_links * p.lanes_per_link);
    TT_FATAL(
        t.scratch.buffer()->num_pages() >= scratch.pages &&
            t.scratch.buffer()->aligned_page_size() == kvd::kScratchPageBytes,
        "sparse_kv_dispatch scratch must hold {} pages of {} B",
        scratch.pages,
        kvd::kScratchPageBytes);
    if (t.batch_index_tensor.has_value()) {
        TT_FATAL(
            t.batch_index_tensor->dtype() == DataType::UINT32 && t.batch_index_tensor->logical_volume() == 1,
            "sparse_kv_dispatch batch_index_tensor must be one UINT32");
    }
    validate_runtime(p, t);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

using namespace CMAKE_UNIQUE_NAMESPACE;

ttsl::hash::hash_t SparseKvDispatchDeviceOperation::compute_program_hash(
    const SparseKvDispatchParams& p, const SparseKvDispatchInputs& t) {
    const auto batch_spec = t.batch_index_tensor.has_value()
                                ? std::optional<tt::tt_metal::TensorSpec>(t.batch_index_tensor->tensor_spec())
                                : std::nullopt;
    return tt::tt_metal::operation::hash_operation<SparseKvDispatchDeviceOperation>(
        ttsl::hash::hash_objects_with_default_seed(
            p.stripe_chunk,
            p.max_rows,
            p.num_links,
            p.rows_per_packet,
            p.landing_slots,
            p.planners,
            p.lanes_per_link,
            std::getenv("KVD_STATS") != nullptr,  // the debug-statistics kernels are a different program
            std::string(std::getenv("KVD_ABL") ? std::getenv("KVD_ABL") : ""),        // ablation kernels, likewise
            std::string(std::getenv("KVD_DESIGN") ? std::getenv("KVD_DESIGN") : ""),  // v3 (experiment)
            std::string(std::getenv("KVD3") ? std::getenv("KVD3") : ""),
            std::getenv("KVD3_OS") != nullptr,
            std::getenv("KVD3_GMAX_CLASSES") != nullptr,
            p.core_grid,
            p.fabric_config,
            t.input_tensor.device()->shape(),
            t.input_tensor.tensor_spec(),
            t.output_tensor.tensor_spec(),
            t.table.tensor_spec(),
            t.inbox.tensor_spec(),
            t.scratch.tensor_spec(),
            batch_spec,
            // the fused plan (kvd4): the indices' spec and the chunk start's form
            t.plan_indices.has_value() ? std::optional<tt::tt_metal::TensorSpec>(t.plan_indices->tensor_spec())
                                       : std::nullopt,
            p.plan_chunk_start.has_value(),
            p.active_from,
            p.part,
            p.skip_exchange,
            t.rank_table.has_value() ? std::optional<tt::tt_metal::TensorSpec>(t.rank_table->tensor_spec())
                                     : std::nullopt,
            t.plan_chunk_start_tensor.has_value()
                ? std::optional<tt::tt_metal::TensorSpec>(t.plan_chunk_start_tensor->tensor_spec())
                : std::nullopt,
            std::string(std::getenv("PLAN_STATS") ? "1" : "")));
}

void SparseKvDispatchDeviceOperation::validate_on_program_cache_miss(
    const SparseKvDispatchParams& p, const SparseKvDispatchInputs& t) {
    validate_config(p, t);
}

void SparseKvDispatchDeviceOperation::validate_on_program_cache_hit(
    const SparseKvDispatchParams& p, const SparseKvDispatchInputs& t) {
    validate_runtime(p, t);
}

SparseKvDispatchDeviceOperation::spec_return_value_t SparseKvDispatchDeviceOperation::compute_output_specs(
    const SparseKvDispatchParams&, const SparseKvDispatchInputs& t) {
    return t.output_tensor.tensor_spec();
}

SparseKvDispatchDeviceOperation::topology_return_value_t SparseKvDispatchDeviceOperation::compute_output_topologies(
    const SparseKvDispatchParams&, const SparseKvDispatchInputs& t) {
    return {t.output_tensor.tensor_topology()};
}

SparseKvDispatchDeviceOperation::tensor_return_value_t SparseKvDispatchDeviceOperation::create_output_tensors(
    const SparseKvDispatchParams&, const SparseKvDispatchInputs& t) {
    return t.output_tensor;
}

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::prim {

Tensor sparse_kv_dispatch(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& table,
    const Tensor& inbox,
    const Tensor& scratch,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    const GlobalSemaphore& fence_semaphore,
    const GlobalSemaphore& arrived_semaphore,
    const GlobalSemaphore& credit_semaphore,
    const GlobalSemaphore& inbox_semaphore,
    uint32_t num_links,
    uint32_t rows_per_packet,
    uint32_t landing_slots,
    uint32_t planners,
    uint32_t batch_index,
    const std::optional<Tensor>& batch_index_tensor,
    uint32_t batch_slot_num_layers,
    uint32_t batch_slot_layer_idx,
    const std::optional<CoreRangeSet>& core_grid,
    uint32_t lanes_per_link,
    const std::optional<Tensor>& plan_indices,
    std::optional<uint32_t> plan_chunk_start,
    const std::optional<Tensor>& plan_chunk_start_tensor,
    uint32_t active_from,
    uint32_t part,
    bool skip_exchange,
    const std::optional<Tensor>& rank_table) {
    using namespace ttnn::operations::experimental::sparse_kv_push;
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "sparse_kv_dispatch input must be on device");
    const auto grid = core_grid.value_or(mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, mesh_device->get_sub_device_ids().at(0)));
    return ttnn::device_operation::launch<SparseKvDispatchDeviceOperation>(
        SparseKvDispatchParams{
            .stripe_chunk = stripe_chunk,
            .max_rows = max_rows,
            .num_links = num_links,
            .rows_per_packet = rows_per_packet,
            .landing_slots = landing_slots,
            .planners = planners,
            .lanes_per_link = lanes_per_link,
            .core_grid = grid,
            .fabric_config = tt::tt_fabric::GetFabricConfig(),
            .fence_semaphore = fence_semaphore,
            .arrived_semaphore = arrived_semaphore,
            .credit_semaphore = credit_semaphore,
            .inbox_semaphore = inbox_semaphore,
            .batch_index = batch_index,
            .batch_slot_num_layers = batch_slot_num_layers,
            .batch_slot_layer_idx = batch_slot_layer_idx,
            .plan_chunk_start = plan_chunk_start,
            .active_from = active_from,
            .part = part,
            .skip_exchange = skip_exchange},
        SparseKvDispatchInputs{
            .input_tensor = input_tensor,
            .output_tensor = output_tensor,
            .table = table,
            .inbox = inbox,
            .scratch = scratch,
            .batch_index_tensor = batch_index_tensor,
            .plan_indices = plan_indices,
            .plan_chunk_start_tensor = plan_chunk_start_tensor,
            .rank_table = rank_table});
}

}  // namespace ttnn::prim
