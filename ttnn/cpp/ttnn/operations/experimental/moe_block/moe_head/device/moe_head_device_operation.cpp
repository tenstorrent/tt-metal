// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_head_device_operation.hpp"
#include <tracy/Tracy.hpp>

#include <unordered_map>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

namespace moe_head_op_detail {

void check_dram(const Tensor& tensor, const char* what) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "moe_head: {} must be on device",
        what);
    TT_FATAL(tensor.memory_config().buffer_type() == BufferType::DRAM, "moe_head: {} must be in DRAM", what);
}

std::vector<ttnn::MeshCoordinate> coords_of(const ttnn::MeshCoordinateRangeSet& range_set) {
    std::vector<ttnn::MeshCoordinate> out;
    for (const auto& range : range_set.ranges()) {
        for (const auto& coord : range) {
            out.push_back(coord);
        }
    }
    return out;
}

// The HeadBuffers of a call: its tensors, the arena and the persistent words.
HeadBuffers head_buffers(const MoEHeadParams& attrs, const MoEHeadInputs& args) {
    HeadBuffers out;
    TT_FATAL(
        SOURCE_FIRST_WORD + attrs.words.size() <= kHeadMaxSources, "moe_head: {} persistent words", attrs.words.size());
    auto bind_source = [&](HeadSource source, const Tensor* tensor) {
        if (tensor != nullptr) {
            out.buffers[source] = tensor->buffer();
            out.addresses[source] = static_cast<uint32_t>(tensor->buffer()->address());
        }
    };
    bind_source(SOURCE_X, &args.x);
    bind_source(SOURCE_W, &args.w);
    bind_source(SOURCE_Y, &args.y);
    bind_source(SOURCE_ROWS, &args.rows);
    bind_source(SOURCE_DISPATCH_BUFFER, &args.dispatch_buffer);
    bind_source(SOURCE_ROUTE, &args.route);
    bind_source(SOURCE_ROUTER_W, &args.w_router);
    bind_source(SOURCE_ROUTER_BIAS, &args.router_bias);
    bind_source(SOURCE_ROUTER_INDEX, &args.router_index);
    bind_source(SOURCE_SLOTS, &args.slots);
    bind_source(SOURCE_COUNTS, &args.counts);
    bind_source(SOURCE_REGIONS, &args.regions);
    bind_source(SOURCE_TOKEN_INDEX, &args.token_index);
    bind_source(SOURCE_TABLE, &args.table);
    bind_source(SOURCE_PLAN, &args.plan);
    bind_source(SOURCE_ARENA, &attrs.arena);
    bind_source(SOURCE_START, args.actual_start ? &*args.actual_start : (args.metadata ? &*args.metadata : nullptr));
    out.padded_forward_metadata = args.metadata.has_value();
    bind_source(SOURCE_END, args.actual_end ? &*args.actual_end : nullptr);
    for (size_t i = 0; i < attrs.words.size(); ++i) {
        out.addresses[SOURCE_FIRST_WORD + i] = attrs.words[i];
    }
    const uint32_t row_blocks = attrs.config.shape.row_blocks;
    out.row_blocks = row_blocks;
    out.block = 0;
    out.split = attrs.split;
    if (row_blocks > 1) {
        // (a block's rows of an interleaved DRAM tensor l x stride a bank further: whole pages of every bank a block)
        auto stride = [&](const Tensor& tensor, const char* what) -> uint32_t {
            TT_FATAL(!tensor.is_sharded(), "moe_head: row blocks take {} interleaved", what);
            const auto* buffer = tensor.buffer();
            const uint32_t pages = buffer->num_pages(),
                           num_banks = buffer->allocator()->get_num_banks(buffer->buffer_type());
            TT_FATAL(
                pages % (row_blocks * num_banks) == 0,
                "moe_head: row blocks need {}'s pages ({}) a whole number of every bank's ({} banks) a block ({} "
                "blocks)",
                what,
                pages,
                num_banks,
                row_blocks);
            return static_cast<uint32_t>(pages / row_blocks / num_banks * buffer->aligned_page_size());
        };
        out.merged = true;
        out.x_stride = stride(args.x, "x");
        out.y_stride = stride(args.y, "y");
        const auto specs = head_words(*attrs.model);
        for (size_t i = 0; i < specs.size(); ++i) {
            if (specs[i].name == "row_blocks") {
                out.row_block_words = attrs.words.at(i);
            }
        }
        TT_FATAL(out.row_block_words != 0, "moe_head: no block words for row blocks in one call");
    }
    return out;
}

}  // namespace moe_head_op_detail
using namespace moe_head_op_detail;

ttsl::hash::hash_t MoEHeadDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& config = attrs.config;
    return tt::tt_metal::operation::hash_operation<MoEHeadDeviceOperation>(
        attrs.plan_key,
        config.layout.engine_rows,
        config.layout.row_tiles_per_core,
        config.layout.row_chunks,
        config.layout.k_step_tiles,
        config.layout.engine_columns,
        config.layout.compact_reduce_scatter_zones,
        config.layout.router_iterations,
        config.shape.rows,
        config.shape.accumulators,
        config.fidelity,
        config.w_dtype,
        config.l1_free,
        config.cb_floor,
        args.x.tensor_spec(),
        args.w.tensor_spec(),
        args.w_router.tensor_spec(),
        args.actual_start.has_value(),
        args.metadata.has_value(),
        attrs.config.shape.row_blocks,
        attrs.split);
}

MoEHeadDeviceOperation::program_factory_t MoEHeadDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoEHeadFactory{};
}

void MoEHeadDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    TT_FATAL(attrs.model != nullptr, "moe_head: the host model is missing");
    const auto shapes = head_tensor_shapes(*attrs.model);
    check_dram(args.x, "x");
    const auto& x_shape = args.x.logical_shape();
    const uint32_t row_blocks = attrs.config.shape.row_blocks;  // (the blocks the call runs: x and y hold them all)
    TT_FATAL(
        x_shape[-2] == attrs.config.shape.rows * row_blocks &&
            x_shape[-1] == attrs.config.shape.hidden_dim / attrs.config.shape.tp_size,
        "moe_head: x must be [.., {}, {}] a chip, got {}",
        attrs.config.shape.rows * row_blocks,
        attrs.config.shape.hidden_dim / attrs.config.shape.tp_size,
        x_shape);
    TT_FATAL(args.x.layout() == Layout::TILE && args.x.dtype() == DataType::BFLOAT16, "moe_head: x is bf16 TILE");
    for (const auto* tensor :
         {&args.w,
          &args.w_router,
          &args.router_bias,
          &args.router_index,
          &args.y,
          &args.rows,
          &args.dispatch_buffer,
          &args.route,
          &args.slots,
          &args.counts,
          &args.regions,
          &args.token_index,
          &args.table,
          &args.plan}) {
        check_dram(*tensor, "an argument");
    }
    TT_FATAL(
        args.actual_start.has_value() == args.actual_end.has_value(),
        "moe_head: a padded chunk takes both actual_start and actual_end");
    TT_FATAL(
        !(args.metadata.has_value() && args.actual_start.has_value()),
        "moe_head: a padded forward as actual_start / actual_end or as metadata, not both");
    TT_FATAL(
        attrs.config.shape.forward_chunks == 1 || !args.actual_start.has_value(),
        "moe_head: a forward of {} chunks takes the forward's metadata words (one chunk's actual_start / actual_end "
        "cannot describe it)",
        attrs.config.shape.forward_chunks);
    if (args.metadata) {
        check_dram(*args.metadata, "metadata");
        const uint32_t forward_chunks = attrs.config.shape.forward_chunks,
                       words = forward_chunks == 1 ? 3 : 2 + 3 * forward_chunks;
        TT_FATAL(
            args.metadata->dtype() == DataType::UINT32 && args.metadata->logical_volume() >= words &&
                !args.metadata->is_sharded(),
            "moe_head: the forward's metadata is {} uint32 words (DRAM interleaved), got {} of {} elements",
            words,
            args.metadata->dtype(),
            args.metadata->logical_volume());
        TT_FATAL(
            args.metadata->buffer()->aligned_page_size() * args.metadata->buffer()->num_pages() >= 4 * words &&
                args.metadata->buffer()->num_pages() == 1,
            "moe_head: the forward's metadata words must be one page");
    }
    for (const auto* optional_tensor : {&args.actual_start, &args.actual_end}) {
        if (*optional_tensor) {
            check_dram(**optional_tensor, "actual_start / actual_end");
            TT_FATAL(
                (*optional_tensor)->dtype() == DataType::UINT32 && (*optional_tensor)->logical_volume() == 1 &&
                    !(*optional_tensor)->is_sharded(),
                "moe_head: actual_start / actual_end are one uint32 each (DRAM interleaved), got {} of {} elements",
                (*optional_tensor)->dtype(),
                (*optional_tensor)->logical_volume());
        }
    }
    TT_FATAL(
        args.y.logical_shape()[-2] == shapes.y_rows * row_blocks && args.y.logical_shape()[-1] == shapes.y_columns,
        "moe_head: y must be [{}, {}]",
        shapes.y_rows * row_blocks,
        shapes.y_columns);
}

void MoEHeadDeviceOperation::validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}

MoEHeadDeviceOperation::spec_return_value_t MoEHeadDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args) {
    return args.y.tensor_spec();
}

MoEHeadDeviceOperation::tensor_return_value_t MoEHeadDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& args) {
    return args.y;
}

MoEHeadFactory::cached_mesh_workload_t MoEHeadFactory::create_mesh_workload(
    const MoEHeadParams& attrs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const MoEHeadInputs& args,
    Tensor& /*output*/) {
    ZoneScopedN("moe_head: create (miss)");
    auto built = build_head_workload(*attrs.model, head_buffers(attrs, args), coords_of(tensor_coords));
    TT_FATAL(
        built.addresses.size() == built.workload.programs.size(),
        "moe_head: {} programs, {} address lists",
        built.workload.programs.size(),
        built.addresses.size());
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    for (size_t i = 0; i < built.workload.programs.size(); ++i) {
        auto& [range, descriptor] = built.workload.programs[i];
        workload.add_program(range, tt::tt_metal::Program(descriptor));
        shared.emplace(range, shared_variables_t{.addresses = std::move(built.addresses[i])});
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void MoEHeadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached, const MoEHeadParams& attrs, const MoEHeadInputs& args, Tensor& /*output*/) {
    ZoneScopedN("moe_head: override (hit)");
    const HeadBuffers buffers = head_buffers(attrs, args);
    for (auto& [range, sv] : cached.shared_variables) {
        auto& program = cached.workload.get_programs().at(range);
        write_program_addresses(program, sv.addresses, buffers.addresses, buffers[SOURCE_ARENA]);
    }
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head

namespace ttnn::prim {

Tensor moe_block_head(
    const ttnn::operations::experimental::moe_block::moe_head::MoEHeadParams& params,
    ttnn::operations::experimental::moe_block::moe_head::MoEHeadInputs inputs) {
    using OperationType = ttnn::operations::experimental::moe_block::moe_head::MoEHeadDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(params, std::move(inputs));
}

}  // namespace ttnn::prim
