// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_program_device_operation.hpp"

#include "ttnn/operation.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::chunk_program {

using namespace tt::tt_metal;

namespace {

// io tensors and values a kind takes (chunk_program.hpp)
void check_counts(const ChunkPlan& p, size_t n_io, size_t n_values) {
    const auto& k = p.consts;
    switch (p.kind) {
        case ChunkKind::PAGE_COPY:
            TT_FATAL(
                n_io == 2 && n_values == 6, "chunk_program page_copy: io [src, dst], values [s0, d0, n, nb, ss, ds]");
            break;
        case ChunkKind::GLUED_ROWS:
            TT_FATAL(
                n_io == 2 * k.at(5) && n_values == p.rows * k.at(4),
                "chunk_program glued_rows: io (table, out) x {}, values {} a mesh row",
                k.at(5),
                k.at(4));
            break;
        case ChunkKind::IDS_REORDER:
            TT_FATAL(n_io == 2 && n_values == 3, "chunk_program ids_reorder: io [src, dst], values [iu, cs, valid]");
            break;
        case ChunkKind::GLUED_IDS:
            TT_FATAL(n_io == 2 && n_values == 0, "chunk_program glued_ids: io [src, dst]");
            break;
        case ChunkKind::DERIVE: {
            size_t outs = 0, vals = 1;
            for (const auto& s : p.groups) {
                outs += s.at(11);
                vals += s.at(13);
            }
            TT_FATAL(
                n_io == 1 + outs && n_values == vals,
                "chunk_program derive: io [header, outputs ({})], values [header offset, section values ({})]",
                outs,
                vals - 1);
            break;
        }
        case ChunkKind::OWNED_SEND:
            TT_FATAL(
                n_io == 2 && n_values == (3 * k.at(6)) + k.at(7),
                "chunk_program owned_send: io [out, a record buffer], values (buffers, lo, hi, map words)");
            break;
        case ChunkKind::PAGE_MULTI:
            TT_FATAL(
                k.at(3) >= 1 && n_io == 2 * k.at(3) && n_values == 3 * k.at(3),
                "chunk_program page_multi: io (src, dst) x {}, values (s0, d0, n) x {}",
                k.at(3),
                k.at(3));
            break;
    }
}

}  // namespace

ChunkProgramDeviceOperation::program_factory_t ChunkProgramDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return ChunkProgramMeshWorkloadFactory{};
}

void ChunkProgramDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(attrs.plan != nullptr, "chunk_program: no plan");
    const ChunkPlan& p = *attrs.plan;
    const auto& io = tensor_args.io;
    check_counts(p, io.size(), attrs.values.size());
    auto* mesh = io.at(0).device();
    TT_FATAL(mesh != nullptr, "chunk_program: io must be on a device");
    TT_FATAL(
        mesh->shape().dims() == 2 && mesh->shape()[0] == p.rows && mesh->shape()[1] == p.cols,
        "chunk_program: the plan was made for a {}x{} mesh",
        p.rows,
        p.cols);
    for (const auto& t : io) {
        TT_FATAL(t.device() == mesh, "chunk_program: every io tensor on one mesh");
        TT_FATAL(t.buffer() != nullptr, "chunk_program: every io tensor allocated");
    }
    if (p.kind == ChunkKind::DERIVE) {
        // the kernel's one accessor config for every output (derive.cpp x_args)
        for (size_t i = 1; i < io.size(); ++i) {
            const auto& mc = io[i].memory_config();
            TT_FATAL(
                mc.buffer_type() == BufferType::DRAM && mc.memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "chunk_program derive: every output DRAM interleaved");
        }
    }
    if (p.kind == ChunkKind::PAGE_MULTI) {
        // the kernel's one accessor config for every source and one for every destination (page_multi.cpp)
        auto ct = [](const Tensor& t) {
            std::vector<uint32_t> v;
            TensorAccessorArgs(*t.buffer()).append_to(v);
            return v;
        };
        const auto s0 = ct(io.at(0)), d0 = ct(io.at(1));
        for (size_t i = 0; i < io.size(); ++i) {
            TT_FATAL(
                io[i].buffer()->page_size() == p.consts.at(0),
                "chunk_program page_multi: every source and destination of one page size {}",
                p.consts.at(0));
            TT_FATAL(
                ct(io[i]) == (i % 2 == 0 ? s0 : d0),
                "chunk_program page_multi: every source (destination) with the first one's accessor config");
        }
    }
    if (p.kind == ChunkKind::GLUED_ROWS) {
        for (size_t i = 0; i < io.size(); ++i) {
            TT_FATAL(
                io[i].buffer()->page_size() == p.consts.at(0),
                "chunk_program glued_rows: every table and output of one page size {}",
                p.consts.at(0));
        }
    }
}

void ChunkProgramDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    check_counts(*attrs.plan, tensor_args.io.size(), attrs.values.size());
}

ttsl::hash::hash_t ChunkProgramDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // The plan and what each io tensor's accessor compile-time args and page size hold: its buffer type, memory
    // layout and page size, and for a sharded one its memory config and padded shape (FIX.32). Never an address or a
    // value.
    auto h = static_cast<ttsl::hash::hash_t>(attrs.plan->hash);
    h = ttsl::hash::hash_objects(h, tensor_args.io.size());
    for (const auto& t : tensor_args.io) {
        const auto& mc = t.memory_config();
        h = ttsl::hash::hash_objects(h, mc.buffer_type(), mc.memory_layout(), t.buffer()->page_size());
        if (mc.is_sharded()) {
            h = ttsl::hash::hash_objects(h, mc, t.padded_shape());
        }
    }
    return h;
}

ChunkProgramDeviceOperation::spec_return_value_t ChunkProgramDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    return tensor_args.io.at(0).tensor_spec();
}

ChunkProgramDeviceOperation::tensor_return_value_t ChunkProgramDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    return tensor_args.io.at(0);
}

}  // namespace ttnn::operations::chunk_program

namespace ttnn::prim {

Tensor chunk_program(
    const std::vector<Tensor>& io,
    const std::shared_ptr<const ttnn::operations::chunk_program::ChunkPlan>& plan,
    const std::vector<uint32_t>& values) {
    using OperationType = ttnn::operations::chunk_program::ChunkProgramDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.plan = plan, .values = values}, OperationType::tensor_args_t{.io = io});
}

}  // namespace ttnn::prim
