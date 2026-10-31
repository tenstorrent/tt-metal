// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rope_chain_device_operation.hpp"

#include <algorithm>
#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::rotary_embedding_indexed {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

constexpr auto kReader = "ttnn/ttnn/bringup/rotary_embedding_indexed/device/kernels/chain/reader_rope_chain.cpp";
constexpr auto kWriter = "ttnn/ttnn/bringup/rotary_embedding_indexed/device/kernels/chain/writer_rope_chain.cpp";
constexpr auto kCompute = "ttnn/ttnn/bringup/rotary_embedding_indexed/device/kernels/chain/compute_rope_chain.cpp";

constexpr uint32_t CB_IN = 0, CB_X = 1, CB_COS = 2, CB_SIN = 3, CB_TRANS = 4, CB_ROT = 5, CB_SIN_I = 6, CB_COS_I = 7,
                   CB_R = 8, CB_ASM = 9, CB_OUT_RM = 10;

struct Geometry {
    uint32_t B, H, S, W, WT, RT, RO, Ht;
    bool in_rm;
};

Geometry geometry(
    const RopeChainDeviceOperation::operation_attributes_t& args,
    const RopeChainDeviceOperation::tensor_args_t& tensor_args) {
    const auto& shape = tensor_args.input.logical_shape();
    Geometry g{};
    g.B = shape[0];
    g.S = shape[2];
    if (args.input_heads > 0) {
        g.H = args.input_heads;
        g.W = shape[3] / args.input_heads;
    } else {
        g.H = shape[1];
        g.W = shape[3];
    }
    g.WT = g.W / TILE_WIDTH;
    g.RT = args.rotary_dim / TILE_WIDTH;
    g.RO = args.rotary_offset / TILE_WIDTH;
    g.Ht = g.S / TILE_HEIGHT;
    g.in_rm = tensor_args.input.layout() == Layout::ROW_MAJOR;
    return g;
}

CoreCoord op_grid(const RopeChainDeviceOperation::operation_attributes_t& args, IDevice* device) {
    auto grid = device->compute_with_storage_grid_size();
    if (args.max_grid.has_value()) {
        grid.x = std::max<std::size_t>(1, std::min<std::size_t>(grid.x, args.max_grid->x));
        grid.y = std::max<std::size_t>(1, std::min<std::size_t>(grid.y, args.max_grid->y));
    }
    return grid;
}

void validate(
    const RopeChainDeviceOperation::operation_attributes_t& args,
    const RopeChainDeviceOperation::tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    for (const Tensor* t : {&input, &tensor_args.cos, &tensor_args.sin, &tensor_args.trans_mat}) {
        TT_FATAL(
            t->storage_type() == StorageType::DEVICE && t->is_allocated(),
            "rotary_embedding_indexed chain: every operand must be an allocated device tensor");
        TT_FATAL(t->device() == input.device(), "rotary_embedding_indexed chain: operands on one device");
        TT_FATAL(
            t->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "rotary_embedding_indexed chain: interleaved operands only");
    }
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 && tensor_args.cos.dtype() == DataType::BFLOAT16 &&
            tensor_args.sin.dtype() == DataType::BFLOAT16 && tensor_args.trans_mat.dtype() == DataType::BFLOAT16,
        "rotary_embedding_indexed chain: bf16 input, cos, sin and trans_mat");
    const auto& shape = input.logical_shape();
    TT_FATAL(shape.rank() == 4, "rotary_embedding_indexed chain: a rank-4 input");
    const Geometry g = geometry(args, tensor_args);
    if (args.input_heads > 0) {
        TT_FATAL(input.layout() == Layout::TILE, "rotary_embedding_indexed chain: input_heads needs a TILE input");
        TT_FATAL(shape[1] == 1, "rotary_embedding_indexed chain: input_heads needs a [B, 1, S, H W] input");
        TT_FATAL(
            shape[3] % (args.input_heads * TILE_WIDTH) == 0,
            "rotary_embedding_indexed chain: width {} is not input_heads {} x a tile-aligned head width",
            shape[3],
            args.input_heads);
    }
    TT_FATAL(g.S % TILE_HEIGHT == 0, "rotary_embedding_indexed chain: rows ({}) must be a multiple of 32", g.S);
    TT_FATAL(g.W % TILE_WIDTH == 0, "rotary_embedding_indexed chain: head width ({}) must be tile-aligned", g.W);
    TT_FATAL(
        args.rotary_dim > 0 && args.rotary_dim % TILE_WIDTH == 0 && args.rotary_offset % TILE_WIDTH == 0 &&
            args.rotary_offset + args.rotary_dim <= g.W,
        "rotary_embedding_indexed chain: rotary region [{}, {}) must be tile-aligned inside the head width {}",
        args.rotary_offset,
        args.rotary_offset + args.rotary_dim,
        g.W);
    const auto& cs = tensor_args.cos.logical_shape();
    TT_FATAL(
        tensor_args.cos.layout() == Layout::TILE && tensor_args.sin.layout() == Layout::TILE &&
            tensor_args.trans_mat.layout() == Layout::TILE,
        "rotary_embedding_indexed chain: TILE cos, sin and trans_mat");
    TT_FATAL(
        cs.rank() == 4 && cs[0] == 1 && cs[1] == 1 && cs[3] == args.rotary_dim && cs[2] >= g.S &&
            tensor_args.sin.logical_shape() == cs,
        "rotary_embedding_indexed chain: cos / sin must be [1, 1, >= {}, {}] (per-chip pre-sliced rows), got {}",
        g.S,
        args.rotary_dim,
        cs);
    TT_FATAL(
        tensor_args.trans_mat.padded_shape()[-2] == TILE_HEIGHT &&
            tensor_args.trans_mat.padded_shape()[-1] == TILE_WIDTH,
        "rotary_embedding_indexed chain: trans_mat must be one tile");
    TT_FATAL(
        args.output_dtype == DataType::BFLOAT16 || args.output_dtype == DataType::BFLOAT8_B ||
            args.output_dtype == DataType::BFLOAT4_B,
        "rotary_embedding_indexed chain: output_dtype bfloat16, bfloat8_b or bfloat4_b");
    TT_FATAL(args.group_heads >= 1 && g.H % args.group_heads == 0, "group_heads must divide the heads");
    if (args.output_layout == Layout::ROW_MAJOR) {
        TT_FATAL(
            args.output_dtype == DataType::BFLOAT16 && args.group_heads == 1,
            "rotary_embedding_indexed chain: a ROW_MAJOR output is bf16 [B, H, S, W] (no group_heads)");
    }
    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "rotary_embedding_indexed chain: interleaved output only");
    if (tensor_args.output.has_value()) {
        const auto& out = *tensor_args.output;
        TT_FATAL(
            out.storage_type() == StorageType::DEVICE && out.is_allocated(),
            "rotary_embedding_indexed chain: output_tensor must be an allocated device tensor");
        TT_FATAL(
            out.tensor_spec() == RopeChainDeviceOperation::compute_output_specs(args, tensor_args),
            "rotary_embedding_indexed chain: output_tensor spec {} is not the op's output spec {}",
            out.tensor_spec(),
            RopeChainDeviceOperation::compute_output_specs(args, tensor_args));
        TT_FATAL(
            out.buffer()->address() != input.buffer()->address(),
            "rotary_embedding_indexed chain: output_tensor must not be the input");
    }
}

}  // namespace

RopeChainDeviceOperation::program_factory_t RopeChainDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MeshWorkloadFactory{};
}

void RopeChainDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate(args, tensor_args);
}

void RopeChainDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Structure is hashed; the output_tensor (not hashed) is checked on every call.
    if (tensor_args.output.has_value()) {
        TT_FATAL(
            tensor_args.output->tensor_spec() == compute_output_specs(args, tensor_args),
            "rotary_embedding_indexed chain: output_tensor spec {} is not the op's output spec {}",
            tensor_args.output->tensor_spec(),
            compute_output_specs(args, tensor_args));
        TT_FATAL(
            tensor_args.output->buffer()->address() != tensor_args.input.buffer()->address(),
            "rotary_embedding_indexed chain: output_tensor must not be the input");
    }
}

RopeChainDeviceOperation::spec_return_value_t RopeChainDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Geometry g = geometry(args, tensor_args);
    const uint32_t G = args.group_heads;
    const ttnn::Shape shape({g.B, g.H / G, g.S, G * g.W});
    return TensorSpec(shape, TensorLayout(args.output_dtype, PageConfig(args.output_layout), args.output_mem_config));
}

RopeChainDeviceOperation::tensor_return_value_t RopeChainDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return *tensor_args.output;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t RopeChainDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto hash = tt::tt_metal::operation::hash_operation<RopeChainDeviceOperation>(
        args.input_heads,
        args.rotary_dim,
        args.rotary_offset,
        args.output_layout,
        args.output_dtype,
        args.group_heads,
        args.output_mem_config,
        args.compute_kernel_config,
        tensor_args.input.tensor_spec(),
        tensor_args.cos.tensor_spec(),
        tensor_args.sin.tensor_spec(),
        tensor_args.trans_mat.tensor_spec());
    if (args.max_grid.has_value()) {
        hash = ttsl::hash::hash_objects(hash, args.max_grid->x, args.max_grid->y);
    }
    return hash;
}

tt::tt_metal::ProgramDescriptor RopeChainDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    auto* device = input.device();
    const Geometry g = geometry(args, tensor_args);
    const bool out_rm = args.output_layout == Layout::ROW_MAJOR;
    const uint32_t G = args.group_heads;

    const auto grid = op_grid(args, device);
    const uint32_t gx = grid.x;
    const uint32_t blocks = g.B * g.H * g.Ht;
    const uint32_t rows_used = std::min<uint32_t>(grid.y, (blocks + gx - 1) / gx);
    const uint32_t ncores = gx * rows_used;
    const uint32_t per = blocks / ncores;
    const uint32_t extra = blocks % ncores;
    const CoreRangeSet all_cores(CoreRange({0, 0}, {gx - 1, rows_used - 1}));

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);
    (void)packer_l1_acc;
    const uint32_t dst_tiles = (fp32_dest_acc_en ? 4u : 8u) * (dst_full_sync_en ? 2u : 1u);

    const auto bf16 = tt::DataFormat::Float16_b;
    const uint32_t bf16_tile = tt::tile_size(bf16);
    const auto out_fmt = datatype_to_dataformat_converter(args.output_dtype);
    const uint32_t out_tile = tt::tile_size(out_fmt);
    const uint32_t row_bytes = g.W * 2;

    ProgramDescriptor desc;
    const auto add_cb = [&](uint32_t index, tt::DataFormat fmt, uint32_t page, uint32_t pages) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = page * pages,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{.buffer_index = index, .data_format = fmt, .page_size = page}}},
        });
    };
    if (g.in_rm) {
        add_cb(CB_IN, bf16, row_bytes, 2 * TILE_HEIGHT);
        add_cb(CB_X, bf16, bf16_tile, g.WT);
    } else {
        add_cb(CB_X, bf16, bf16_tile, 2 * g.WT);  // the reader's target (cb_in == cb_x)
    }
    add_cb(CB_COS, bf16, bf16_tile, 2 * g.RT);
    add_cb(CB_SIN, bf16, bf16_tile, 2 * g.RT);
    add_cb(CB_TRANS, bf16, bf16_tile, 1);
    for (const uint32_t cb : {CB_ROT, CB_SIN_I, CB_COS_I, CB_R}) {
        add_cb(cb, bf16, bf16_tile, g.RT);
    }
    if (out_rm) {
        add_cb(CB_ASM, bf16, bf16_tile, g.WT);
        add_cb(CB_OUT_RM, bf16, bf16_tile, 2 * g.WT);
    } else {
        add_cb(CB_ASM, out_fmt, out_tile, 2 * g.WT);
    }
    const uint32_t cb_in = g.in_rm ? CB_IN : CB_X;
    const uint32_t in_mode = g.in_rm ? 2u : (args.input_heads > 0 ? 1u : 0u);

    KernelDescriptor::CompileTimeArgs reader_ct = {
        cb_in, CB_COS, CB_SIN, CB_TRANS, in_mode, g.H, g.Ht, g.WT, g.RT, gx, g.in_rm ? row_bytes : bf16_tile};
    TensorAccessorArgs(input.buffer()).append_to(reader_ct);
    TensorAccessorArgs(tensor_args.cos.buffer()).append_to(reader_ct);
    TensorAccessorArgs(tensor_args.sin.buffer()).append_to(reader_ct);
    TensorAccessorArgs(tensor_args.trans_mat.buffer()).append_to(reader_ct);
    KernelDescriptor reader;
    reader.kernel_source = kReader;
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};
    reader.common_runtime_args = {
        input.buffer()->address(),                  // smuggled-rta-ok: patched per hit, override_runtime_arguments
        tensor_args.cos.buffer()->address(),        // smuggled-rta-ok: patched per hit, override_runtime_arguments
        tensor_args.sin.buffer()->address(),        // smuggled-rta-ok: patched per hit, override_runtime_arguments
        tensor_args.trans_mat.buffer()->address(),  // smuggled-rta-ok: patched per hit, override_runtime_arguments
        per,
        extra};

    KernelDescriptor::CompileTimeArgs writer_ct = {
        out_rm ? CB_OUT_RM : CB_ASM,
        static_cast<uint32_t>(out_rm),
        g.H,
        g.Ht,
        g.WT,
        G,
        out_rm ? row_bytes : out_tile,
        gx};
    TensorAccessorArgs(output.buffer()).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = kWriter;
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};
    writer.common_runtime_args = {
        output.buffer()->address(),  // smuggled-rta-ok: patched per hit, override_runtime_arguments
        per,
        extra};

    const uint32_t mbits = args.output_dtype == DataType::BFLOAT4_B ? 3u : 0u;
    const bool pack_precise = args.output_dtype == DataType::BFLOAT8_B;
    KernelDescriptor compute;
    compute.kernel_source = kCompute;
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {
        cb_in,
        CB_X,
        CB_COS,
        CB_SIN,
        CB_TRANS,
        CB_ROT,
        CB_SIN_I,
        CB_COS_I,
        CB_R,
        CB_ASM,
        CB_OUT_RM,
        static_cast<uint32_t>(g.in_rm),
        static_cast<uint32_t>(out_rm),
        g.WT,
        g.RT,
        g.RO,
        gx,
        mbits,
        dst_tiles};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        // bfloat8_b: one rounding from DEST at the group's shared exponent, as ttnn.typecast packs (DS41-137); a
        // define too, since the kernel binary's cache key leaves bfp8_pack_precise out
        .bfp8_pack_precise = pack_precise,
        .math_approx_mode = math_approx_mode};
    if (pack_precise) {
        compute.defines.emplace_back("CHAIN_BFP8_PACK_PRECISE", "1");
    }
    compute.common_runtime_args = {per, extra};

    desc.kernels.push_back(std::move(reader));   // handle 0
    desc.kernels.push_back(std::move(writer));   // handle 1
    desc.kernels.push_back(std::move(compute));  // handle 2
    return desc;
}

void RopeChainDeviceOperation::ProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*args*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*coord*/) {
    auto& reader = GetCommonRuntimeArgs(program, 0);
    reader[0] = tensor_args.input.buffer()->address();
    reader[1] = tensor_args.cos.buffer()->address();
    reader[2] = tensor_args.sin.buffer()->address();
    reader[3] = tensor_args.trans_mat.buffer()->address();
    auto& writer = GetCommonRuntimeArgs(program, 1);
    writer[0] = output.buffer()->address();
}

RopeChainDeviceOperation::MeshWorkloadFactory::cached_mesh_workload_t
RopeChainDeviceOperation::MeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output);
}

void RopeChainDeviceOperation::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output);
}

}  // namespace ttnn::operations::bringup::rotary_embedding_indexed

namespace ttnn::prim::bringup {

ttnn::Tensor rotary_embedding_indexed_chain(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    uint32_t input_heads,
    uint32_t rotary_dim,
    uint32_t rotary_offset,
    tt::tt_metal::Layout output_layout,
    tt::tt_metal::DataType output_dtype,
    uint32_t group_heads,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& output_tensor,
    std::optional<CoreCoord> max_grid) {
    using OperationType = ttnn::operations::bringup::rotary_embedding_indexed::RopeChainDeviceOperation;
    // The stock op's default compute config (HiFi4, fp32 DEST): the rotated values equal its own.
    auto kernel_config_val = init_device_compute_kernel_config(
        input.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, true, false, false);
    auto attrs = OperationType::operation_attributes_t{
        .input_heads = input_heads,
        .rotary_dim = rotary_dim,
        .rotary_offset = rotary_offset,
        .output_layout = output_layout,
        .output_dtype = output_dtype,
        .group_heads = std::max<uint32_t>(1, group_heads),
        .output_mem_config = memory_config.value_or(
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}),
        .compute_kernel_config = kernel_config_val,
        .max_grid = max_grid,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input = input, .cos = cos, .sin = sin, .trans_mat = trans_mat, .output = output_tensor};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim::bringup
