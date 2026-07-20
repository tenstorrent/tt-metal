// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <bit>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/types.hpp"
#include "rand_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::rand {

using namespace tt;
using namespace tt::tt_metal;

namespace {

// Metal 2.0 op-local kernels: named resource bindings (dfb::/args::/tensor::) instead of positional
// compile/runtime args. See writer_rand.cpp / compute_rand.cpp for the accessor names referenced below.
constexpr const char* WRITER_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/writer_rand.cpp";
constexpr const char* COMPUTE_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/compute_rand.cpp";

// Work split + per-device seed offset, shared by create_descriptor (cache miss) and
// override_runtime_arguments (cache hit) so both derive the identical core list and seed offset.
struct RandWorkSplit {
    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
    uint32_t device_seed_offset = 0;
};

RandWorkSplit compute_rand_work_split(
    const RandDeviceOperation::operation_attributes_t& attrs,
    RandDeviceOperation::tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    auto grid = output.device()->compute_with_storage_grid_size();
    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    auto cores = grid_to_cores(num_cores, grid.x, grid.y);

    const ttnn::MeshCoordinate mesh_coordinate =
        mesh_dispatch_coordinate.value_or(ttnn::MeshCoordinate::zero_coordinate(attrs.device->shape().dims()));
    uint32_t device_seed_offset = 0;
    const auto& shard_mask = attrs.mesh_dim_is_sharded;
    if (!shard_mask.empty()) {
        const auto& mesh_shape = attrs.device->shape();
        size_t shard_linear_idx = 0;
        size_t shard_stride = 1;
        for (int i = static_cast<int>(shard_mask.size()) - 1; i >= 0; --i) {
            if (shard_mask[i]) {
                shard_linear_idx += mesh_coordinate[i] * shard_stride;
                shard_stride *= mesh_shape[i];
            }
        }
        device_seed_offset = static_cast<uint32_t>(shard_linear_idx) * static_cast<uint32_t>(cores.size());
    }
    return {
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_group_1,
        units_per_core_group_2,
        std::move(cores),
        device_seed_offset};
}

// Per-core seed; shared so the miss-build and the hit-patch produce identical values. attrs.seed is
// always concrete here (seed == 0 is resolved to a random base at the op entry, see resolve_seed).
uint32_t rand_seed_for_core(
    const RandDeviceOperation::operation_attributes_t& attrs, int i, uint32_t device_seed_offset) {
    return attrs.seed + i + device_seed_offset;
}

}  // namespace

ProgramDescriptor RandDeviceOperation::RandProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         units_per_core_group_1,
         units_per_core_group_2,
         cores,
         device_seed_offset] = compute_rand_work_split(operation_attributes, output, mesh_dispatch_coordinate);
    const auto num_cores_total = cores.size();

    // rand.cpp always invokes ttnn::prim::uniform with DataType::FLOAT32 / Layout::TILE (any user dtype
    // or layout is applied afterward by a separate typecast/to_layout op), so this device op's output is
    // always FLOAT32 tiles. The Metal 2.0 writer streams intermed tiles straight to the output tensor
    // (no dtype-conversion staging CB), so only the FLOAT32 path exists here.
    TT_FATAL(
        output.dtype() == DataType::FLOAT32,
        "RandDeviceOperation: device output must be FLOAT32 (got {}); rand.cpp forces FLOAT32 then typecasts",
        output.dtype());
    const uint32_t intermed_tile_size = tile_size(tt::DataFormat::Float32);

    constexpr uint32_t intermed_num_tiles = 2;
    constexpr uint32_t intermed_cb_id = CBIndex::c_24;

    ProgramDescriptor desc;

    // Named intermed CB: referenced kernel-side as dfb::cb_intermed by both the compute (producer) and
    // writer (consumer) via their DFB bindings below.
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed_num_tiles * intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = intermed_cb_id,
            .data_format = tt::DataFormat::Float32,
            .page_size = intermed_tile_size,
        }}},
        .name = "intermed",
    });

    // Writer compile-time args: only the output tensor's static TensorAccessorArgs payload, starting at
    // word 0 (cta_offset below). The CB id is no longer a positional CTA — it flows through the DFB binding.
    KernelDescriptor::CompileTimeArgs writer_ct_args;
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    // Metal 2.0 named bindings for the writer:
    //  - dfb::cb_intermed  -> the "intermed" CB above
    //  - args::page_size   -> compile-time constant (bytes per tile), read as `constexpr get_arg(args::page_size)`
    //  - args::start_id / args::num_tiles -> per-core RTAs (order matches the runtime_args pushed below)
    //  - tensor::output    -> pass-through TensorBinding: static layout at cta_offset 0, base address in CRTA word 0
    writer_desc.named_compile_time_args = {{"page_size", intermed_tile_size}};
    writer_desc.dfb_bindings = {{.accessor_name = "cb_intermed", .cb_name = "intermed"}};
    writer_desc.runtime_arg_names = {"start_id", "num_tiles"};
    writer_desc.tensor_bindings = {
        {.accessor_name = "output", .cta_offset = 0, .addr_crta_offset = 0, .num_runtime_field_crta_words = 0}};
    // Output base address lives in the writer's single CRTA word (the tensor binding's address slot).
    // DYNAMIC across dispatches: re-applied on every cache hit by override_runtime_arguments().
    writer_desc.common_runtime_args = {static_cast<uint32_t>(output.buffer()->address())};
    writer_desc.runtime_args.reserve(num_cores_total);

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    // No positional CTAs: the intermed CB flows through the DFB binding.
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,  // if fp32_dest_acc_en set to false a precision error may occur which makes
                                   // generated number out of range [from, to)
        .dst_full_sync_en = false,
        .math_approx_mode = true,
    };
    // Metal 2.0 named bindings for the compute kernel:
    //  - dfb::cb_intermed -> the "intermed" CB
    //  - args::seed / from_bits / to_bits / start_id / num_tiles -> per-core RTAs (order matches runtime_args)
    compute_desc.dfb_bindings = {{.accessor_name = "cb_intermed", .cb_name = "intermed"}};
    compute_desc.runtime_arg_names = {"seed", "from_bits", "to_bits", "start_id", "num_tiles"};
    compute_desc.runtime_args.reserve(num_cores_total);

    const float eps = 1e-6f;
    const uint32_t from_bits = std::bit_cast<uint32_t>(operation_attributes.from);
    const uint32_t to_bits = std::bit_cast<uint32_t>(operation_attributes.to - eps);

    uint32_t tile_offset = 0;
    for (int i = 0; i < static_cast<int>(cores.size()); ++i) {
        const auto& core = cores[i];
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t seed = rand_seed_for_core(operation_attributes, i, device_seed_offset);

        // Values are pushed positionally in runtime_arg_names order.
        // compute: {seed, from_bits, to_bits, start_id, num_tiles}
        //   seed/from/to are DYNAMIC (excluded from the cache key): baked here for the cache-miss build,
        //   re-applied on every cache hit via override_runtime_arguments().
        compute_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{seed, from_bits, to_bits, tile_offset, units_per_core});
        // writer: {start_id, num_tiles}
        writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{tile_offset, units_per_core});

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

void RandDeviceOperation::RandProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // Re-derive every per-dispatch arg on each cache hit from the same builder create_descriptor uses:
    // compute's seed/from/to and the writer's output address. override replaces resolve_bindings, so
    // the address is ours to re-apply too. Push order in create_descriptor: writer 0, compute 1.
    constexpr uint32_t writer_kernel_idx = 0;
    constexpr uint32_t compute_kernel_idx = 1;

    const auto ws = compute_rand_work_split(operation_attributes, output, mesh_dispatch_coordinate);
    const float eps = 1e-6f;
    const uint32_t from_bits = std::bit_cast<uint32_t>(operation_attributes.from);
    const uint32_t to_bits = std::bit_cast<uint32_t>(operation_attributes.to - eps);
    const uint32_t out_addr = output.buffer()->address();

    // The writer's output base address is the tensor binding's CRTA slot (common across all cores):
    // re-point it here so the cache-hit dispatch writes into THIS call's output buffer, not the
    // first-miss buffer. Named-arg layout: CRTA word 0 == tensor::output address slot.
    auto& writer_common = tt::tt_metal::GetCommonRuntimeArgs(program, writer_kernel_idx);
    writer_common[0] = out_addr;

    uint32_t tile_offset = 0;
    for (int i = 0; i < static_cast<int>(ws.cores.size()); ++i) {
        const auto& core = ws.cores[i];
        uint32_t units_per_core;
        if (ws.core_group_1.contains(core)) {
            units_per_core = ws.units_per_core_group_1;
        } else if (ws.core_group_2.contains(core)) {
            units_per_core = ws.units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        const uint32_t seed = rand_seed_for_core(operation_attributes, i, ws.device_seed_offset);

        // Named-arg RTA layout (values sit at the byte offsets the generated args:: header assigns,
        // in runtime_arg_names order — named args occupy positions [0..N) so positional indexing holds):
        //   compute: [seed, from_bits, to_bits, start_id, num_tiles]
        auto& compute_args = tt::tt_metal::GetRuntimeArgs(program, compute_kernel_idx, core);
        compute_args[0] = seed;
        compute_args[1] = from_bits;
        compute_args[2] = to_bits;
        compute_args[3] = tile_offset;
        compute_args[4] = units_per_core;

        //   writer: [start_id, num_tiles]
        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_idx, core);
        writer_args[0] = tile_offset;
        writer_args[1] = units_per_core;

        tile_offset += units_per_core;
    }
}

}  // namespace ttnn::operations::rand
