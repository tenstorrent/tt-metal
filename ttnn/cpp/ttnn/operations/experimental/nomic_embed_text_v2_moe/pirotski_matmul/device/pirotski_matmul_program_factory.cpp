// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pirotski_matmul_device_operation.hpp"

#include <algorithm>
#include <map>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/compute_throttle_utils.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

const std::string kKernelDirectory =
    "ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/pirotski_matmul/device/kernels/";
// minimal_matmul's compute kernel, unchanged: its CB indices, argument order and defines are this op's.
const std::string kComputeKernel = "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp";

// Two slots per input ring: the next block loads while the current one computes.
constexpr uint32_t kRingSlots = 2;

enum CircularBufferIndex : uint8_t {
    kInputRing = tt::CBIndex::c_0,
    kWeightRing = tt::CBIndex::c_1,
    kOutput = tt::CBIndex::c_2,
    kPartials = tt::CBIndex::c_3,
    kBias = tt::CBIndex::c_4,
};

enum SemaphoreId : uint32_t {
    kInputGranted,
    kInputArrived,
    kWeightGranted,
    kWeightArrived,
    kDrainPosted,
    kDrainDone,
};

tt::tt_metal::CBDescriptor ring(const CoreRangeSet& cores, uint8_t index, tt::DataFormat format, uint32_t tiles) {
    const uint32_t tile_bytes = tt::tile_size(format);
    return tt::tt_metal::CBDescriptor{
        .total_size = tile_bytes * tiles,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = index,
            .data_format = format,
            .page_size = tile_bytes,
        }}},
    };
}

}  // namespace

tt::tt_metal::ProgramDescriptor PirotskiMatmulDeviceOperation::create_descriptor(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using tt::tt_metal::ComputeConfigDescriptor;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::DataMovementProcessor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::NOC;
    using tt::tt_metal::TensorAccessorArgs;

    const auto layout = compute_layout(attributes, tensor_args);
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    const auto& config = attributes.config;
    const bool has_bias = tensor_args.bias.has_value();
    auto* device = input.device();
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attributes.compute_kernel_config);

    const uint32_t columns = layout.columns;
    const uint32_t rows = layout.rows;
    const bool has_strip = layout.strip_m_tiles > 0;
    const CoreRangeSet main_cores{CoreRange(CoreCoord(0, 0), CoreCoord(columns - 1, rows - 1))};
    const CoreRangeSet strip_cores =
        has_strip ? CoreRangeSet{CoreRange(CoreCoord(0, rows), CoreCoord(columns - 1, rows + layout.strip_rows - 1))}
                  : CoreRangeSet{};
    const CoreRangeSet all_cores = has_strip ? main_cores.merge(strip_cores) : main_cores;
    const uint32_t m_block = config.M_block_size;
    const uint32_t k_block = config.K_block_size;
    const uint32_t n_block = config.N_block_size;
    // An interleaved output's last block goes out on both RISCs; a direct-packed one is never written.
    const bool split_drain = !layout.direct_pack;

    const auto input_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto weight_format = tt::tt_metal::datatype_to_dataformat_converter(weight.dtype());
    const auto output_format = tt::tt_metal::datatype_to_dataformat_converter(attributes.dtype);
    const auto partials_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const auto bias_format = has_bias ? tt::tt_metal::datatype_to_dataformat_converter(tensor_args.bias->dtype())
                                      : tt::DataFormat::Float16_b;

    tt::tt_metal::ProgramDescriptor descriptor;
    descriptor.cbs.push_back(ring(main_cores, kInputRing, input_format, kRingSlots * m_block * k_block));
    descriptor.cbs.push_back(ring(
        main_cores,
        kWeightRing,
        weight_format,
        (attributes.resident_weights ? layout.k_blocks * layout.n_blocks_per_core : kRingSlots) * k_block * n_block));
    if (layout.direct_pack) {
        // Compute packs straight into the output's shard; rows past M land in the last column's shard padding.
        auto out = ring(main_cores, kOutput, output_format, layout.m_tiles_per_core * n_block);
        out.buffer = output.buffer();
        descriptor.cbs.push_back(out);
    } else {
        descriptor.cbs.push_back(ring(main_cores, kOutput, output_format, m_block * n_block));
    }
    descriptor.cbs.push_back(ring(main_cores, kPartials, partials_format, m_block * n_block));
    if (has_bias) {
        // One N block: with one a core it is read once and re-pushed unread for every output block.
        descriptor.cbs.push_back(ring(main_cores, kBias, bias_format, n_block));
    }
    if (has_strip) {
        const uint32_t strip_block = layout.strip_m_tiles * layout.strip_n_tiles;
        descriptor.cbs.push_back(
            ring(strip_cores, kInputRing, input_format, kRingSlots * layout.strip_m_tiles * k_block));
        descriptor.cbs.push_back(
            ring(strip_cores, kWeightRing, weight_format, kRingSlots * k_block * layout.strip_n_tiles));
        descriptor.cbs.push_back(ring(strip_cores, kOutput, output_format, strip_block));
        descriptor.cbs.push_back(ring(strip_cores, kPartials, partials_format, strip_block));
        if (has_bias) {
            descriptor.cbs.push_back(ring(strip_cores, kBias, bias_format, layout.strip_n_tiles));
        }
    }
    for (uint32_t id : {kInputGranted, kInputArrived, kWeightGranted, kWeightArrived, kDrainPosted, kDrainDone}) {
        descriptor.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
    }

    // One region's blocking: the config grid's, or the strip's (one M and one N block a core).
    struct Region {
        CoreRangeSet cores;
        uint32_t m_block;
        uint32_t m_blocks;
        uint32_t n_block;
        uint32_t subblock_h;
        uint32_t subblock_w;
        bool split_drain;
        bool direct_pack;
        bool shard_drain;
        bool read_ahead;
    };

    // RISCV_0 on NOC0 carries the activations down each column from its top core, RISCV_1 on NOC1 the weights west
    // along each row from its right core. A NOC1 multicast runs from the far corner of its rectangle to the near one.
    const auto in0_kernel_for = [&](const Region& region) {
        KernelDescriptor kernel;
        kernel.kernel_source = kKernelDirectory + "dataflow/in0_stream.cpp";
        kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernel.core_ranges = region.cores;
        kernel.named_compile_time_args = {
            {"K_tiles", layout.k_tiles},
            {"M_block_tiles", region.m_block},
            {"K_block_tiles", k_block},
            {"M_blocks_per_core", region.m_blocks},
            {"N_blocks_per_core", layout.n_blocks_per_core},
            {"K_num_blocks", layout.k_blocks},
            {"in0_cb", kInputRing},
            {"shard_width", layout.input_shard_width},
            {"split_drain", region.split_drain ? 1u : 0u},
            {"N_tiles", layout.n_tiles},
            {"N_block_tiles", region.n_block},
            {"out_cb", kOutput},
            {"in0_granted", kInputGranted},
            {"in0_arrived", kInputArrived},
            {"drain_posted", kDrainPosted},
            {"drain_done", kDrainDone},
        };
        TensorAccessorArgs(*input.buffer()).append_to(kernel.compile_time_args);
        TensorAccessorArgs(*output.buffer()).append_to(kernel.compile_time_args);
        kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
        return kernel;
    };
    const auto in1_kernel_for = [&](const Region& region) {
        KernelDescriptor kernel;
        kernel.kernel_source = kKernelDirectory + "dataflow/in1_stream.cpp";
        kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernel.core_ranges = region.cores;
        kernel.named_compile_time_args = {
            {"N_tiles", layout.n_tiles},
            {"M_block_tiles", region.m_block},
            {"K_block_tiles", k_block},
            {"N_block_tiles", region.n_block},
            {"M_blocks_per_core", region.m_blocks},
            {"N_blocks_per_core", layout.n_blocks_per_core},
            {"K_num_blocks", layout.k_blocks},
            {"has_bias", has_bias ? 1u : 0u},
            {"resident", attributes.resident_weights ? 1u : 0u},
            {"split_drain", region.split_drain ? 1u : 0u},
            {"direct_pack", region.direct_pack ? 1u : 0u},
            {"shard_drain", region.shard_drain ? 1u : 0u},
            {"read_ahead", region.read_ahead ? 1u : 0u},
            {"in1_cb", kWeightRing},
            {"out_cb", kOutput},
            {"bias_cb", kBias},
            {"in1_granted", kWeightGranted},
            {"in1_arrived", kWeightArrived},
            {"drain_posted", kDrainPosted},
            {"drain_done", kDrainDone},
        };
        TensorAccessorArgs(*weight.buffer()).append_to(kernel.compile_time_args);
        // Without a bias the accessor is never built; the weight's arguments hold its place.
        TensorAccessorArgs(*(has_bias ? tensor_args.bias->buffer() : weight.buffer()))
            .append_to(kernel.compile_time_args);
        TensorAccessorArgs(*output.buffer()).append_to(kernel.compile_time_args);
        kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
        return kernel;
    };
    std::map<std::string, std::string> compute_defines;
    if (has_bias) {
        compute_defines["FUSE_BIAS"] = "1";
    }
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(),
        static_cast<int>(columns * (rows + layout.strip_rows)),
        compute_defines,
        ttnn::get_throttle_level(attributes.compute_kernel_config));
    const auto compute_kernel_for = [&](const Region& region) {
        KernelDescriptor kernel;
        kernel.kernel_source = kComputeKernel;
        kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernel.core_ranges = region.cores;
        kernel.compile_time_args = {
            layout.k_blocks,
            region.m_block,
            k_block,
            region.n_block,
            region.m_blocks,
            layout.n_blocks_per_core,
            region.subblock_h,
            region.subblock_w};
        kernel.defines.assign(compute_defines.begin(), compute_defines.end());
        // As minimal_matmul's factory builds it: a double-buffered destination whatever dst_full_sync_en says, and
        // fp32 partials unpacked to the source registers (the default mode). The legacy descriptor compiles at O3.
        kernel.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = false,
            .math_approx_mode = math_approx_mode,
        };
        return kernel;
    };

    const Region main_region{
        .cores = main_cores,
        .m_block = m_block,
        .m_blocks = layout.m_blocks_per_core,
        .n_block = n_block,
        .subblock_h = config.subblock_h,
        .subblock_w = config.subblock_w,
        .split_drain = split_drain,
        .direct_pack = layout.direct_pack,
        .shard_drain = false,
        .read_ahead = attributes.read_ahead};
    auto in0_kernel = in0_kernel_for(main_region);
    auto in1_kernel = in1_kernel_for(main_region);
    auto compute = compute_kernel_for(main_region);

    const auto noc_coordinates = [&](uint32_t x, uint32_t y) { return device->worker_core_from_logical_core({x, y}); };
    // The column's activations: multicast south from its top core on NOC0 (start <= end), or read by the core
    // itself when it has no receivers.
    const auto in0_args = [&](uint32_t x,
                              uint32_t y,
                              bool is_sender,
                              uint32_t m_start,
                              uint32_t m_end,
                              uint32_t receivers,
                              uint32_t n_start) {
        const auto sender = noc_coordinates(x, is_sender ? y : 0);
        const auto first = noc_coordinates(x, receivers > 0 ? 1 : y);
        const auto last = noc_coordinates(x, receivers > 0 ? rows - 1 : y);
        KernelDescriptor::RTArgList args;
        args.push_back(input.buffer());
        args.push_back(is_sender ? 1u : 0u);
        args.push_back(m_start);
        args.push_back(m_end);
        args.push_back(static_cast<uint32_t>(sender.x));
        args.push_back(static_cast<uint32_t>(sender.y));
        args.push_back(static_cast<uint32_t>(first.x));
        args.push_back(static_cast<uint32_t>(first.y));
        args.push_back(static_cast<uint32_t>(last.x));
        args.push_back(static_cast<uint32_t>(last.y));
        args.push_back(receivers);
        if (layout.input_shard_width > 0) {
            for (uint32_t shard = 0; shard < layout.k_tiles / layout.input_shard_width; ++shard) {
                args.push_back(static_cast<uint32_t>(noc_coordinates(x, shard).y));
            }
        }
        args.push_back(output.buffer());
        args.push_back(n_start);
        return args;
    };
    // The row's weights: multicast west from its right core on NOC1, start at the far corner.
    const auto in1_args =
        [&](uint32_t x, uint32_t y, uint32_t m_start, uint32_t m_end, uint32_t n_start, uint32_t defer_write_k_block) {
            const auto sender = noc_coordinates(columns - 1, y);
            const auto far = noc_coordinates(columns > 1 ? columns - 2 : 0, y);
            const auto near = noc_coordinates(0, y);
            KernelDescriptor::RTArgList args;
            args.push_back(weight.buffer());
            args.push_back(has_bias ? tensor_args.bias->buffer() : weight.buffer());
            args.push_back(output.buffer());
            for (uint32_t value :
                 {x == columns - 1 ? 1u : 0u,
                  m_start,
                  m_end,
                  n_start,
                  defer_write_k_block,
                  static_cast<uint32_t>(sender.x),
                  static_cast<uint32_t>(sender.y),
                  static_cast<uint32_t>(far.x),
                  static_cast<uint32_t>(far.y),
                  static_cast<uint32_t>(near.x),
                  static_cast<uint32_t>(near.y),
                  columns - 1}) {
                args.push_back(value);
            }
            return args;
        };

    // minimal_matmul's rule: block m - 1 goes out at a K block that grows with the core's row, never K block 0.
    const uint32_t k_blocks_per_row = tt::div_up(layout.k_blocks, rows);
    for (uint32_t x = 0; x < columns; ++x) {
        for (uint32_t y = 0; y < rows; ++y) {
            const CoreCoord core(x, y);
            const uint32_t m_start = x * layout.m_tiles_per_core;
            const uint32_t m_end = std::min(m_start + layout.main_m_tiles, layout.m_tiles);
            const uint32_t n_start = y * layout.n_tiles_per_core;
            in0_kernel.emplace_runtime_args(core, in0_args(x, y, y == 0, m_start, m_end, rows - 1, n_start));
            uint32_t defer_write_k_block = std::min(y * k_blocks_per_row, layout.k_blocks - 1);
            if (layout.k_blocks > 1) {
                defer_write_k_block = std::max(defer_write_k_block, 1u);
            }
            in1_kernel.emplace_runtime_args(core, in1_args(x, y, m_start, m_end, n_start, defer_write_k_block));
            // As minimal_matmul passes them: the core's M share unclamped, its padding rows computed.
            compute.emplace_runtime_args(
                core, {m_start, m_start + layout.main_m_tiles, n_start, n_start + layout.n_tiles_per_core});
        }
    }
    descriptor.kernels.push_back(std::move(in0_kernel));
    descriptor.kernels.push_back(std::move(in1_kernel));
    descriptor.kernels.push_back(std::move(compute));

    if (has_strip) {
        // Each strip core reads its own rows; each strip row's weights come from its right core, as a main row's.
        const Region strip_region{
            .cores = strip_cores,
            .m_block = layout.strip_m_tiles,
            .m_blocks = 1,
            .n_block = layout.strip_n_tiles,
            .subblock_h = layout.strip_subblock_h,
            .subblock_w = layout.strip_subblock_w,
            .split_drain = false,
            .direct_pack = false,
            .shard_drain = layout.direct_pack,
            .read_ahead = attributes.strip_read_ahead};
        auto strip_in0 = in0_kernel_for(strip_region);
        auto strip_in1 = in1_kernel_for(strip_region);
        auto strip_compute = compute_kernel_for(strip_region);
        for (uint32_t x = 0; x < columns; ++x) {
            for (uint32_t j = 0; j < layout.strip_rows; ++j) {
                const CoreCoord core(x, rows + j);
                const uint32_t m_start = x * layout.m_tiles_per_core + layout.main_m_tiles;
                const uint32_t m_end = std::min(m_start + layout.strip_m_tiles, layout.m_tiles);
                const uint32_t n_start = j * layout.strip_n_tiles;
                strip_in0.emplace_runtime_args(core, in0_args(x, rows + j, true, m_start, m_end, 0, n_start));
                auto args = in1_args(x, rows + j, m_start, m_end, n_start, 0);
                if (layout.direct_pack) {
                    // The rows go into the shards of grid rows [j * strip_lanes, (j + 1) * strip_lanes) of column x.
                    args.push_back(layout.main_m_tiles);
                    args.push_back(layout.n_tiles_per_core);
                    args.push_back(layout.strip_lanes);
                    for (uint32_t lane = 0; lane < layout.strip_lanes; ++lane) {
                        const auto shard = noc_coordinates(x, j * layout.strip_lanes + lane);
                        args.push_back(static_cast<uint32_t>(shard.x));
                        args.push_back(static_cast<uint32_t>(shard.y));
                    }
                }
                strip_in1.emplace_runtime_args(core, args);
                strip_compute.emplace_runtime_args(
                    core, {m_start, m_start + layout.strip_m_tiles, n_start, n_start + layout.strip_n_tiles});
            }
        }
        descriptor.kernels.push_back(std::move(strip_in0));
        descriptor.kernels.push_back(std::move(strip_in1));
        descriptor.kernels.push_back(std::move(strip_compute));
    }
    return descriptor;
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
