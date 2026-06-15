// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_multi_core_hw_program_factory.hpp"

#include <optional>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

// Per-core runtime args for one dispatch, derived purely from (operation_attributes, inputs, output).
// Reader arg0 (src0) and arg1 (src1) and writer arg0 (dst) are tensor ADDRESSES, left as placeholder 0
// here (the factory binds them as patchable Buffer* on a cache miss; get_dynamic writes the live address
// on a cache hit). Every other slot is shape/geometry-derived. SINGLE SOURCE OF TRUTH shared by both
// create_descriptor() (cache miss) and get_dynamic_runtime_args() (cache hit re-apply).
struct BcastHWPerCoreArgs {
    std::vector<CoreCoord> cores;
    std::vector<KernelDescriptor::CoreRuntimeArgs> reader_args;
    std::vector<KernelDescriptor::CoreRuntimeArgs> writer_args;
    std::vector<KernelDescriptor::CoreRuntimeArgs> compute_args;
    std::vector<bool> is_work_core;
};

BcastHWPerCoreArgs compute_bcast_hw_per_core_args(
    const BcastParams& /*operation_attributes*/, const Tensor& a, const Tensor& b, const Tensor& output) {
    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    const uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    const uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    const uint32_t H = ashape[-2];
    const uint32_t W = ashape[-1];
    const uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    const uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    const uint32_t NC = N * C;

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;
    const uint32_t HtWt = Ht * Wt;

    const uint32_t num_tensor_tiles = NC * Ht * Wt;
    const uint32_t bnc1 = (bN * bC == 1);

    IDevice* device = a.device();

    std::optional<ShardSpec> shard_spec = std::nullopt;
    const bool src0_sharded = a.memory_config().is_sharded();
    const bool output_sharded = output.memory_config().is_sharded();
    if (src0_sharded) {
        shard_spec = a.shard_spec().value();
    } else if (output_sharded) {
        shard_spec = output.shard_spec().value();
    }

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cores_total = num_cores_x * num_cores_y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);
    (void)num_cores;

    if (shard_spec.has_value()) {
        const uint32_t num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        num_tiles_per_core_group_1 = num_tiles_per_shard;
        num_tiles_per_core_group_2 = 0;
        all_cores = shard_spec.value().grid;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
    }

    BcastHWPerCoreArgs result;
    result.cores.resize(num_cores_total);
    result.reader_args.resize(num_cores_total);
    result.writer_args.resize(num_cores_total);
    result.compute_args.resize(num_cores_total);
    result.is_work_core.resize(num_cores_total, false);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        result.cores[i] = core;
        uint32_t num_tensor_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            result.reader_args[i] = KernelDescriptor::CoreRuntimeArgs(7, 0);
            result.compute_args[i] = KernelDescriptor::CoreRuntimeArgs{1, 1, 0};
            result.writer_args[i] = KernelDescriptor::CoreRuntimeArgs(3, 0);
            continue;
        }

        // Addresses at reader args 0 (src0) and 1 (src1) as placeholder 0.
        result.reader_args[i] = KernelDescriptor::CoreRuntimeArgs{
            0u,  // 0  src0 address
            0u,  // 1  src1 address
            num_tensor_tiles_per_core,
            HtWt,
            num_tiles_read / HtWt * HtWt,
            num_tiles_read % HtWt,
            bnc1 ? 0u : num_tiles_read / HtWt};

        result.compute_args[i] = KernelDescriptor::CoreRuntimeArgs{
            1,                         // B
            1,                         // Ht
            num_tensor_tiles_per_core  // Wt
        };

        // Address at writer arg 0 (dst) as placeholder 0.
        result.writer_args[i] = KernelDescriptor::CoreRuntimeArgs{0u, num_tensor_tiles_per_core, num_tiles_read};
        result.is_work_core[i] = true;

        num_tiles_read += num_tensor_tiles_per_core;
    }

    return result;
}

}  // namespace

tt::tt_metal::ProgramDescriptor BcastMultiCoreHWProgramFactory::create_descriptor(
    const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value) {
    const Tensor& a = tensor_args.input_a;
    const Tensor& b = tensor_args.input_b;
    Tensor& output = tensor_return_value;

    const auto& bshape = b.padded_shape();
    const uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    const uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;

    const uint32_t bnc1 = (bN * bC == 1);

    IDevice* device = a.device();

    std::optional<ShardSpec> shard_spec = std::nullopt;
    const bool src0_sharded = a.memory_config().is_sharded();
    const bool output_sharded = output.memory_config().is_sharded();
    if (src0_sharded) {
        shard_spec = a.shard_spec().value();
    } else if (output_sharded) {
        shard_spec = output.shard_spec().value();
    }

    const tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat src1_cb_data_format = datatype_to_dataformat_converter(b.dtype());
    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    const uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    const uint32_t src1_single_tile_size = tt::tile_size(src1_cb_data_format);
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    Buffer* src0_buffer = a.buffer();
    Buffer* src1_buffer = b.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t src0_cb_index = 0;
    const uint32_t num_input_tiles = 2;
    uint32_t num_tiles_per_shard = 0;
    if (shard_spec.has_value()) {
        num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
    }

    const uint32_t num_input_tiles_cb0 = src0_sharded ? num_tiles_per_shard : num_input_tiles;

    const uint32_t src1_cb_index = 1;
    const uint32_t output_cb_index = tt::CBIndex::c_16;
    const uint32_t num_output_tiles = output_sharded ? num_tiles_per_shard : 2;

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles_cb0 * src0_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
        .buffer = src0_sharded ? src0_buffer : nullptr,
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src1_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = src1_cb_data_format,
            .page_size = src1_single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
        .buffer = output_sharded ? dst_buffer : nullptr,
    });

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines;
    std::vector<uint32_t> reader_compile_time_args;
    std::map<std::string, std::string> bcast_compute_defines =
        bcast_op_utils::get_defines(BcastOpDim::HW, operation_attributes.math_op);
    if (bnc1) {
        reader_defines["BCAST_SCALAR"] = "1";
        bcast_compute_defines["BCAST_SCALAR"] = "1";
    }
    if (src0_sharded) {
        reader_defines["IN0_SHARDED"] = "1";
    } else {
        TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    }
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    static constexpr const char* READER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/"
        "reader_bcast_hw_interleaved_partitioned.cpp";
    static constexpr const char* WRITER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    static constexpr const char* BCAST_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_hw.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_device_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = {reader_defines.begin(), reader_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};

    std::map<std::string, std::string> writer_defines;
    if (output_sharded) {
        writer_defines["OUT_SHARDED"] = "1";
    }
    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_device_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = {writer_defines.begin(), writer_defines.end()};
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = BCAST_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_device_cores;
    compute_desc.defines = {bcast_compute_defines.begin(), bcast_compute_defines.end()};
    compute_desc.config = ComputeConfigDescriptor{};

    // ---- Per-core runtime args ----
    // Single source of truth: compute_bcast_hw_per_core_args() derives every per-core arg the same way
    // for both this cache-miss build and the cache-hit re-apply in get_dynamic_runtime_args(). Avoids
    // the #46506 slow-path rebuild on a cache hit; re-applying every core also covers work-core-set
    // changes across a shared cache entry.
    const auto per_core = compute_bcast_hw_per_core_args(operation_attributes, a, b, output);

    for (uint32_t i = 0; i < per_core.cores.size(); i++) {
        const CoreCoord& core = per_core.cores[i];
        const auto& r = per_core.reader_args[i];
        const auto& w = per_core.writer_args[i];

        if (!per_core.is_work_core[i]) {
            reader_desc.runtime_args.emplace_back(core, r);
            compute_desc.runtime_args.emplace_back(core, per_core.compute_args[i]);
            writer_desc.runtime_args.emplace_back(core, w);
            continue;
        }

        KernelDescriptor::RTArgList reader_rt;
        reader_rt.reserve(r.size());
        reader_rt.push_back(src0_buffer);  // 0
        reader_rt.push_back(src1_buffer);  // 1
        for (size_t aIdx = 2; aIdx < r.size(); ++aIdx) {
            reader_rt.push_back(r[aIdx]);
        }
        reader_desc.emplace_runtime_args(core, std::move(reader_rt));

        compute_desc.runtime_args.emplace_back(core, per_core.compute_args[i]);

        KernelDescriptor::RTArgList writer_rt;
        writer_rt.reserve(w.size());
        writer_rt.push_back(dst_buffer);  // 0
        for (size_t aIdx = 1; aIdx < w.size(); ++aIdx) {
            writer_rt.push_back(w[aIdx]);
        }
        writer_desc.emplace_runtime_args(core, std::move(writer_rt));
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

std::vector<tt::tt_metal::DynamicRuntimeArg> BcastMultiCoreHWProgramFactory::get_dynamic_runtime_args(
    const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value) {
    const Tensor& a = tensor_args.input_a;
    const Tensor& b = tensor_args.input_b;
    Tensor& output = tensor_return_value;

    // Kernel order matches create_descriptor(): reader(0), writer(1), compute(2). Re-apply EVERY core's
    // args (work AND ex-work-now-noop cores) since the work-split changes the work-core set with shape.
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;
    constexpr uint32_t kComputeKernelIdx = 2;

    const auto per_core = compute_bcast_hw_per_core_args(operation_attributes, a, b, output);

    const auto addr_of = [](const Tensor& t) -> uint32_t {
        return t.buffer() != nullptr ? static_cast<uint32_t>(t.buffer()->address()) : 0u;
    };
    const uint32_t src0_addr = addr_of(a);
    const uint32_t src1_addr = addr_of(b);
    const uint32_t dst_addr = addr_of(output);

    std::vector<tt::tt_metal::DynamicRuntimeArg> dynamic_args;
    for (uint32_t i = 0; i < per_core.cores.size(); i++) {
        const CoreCoord& core = per_core.cores[i];
        const auto& r = per_core.reader_args[i];
        const auto& w = per_core.writer_args[i];
        const auto& c = per_core.compute_args[i];

        const bool work = per_core.is_work_core[i];
        for (uint32_t aIdx = 0; aIdx < r.size(); ++aIdx) {
            uint32_t value = r[aIdx];
            if (work && aIdx == 0) {
                value = src0_addr;
            } else if (work && aIdx == 1) {
                value = src1_addr;
            }
            dynamic_args.push_back({kReaderKernelIdx, core, aIdx, value});
        }
        for (uint32_t aIdx = 0; aIdx < w.size(); ++aIdx) {
            const uint32_t value = (work && aIdx == 0) ? dst_addr : w[aIdx];
            dynamic_args.push_back({kWriterKernelIdx, core, aIdx, value});
        }
        for (uint32_t aIdx = 0; aIdx < c.size(); ++aIdx) {
            dynamic_args.push_back({kComputeKernelIdx, core, aIdx, c[aIdx]});
        }
    }
    return dynamic_args;
}

}  // namespace ttnn::prim
