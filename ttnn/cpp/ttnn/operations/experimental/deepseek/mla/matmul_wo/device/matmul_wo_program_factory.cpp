// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_wo_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace ttnn::operations::experimental::deepseek::mla {

namespace {

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ReaderConfigDescriptor;
using tt::tt_metal::SemaphoreDescriptor;
using tt::tt_metal::WriterConfigDescriptor;

constexpr uint32_t kReduceSemaphoreId = 0;

std::vector<CoreCoord> find_collector_core_coords(
    const CoreCoord& full_grid_size, const std::set<CoreCoord>& dram_cores_set, size_t num_collectors) {
    std::vector<CoreCoord> collector_core_coords;
    collector_core_coords.reserve(num_collectors);
    for (int32_t y = full_grid_size.y - 1; y >= 0; --y) {
        for (int32_t x = full_grid_size.x - 1; x >= 0; --x) {
            const auto core_coord = CoreCoord(x, y);
            if (!dram_cores_set.contains(core_coord)) {
                collector_core_coords.push_back(core_coord);
                if (collector_core_coords.size() == num_collectors) {
                    return collector_core_coords;
                }
            }
        }
    }

    // We come here if we did not find enough collector cores; TT_FATAL is expected not to return.
    TT_FATAL(false, "Failed to find {} cores", num_collectors);
    return {};
}

CBDescriptor make_cb(
    tt::CBIndex index,
    tt::DataFormat data_format,
    bool is_tile,
    uint32_t tiles_per_cb,
    const CoreRangeSet& core_ranges,
    tt::tt_metal::Buffer* buffer) {
    const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
    CBDescriptor cb_desc;
    cb_desc.total_size = tiles_per_cb * bytes_per_tile;
    cb_desc.core_ranges = core_ranges;
    cb_desc.format_descriptors.push_back(CBFormatDescriptor{
        .buffer_index = static_cast<uint8_t>(index),
        .data_format = data_format,
        .page_size = bytes_per_tile,
    });
    cb_desc.buffer = buffer;
    return cb_desc;
}

}  // namespace

tt::tt_metal::ProgramDescriptor MatmulWODeviceOperation::create_descriptor(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t&) {
    // Get the cores for the program
    const auto dram_bank2core_coords =
        tensor_args.input_tensor.device()->get_optimal_dram_bank_to_logical_worker_assignment(
            tt::tt_metal::NOC::RISCV_0_default);

    const uint32_t num_cores = dram_bank2core_coords.size();
    auto dram_cores = CoreRangeSet(dram_bank2core_coords);

    // Let us find 7 cores that are not taken by the DRAM reader/writer kernels
    const auto full_grid_size = tensor_args.input_tensor.device()->compute_with_storage_grid_size();

    // Let us create a set of DRAM cores so we can do an easy find operation
    const auto dram_cores_set = std::set<CoreCoord>(dram_bank2core_coords.begin(), dram_bank2core_coords.end());

    const auto collector_core_coords = find_collector_core_coords(full_grid_size, dram_cores_set, 7);
    const auto collector_cores = CoreRangeSet(collector_core_coords);

    // Convert the collector core coordinates to physical coordinates
    std::vector<uint32_t> collector_core_physical_coords;
    collector_core_physical_coords.reserve(2 * collector_core_coords.size());
    for (const auto& core_coord : collector_core_coords) {
        const auto physical_core_coord = tensor_args.input_tensor.device()->worker_core_from_logical_core(core_coord);
        collector_core_physical_coords.push_back(physical_core_coord.x);
        collector_core_physical_coords.push_back(physical_core_coord.y);
    }

    // Put them in defines for kernel to access
    const KernelDescriptor::Defines kernel_defines = {
        {"COLLECTOR_CORE_COORDS", ttnn::operations::ccl::common::stringify(collector_core_physical_coords)}};

    auto all_cores = dram_cores.merge(collector_cores);

    // CBs used in the Matmul WO operation
    /*
        ------------------------------------------------------------------------------------
        |     Name       |   CB Index    |   Dtype    | Tile? | Tiles/CB |  Total size (B) |
        ------------------------------------------------------------------------------------
        | cb_r2c_w       | CBIndex::c_0  | Bfp8_b     | true  |    7*3*2 |      45696      |
        | cb_s2c_in(sh)  | CBIndex::c_1  | Float16_b  | true  |    512   |      1048576    |
        | cb_c2w_out     | CBIndex::c_2  | Float16_b  | true  |    28    |      57344      |
        | cb_s2c_in2     | CBIndex::c_3  | Float16_b  | true  |    48    |      98304      |
        | cb_s2c_out(sh) | CBIndex::c_4  | Float16_b  | true  |    4     |      8192       |
        ------------------------------------------------------------------------------------
    */

    auto* input_buffer = tensor_args.input_tensor.buffer();
    auto* weight_buffer = tensor_args.w_tensor.buffer();
    auto* output_buffer = tensor_args.output_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "matmul_wo input tensor buffer is null");
    TT_FATAL(weight_buffer != nullptr, "matmul_wo weight tensor buffer is null");
    TT_FATAL(output_buffer != nullptr, "matmul_wo output tensor buffer is null");

    tt::tt_metal::ProgramDescriptor program_desc;
    // Buffer* runtime args and CBDescriptor::buffer are patched on a program-cache hit.
    program_desc.cbs = {
        make_cb(tt::CBIndex::c_0, tt::DataFormat::Bfp8_b, true, 7 * 3 * 2, dram_cores, nullptr),
        make_cb(tt::CBIndex::c_2, tt::DataFormat::Float16_b, true, 28, dram_cores, nullptr),
        make_cb(tt::CBIndex::c_3, tt::DataFormat::Float16_b, true, 48, all_cores, nullptr),
        make_cb(tt::CBIndex::c_1, tt::DataFormat::Float16_b, true, 512, dram_cores, input_buffer),
        make_cb(tt::CBIndex::c_4, tt::DataFormat::Float16_b, true, 4, collector_cores, output_buffer),
    };

    // Create compile args for the program
    const auto tensors = std::vector<tt::tt_metal::Buffer*>{input_buffer, weight_buffer, output_buffer};

    std::vector<uint32_t> compile_args;
    for (const auto* buffer : tensors) {
        tt::tt_metal::TensorAccessorArgs(*buffer).append_to(compile_args);
    }

    const KernelDescriptor::NamedCompileTimeArgs named_compile_time_args = {
        {"layer_id", operation_attributes.layer_id},
        {"num_cores", num_cores},
        {"reduce_semaphore_id", kReduceSemaphoreId},
    };

    // Create semaphores for reducing the partials at the end
    program_desc.semaphores.push_back(SemaphoreDescriptor{
        .id = kReduceSemaphoreId,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = collector_cores,
        .initial_value = 0,
    });

    const std::string dm0_kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/dm0.cpp";
    const std::string dm1_kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/dm1.cpp";
    const std::string compute_kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/compute.cpp";
    const std::string dm1_collector_kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/dm1_collector.cpp";
    const std::string compute_collector_kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/compute_collector.cpp";

    KernelDescriptor dm0_kernel{
        .kernel_source = dm0_kernel_source,
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = dram_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .defines = kernel_defines,
        .config = ReaderConfigDescriptor{},
    };

    KernelDescriptor dm1_kernel{
        .kernel_source = dm1_kernel_source,
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = dram_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .defines = kernel_defines,
        .config = WriterConfigDescriptor{},
    };

    KernelDescriptor compute_kernel{
        .kernel_source = compute_kernel_source,
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = dram_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .defines = kernel_defines,
        .config =
            ComputeConfigDescriptor{
                .math_fidelity = tt::tt_metal::MathFidelity::LoFi,
                .fp32_dest_acc_en = false,
                .dst_full_sync_en = false,
                .bfp8_pack_precise = false,
                .math_approx_mode = true,
            },
    };

    //-------------------------------------------------------------------------
    // Collector cores - these collect all data and reduce them.
    //-------------------------------------------------------------------------
    KernelDescriptor dm1_collector_kernel{
        .kernel_source = dm1_collector_kernel_source,
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = collector_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .defines = kernel_defines,
        .config = WriterConfigDescriptor{},
    };

    KernelDescriptor compute_collector_kernel{
        .kernel_source = compute_collector_kernel_source,
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = collector_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .defines = kernel_defines,
        .config =
            ComputeConfigDescriptor{
                .math_fidelity = tt::tt_metal::MathFidelity::LoFi,
                .fp32_dest_acc_en = false,
                .dst_full_sync_en = false,
                .bfp8_pack_precise = false,
                .math_approx_mode = true,
            },
    };

    // Set the runtime arguments for the kernels
    std::vector<uint32_t> vchannels;
    vchannels.reserve(dram_bank2core_coords.size());
    uint32_t dram_bank = 0;
    for (auto core : dram_bank2core_coords) {
        uint32_t vchannel = dram_bank & 0x3;

        // Check if there is any core with the same row
        auto it = std::find_if(
            dram_bank2core_coords.begin(), dram_bank2core_coords.begin() + dram_bank, [&](const auto& core_prev) {
                return core_prev.y == core.y;
            });

        // If there is any core with the same row, make sure the VChannel is different
        if (it != dram_bank2core_coords.begin() + dram_bank) {
            size_t j = std::distance(dram_bank2core_coords.begin(), it);
            if (vchannel == vchannels[j]) {
                vchannel = (vchannel + 1) & 0x3;
            }
        }
        vchannels.push_back(vchannel);

        KernelDescriptor::RTArgList runtime_args;
        runtime_args.reserve(5);
        runtime_args.push_back(dram_bank);
        runtime_args.push_back(vchannel);
        runtime_args.push_back(input_buffer);
        runtime_args.push_back(weight_buffer);
        runtime_args.push_back(output_buffer);
        dm0_kernel.emplace_runtime_args(core, runtime_args);
        dm1_kernel.emplace_runtime_args(core, runtime_args);
        compute_kernel.emplace_runtime_args(core, runtime_args);
        dram_bank++;
    }

    uint32_t core_id = 0;
    for (const auto& core : collector_core_coords) {
        dm1_collector_kernel.emplace_runtime_args(core, {core_id});
        compute_collector_kernel.emplace_runtime_args(core, {core_id});
        core_id++;
    }

    program_desc.kernels.reserve(5);
    program_desc.kernels.push_back(std::move(dm0_kernel));
    program_desc.kernels.push_back(std::move(dm1_kernel));
    program_desc.kernels.push_back(std::move(compute_kernel));
    program_desc.kernels.push_back(std::move(dm1_collector_kernel));
    program_desc.kernels.push_back(std::move(compute_collector_kernel));
    return program_desc;
}

}  // namespace ttnn::operations::experimental::deepseek::mla
