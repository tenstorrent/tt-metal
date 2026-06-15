// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_wo_program_factory.hpp"
#include "matmul_wo_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace ttnn::operations::experimental::deepseek::mla::program {

using namespace tt::tt_metal;

static std::vector<CoreCoord> find_collector_core_coords(
    const CoreCoord& full_grid_size, const std::set<CoreCoord>& dram_cores_set, size_t num_collectors) {
    std::vector<CoreCoord> collector_core_coords;
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

tt::tt_metal::ProgramDescriptor MatmulWOProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    ProgramDescriptor desc;

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

    // Define the (non-sharded) CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb, core_range_set.
    const std::vector<std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t, CoreRangeSet>> cb_specs0 = {
        {"cb_r2c_w", tt::CBIndex::c_0, tt::DataFormat::Bfp8_b, true, 7 * 3 * 2, dram_cores},
        {"cb_c2w_out", tt::CBIndex::c_2, tt::DataFormat::Float16_b, true, 28, dram_cores},
        {"cb_s2c_in2", tt::CBIndex::c_3, tt::DataFormat::Float16_b, true, 48, all_cores},
    };

    for (const auto& [name, index, data_format, is_tile, tiles_per_cb, core_range_set] : cb_specs0) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles_per_cb * bytes_per_tile,
            .core_ranges = core_range_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index),
                .data_format = data_format,
                .page_size = bytes_per_tile,
            }}},
        });
    }

    // Create sharded CBs (bound via .buffer for dynamic address re-application).
    // Define the CB configuration as a tuple: name, CBIndex, DataFormat, tiles_per_cb, Buffer*, core_range_set.
    const std::vector<
        std::tuple<std::string, tt::CBIndex, tt::DataFormat, bool, uint32_t, tt::tt_metal::Buffer*, CoreRangeSet>>
        sharded_cb_specs = {
            {"cb_s2c_in",
             tt::CBIndex::c_1,
             tt::DataFormat::Float16_b,
             true,
             512,
             tensor_args.input_tensor.buffer(),
             dram_cores},
            {"cb_s2c_out",
             tt::CBIndex::c_4,
             tt::DataFormat::Float16_b,
             true,
             4,
             tensor_args.output_tensor.buffer(),
             collector_cores}};

    for (const auto& [name, index, data_format, is_tile, tiles_per_cb, p_buffer, core_range_set] : sharded_cb_specs) {
        const uint32_t bytes_per_tile = is_tile ? tt::tile_size(data_format) : tt::datum_size(data_format);
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles_per_cb * bytes_per_tile,
            .core_ranges = core_range_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index),
                .data_format = data_format,
                .page_size = bytes_per_tile,
            }}},
            .buffer = p_buffer,
        });
    }

    // Create compile args for the program
    const auto tensors =
        std::vector<const Tensor*>{&tensor_args.input_tensor, &tensor_args.w_tensor, &tensor_args.output_tensor};

    std::vector<uint32_t> compile_args;
    for (const auto& tensor : tensors) {
        tt::tt_metal::TensorAccessorArgs(*tensor->buffer()).append_to(compile_args);
    }

    // Create the reduce-partials semaphore on the collector cores.  Semaphore id is
    // its index in desc.semaphores (0-based, declaration order).
    const uint32_t reduce_semaphore_id = 0;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .core_ranges = collector_cores,
        .initial_value = 0,
    });

    KernelDescriptor::NamedCompileTimeArgs named_compile_time_args = {
        {"layer_id", operation_attributes.layer_id},
        {"num_cores", static_cast<uint32_t>(num_cores)},
        {"reduce_semaphore_id", reduce_semaphore_id},
    };

    // ---- Kernels: DRAM worker cores ----
    KernelDescriptor dm0_desc;
    dm0_desc.kernel_source = "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/dm0.cpp";
    dm0_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    dm0_desc.core_ranges = dram_cores;
    dm0_desc.compile_time_args = compile_args;
    dm0_desc.named_compile_time_args = named_compile_time_args;
    dm0_desc.defines = kernel_defines;
    dm0_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_0,
    };

    KernelDescriptor dm1_desc;
    dm1_desc.kernel_source = "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/dm1.cpp";
    dm1_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    dm1_desc.core_ranges = dram_cores;
    dm1_desc.compile_time_args = compile_args;
    dm1_desc.named_compile_time_args = named_compile_time_args;
    dm1_desc.defines = kernel_defines;
    dm1_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_1,
    };

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/compute.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = dram_cores;
    compute_desc.compile_time_args = compile_args;
    compute_desc.named_compile_time_args = named_compile_time_args;
    compute_desc.defines = kernel_defines;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::LoFi,
        .fp32_dest_acc_en = false,
        .dst_full_sync_en = false,
        .bfp8_pack_precise = false,
        .math_approx_mode = true,
    };

    //-------------------------------------------------------------------------
    // Collector cores - these collect all data and reduce them.
    //-------------------------------------------------------------------------
    KernelDescriptor dm1_collector_desc;
    dm1_collector_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/dm1_collector.cpp";
    dm1_collector_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    dm1_collector_desc.core_ranges = collector_cores;
    dm1_collector_desc.compile_time_args = compile_args;
    dm1_collector_desc.named_compile_time_args = named_compile_time_args;
    dm1_collector_desc.defines = kernel_defines;
    dm1_collector_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_1,
    };

    KernelDescriptor compute_collector_desc;
    compute_collector_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/compute_collector.cpp";
    compute_collector_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_collector_desc.core_ranges = collector_cores;
    compute_collector_desc.compile_time_args = compile_args;
    compute_collector_desc.named_compile_time_args = named_compile_time_args;
    compute_collector_desc.defines = kernel_defines;
    compute_collector_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::LoFi,
        .fp32_dest_acc_en = false,
        .dst_full_sync_en = false,
        .bfp8_pack_precise = false,
        .math_approx_mode = true,
    };

    // ---- Per-core runtime args ----
    // Runtime args layout for DRAM worker cores: [dram_bank_id, vchannel, input_addr, w_addr, output_addr].
    // The previous override_runtime_arguments only patched [3] = w_tensor address, but a fresh
    // create_descriptor recomputes every entry, so the addresses are always current.
    std::vector<uint32_t> base_runtime_args;
    base_runtime_args.push_back(0);  // DRAM Bank ID placeholder
    base_runtime_args.push_back(0);  // VChannel placeholder
    for (const auto& tensor : tensors) {
        base_runtime_args.push_back(tensor->buffer()->address());
    }

    dm0_desc.runtime_args.reserve(dram_bank2core_coords.size());
    dm1_desc.runtime_args.reserve(dram_bank2core_coords.size());
    compute_desc.runtime_args.reserve(dram_bank2core_coords.size());

    std::vector<uint32_t> vchannels;
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

        std::vector<uint32_t> per_core_args = base_runtime_args;
        per_core_args[0] = dram_bank++;
        per_core_args[1] = vchannel;

        dm0_desc.runtime_args.emplace_back(core, per_core_args);
        dm1_desc.runtime_args.emplace_back(core, per_core_args);
        compute_desc.runtime_args.emplace_back(core, std::move(per_core_args));
    }

    // Collector-core runtime args: [core_id]
    dm1_collector_desc.runtime_args.reserve(collector_core_coords.size());
    compute_collector_desc.runtime_args.reserve(collector_core_coords.size());
    uint32_t core_id = 0;
    for (const auto& core : collector_core_coords) {
        std::vector<uint32_t> collector_runtime_args = {core_id++};
        dm1_collector_desc.runtime_args.emplace_back(core, collector_runtime_args);
        compute_collector_desc.runtime_args.emplace_back(core, std::move(collector_runtime_args));
    }

    desc.kernels.push_back(std::move(dm0_desc));
    desc.kernels.push_back(std::move(dm1_desc));
    desc.kernels.push_back(std::move(compute_desc));
    desc.kernels.push_back(std::move(dm1_collector_desc));
    desc.kernels.push_back(std::move(compute_collector_desc));

    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek::mla::program
