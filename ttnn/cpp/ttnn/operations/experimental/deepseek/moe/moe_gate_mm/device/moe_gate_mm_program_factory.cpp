// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gate_mm_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ttnn::operations::experimental::deepseek::moe::moe_gate_mm {

namespace {

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ReaderConfigDescriptor;
using tt::tt_metal::SemaphoreDescriptor;
using tt::tt_metal::WriterConfigDescriptor;

constexpr uint32_t kPartialSemaphoreId = 0;
constexpr uint32_t kRawScoresSemaphoreId = 1;

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

tt::tt_metal::ProgramDescriptor MoEGateMMDeviceOperation::create_descriptor(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t&) {
    // Get the cores for the program
    const auto dram_bank2core_coords =
        tensor_args.input_tensor.device()->get_optimal_dram_bank_to_logical_worker_assignment(
            tt::tt_metal::NOC::RISCV_0_default);

    const uint32_t num_cores = dram_bank2core_coords.size();
    constexpr uint32_t required_cores = 12;
    TT_FATAL(
        num_cores == required_cores,
        "moe_gate_mm requires exactly {} DRAM-aligned cores (Wormhole); got {}. "
        "This op's ring algorithm is hardcoded for Wormhole's 12 DRAM views and does not support other "
        "architectures.",
        required_cores,
        num_cores);
    auto all_cores = CoreRangeSet(dram_bank2core_coords);

    // CBs used in the MoE Gate MM operation
    /*
        ------------------------------------------------------------------------------------
        |     Name       |   CB Index    |   Dtype    | Tile? | Tiles/CB |  Total size (B) |
        ------------------------------------------------------------------------------------
        | cb_r2c_w       | CBIndex::c_0  | Float16_b  | true  |    32*3  |      196608     |
        | cb_s2c_in(sh)  | CBIndex::c_1  | Float16_b  | true  |    224   |      458752     |
        | cb_c2w_rdy     | CBIndex::c_2  | Float32    | false |    1     |      4          |
        | cb_w2c_in2     | CBIndex::c_3  | Float32    | true  |    1     |      2048       |
        | cb_s2c_out(sh) | CBIndex::c_4  | Float16_b  | true  |    1     |      2048       |
        | cb_w2c_in3     | CBIndex::c_5  | Float16_b  | true  |    1     |      2048       |
        | cb_w2c_in4     | CBIndex::c_6  | Float16_b  | true  |    1     |      2048       |
        | cb_w2c_in5     | CBIndex::c_7  | Float16_b  | true  |    1     |      2048       |
        | cb_w2c_in6     | CBIndex::c_8  | Float16_b  | true  |    4     |      8192       |
        | cb_w2c_in7     | CBIndex::c_9  | Float16_b  | true  |    4     |      8192      |
        ------------------------------------------------------------------------------------
    */

    auto* input_buffer = tensor_args.input_tensor.buffer();
    auto* weight_buffer = tensor_args.w_tensor.buffer();
    auto* output_buffer = tensor_args.output_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "moe_gate_mm input tensor buffer is null");
    TT_FATAL(weight_buffer != nullptr, "moe_gate_mm weight tensor buffer is null");
    TT_FATAL(output_buffer != nullptr, "moe_gate_mm output tensor buffer is null");

    tt::tt_metal::ProgramDescriptor program_desc;
    // Buffer* runtime args and CBDescriptor::buffer are patched on a program-cache hit.
    program_desc.cbs = {
        make_cb(tt::CBIndex::c_0, tt::DataFormat::Float16_b, true, 32 * 3, all_cores, nullptr),
        make_cb(tt::CBIndex::c_2, tt::DataFormat::Float32, false, 1, all_cores, nullptr),
        make_cb(tt::CBIndex::c_3, tt::DataFormat::Float32, true, 1, all_cores, nullptr),
        make_cb(tt::CBIndex::c_5, tt::DataFormat::Float16_b, true, 1, all_cores, nullptr),
        make_cb(tt::CBIndex::c_6, tt::DataFormat::Float16_b, true, 1, all_cores, nullptr),
        make_cb(tt::CBIndex::c_7, tt::DataFormat::Float16_b, true, 1, all_cores, nullptr),
        make_cb(tt::CBIndex::c_8, tt::DataFormat::Float16_b, true, 4, all_cores, nullptr),
        make_cb(tt::CBIndex::c_9, tt::DataFormat::Float16_b, true, 4, all_cores, nullptr),
        make_cb(tt::CBIndex::c_1, tt::DataFormat::Float16_b, true, 224, all_cores, input_buffer),
        make_cb(tt::CBIndex::c_4, tt::DataFormat::Float16_b, true, 1, all_cores, output_buffer),
    };

    // Create compile args for the program
    const auto tensors = std::vector<tt::tt_metal::Buffer*>{input_buffer, weight_buffer, output_buffer};

    std::vector<uint32_t> compile_args;
    for (const auto* buffer : tensors) {
        tt::tt_metal::TensorAccessorArgs(*buffer).append_to(compile_args);
    }

    // Create optimal ring ordering for NOC1 to minimize traffic conflicts
    // NOC1 routes: decreasing y (top) first, then decreasing x (left)
    // Sort cores by (descending y, descending x) to create a ring that flows naturally
    std::vector<uint32_t> ring_pos2bank_id(num_cores);
    std::iota(ring_pos2bank_id.begin(), ring_pos2bank_id.end(), 0);
    auto* device = tensor_args.input_tensor.device();

    std::sort(
        ring_pos2bank_id.begin(),
        ring_pos2bank_id.end(),
        [device, &dram_bank2core_coords](uint32_t bank_id_a, uint32_t bank_id_b) {
            const auto& pa = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_a]);
            const auto& pb = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_b]);
            if (pa.y != pb.y) {
                return pa.y > pb.y;  // Descending y
            }
            return pa.x > pb.x;  // Descending x
        });

    // For every third core, figure out the physical coords of the two cores after it.
    std::unordered_map<uint32_t, std::array<uint32_t, 5>> dram_bank2neighbors;
    for (uint32_t ring_pos = 0; ring_pos < num_cores; ring_pos += 3) {
        auto bank_id = ring_pos2bank_id[ring_pos];
        auto bank_id_next1 = ring_pos2bank_id[ring_pos + 1];
        auto bank_id_next2 = ring_pos2bank_id[ring_pos + 2];

        const auto& core_next1 = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_next1]);
        const auto& core_next2 = device->worker_core_from_logical_core(dram_bank2core_coords[bank_id_next2]);

        dram_bank2neighbors[bank_id] = {1, core_next1.x, core_next1.y, core_next2.x, core_next2.y};
    }

    // We also need the reverse mapping for bank_id to N tile_id
    std::unordered_map<uint32_t, uint32_t> bank2tile_id;
    uint32_t tile_id = 0;
    for (uint32_t core_id = 0; core_id < num_cores; core_id++) {
        if ((core_id % 3) == 0) {
            continue;
        }
        bank2tile_id[ring_pos2bank_id[core_id]] = tile_id++;
    }

    const auto& collector_core = device->worker_core_from_logical_core(dram_bank2core_coords[ring_pos2bank_id[11]]);
    const auto& first_core = device->worker_core_from_logical_core(dram_bank2core_coords[ring_pos2bank_id[0]]);

    const KernelDescriptor::NamedCompileTimeArgs named_compile_time_args = {
        {"layer_id", operation_attributes.layer_id},
        {"num_cores", num_cores},
        {"collector_physical_x", collector_core.x},
        {"collector_physical_y", collector_core.y},
        {"first_physical_x", first_core.x},
        {"first_physical_y", first_core.y},
        {"column_id", operation_attributes.column_id},
    };

    const std::string dm0_kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/kernels/dm0.cpp";
    const std::string dm1_kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/kernels/dm1.cpp";
    const std::string compute_kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/kernels/compute.cpp";

    KernelDescriptor dm0_kernel{
        .kernel_source = dm0_kernel_source,
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .config = ReaderConfigDescriptor{},
    };

    KernelDescriptor dm1_kernel{
        .kernel_source = dm1_kernel_source,
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .config = WriterConfigDescriptor{},
    };

    KernelDescriptor compute_kernel{
        .kernel_source = compute_kernel_source,
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = compile_args,
        .named_compile_time_args = named_compile_time_args,
        .config =
            ComputeConfigDescriptor{
                .math_fidelity = tt::tt_metal::MathFidelity::LoFi,
                .fp32_dest_acc_en = false,
                .dst_full_sync_en = false,
                .bfp8_pack_precise = false,
                .math_approx_mode = true,
            },
    };

    // Create semaphores to wait for the partial to arrive from the other core
    // There will be 8 cores, each waiting for partial to come from 4 other cores.
    // The 4 cores will send partial to two cores each.
    program_desc.semaphores.push_back(SemaphoreDescriptor{
        .id = kPartialSemaphoreId,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });
    program_desc.semaphores.push_back(SemaphoreDescriptor{
        .id = kRawScoresSemaphoreId,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });

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

        const bool is_sender = dram_bank2neighbors.contains(dram_bank);
        const auto neighbors = is_sender ? dram_bank2neighbors[dram_bank] : std::array<uint32_t, 5>{};
        const uint32_t core_id = is_sender ? 0u : bank2tile_id[dram_bank];

        KernelDescriptor::RTArgList runtime_args;
        runtime_args.reserve(13);
        runtime_args.push_back(dram_bank);
        runtime_args.push_back(vchannel);
        runtime_args.push_back(input_buffer);
        runtime_args.push_back(weight_buffer);
        runtime_args.push_back(output_buffer);
        runtime_args.push_back(kPartialSemaphoreId);
        runtime_args.push_back(is_sender ? neighbors[0] : 0u);
        runtime_args.push_back(is_sender ? neighbors[1] : 0u);
        runtime_args.push_back(is_sender ? neighbors[2] : 0u);
        runtime_args.push_back(is_sender ? neighbors[3] : 0u);
        runtime_args.push_back(is_sender ? neighbors[4] : 0u);
        runtime_args.push_back(core_id);
        runtime_args.push_back(kRawScoresSemaphoreId);

        dm0_kernel.emplace_runtime_args(core, runtime_args);
        dm1_kernel.emplace_runtime_args(core, runtime_args);
        compute_kernel.emplace_runtime_args(core, runtime_args);

        dram_bank++;
    }

    program_desc.kernels.reserve(3);
    program_desc.kernels.push_back(std::move(dm0_kernel));
    program_desc.kernels.push_back(std::move(dm1_kernel));
    program_desc.kernels.push_back(std::move(compute_kernel));
    return program_desc;
}

}  // namespace ttnn::operations::experimental::deepseek::moe::moe_gate_mm
