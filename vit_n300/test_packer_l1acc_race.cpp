// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Test to trigger the PACKER_L1_ACC reconfiguration race condition.
// See vit_n300/memory.md for full analysis.

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <map>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal {

using std::vector;

bool run_l1acc_race_test(const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t num_tiles) {
    bool pass = true;

    constexpr uint32_t in_cb_index = 0;
    constexpr uint32_t out_cb_index = 16;
    constexpr uint32_t single_tile_size = 2 * 32 * 32;  // bfloat16, 32x32
    const size_t output_buffer_size = num_tiles * single_tile_size;

    CoreCoord core = {0, 0};

    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_out_config{
        .device = device,
        .size = output_buffer_size,
        .page_size = output_buffer_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    auto output_dram_buffer = CreateBuffer(dram_out_config);
    uint32_t output_dram_addr = output_dram_buffer->address();

    // Input CB: double-buffered
    tt_metal::CircularBufferConfig in_cb_config =
        tt_metal::CircularBufferConfig(2 * single_tile_size, {{in_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(in_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program_, core, in_cb_config);

    // Output CB: double-buffered
    tt_metal::CircularBufferConfig out_cb_config =
        tt_metal::CircularBufferConfig(2 * single_tile_size, {{out_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(out_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program_, core, out_cb_config);

    // Reader kernel: generates constant-value tiles (bfloat16 1.0) in L1
    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "vit_n300/kernels/reader_l1acc_hammer.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    // Compute kernel: copy in_cb -> out_cb with L1_ACC toggling
    vector<uint32_t> compute_compile_args = {num_tiles};
    std::map<std::string, std::string> compute_defines = {{"PACKER_L1_ACC", "1"}};
    [[maybe_unused]] auto compute_kernel = tt_metal::CreateKernel(
        program_,
        "vit_n300/kernels/compute_l1acc_hammer.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = true,
            .compile_args = compute_compile_args,
            .defines = compute_defines});

    // Writer kernel: reads from out_cb, writes to DRAM
    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/"
        "unit_tests/dram/direct_writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    // Reader runtime args: just total tile count
    tt_metal::SetRuntimeArgs(program_, reader_kernel, core, {num_tiles});

    // Writer runtime args: DRAM addr, bank_id, num_tiles
    tt_metal::SetRuntimeArgs(program_, writer_kernel, core, {output_dram_addr, (uint32_t)0, num_tiles});

    fmt::print("Running L1_ACC race test: {} tiles on core (0,0)\n", num_tiles);

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Read back output
    std::vector<uint32_t> result_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, result_data);

    // Each tile should contain bfloat16(1.0) = 0x3F80 in every position
    // (When L1_ACC=0: overwrite with dest value = 1.0)
    // (When L1_ACC=1: accumulate dest value to existing L1 content)
    // If race causes wrong L1_ACC mode, values will differ from expected.
    float expected_val = 1.0f;
    float tolerance = 0.01f;
    uint32_t errors = 0;

    for (uint32_t i = 0; i < result_data.size(); i++) {
        uint16_t lo = result_data[i] & 0xFFFF;
        uint16_t hi = (result_data[i] >> 16) & 0xFFFF;
        float flo = static_cast<float>(bfloat16(lo));
        float fhi = static_cast<float>(bfloat16(hi));

        if (std::abs(flo - expected_val) > tolerance) {
            if (errors < 10) {
                fmt::print("  ERR word {} lo: {} != {}\n", i, flo, expected_val);
            }
            errors++;
        }
        if (std::abs(fhi - expected_val) > tolerance) {
            if (errors < 10) {
                fmt::print("  ERR word {} hi: {} != {}\n", i, fhi, expected_val);
            }
            errors++;
        }
    }

    if (errors > 0) {
        fmt::print("\n*** L1_ACC RACE DETECTED! {} errors / {} elements ***\n\n", errors, result_data.size() * 2);
        pass = false;
    } else {
        fmt::print("  All {} tiles OK (all elems ~= {}). No race detected.\n", num_tiles, expected_val);
    }

    return pass;
}

TEST_F(MeshDeviceFixture, TensixPackerL1AccRaceSmall) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_l1acc_race_test(devices_.at(id), 10));
    }
}

TEST_F(MeshDeviceFixture, TensixPackerL1AccRace) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_l1acc_race_test(devices_.at(id), 10000));
    }
}

TEST_F(MeshDeviceFixture, TensixPackerL1AccRaceStress) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_l1acc_race_test(devices_.at(id), 100000));
    }
}

}  // namespace tt::tt_metal
