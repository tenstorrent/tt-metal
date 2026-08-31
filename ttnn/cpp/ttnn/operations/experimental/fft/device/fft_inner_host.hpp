// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_inner_host.hpp — Tile constants and small helpers shared by the FFT
//                      ProgramDescriptor factories in device/*_factory.cpp.
//
// Provides:
//   * kTileHW / kTileElems / kTileSizeFp32 — Tensix tile geometry
//   * make_mesh_buf() — thin helper that allocates a DRAM ReplicatedBuffer
//   * buf_addr()      — device-buffer address extractor
//   * log2u() / bit_rev() — tiny arithmetic helpers used by twiddle math
//
// All host-side end-to-end reference FFT code that previously lived in
// this header (run_fft, FFTPlan, make_plan, execute, plan_cache, the
// std::vector<Complex>-based fft() overloads) has been removed — the
// runtime path goes exclusively through the ProgramDescriptor factories
// in device/*_factory.cpp.

#pragma once

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_buffer.hpp"

#include <cstdint>
#include <memory>

namespace fft_example {

using tt::tt_metal::BufferType;
using tt::tt_metal::distributed::DeviceLocalBufferConfig;
using tt::tt_metal::distributed::MeshBuffer;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::ReplicatedBufferConfig;

constexpr uint32_t kTileHW = 32;
constexpr uint32_t kTileElems = kTileHW * kTileHW;              // 1024
constexpr uint32_t kTileSizeFp32 = kTileElems * sizeof(float);  // 4096 bytes

inline std::shared_ptr<MeshBuffer> make_mesh_buf(std::shared_ptr<MeshDevice> md, uint32_t size, uint32_t page_size) {
    ReplicatedBufferConfig rep{.size = size};
    DeviceLocalBufferConfig dev{.page_size = page_size, .buffer_type = BufferType::DRAM};
    return MeshBuffer::create(rep, dev, md.get());
}

inline uint32_t buf_addr(const std::shared_ptr<MeshBuffer>& mb) {
    return mb->get_device_buffer(MeshCoordinate(0, 0))->address();
}

inline uint32_t log2u(uint32_t x) {
    uint32_t r = 0;
    while ((1u << r) < x) {
        ++r;
    }
    return r;
}

inline uint32_t bit_rev(uint32_t x, uint32_t bits) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1u);
        x >>= 1u;
    }
    return r;
}

}  // namespace fft_example
