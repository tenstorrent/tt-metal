// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device.hpp"
#include "tt_metal/impl/device/device_pool.hpp"

namespace ttnn {

namespace device {

Device &open_device(int device_id, size_t l1_small_size, size_t trace_region_size, tt::tt_metal::DispatchCoreType dispatch_core_type) {
    tt::DevicePool::initialize({device_id}, 1, l1_small_size, trace_region_size, dispatch_core_type, {});
    return *(tt::DevicePool::instance().get_active_device(device_id));
}

bool is_device_open(int device_id){
    return tt::DevicePool::instance().is_device_active(device_id);
}

void enable_program_cache(Device &device) {
    device.enable_program_cache();
}

void disable_and_clear_program_cache(Device &device) {
    device.disable_and_clear_program_cache();
}

float sfpu_positive_nan(DataType dtype) {
    union {
        uint32_t i;
        float f;
    } u;
    switch (dtype) {
        case DataType::BFLOAT16: {
            u.i = { 0x7FFF };
            return u.f;
        }
        case DataType::FLOAT32: {
            u.i = { 0x7FFFFFFF };
            return u.f;
        }
        default:
            return std::numeric_limits<float>::quiet_NaN();
    }
}

float sfpu_negative_nan(DataType dtype) {
    union {
        uint32_t i;
        float f;
    } u;
    switch (dtype) {
        case DataType::BFLOAT16: {
            u.i = { 0xFFFF };
            return u.f;
        }
        case DataType::FLOAT32: {
            u.i = { 0xFFFFFFFF };
            return u.f;
        }
        default:
            return -std::numeric_limits<float>::quiet_NaN();
    }
}

float sfpu_positive_inf(DataType dtype) {

    switch (dtype) {
        case DataType::BFLOAT16:
            return 0x7F80;
        case DataType::FLOAT32:
            return 0x7F800000;
        default:
            return std::numeric_limits<float>::infinity();
    }
    return std::numeric_limits<float>::infinity();
}

float sfpu_negative_inf(DataType dtype) {

    switch (dtype) {
        case DataType::BFLOAT16:
            return 0xFF80;
        case DataType::FLOAT32:
            return 0xFF800000;
        default:
            return -std::numeric_limits<float>::infinity();
    }
    return -std::numeric_limits<float>::infinity();
}

void close_device(Device &device) {
    tt::DevicePool::instance().close_device(device.id());

}

bool is_wormhole_or_blackhole(tt::ARCH arch) {
    return arch == tt::ARCH::WORMHOLE_B0 or arch == tt::ARCH::BLACKHOLE;
}

void deallocate_buffers(Device* device) {
        device->push_work([device] () mutable {
            device->deallocate_buffers();
        });
}

}  // namespace device

using namespace device;

}  // namespace ttnn
