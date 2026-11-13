// Zone-profile helpers of moe_compute (compiled out unless the device profiler is on: MZ_WALL() = 0 and the timestamped data events vanish).
#pragma once
#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"

#if defined(PROFILE_KERNEL) && (PROFILE_KERNEL)
#define MZ_WALL() (*reinterpret_cast<volatile uint32_t*>(0xFFB121F0))
#else
#define MZ_WALL() 0u
#endif
// timestamped event: the event time is the time of the call, data = 64 bit payload
#define MZ_TS(name, v) DeviceTimestampedData(name, (uint64_t)(v))
#define MZ_PACK(hi, lo) ((((uint64_t)(hi)) << 32) | (uint64_t)(lo))
