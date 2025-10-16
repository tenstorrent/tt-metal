#pragma once

#include <stdint.h>

FORCE_INLINE uint32_t div_up(const uint32_t a, const uint32_t b) { return static_cast<uint32_t>((a + b - 1) / b); }
