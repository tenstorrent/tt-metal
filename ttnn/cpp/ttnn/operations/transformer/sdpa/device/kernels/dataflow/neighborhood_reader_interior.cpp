// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

// Interior program. Skip value comes from factory -DNA_PATH_KIND=1 -> NA_SKIP_IF 2.
#define NA_HAS_PATH_SKIP
#define NA_SKIP_REV 8
#include "neighborhood_reader.cpp"
