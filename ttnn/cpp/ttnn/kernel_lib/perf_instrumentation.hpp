// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// MaybeDeviceZoneScope(name): a named device-profiler zone (RAII, start at the macro, end at the closing
// brace of the enclosing block). Free when the device profiler is off (DeviceZoneScopedN compiles to
// nothing), so kernels keep their stage zones PERMANENTLY.
// Durability contract: stage zones placed by perf passes are never removed; extend them to new paths.
// Budget: ~125 zone executions per RISC per dispatch, silently dropped beyond it — keep zones out of
// per-tile loops. See .claude/references/device-zone-scope-attribution.md.
#include "api/debug/kernel_profiler.hpp"

#ifndef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#endif
