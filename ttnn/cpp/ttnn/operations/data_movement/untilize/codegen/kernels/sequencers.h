// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilize-local identity sequencer only. Full unified sequencers live in
// data_movement/common/kernels/codegen/ (see PR #52806); switch untilize to
// those once that lands.
#pragma once

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t SEQ_IDENTITY = 0;

struct SeqIdentityState {
    uint32_t page_id;
};

inline __attribute__((always_inline)) SeqIdentityState seq_identity_init(uint32_t start_id) { return {start_id}; }

inline __attribute__((always_inline)) uint32_t seq_identity_next(SeqIdentityState& st) { return st.page_id++; }
