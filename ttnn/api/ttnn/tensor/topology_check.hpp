// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

// Checks that a host tensor's topology label describes its shards, and throws (TT_FATAL) naming the mesh coordinates
// and the label where it does not:
//  - every populated local shard sits at a coordinate the label lists, and every labelled coordinate holds a shard
//    (or is remote to this host, as in multi-host LOCAL dumps);
//  - shards the label calls replicas of each other (coordinates that differ only along Replicate axes) have the
//    same size and, with `compare_replica_contents`, the same bytes.
// Device ops relabel their outputs, so a label can be wrong, and anything that keeps one copy per replica group (the
// flatbuffer writer, tt-train's checkpoint gather) would then drop a shard silently; those callers run this first.
void verify_shards_match_topology(const Tensor& tensor, bool compare_replica_contents = true);

}  // namespace ttnn
