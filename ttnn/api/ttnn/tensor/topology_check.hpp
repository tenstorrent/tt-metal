// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

// Replica groups of a label: distribution coordinates that differ only along Replicate dimensions belong to one
// group, keyed by their coordinates at the Shard dimensions in row-major order; keys run from 0 to
// num_replica_groups(topology) - 1. verify_shards_match_topology and the flatbuffer writer group shards this way.
size_t num_replica_groups(const tt::tt_metal::TensorTopology& topology);
size_t replica_group_key(
    const tt::tt_metal::TensorTopology& topology, const tt::tt_metal::distributed::MeshCoordinate& distribution_coord);

// Checks that a host tensor's topology label describes its shards, and throws (TT_FATAL) naming the mesh coordinates
// and the label where it does not:
//  - every populated local shard sits at a coordinate the label lists, and every labelled coordinate holds a shard
//    (or is remote to this host, as in multi-host LOCAL dumps);
//  - shards the label calls replicas of each other (coordinates that differ only along Replicate axes) have the
//    same size and, with `compare_replica_contents`, the same values (tile padding and alignment bytes, which hold
//    whatever the producing op left there, are not compared; whatever keeps one copy per group keeps that copy's
//    padding).
// Device ops relabel their outputs, so a label can be wrong, and anything that keeps one copy per replica group
// would then drop a shard silently. The flatbuffer writer runs this first; it is exported for any other caller that
// keeps one copy per group, such as a checkpoint gather that composes a tensor by its label.
void verify_shards_match_topology(const Tensor& tensor, bool compare_replica_contents = true);

}  // namespace ttnn
