// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Evaluator for a LightGBM gradient-boosted tree ensemble (numerical splits only), exported to the blob format read
// below. predict() reproduces LightGBM's
// Booster.predict(raw_score=True) bit for bit: same split rule, same missing-value handling and the
// same summation order over trees.
namespace ttnn::operations::matmul::model {

// decision bits (the blob's compaction of LightGBM's decision_type)
inline constexpr std::uint8_t kGbdtDefaultLeft = 1u << 0;
inline constexpr std::uint8_t kGbdtMissingShift = 1;
inline constexpr std::uint8_t kGbdtMissingNone = 0;
inline constexpr std::uint8_t kGbdtMissingZero = 1;
inline constexpr std::uint8_t kGbdtMissingNaN = 2;

// Upper bound on num_features (split_feature is stored as u8).
inline constexpr std::size_t kGbdtMaxFeatures = 256;

// Non-owning flat struct-of-arrays view of a tree ensemble. Tree t owns internal nodes
// [node_begin[t], node_begin[t + 1]) and leaves [leaf_begin[t], leaf_begin[t + 1]); its root is its first
// internal node, or its only leaf when it has no internal node. Child indices are tree-local: >= 0 is an
// internal node, < 0 is leaf ~child.
struct GbdtModel {
    std::uint32_t num_features = 0;
    std::span<const std::uint32_t> node_begin;  // num_trees + 1
    std::span<const std::uint32_t> leaf_begin;  // num_trees + 1
    // per internal node
    std::span<const std::uint8_t> split_feature;
    std::span<const std::uint8_t> decision;
    std::span<const std::uint16_t> threshold_index;  // into thresholds
    std::span<const std::int16_t> left_child;
    std::span<const std::int16_t> right_child;
    // shared tables
    std::span<const double> thresholds;  // deduplicated split thresholds; a row goes left iff x <= threshold
    std::span<const double> leaf_value;  // per leaf

    std::size_t num_trees() const { return node_begin.empty() ? 0 : node_begin.size() - 1; }

    // Raw score: the sum of the trees' leaf values. x holds num_features values in the model's feature order
    // (NaN marks a missing value); returns NaN if x is shorter. Allocation-free and safe to call concurrently.
    double predict(std::span<const double> x) const;

    // Raw scores of rows.size() / num_features row-major rows into out (same values as predict(), bit for bit).
    // Walks tree by tree over all rows, which keeps each tree cache-resident: use it to score a candidate list.
    void predict_batch(std::span<const double> rows, std::span<double> out) const;

    // Structural check (sizes, index ranges, acyclic children). Returns an empty string when valid.
    std::string structure_error() const;

private:
    std::int32_t step(std::uint32_t i, const double* x) const;
    std::uint32_t leaf_of(std::size_t t, const double* x) const;
};

// Owning storage for a model read from a binary blob.
class GbdtModelStorage {
public:
    // nullopt, with the reason in `error`, for a malformed or truncated blob
    static std::optional<GbdtModelStorage> from_blob(std::span<const std::byte> blob, std::string& error);

    const GbdtModel& model() const { return model_; }
    const std::vector<std::string>& feature_names() const { return feature_names_; }
    // sha256 of the LightGBM text model the blob was exported from
    const std::string& source_sha256() const { return source_sha256_; }

    GbdtModelStorage(const GbdtModelStorage&) = delete;
    GbdtModelStorage& operator=(const GbdtModelStorage&) = delete;
    GbdtModelStorage(GbdtModelStorage&&) = default;  // vector moves keep their buffers, so model_ stays valid
    GbdtModelStorage& operator=(GbdtModelStorage&&) = default;

private:
    GbdtModelStorage() = default;

    std::vector<std::uint32_t> node_begin_, leaf_begin_;
    std::vector<std::uint8_t> split_feature_, decision_;
    std::vector<std::uint16_t> threshold_index_;
    std::vector<std::int16_t> left_child_, right_child_;
    std::vector<double> thresholds_, leaf_value_;
    std::vector<std::string> feature_names_;
    std::string source_sha256_;
    GbdtModel model_;
};

}  // namespace ttnn::operations::matmul::model
