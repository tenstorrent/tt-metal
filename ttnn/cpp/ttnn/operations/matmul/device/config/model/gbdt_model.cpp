// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/model/gbdt_model.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <string_view>

namespace ttnn::operations::matmul::model {

namespace {

// LightGBM's kZeroThreshold (a float constant, compared after promotion to double)
constexpr double kZeroThreshold = static_cast<double>(1e-35f);

// LightGBM's dense-row loader drops |x| <= kZeroThreshold (so -0.0 and denormals read as 0.0); NaN passes through.
inline double canonical(double v) { return (std::fabs(v) > kZeroThreshold || std::isnan(v)) ? v : 0.0; }

}  // namespace

// One step of Tree::GetLeaf at global node i (Tree::NumericalDecision): the tree-local child row x goes to.
inline std::int32_t GbdtModel::step(std::uint32_t i, const double* x) const {
    const std::uint8_t d = decision[i];
    const std::uint8_t missing = d >> kGbdtMissingShift;
    double v = canonical(x[split_feature[i]]);
    if (std::isnan(v) && missing != kGbdtMissingNaN) {
        v = 0.0;
    }
    bool go_left;
    if ((missing == kGbdtMissingZero && v >= -kZeroThreshold && v <= kZeroThreshold) ||
        (missing == kGbdtMissingNaN && std::isnan(v))) {
        go_left = (d & kGbdtDefaultLeft) != 0;
    } else {
        go_left = v <= thresholds[threshold_index[i]];
    }
    return go_left ? left_child[i] : right_child[i];
}

// Leaf index (into leaf_value) that row x reaches in tree t.
inline std::uint32_t GbdtModel::leaf_of(std::size_t t, const double* x) const {
    const std::uint32_t base = node_begin[t];
    if (node_begin[t + 1] == base) {  // single-leaf tree
        return leaf_begin[t];
    }
    std::int32_t node = 0;
    while (node >= 0) {
        node = step(base + static_cast<std::uint32_t>(node), x);
    }
    return leaf_begin[t] + static_cast<std::uint32_t>(~node);
}

double GbdtModel::predict(std::span<const double> x) const {
    if (x.size() < num_features) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // GBDT::PredictRaw: a plain left-to-right double sum over trees
    double sum = 0.0;
    for (std::size_t t = 0; t < num_trees(); ++t) {
        sum += leaf_value[leaf_of(t, x.data())];
    }
    return sum;
}

void GbdtModel::predict_batch(std::span<const double> rows, std::span<double> out) const {
    const std::size_t n = num_features == 0 ? 0 : std::min(out.size(), rows.size() / num_features);
    std::fill(out.begin(), out.end(), std::numeric_limits<double>::quiet_NaN());
    std::fill_n(out.begin(), n, 0.0);
    // Tree-major: each tree stays in cache across the rows, and consecutive rows' walks are independent load
    // chains the CPU overlaps. Each row still sums its trees in order, so results match predict() bit for bit.
    for (std::size_t t = 0; t < num_trees(); ++t) {
        for (std::size_t r = 0; r < n; ++r) {
            out[r] += leaf_value[leaf_of(t, rows.data() + r * num_features)];
        }
    }
}

std::string GbdtModel::structure_error() const {
    if (num_features == 0 || num_features > kGbdtMaxFeatures) {
        return "num_features out of range";
    }
    if (node_begin.size() != leaf_begin.size() || node_begin.empty()) {
        return "node_begin/leaf_begin size mismatch";
    }
    const std::size_t nodes = split_feature.size();
    if (decision.size() != nodes || threshold_index.size() != nodes || left_child.size() != nodes ||
        right_child.size() != nodes) {
        return "per-node array size mismatch";
    }
    if (node_begin.front() != 0 || node_begin.back() != nodes || leaf_begin.front() != 0 ||
        leaf_begin.back() != leaf_value.size()) {
        return "tree offsets do not cover the node/leaf arrays";
    }
    for (std::size_t t = 0; t < num_trees(); ++t) {
        // Bound this tree's end offsets before walking its nodes: a later tree's offsets are only checked later
        if (node_begin[t + 1] < node_begin[t] || leaf_begin[t + 1] < leaf_begin[t] || node_begin[t + 1] > nodes ||
            leaf_begin[t + 1] > leaf_value.size()) {
            return "tree offsets not monotone or out of range";
        }
        const std::int64_t tree_nodes = node_begin[t + 1] - node_begin[t];
        const std::int64_t tree_leaves = leaf_begin[t + 1] - leaf_begin[t];
        if (tree_leaves != tree_nodes + 1) {
            return "tree " + std::to_string(t) + ": leaves != internal nodes + 1";
        }
        for (std::int64_t j = 0; j < tree_nodes; ++j) {
            const std::size_t i = node_begin[t] + j;
            if (split_feature[i] >= num_features || threshold_index[i] >= thresholds.size() ||
                (decision[i] >> kGbdtMissingShift) > kGbdtMissingNaN) {
                return "tree " + std::to_string(t) + ": bad node " + std::to_string(j);
            }
            for (const std::int16_t c : {left_child[i], right_child[i]}) {
                // children of a node come after it, so every walk terminates
                if (c >= 0 ? (c <= j || c >= tree_nodes) : (~c >= tree_leaves)) {
                    return "tree " + std::to_string(t) + ": bad child at node " + std::to_string(j);
                }
            }
        }
    }
    return {};
}

namespace {

constexpr std::string_view kMagic = "TTGBDT01";
constexpr std::uint32_t kBlobVersion = 1;
constexpr std::size_t kHeaderBytes = 8 + 8 * 4 + 32;

static_assert(std::endian::native == std::endian::little, "the blob is little-endian");

// Reads the blob front to back; a read past its end reads nothing and marks the blob truncated
class Reader {
public:
    explicit Reader(std::span<const std::byte> b) : b_(b) {}

    bool truncated() const { return truncated_; }

    template <typename T>
    void read(std::vector<T>& out, std::size_t count) {
        const std::size_t bytes = count * sizeof(T);
        if (!need(bytes)) {
            return;
        }
        out.resize(count);
        if (bytes != 0) {
            std::memcpy(out.data(), b_.data() + pos_, bytes);
        }
        pos_ += bytes;
        pos_ += (8 - pos_ % 8) % 8;  // sections are 8-byte aligned
    }

    std::uint32_t u32() {
        std::uint32_t v = 0;
        if (need(4)) {
            std::memcpy(&v, b_.data() + pos_, 4);
            pos_ += 4;
        }
        return v;
    }

    std::span<const std::byte> take(std::size_t bytes) {
        if (!need(bytes)) {
            return {};
        }
        auto s = b_.subspan(pos_, bytes);
        pos_ += bytes;
        return s;
    }

private:
    bool need(std::size_t bytes) {
        if (truncated_ || pos_ > b_.size() || b_.size() - pos_ < bytes) {
            truncated_ = true;
        }
        return !truncated_;
    }

    std::span<const std::byte> b_;
    std::size_t pos_ = 0;
    bool truncated_ = false;
};

}  // namespace

std::optional<GbdtModelStorage> GbdtModelStorage::from_blob(std::span<const std::byte> blob, std::string& error) {
    if (blob.size() < kHeaderBytes || std::memcmp(blob.data(), kMagic.data(), kMagic.size()) != 0) {
        error = "not a gbdt blob";
        return std::nullopt;
    }
    Reader r(blob);
    r.take(kMagic.size());
    if (const std::uint32_t version = r.u32(); version != kBlobVersion) {
        error = "unsupported gbdt blob version " + std::to_string(version) + " (expected " +
                std::to_string(kBlobVersion) + ")";
        return std::nullopt;
    }
    const std::uint32_t num_features = r.u32();
    const std::uint32_t num_trees = r.u32();
    const std::uint32_t num_nodes = r.u32();
    const std::uint32_t num_leaves = r.u32();
    const std::uint32_t num_thresholds = r.u32();
    const std::uint32_t names_bytes = r.u32();
    r.u32();  // reserved

    GbdtModelStorage s;
    static constexpr char kHex[] = "0123456789abcdef";
    for (const std::byte c : r.take(32)) {
        s.source_sha256_ += kHex[std::to_integer<unsigned>(c) >> 4];
        s.source_sha256_ += kHex[std::to_integer<unsigned>(c) & 15];
    }
    r.read(s.node_begin_, std::size_t{num_trees} + 1);
    r.read(s.leaf_begin_, std::size_t{num_trees} + 1);
    r.read(s.thresholds_, num_thresholds);
    r.read(s.leaf_value_, num_leaves);
    r.read(s.threshold_index_, num_nodes);
    r.read(s.left_child_, num_nodes);
    r.read(s.right_child_, num_nodes);
    r.read(s.split_feature_, num_nodes);
    r.read(s.decision_, num_nodes);
    const std::span<const std::byte> names_section = r.take(names_bytes);
    if (r.truncated()) {
        error = "gbdt blob truncated (" + std::to_string(blob.size()) + " bytes)";
        return std::nullopt;
    }
    std::string names(reinterpret_cast<const char*>(names_section.data()), names_section.size());
    for (std::size_t p = 0; p < names.size();) {
        const std::size_t e = names.find('\0', p);
        if (e == std::string::npos) {
            error = "gbdt blob: unterminated feature name";
            return std::nullopt;
        }
        s.feature_names_.push_back(names.substr(p, e - p));
        p = e + 1;
    }
    if (s.feature_names_.size() != num_features) {
        error = "gbdt blob: " + std::to_string(s.feature_names_.size()) + " feature names for " +
                std::to_string(num_features) + " features";
        return std::nullopt;
    }

    s.model_ = GbdtModel{
        .num_features = num_features,
        .node_begin = s.node_begin_,
        .leaf_begin = s.leaf_begin_,
        .split_feature = s.split_feature_,
        .decision = s.decision_,
        .threshold_index = s.threshold_index_,
        .left_child = s.left_child_,
        .right_child = s.right_child_,
        .thresholds = s.thresholds_,
        .leaf_value = s.leaf_value_,
    };
    if (const std::string err = s.model_.structure_error(); !err.empty()) {
        error = "gbdt blob invalid: " + err;
        return std::nullopt;
    }
    return s;
}

}  // namespace ttnn::operations::matmul::model
