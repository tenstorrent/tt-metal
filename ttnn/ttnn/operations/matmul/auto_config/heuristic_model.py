# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Heuristic and lightweight DNN-based model for predicting optimal matmul
configurations on Tenstorrent hardware.

The HeuristicConfigPredictor uses a multi-signal scoring function that
captures the key performance drivers:
  1. Compute utilization (maximize tile-level parallelism across cores)
  2. Memory efficiency (maximize data reuse within L1)
  3. Load balance (even distribution of work across the core grid)
  4. Subblock efficiency (larger subblocks reduce overhead)

The DNNConfigPredictor uses a small feedforward network trained on
profiling data. It can be retrained when the underlying matmul
implementation changes.
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config_space import (
    DeviceConstraints,
    MatmulConfig,
    MatmulShape,
    enumerate_candidate_configs,
    estimate_l1_usage,
    get_valid_subblock,
)


@dataclass
class ScoringWeights:
    """Weights for the heuristic scoring function. Tunable."""
    compute_utilization: float = 0.35
    memory_efficiency: float = 0.25
    load_balance: float = 0.25
    subblock_efficiency: float = 0.15


def score_config(config: MatmulConfig, shape: MatmulShape,
                 constraints: DeviceConstraints,
                 weights: Optional[ScoringWeights] = None) -> float:
    if weights is None:
        weights = ScoringWeights()
    M_tiles = shape.M_tiles
    K_tiles = shape.K_tiles
    N_tiles = shape.N_tiles
    num_cores = constraints.num_cores
    tiles_per_core = config.M_block_size * config.N_block_size
    total_output_tiles = M_tiles * N_tiles
    num_m_blocks = math.ceil(M_tiles / config.M_block_size)
    num_n_blocks = math.ceil(N_tiles / config.N_block_size)
    total_blocks = num_m_blocks * num_n_blocks
    active_cores = min(total_blocks, num_cores)
    compute_util = active_cores / num_cores
    k_reuse = config.K_block_size / K_tiles
    l1_usage = estimate_l1_usage(config.M_block_size, config.K_block_size, config.N_block_size, constraints.tile_size_bytes)
    l1_ratio = l1_usage / constraints.max_l1_bytes
    if l1_ratio < 0.4:
        mem_eff = 0.5 + l1_ratio
    elif l1_ratio < 0.8:
        mem_eff = 1.0
    else:
        mem_eff = max(0.1, 1.0 - (l1_ratio - 0.8) * 5)
    memory_score = 0.5 * k_reuse + 0.5 * mem_eff
    if total_blocks >= num_cores:
        blocks_per_core = total_blocks / num_cores
        fractional_waste = blocks_per_core - math.floor(blocks_per_core)
        load_bal = 1.0 - fractional_waste
    else:
        load_bal = total_blocks / num_cores
    subblock_product = config.subblock_h * config.subblock_w
    max_subblock = constraints.subblock_max_product
    subblock_eff = subblock_product / max_subblock
    score = (weights.compute_utilization * compute_util + weights.memory_efficiency * memory_score + weights.load_balance * load_bal + weights.subblock_efficiency * subblock_eff)
    return score


class HeuristicConfigPredictor:
    def __init__(self, weights=None):
        self.weights = weights or ScoringWeights()
    def predict(self, shape, constraints, top_k=1):
        candidates = enumerate_candidate_configs(shape, constraints)
        if not candidates:
            sub_h, sub_w = get_valid_subblock(1, 1, constraints.fp32_dest_acc_en)
            return [MatmulConfig(M_block_size=1, K_block_size=1, N_block_size=1, subblock_h=sub_h, subblock_w=sub_w, grid_x=constraints.grid_x, grid_y=constraints.grid_y)]
        scored = [(score_config(c, shape, constraints, self.weights), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]


class DNNConfigPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self._model = None
        self._config_index = None
        self._feature_stats = None
    def _featurize(self, shape, constraints):
        M_tiles = shape.M_tiles
        K_tiles = shape.K_tiles
        N_tiles = shape.N_tiles
        num_cores = constraints.num_cores
        return [math.log2(max(M_tiles, 1)), math.log2(max(K_tiles, 1)), math.log2(max(N_tiles, 1)), math.log2(max(num_cores, 1)), M_tiles / max(K_tiles, 1), N_tiles / max(K_tiles, 1), M_tiles * N_tiles / max(num_cores, 1), constraints.grid_x / max(constraints.grid_y, 1), 1.0 if constraints.fp32_dest_acc_en else 0.0, math.log2(max(shape.batch_size, 1))]
    def load(self, model_path=None):
        path = model_path or self.model_path
        if path is None or not os.path.exists(path): return False
        with open(path, 'r') as f: data = json.load(f)
        self._model = {'w1': data['weights']['w1'], 'b1': data['weights']['b1'], 'w2': data['weights']['w2'], 'b2': data['weights']['b2']}
        self._config_index = {int(k): tuple(v) for k, v in data['config_index'].items()}
        self._feature_stats = data.get('feature_stats')
        return True
    def predict(self, shape, constraints, top_k=1):
        if self._model is None: return HeuristicConfigPredictor().predict(shape, constraints, top_k)
        features = self._featurize(shape, constraints)
        if self._feature_stats:
            means = self._feature_stats['mean']
            stds = self._feature_stats['std']
            features = [(f - m) / max(s, 1e-8) for f, m, s in zip(features, means, stds)]
        hidden = self._matmul_vec(self._model['w1'], features, self._model['b1'])
        hidden = [max(0, x) for x in hidden]
        logits = self._matmul_vec(self._model['w2'], hidden, self._model['b2'])
        indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
        results = []
        for idx, _ in indexed[:top_k]:
            if idx in self._config_index:
                m, k, n, sh, sw = self._config_index[idx]
                sub_h, sub_w = get_valid_subblock(m, n, constraints.fp32_dest_acc_en)
                results.append(MatmulConfig(M_block_size=m, K_block_size=k, N_block_size=n, subblock_h=sub_h, subblock_w=sub_w, grid_x=constraints.grid_x, grid_y=constraints.grid_y))
        if not results: return HeuristicConfigPredictor().predict(shape, constraints, top_k)
        return results
    @staticmethod
    def _matmul_vec(weights, vec, bias):
        return [sum(w * v for w, v in zip(row, vec)) + bias[i] for i, row in enumerate(weights)]


class TrainingDataCollector:
    def __init__(self): self.data = []
    def record(self, shape, constraints, config, latency_us, throughput_tflops=None):
        self.data.append({'M': shape.M, 'K': shape.K, 'N': shape.N, 'batch_size': shape.batch_size, 'grid_x': constraints.grid_x, 'grid_y': constraints.grid_y, 'fp32_dest_acc_en': constraints.fp32_dest_acc_en, 'M_block': config.M_block_size, 'K_block': config.K_block_size, 'N_block': config.N_block_size, 'subblock_h': config.subblock_h, 'subblock_w': config.subblock_w, 'latency_us': latency_us, 'throughput_tflops': throughput_tflops})
    def save(self, path):
        with open(path, 'w') as f: json.dump(self.data, f, indent=2)
    def load(self, path):
        with open(path, 'r') as f: self.data = json.load(f)
    def train_model(self, output_path, hidden_dim=64, epochs=100, lr=0.01):
        if not self.data: raise ValueError("No training data collected")
        groups = {}
        for d in self.data:
            key = (d['M'], d['K'], d['N'], d['grid_x'], d['grid_y'])
            groups.setdefault(key, []).append(d)
        config_set = set()
        for d in self.data: config_set.add((d['M_block'], d['K_block'], d['N_block'], d['subblock_h'], d['subblock_w']))
        config_list = sorted(config_set)
        config_to_idx = {c: i for i, c in enumerate(config_list)}
        X, y = [], []
        for key, entries in groups.items():
            best = min(entries, key=lambda e: e['latency_us'])
            M, K, N, gx, gy = key
            p = DNNConfigPredictor()
            X.append(p._featurize(MatmulShape(M, K, N), DeviceConstraints(gx, gy)))
            y.append(config_to_idx[(best['M_block'], best['K_block'], best['N_block'], best['subblock_h'], best['subblock_w'])])
        nf = len(X[0])
        means = [sum(x[i] for x in X) / len(X) for i in range(nf)]
        stds = [math.sqrt(sum((x[i] - means[i])**2 for x in X) / len(X)) for i in range(nf)]
        Xn = [[(x[i] - means[i]) / max(stds[i], 1e-8) for i in range(nf)] for x in X]
        import random
        nc = len(config_list)
        rm = lambda r, c, s=0.1: [[random.gauss(0, s) for _ in range(c)] for _ in range(r)]
        w1, b1, w2, b2 = rm(hidden_dim, nf), [0.0]*hidden_dim, rm(nc, hidden_dim), [0.0]*nc
        losses = []
        for epoch in range(epochs):
            tl = 0
            for xi, yi in zip(Xn, y):
                h = [max(0, sum(w1[j][k] * xi[k] for k in range(nf)) + b1[j]) for j in range(hidden_dim)]
                log = [sum(w2[j][k] * h[k] for k in range(hidden_dim)) + b2[j] for j in range(nc)]
                mx = max(log)
                exp = [math.exp(l - mx) for l in log]
                se = sum(exp)
                pr = [e / se for e in exp]
                tl += -math.log(max(pr[yi], 1e-10))
                dl = list(pr)
                dl[yi] -= 1.0
                for j in range(nc):
                    for k in range(hidden_dim): w2[j][k] -= lr * dl[j] * h[k]
                    b2[j] -= lr * dl[j]
                dh = [sum(w2[j][k] * dl[j] for j in range(nc)) for k in range(hidden_dim)]
                dhr = [dh[k] if h[k] > 0 else 0.0 for k in range(hidden_dim)]
                for j in range(hidden_dim):
                    for k in range(nf): w1[j][k] -= lr * dhr[j] * xi[k]
                    b1[j] -= lr * dhr[j]
            losses.append(tl / len(Xn))
        md = {'weights': {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}, 'config_index': {str(i): list(c) for i, c in enumerate(config_list)}, 'feature_stats': {'mean': means, 'std': stds}, 'training_info': {'n_samples': len(X), 'n_configs': nc, 'final_loss': losses[-1] if losses else None}}
        with open(output_path, 'w') as f: json.dump(md, f, indent=2)
        return {'n_samples': len(X), 'n_configs': nc, 'final_loss': losses[-1] if losses else None, 'losses': losses}
