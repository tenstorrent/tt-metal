# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Auto-optimal matmul API -- ttnn.matmul_auto."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ttnn._experimental.auto_config.base import ConfigCandidate, SelectionResult
from ttnn._experimental.auto_config.candidate_generator import generate_matmul_candidates
from ttnn._experimental.auto_config.config_cache import ConfigCache
from ttnn._experimental.auto_config.constraint_validator import validate_candidate
from ttnn._experimental.auto_config.feature_extraction import extract_matmul_features, get_cache_key_from_features
from ttnn._experimental.auto_config.scorer.dnn_scorer import DNNConfigGenerator
from ttnn._experimental.auto_config.scorer.heuristic import HeuristicScorer

import ttnn

logger = logging.getLogger(__name__)

_global_cache: Optional[ConfigCache] = None
_global_scorer: Optional[HeuristicScorer] = None
_global_dnn_generator: Optional[DNNConfigGenerator] = None


def _get_global_dnn_generator() -> DNNConfigGenerator:
    global _global_dnn_generator
    if _global_dnn_generator is None:
        _global_dnn_generator = DNNConfigGenerator()
    return _global_dnn_generator


def _get_global_cache() -> ConfigCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = ConfigCache()
    return _global_cache


_last_selected_config = None
_last_selected_api = None


def get_last_selected_config():
    return _last_selected_config, _last_selected_api


def _get_global_scorer() -> HeuristicScorer:
    global _global_scorer
    if _global_scorer is None:
        _global_scorer = HeuristicScorer()
    return _global_scorer


class MatmulAutoConfig:
    """Auto-config selector for matmul operations.

    Implements feature extraction, candidate generation, validation,
    scoring, selection, and caching.
    """

    def __init__(self, cache: Optional[ConfigCache] = None, scorer: Optional[HeuristicScorer] = None):
        self._cache = cache if cache is not None else _get_global_cache()
        self._scorer = scorer or _get_global_scorer()

    def extract_features(self, *tensors, **kwargs) -> Dict[str, Any]:
        """Extract matmul features from input tensors."""
        if len(tensors) < 2:
            raise ValueError("matmul_auto requires at least 2 input tensors")
        return extract_matmul_features(
            tensors[0],
            tensors[1],
            bias=kwargs.get("bias"),
            transpose_a=kwargs.get("transpose_a", False),
            transpose_b=kwargs.get("transpose_b", False),
            activation=kwargs.get("activation"),
            dtype=kwargs.get("dtype"),
            memory_config=kwargs.get("memory_config"),
        )

    def get_cache_key(self, features: Dict[str, Any]) -> str:
        """Generate cache key from matmul features."""
        return get_cache_key_from_features(features)

    def generate_candidates(self, features: Dict[str, Any]) -> List[ConfigCandidate]:
        """Generate all matmul config candidates."""
        return generate_matmul_candidates(features)

    def validate(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a matmul config candidate."""
        return validate_candidate(
            candidate.config,
            candidate.config_family,
            candidate.backend,
            features,
            math_fidelity=getattr(candidate, "math_fidelity", None),
        )

    def score(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score a matmul config candidate."""
        return self._scorer.score(candidate, features)

    def select(self, *tensors, **kwargs) -> SelectionResult:
        """Select the optimal configuration for the given inputs."""
        start = time.perf_counter()

        features = self.extract_features(*tensors, **kwargs)

        # Check cache
        if self._cache is not None:
            cache_key = self.get_cache_key(features)
            cached = self._cache.get(cache_key, features=features)
            if cached is not None:
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug(f"Cache hit for key: {cache_key[:60]}...")
                return SelectionResult(
                    selected_config=cached,
                    all_candidates=[cached],
                    cache_hit=True,
                    selection_time_ms=elapsed,
                )

        # Generate candidates
        candidates = self.generate_candidates(features)
        logger.debug(f"Generated {len(candidates)} candidates")

        # Validate
        valid_candidates = []
        for candidate in candidates:
            is_valid, reason = self.validate(candidate, features)
            candidate.is_valid = is_valid
            candidate.validation_reason = reason
            if is_valid:
                valid_candidates.append(candidate)

        logger.debug(f"{len(valid_candidates)}/{len(candidates)} candidates passed validation")

        if not valid_candidates:
            logger.warning(f"No valid candidates found. All {len(candidates)} candidates were invalid.")
            elapsed = (time.perf_counter() - start) * 1000
            return SelectionResult(
                selected_config=None,
                all_candidates=candidates,
                cache_hit=False,
                selection_time_ms=elapsed,
            )

        # Score and select
        for candidate in valid_candidates:
            candidate.score = self.score(candidate, features)

        valid_candidates.sort(key=lambda c: c.score, reverse=True)
        selected = valid_candidates[0]

        # Cache the result
        if self._cache is not None:
            cache_key = self.get_cache_key(features)
            self._cache.put(cache_key, selected)

        elapsed = (time.perf_counter() - start) * 1000

        return SelectionResult(
            selected_config=selected,
            all_candidates=candidates,
            cache_hit=False,
            selection_time_ms=elapsed,
        )


def _build_compute_kernel_config(selected: ConfigCandidate) -> Optional[Any]:
    """Build a WormholeComputeKernelConfig with the candidate's math_fidelity."""
    fidelity = getattr(selected, "math_fidelity", None)
    if fidelity is None:
        return None
    try:
        from ttnn._experimental.auto_config.math_fidelity import MathFidelity

        fidelity_map = {
            MathFidelity.LoFi: ttnn.MathFidelity.LoFi,
            MathFidelity.HiFi2: ttnn.MathFidelity.HiFi2,
            MathFidelity.HiFi3: ttnn.MathFidelity.HiFi3,
            MathFidelity.HiFi4: ttnn.MathFidelity.HiFi4,
        }
        ttnn_fidelity = fidelity_map.get(fidelity)
        if ttnn_fidelity is None:
            return None
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn_fidelity,
            math_approx_mode=True,
            fp32_dest_acc_en=(fidelity == MathFidelity.HiFi4),
            packer_l1_acc=True,
        )
    except Exception:
        return None


def _execute_with_config(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    result: SelectionResult,
    *,
    bias: Optional[ttnn.Tensor] = None,
    activation: Optional[str] = None,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> ttnn.Tensor:
    """Execute matmul with the selected configuration."""
    selected = result.selected_config
    compute_kernel_config = _build_compute_kernel_config(selected)

    program_config = selected.config

    matmul_kwargs: Dict[str, Any] = {
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "memory_config": memory_config,
        "dtype": dtype,
        "program_config": program_config,
        "activation": activation,
    }
    if compute_kernel_config is not None:
        matmul_kwargs["compute_kernel_config"] = compute_kernel_config

    if bias is not None:
        return ttnn.linear(
            input_tensor_a,
            input_tensor_b,
            bias=bias,
            **matmul_kwargs,
        )

    import ttnn._experimental.auto_config.matmul_auto as _selfmod

    _selfmod._last_selected_config = program_config
    _selfmod._last_selected_api = "matmul"
    return ttnn.matmul(input_tensor_a, input_tensor_b, **matmul_kwargs)


def _execute_fallback(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    bias: Optional[ttnn.Tensor] = None,
    activation: Optional[str] = None,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> ttnn.Tensor:
    """Fallback: use default ttnn.matmul with no program_config."""
    logger.warning("[matmul_auto] All candidates failed validation, using default ttnn.matmul (no program_config)")
    if bias is not None:
        return ttnn.linear(
            input_tensor_a,
            input_tensor_b,
            bias=bias,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            memory_config=memory_config,
            dtype=dtype,
            activation=activation,
        )
    return ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        memory_config=memory_config,
        dtype=dtype,
        activation=activation,
    )


def _ensure_on_device(tensor: ttnn.Tensor, device: Optional[ttnn.Device]) -> ttnn.Tensor:
    """Ensure a tensor is on device. Moves host tensors to device automatically."""
    try:
        if not ttnn.is_tensor_storage_on_device(tensor):
            if device is None:
                raise ValueError(
                    "matmul_auto received a host tensor but no device was specified. "
                    "Pass device= to matmul_auto() when using host weights."
                )
            return ttnn.to_device(tensor, device)
        return tensor
    except ValueError:
        raise
    except Exception:
        if device is None:
            raise ValueError(
                "matmul_auto received a host tensor but no device was specified. "
                "Pass device= to matmul_auto() when using host weights."
            )
        return ttnn.to_device(tensor, device)


def matmul_auto(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    device: Optional[ttnn.Device] = None,
    bias: Optional[ttnn.Tensor] = None,
    activation: Optional[str] = None,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
    force_program_config: Optional[Any] = None,
    benchmark_mode: bool = False,
) -> ttnn.Tensor:
    """Auto-optimal matmul that selects the best config for given inputs.

    Accepts host weights, caches configs, and falls back to default
    ttnn.matmul when all candidates fail validation.
    """
    if device is None:
        try:
            device = input_tensor_a.device()
        except Exception:
            pass

    input_tensor_a = _ensure_on_device(input_tensor_a, device)
    input_tensor_b = _ensure_on_device(input_tensor_b, device)
    if bias is not None:
        bias = _ensure_on_device(bias, device)

    # Manual override path
    if force_program_config is not None:
        if bias is not None:
            return ttnn.linear(
                input_tensor_a,
                input_tensor_b,
                bias=bias,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                memory_config=memory_config,
                dtype=dtype,
                program_config=force_program_config,
                activation=activation,
            )
        return ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            memory_config=memory_config,
            dtype=dtype,
            program_config=force_program_config,
            activation=activation,
        )

    # Auto-selection path
    selector = MatmulAutoConfig()
    dnn_gen = _get_global_dnn_generator()

    result = None
    if dnn_gen.is_available():
        try:
            features = selector.extract_features(
                input_tensor_a,
                input_tensor_b,
                bias=bias,
                activation=activation,
                dtype=dtype,
                memory_config=memory_config,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
            )
            generated = dnn_gen.generate(features)
            if generated is not None:
                from ttnn._experimental.auto_config.candidate_generator import _build_config_from_params

                dnn_candidate = _build_config_from_params(generated, features)
                if dnn_candidate is not None:
                    is_valid, reason = selector.validate(dnn_candidate, features)
                    if is_valid:
                        dnn_candidate.score = 1.0
                        dnn_candidate.is_valid = True
                        result = SelectionResult(
                            selected_config=dnn_candidate,
                            all_candidates=[dnn_candidate],
                            cache_hit=False,
                            selection_time_ms=0.0,
                        )
                        logger.info(
                            "[matmul_auto] DNN generator selected: %s", generated.get("config_family", "unknown")
                        )
        except Exception as e:
            logger.debug("DNN generator path failed: %s, falling back to heuristic", e)

    if result is None:
        result = selector.select(
            input_tensor_a,
            input_tensor_b,
            bias=bias,
            activation=activation,
            dtype=dtype,
            memory_config=memory_config,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )

    if result is None or result.selected_config is None:
        return _execute_fallback(
            input_tensor_a,
            input_tensor_b,
            bias=bias,
            activation=activation,
            dtype=dtype,
            memory_config=memory_config,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )

    selected = result.selected_config
    logger.info(
        f"[matmul_auto] Selected: {selected.config_family} "
        f"(backend={selected.backend}, score={selected.score:.3f}, "
        f"cache_hit={result.cache_hit}, time={result.selection_time_ms:.1f}ms)"
    )

    if benchmark_mode:
        return _benchmark_all_candidates(
            input_tensor_a,
            input_tensor_b,
            result,
            bias=bias,
            activation=activation,
            dtype=dtype,
            memory_config=memory_config,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )

    return _execute_with_config(
        input_tensor_a,
        input_tensor_b,
        result,
        bias=bias,
        activation=activation,
        dtype=dtype,
        memory_config=memory_config,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )


def _benchmark_all_candidates(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    result: SelectionResult,
    **kwargs,
) -> ttnn.Tensor:
    """Run all valid candidates and report timing with a human-readable table."""
    valid = [c for c in result.all_candidates if c.is_valid]
    best_time = float("inf")
    best_output = None
    best_candidate = None
    measured = []

    logger.info(f"[matmul_auto] Benchmark mode: testing {len(valid)} candidates")

    for candidate in valid:
        try:
            warmup_result = SelectionResult(selected_config=candidate, all_candidates=[candidate])
            _ = _execute_with_config(input_tensor_a, input_tensor_b, warmup_result, **kwargs)
            ttnn.synchronize_device(input_tensor_a.device())

            ttnn.synchronize_device(input_tensor_a.device())
            start = time.perf_counter()
            for _ in range(5):
                output = _execute_with_config(input_tensor_a, input_tensor_b, warmup_result, **kwargs)
            ttnn.synchronize_device(input_tensor_a.device())
            elapsed = (time.perf_counter() - start) * 1e6

            latency_us = elapsed / 5
            candidate.measured_latency_us = latency_us

            if latency_us < best_time:
                best_time = latency_us
                best_output = output
                best_candidate = candidate

            measured.append((candidate, latency_us))
        except Exception as e:
            candidate.measured_latency_us = None
            logger.info(f"{candidate.config_family:<45} | {'FAILED':>12} | {candidate.score:>5.2f} | {str(e)[:30]}")

    if measured:
        measured.sort(key=lambda x: x[1])
        logger.info(f"{'Config':<45} | {'Latency (us)':>12} | {'Score':>6} | {'Selected'}")
        logger.info(f"{'-'*45}-+-{'-'*12}-+-{'-'*6}-+-{'-'*10}")
        for cand, lat in measured:
            marker = "  *" if cand is best_candidate else ""
            logger.info(f"{cand.config_family:<45} | {lat:>12.0f} | {cand.score:>5.2f} |{marker}")

    if best_candidate:
        logger.info(f"[matmul_auto] Benchmark winner: {best_candidate.config_family} ({best_time:.0f} us)")
        selector = MatmulAutoConfig()
        features = selector.extract_features(input_tensor_a, input_tensor_b, **kwargs)
        cache_key = selector.get_cache_key(features)
        _get_global_cache().put(cache_key, best_candidate)

    return best_output


__all__ = ["matmul_auto", "MatmulAutoConfig"]

_parent_package = sys.modules.get(__package__)
if _parent_package is not None:
    setattr(_parent_package, "matmul_auto", matmul_auto)
