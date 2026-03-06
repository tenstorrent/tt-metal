# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Abstract base class for automatic configuration selection.

AutoConfigSelector provides a reusable framework for any TTNN operation
that needs automatic config selection. Concrete implementations (e.g.,
MatmulAutoConfig) override the abstract methods for their specific op.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConfigCandidate:
    """A candidate configuration with its metadata."""

    config: Any  # The actual program config object
    config_family: str  # e.g., "MultiCast1D", "MultiCast2D", "DRAMSharded"
    backend: str  # "matmul" or "minimal_matmul"
    params: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    is_valid: bool = True
    validation_reason: str = "valid"
    measured_latency_us: Optional[float] = None

    def __repr__(self) -> str:
        status = "valid" if self.is_valid else f"invalid({self.validation_reason})"
        return (
            f"ConfigCandidate(family={self.config_family}, backend={self.backend}, "
            f"score={self.score:.2f}, status={status})"
        )


@dataclass
class SelectionResult:
    """Result of the auto-config selection process."""

    selected_config: Optional[ConfigCandidate]
    all_candidates: List[ConfigCandidate]
    cache_hit: bool = False
    suggestions: List[str] = field(default_factory=list)
    selection_time_ms: float = 0.0

    @property
    def config(self) -> Any:
        if self.selected_config is None:
            return None
        return self.selected_config.config

    @property
    def backend(self) -> str:
        if self.selected_config is None:
            return "matmul"
        return self.selected_config.backend


class AutoConfigSelector(ABC):
    """
    Abstract base class for automatic configuration selection.

    This class provides a reusable framework for selecting optimal configurations
    for TTNN operations. Subclasses implement operation-specific logic.

    The selection pipeline:
        1. extract_features() - Extract normalized features from inputs
        2. generate_candidates() - Generate valid config candidates
        3. validate() - Filter invalid candidates
        4. score() - Score remaining candidates
        5. select() - Pick the best candidate, optionally benchmark top-K
    """

    def __init__(self, cache=None, scorer=None):
        self._cache = cache
        self._scorer = scorer

    @abstractmethod
    def extract_features(self, *tensors, **kwargs) -> Dict[str, Any]:
        """Extract a normalized feature dict from input tensors."""
        ...

    @abstractmethod
    def generate_candidates(self, features: Dict[str, Any]) -> List[ConfigCandidate]:
        """Generate a list of candidate configs for the given features."""
        ...

    @abstractmethod
    def validate(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a candidate configuration.

        Returns:
            (is_valid, reason) tuple.
        """
        ...

    @abstractmethod
    def score(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score a candidate configuration (higher = better)."""
        ...

    def get_cache_key(self, features: Dict[str, Any]) -> str:
        """Generate a cache key from features. Override for custom keying."""
        key_parts = []
        for k in sorted(features.keys()):
            v = features[k]
            key_parts.append(f"{k}={v}")
        return "|".join(key_parts)

    def select(self, *tensors, benchmark_top_k: int = 0, **kwargs) -> SelectionResult:
        """
        Select the optimal configuration for the given inputs.

        Args:
            *tensors: Input tensors for the operation.
            benchmark_top_k: If > 0, benchmark the top K candidates on-device
                           and select the fastest. If 0, use heuristic scoring only.
            **kwargs: Additional operation-specific parameters.

        Returns:
            SelectionResult containing the selected config and metadata.
        """
        import time

        start = time.perf_counter()

        # Step 1: Extract features
        features = self.extract_features(*tensors, **kwargs)

        # Step 2: Check cache
        if self._cache is not None:
            cache_key = self.get_cache_key(features)
            cached = self._cache.get(cache_key)
            if cached is not None:
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug(f"Cache hit for key: {cache_key[:60]}...")
                return SelectionResult(
                    selected_config=cached,
                    all_candidates=[cached],
                    cache_hit=True,
                    selection_time_ms=elapsed,
                )

        # Step 3: Generate candidates
        candidates = self.generate_candidates(features)
        logger.debug(f"Generated {len(candidates)} candidates")

        # Step 4: Validate
        valid_candidates = []
        for candidate in candidates:
            is_valid, reason = self.validate(candidate, features)
            candidate.is_valid = is_valid
            candidate.validation_reason = reason
            if is_valid:
                valid_candidates.append(candidate)

        logger.debug(f"{len(valid_candidates)}/{len(candidates)} candidates passed validation")

        if not valid_candidates:
            # Return with selected_config=None to signal fallback to caller.
            # matmul_auto.py handles this by falling back to default ttnn.matmul.
            logger.warning(
                "No valid candidates found. "
                f"All {len(candidates)} candidates were invalid."
            )
            elapsed = (time.perf_counter() - start) * 1000
            return SelectionResult(
                selected_config=None,
                all_candidates=candidates,
                cache_hit=False,
                suggestions=[],
                selection_time_ms=elapsed,
            )

        # Step 5: Score
        for candidate in valid_candidates:
            candidate.score = self.score(candidate, features)

        # Step 6: Sort by score (descending) and select
        valid_candidates.sort(key=lambda c: c.score, reverse=True)

        selected = valid_candidates[0]

        # Step 7: Cache the result
        if self._cache is not None:
            cache_key = self.get_cache_key(features)
            self._cache.put(cache_key, selected)

        elapsed = (time.perf_counter() - start) * 1000

        # Step 8: Generate suggestions
        suggestions = self._generate_suggestions(features, valid_candidates)

        return SelectionResult(
            selected_config=selected,
            all_candidates=candidates,
            cache_hit=False,
            suggestions=suggestions,
            selection_time_ms=elapsed,
        )

    def _generate_suggestions(
        self, features: Dict[str, Any], candidates: List[ConfigCandidate]
    ) -> List[str]:
        """Generate user-facing suggestions for better performance. Override as needed."""
        return []
