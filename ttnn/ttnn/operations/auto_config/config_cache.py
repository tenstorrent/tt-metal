# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Persistent configuration cache for auto-config selection.

Caches optimal configurations by input signature for cross-session reuse.
Cache entries are invalidated by selector_version changes (e.g., when the
underlying matmul implementations change).

Cache directory is configurable via TTNN_AUTO_CONFIG_CACHE_DIR environment variable
for CI compatibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from ttnn.operations.auto_config.base import ConfigCandidate

logger = logging.getLogger(__name__)

# Current selector version — bump when config selection logic changes
SELECTOR_VERSION = "1.0.0"

# Default cache directory, configurable via env var for CI
DEFAULT_CACHE_DIR = os.environ.get(
    "TTNN_AUTO_CONFIG_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".ttnn", "auto_config_cache"),
)


class ConfigCache:
    """
    Thread-safe, persistent cache for optimal matmul configurations.

    Cache entries are keyed by a hash of the input signature (shapes, dtypes,
    memory layouts, device arch, grid size). The cache persists across sessions via JSON.

    The cache key includes the silicon architecture (e.g., "wormhole_b0" vs "grayskull")
    to prevent wrongly sharing entries across different hardware.

    Invalidation:
        - Entries are tagged with selector_version
        - Version mismatch = stale entry = cache miss
        - Manual flush via clear()

    Configuration:
        - Set TTNN_AUTO_CONFIG_CACHE_DIR env var to override the cache directory
          (useful for CI environments that need a temp directory)
    """

    def __init__(
        self, cache_dir: Optional[str] = None, max_entries: int = 10000
    ):
        self._lock = threading.Lock()
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._max_entries = max_entries

        # Use env var, then constructor arg, then default
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR
        self._cache_dir = Path(cache_dir)
        self._cache_file = self._cache_dir / "matmul_configs.json"

        # Load existing cache from disk
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load cache entries from disk. Auto-deletes on corruption."""
        if not self._cache_file.exists():
            return
        try:
            with open(self._cache_file, "r") as f:
                data = json.load(f)
            # Only load entries matching current version
            for key, entry in data.items():
                if entry.get("version") == SELECTOR_VERSION:
                    self._memory_cache[key] = entry
            logger.debug(
                f"Loaded {len(self._memory_cache)} cache entries from {self._cache_file}"
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load config cache (auto-deleting corrupt cache): {e}")
            # Auto-delete corrupt cache file
            try:
                self._cache_file.unlink()
            except OSError:
                pass

    def _save_to_disk(self) -> None:
        """Persist current cache to disk."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump(self._memory_cache, f, indent=2, default=str)
        except OSError as e:
            logger.warning(f"Failed to save config cache: {e}")

    def _hash_key(self, key: str) -> str:
        """Create a compact hash of the cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[ConfigCandidate]:
        """
        Look up a cached config by key.

        Returns None on cache miss or version mismatch.
        """
        hashed = self._hash_key(key)
        with self._lock:
            entry = self._memory_cache.get(hashed)
            if entry is None:
                return None
            if entry.get("version") != SELECTOR_VERSION:
                # Stale entry
                del self._memory_cache[hashed]
                return None

            # Reconstruct ConfigCandidate from cached data
            return ConfigCandidate(
                config=None,  # Config will be regenerated from params
                config_family=entry["config_family"],
                backend=entry["backend"],
                params=entry["params"],
                score=entry.get("score", 0.0),
                is_valid=True,
                validation_reason="cached",
            )

    def put(self, key: str, candidate: ConfigCandidate) -> None:
        """Store a config candidate in the cache."""
        hashed = self._hash_key(key)
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._memory_cache) >= self._max_entries:
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]

            self._memory_cache[hashed] = {
                "version": SELECTOR_VERSION,
                "config_family": candidate.config_family,
                "backend": candidate.backend,
                "params": candidate.params,
                "score": candidate.score,
                "original_key": key[:200],  # Truncated for readability
            }

            self._save_to_disk()

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._memory_cache.clear()
            if self._cache_file.exists():
                try:
                    self._cache_file.unlink()
                except OSError:
                    pass
            logger.info("Config cache cleared")

    def __len__(self) -> int:
        return len(self._memory_cache)

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            families = {}
            for entry in self._memory_cache.values():
                family = entry.get("config_family", "unknown")
                families[family] = families.get(family, 0) + 1
            return {
                "total_entries": len(self._memory_cache),
                "max_entries": self._max_entries,
                "cache_file": str(self._cache_file),
                "selector_version": SELECTOR_VERSION,
                "families": families,
            }
