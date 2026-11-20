# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The matmul config model blob (ttnn.CONFIG.matmul_model_config): its default path, and its download into ttnn's
cache for an install without the blob the build fetches (ttnn/cpp/ttnn/operations/matmul/device/config/model/README.md,
"Turning it on" and "The model blob").

    python -m ttnn.matmul_model
"""

import hashlib
import http.client
import json
import os
import pathlib
import urllib.request
import uuid

from loguru import logger

# The pin the build reads too. A wheel carries it under the ttnn package, a source tree has it two levels up.
_PIN = pathlib.Path("ttnn/cpp/ttnn/operations/matmul/device/config/model/matmul_model.json")
_TIMEOUT_S = 10

# Cache paths whose automatic download failed in this process, so it is tried once, not on every device open
_failed = set()


def _pin():
    package = pathlib.Path(__file__).resolve().parent.parent
    for root in (package, package.parent.parent):
        if (root / _PIN).is_file():
            return json.loads((root / _PIN).read_text())
    raise FileNotFoundError(f"{_PIN} is not in the ttnn package or its source tree")


def cache_path():
    """Where the pinned blob is cached: ttnn.CONFIG.cache_path/matmul_model/<sha256>.blob."""
    import ttnn

    return pathlib.Path(ttnn.CONFIG.cache_path) / "matmul_model" / f"{_pin()['sha256']}.blob"


def _verified(path, sha256):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest() == sha256


def download(path=None):
    """Downloads the pinned blob to `path` and returns `path`, unless the pinned blob is already there.

    Without `path`: to ttnn.CONFIG.matmul_model_path when that is the cache path or unset (then it is set to the cache
    path). A matmul_model_path set to another file that doesn't exist raises FileNotFoundError: it is likely a mistake,
    and is not filled with the pinned blob. Raises if the download fails or its sha256 is not the pinned one; the file
    at `path` is only ever the complete, verified blob.
    """
    import ttnn

    pin = _pin()
    if path is None:
        configured = pathlib.Path(ttnn.CONFIG.matmul_model_path)
        path = cache_path()
        if str(configured) in ("", "."):
            ttnn.CONFIG.matmul_model_path = str(path)
        elif configured != path:
            if configured.is_file():
                return configured
            raise FileNotFoundError(f"ttnn.CONFIG.matmul_model_path is {configured}, which doesn't exist")
    path = pathlib.Path(path)
    if path.is_file() and _verified(path, pin["sha256"]):
        return path

    repository, sha256 = pin["repository"], pin["sha256"]
    logger.info(f"Downloading the matmul config model ghcr.io/{repository}@sha256:{sha256} to {path}")
    # The same anonymous pull as fetch_matmul_model.cmake: ghcr serves public packages to an anonymous pull token
    with urllib.request.urlopen(f"https://ghcr.io/token?scope=repository:{repository}:pull", timeout=_TIMEOUT_S) as r:
        token = json.load(r)["token"]
    request = urllib.request.Request(
        f"https://ghcr.io/v2/{repository}/blobs/sha256:{sha256}", headers={"Authorization": f"Bearer {token}"}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Download next to the target and rename, so an interrupted or concurrent download never looks like the blob
    partial = path.with_name(f"{path.name}.{uuid.uuid4().hex}.partial")
    try:
        digest = hashlib.sha256()
        with open(partial, "xb") as out, urllib.request.urlopen(request, timeout=_TIMEOUT_S) as r:
            while chunk := r.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError(f"downloaded blob is sha256:{digest.hexdigest()}, expected sha256:{sha256}")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def download_if_enabled():
    """With ttnn.CONFIG.matmul_model_config on and matmul_model_path the cache path but no file there, downloads the
    blob. Called by `import ttnn` and on device open; a failure is logged once per process and leaves the default
    program configs. Never touches the network with the flag off, or for a matmul_model_path set to another file."""
    import ttnn

    if not ttnn.CONFIG.matmul_model_config:
        return
    path = pathlib.Path(ttnn.CONFIG.matmul_model_path)
    if path.is_file() or path in _failed:
        return
    try:
        if path != cache_path():
            return
        download(path)
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as e:
        _failed.add(path)
        logger.warning(
            f"matmul_model_config: could not download the matmul config model ({e}); matmul keeps its "
            "default program configs"
        )


def on_import(runtime_root):
    """Called by `import ttnn`. Unless matmul_model_path was set, points it at the blob a source build fetched into
    runtime/ under runtime_root, or else at cache_path(); then downloads the blob if the flag is on. An unset path
    reads back as pathlib.Path("") == "."."""
    import ttnn

    if str(ttnn.CONFIG.matmul_model_path) in ("", "."):
        build_blob = pathlib.Path(runtime_root) / "runtime/matmul_model/matmul_model.blob"
        if build_blob.is_file():
            ttnn.CONFIG.matmul_model_path = str(build_blob)
        else:
            try:
                ttnn.CONFIG.matmul_model_path = str(cache_path())
            except (OSError, ValueError, KeyError, TypeError):
                pass  # no readable pin (setup.py checks a wheel has it): the path stays unset
    download_if_enabled()
