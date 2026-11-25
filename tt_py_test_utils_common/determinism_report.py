# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Failure diagnostics for the "run the op twice and compare" determinism helpers.

``assert_deterministic`` is the single entry point. When the two outputs are equal it does one
comparison and returns, printing nothing. Only when they differ does it build a report:

  * how many elements differ, the first few differing values, magnitude of the difference
    (and how many are an exact power of two, which is what a shared-exponent / format error
    looks like),
  * where the differences are: global tile-rows, tile-columns, and 16-wide row groups (the
    bfp8 shared-exponent granularity),
  * whether re-reading the same device buffers gives the same bytes (readback vs compute),
  * a third run of the op, to say which of the first two runs was the outlier.

Everything here is bounded for large tensors: only ``_MAX_LOCATED`` mismatching positions are
analysed, only ``_MAX_SAMPLES`` values are printed, range lists are capped, and the whole
message is truncated at ``_MAX_MESSAGE_CHARS``. Diagnostics never replace the original failure:
if building the report raises, a short message is raised instead.
"""

import torch
import ttnn

_MAX_LOCATED = 1_000_000  # mismatching positions analysed for location statistics
_MAX_SAMPLES = 10  # individual mismatching values printed
_MAX_RANGES = 20  # entries printed in any range list
_MAX_LEAVES = 3  # differing outputs reported in detail for tuple-returning ops
_MAX_MESSAGE_CHARS = 8000
_SCAN_CHUNK = 1 << 24  # elements scanned per step when collecting mismatch indices


def _to_torch(x):
    return x if isinstance(x, torch.Tensor) else ttnn.to_torch(x)


def _leaves(output, path=""):
    """Yield (path, tensor) for a tensor, or for each tensor in nested tuples/lists; skips None."""
    if output is None:
        return
    if isinstance(output, (tuple, list)):
        for i, item in enumerate(output):
            yield from _leaves(item, f"{path}[{i}]")
    else:
        yield path, output


def _differs(a, b, nan_equal):
    """Boolean mask of differing elements. ``nan_equal`` treats NaN == NaN (floating dtypes only)."""
    if a.is_floating_point():
        same = a == b
        if nan_equal:
            same |= torch.isnan(a) & torch.isnan(b)
        return ~same
    return a != b


def _equal(a, b, nan_equal):
    if a.shape != b.shape:
        return False
    return not bool(_differs(a, b, True).any()) if nan_equal else torch.equal(a, b)


def _first_indices(flat_mask, cap):
    """Flat indices of the first ``cap`` True entries, scanning in chunks to bound memory."""
    found, got = [], 0
    for start in range(0, flat_mask.numel(), _SCAN_CHUNK):
        nz = flat_mask[start : start + _SCAN_CHUNK].nonzero().squeeze(1) + start
        nz = nz[: cap - got]
        found.append(nz)
        got += nz.numel()
        if got >= cap:
            break
    return torch.cat(found) if found else torch.zeros(0, dtype=torch.long)


def _compress_ranges(values):
    """[3,4,5,9] -> '3-5, 9' (first ``_MAX_RANGES`` ranges)."""
    ranges = []
    for v in sorted(int(v) for v in values):
        if ranges and v == ranges[-1][1] + 1:
            ranges[-1][1] = v
        else:
            ranges.append([v, v])
    text = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges[:_MAX_RANGES])
    if len(ranges) > _MAX_RANGES:
        text += f", ... ({len(ranges)} ranges total)"
    return text


def _describe_inputs(args, kwargs):
    lines = []
    for name, value in [*enumerate(args), *kwargs.items()]:
        if isinstance(value, ttnn.Tensor):
            try:
                mem = value.memory_config()
            except Exception:  # noqa: BLE001 - diagnostics must never mask the real failure
                mem = "?"
            lines.append(
                f"  input {name}: shape={tuple(value.shape)} dtype={value.dtype} layout={value.layout} mem={mem}"
            )
    return lines


def _mismatch_lines(a, b):
    """Where and how two same-shaped, same-dtype tensors differ (NaN at matching positions counts as equal)."""
    differ = _differs(a, b, True)
    flat = differ.reshape(-1)
    n = int(flat.sum())
    lines = [f"shape={tuple(a.shape)} torch dtype={a.dtype} elements={a.numel()}"]
    lines.append(f"mismatching elements={n} ({100.0 * n / max(a.numel(), 1):.5f}%)")
    if a.is_floating_point():
        lines.append(f"NaNs run1={int(torch.isnan(a).sum())} run2={int(torch.isnan(b).sum())}")
    if n == 0:
        lines.append(
            "no element differs: the strict comparison failed only because both outputs contain NaN at the same "
            "positions (NaN != NaN)."
        )
        return lines

    idx_flat = _first_indices(flat, _MAX_LOCATED)
    located = idx_flat.numel()
    if located < n:
        lines.append(f"(value and location statistics below use the first {located} of {n} mismatches)")
    av_all = a.reshape(-1)[idx_flat]
    bv_all = b.reshape(-1)[idx_flat]

    shape = tuple(a.shape)
    coords, rem = [], idx_flat
    for dim in reversed(shape):
        coords.append(rem % dim)
        rem = rem // dim
    coords.reverse()

    for k in range(min(_MAX_SAMPLES, located)):
        pos = tuple(int(c[k]) for c in coords)
        lines.append(f"  [{pos}] run1={av_all[k].item()!r} run2={bv_all[k].item()!r}")

    if a.is_floating_point():
        af, bf = av_all.float(), bv_all.float()
        finite = torch.isfinite(af) & torch.isfinite(bf)
        if bool(finite.any()):
            d = (af - bf).abs()[finite]
            lines.append(f"abs diff over finite mismatches: max={d.max().item():.6g} mean={d.mean().item():.6g}")
            nz = finite & (af != 0) & (bf != 0)
            if bool(nz.any()):
                log_ratio = torch.log2((af[nz] / bf[nz]).abs())
                pow2 = (log_ratio - log_ratio.round()).abs() < 1e-6
                lines.append(
                    f"{int(pow2.sum())}/{int(nz.sum())} nonzero mismatches differ by an exact power of two "
                    "(a shared-exponent / format-config error looks like this)"
                )

    if len(shape) >= 2:
        height, width = shape[-2], shape[-1]
        tile_rows_per_batch = -(-height // 32)
        batch_flat = torch.zeros(located, dtype=torch.long)
        for dim_size, c in zip(shape[:-2], coords[:-2]):
            batch_flat = batch_flat * dim_size + c
        rows, cols = coords[-2], coords[-1]
        tile_row = batch_flat * tile_rows_per_batch + rows // 32
        total_tile_rows = (a.numel() // max(height * width, 1)) * tile_rows_per_batch
        uniq_rows, row_counts = torch.unique(tile_row, return_counts=True)
        lines.append(
            f"affected tile-rows (global index, {total_tile_rows} total): {len(uniq_rows)} -> "
            f"{_compress_ranges(uniq_rows.tolist())}"
        )
        uniq_cols = torch.unique(cols // 32)
        lines.append(
            f"affected tile-columns: {len(uniq_cols)} of {-(-width // 32)} -> {_compress_ranges(uniq_cols.tolist())}"
        )
        # bfp8 shares one exponent per 16 elements along the row.
        group_key = (batch_flat * height + rows) * (-(-width // 16)) + cols // 16
        _, group_counts = torch.unique(group_key, return_counts=True)
        lines.append(
            f"affected 16-wide row groups: {len(group_counts)}; mismatching elements per group: "
            f"min={int(group_counts.min())} max={int(group_counts.max())} "
            "(16 everywhere = whole groups, i.e. shared-exponent-like)"
        )
        top = row_counts.argsort(descending=True)[:5]
        lines.append(
            "most affected tile-rows (row: #mismatches): "
            + ", ".join(f"{int(uniq_rows[i])}: {int(row_counts[i])}" for i in top)
        )
    return lines


def _structure(output):
    """Recursive signature: nesting, lengths and None positions (list vs tuple is not distinguished)."""
    if isinstance(output, (tuple, list)):
        return tuple(_structure(item) for item in output)
    return "none" if output is None else "tensor"


def _nondeterminism_message(op, args, kwargs, out1, out2, host1, host2, rerun, nan_equal):
    """``host1``/``host2`` are the host copies ({path: torch tensor}) already read for the failed comparison."""
    lines = [
        f"{getattr(op, 'python_fully_qualified_name', op)}: two runs with identical inputs gave different outputs "
        "(both outputs were still alive on device when compared).",
        *_describe_inputs(args, kwargs),
    ]
    device1, device2 = dict(_leaves(out1)), dict(_leaves(out2))

    first_bad = None
    reported = 0
    for path, t1 in host1.items():
        t2 = host2[path]
        label = f"output{path}" if path else "output"
        if t1.shape != t2.shape or t1.dtype != t2.dtype:
            lines.append(
                f"{label}: shape/dtype differ: run1={tuple(t1.shape)}/{t1.dtype} run2={tuple(t2.shape)}/{t2.dtype}"
            )
        elif not _equal(t1, t2, nan_equal):
            if reported < _MAX_LEAVES:
                lines.append(f"{label}:")
                lines.extend("  " + line for line in _mismatch_lines(t1, t2))
            reported += 1
        else:
            continue
        if first_bad is None:
            first_bad = (path, t1, t2)
    if reported > _MAX_LEAVES:
        lines.append(f"... and {reported - _MAX_LEAVES} more differing outputs")
    if first_bad is None:
        return lines
    path, t1, t2 = first_bad

    # Re-reading the same device buffers and comparing with the ORIGINAL reads: different => readback problem,
    # same => the device data really differs. Matching NaNs count as identical here regardless of the caller's policy,
    # since a stable NaN says nothing about readback.
    o1, o2 = device1[path], device2[path]
    if not isinstance(o1, torch.Tensor) and not isinstance(o2, torch.Tensor):
        try:
            r1, r2 = ttnn.to_torch(o1), ttnn.to_torch(o2)
            lines.append(
                "re-read of the SAME output buffers: "
                f"run1 identical to first read={_equal(r1, t1, True)}, run2 identical to first read={_equal(r2, t2, True)}"
            )
        except Exception as e:  # noqa: BLE001
            lines.append(f"re-read failed: {e!r}")

    # A third run says which of the first two was the outlier (same NaN policy as the failed comparison).
    if rerun is not None:
        try:
            t3 = _to_torch(dict(_leaves(rerun()))[path])
            lines.append(
                f"third run: identical to run1={_equal(t3, t1, nan_equal)}, identical to run2={_equal(t3, t2, nan_equal)} "
                "(matches run2 only => run1 was the outlier, i.e. the first launch after earlier work on this device; "
                "matches run1 only => run2 was the outlier)"
            )
        except Exception as e:  # noqa: BLE001
            lines.append(f"third run failed: {e!r}")
    return lines


def assert_deterministic(op, args, kwargs, out1, out2, rerun=None, nan_equal=False):
    """Raise AssertionError (with diagnostics) unless ``out1`` and ``out2`` are identical.

    ``out1``/``out2`` may be ttnn or torch tensors, None, or nested tuples/lists of those; their structure
    (nesting, lengths, None positions) must match.
    ``nan_equal=False`` keeps ``torch.equal`` semantics (NaN != NaN); True treats NaN at matching positions as equal.
    ``rerun`` is an optional zero-argument callable that runs the op once more, used only on failure.
    """
    structure1, structure2 = _structure(out1), _structure(out2)
    if structure1 != structure2:
        raise AssertionError(f"the two runs returned different output structures: {structure1!r} vs {structure2!r}")
    # Read every output once, up front, and keep the host copies: the report must analyse exactly what failed the
    # comparison, and re-reading would hide a transient readback error.
    host1 = {path: _to_torch(o) for path, o in _leaves(out1)}
    host2 = {path: _to_torch(o) for path, o in _leaves(out2)}
    if all(_equal(host1[path], host2[path], nan_equal) for path in host1):
        return
    try:
        message = "\n".join(_nondeterminism_message(op, args, kwargs, out1, out2, host1, host2, rerun, nan_equal))
    except Exception as e:  # noqa: BLE001 - never let diagnostics hide the failure
        message = f"the two runs produced different outputs (diagnostics failed: {e!r})"
    if len(message) > _MAX_MESSAGE_CHARS:
        message = message[:_MAX_MESSAGE_CHARS] + "\n... (truncated)"
    raise AssertionError(message)
