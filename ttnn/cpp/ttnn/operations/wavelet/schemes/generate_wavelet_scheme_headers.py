# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SCHEME_CATALOG = SCRIPT_DIRECTORY / "wavelet_schemes.json"
GENERATED_DIRECTORY = SCRIPT_DIRECTORY.parent / "generated" / "wavelet_schemes"
GENERATED_INCLUDE_DIRECTORY = PurePosixPath("ttnn/cpp/ttnn/operations/wavelet/generated/wavelet_schemes")
SCHEME_COUNT = 106
LICENSE_HEADER = """// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
"""

STEP_TYPES = {
    "predict": "StepType::kPredict",
    "update": "StepType::kUpdate",
    "scale-even": "StepType::kScaleEven",
    "scale-odd": "StepType::kScaleOdd",
    "swap": "StepType::kSwap",
}


@dataclass(frozen=True)
class Step:
    kind: str
    shift: int
    coefficient_bits: tuple[int, ...]


@dataclass(frozen=True)
class Scheme:
    name: str
    identifier: str
    tap_size: int
    delay_even: int
    delay_odd: int
    steps: tuple[Step, ...]


def make_identifier(name: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not identifier or identifier[0].isdigit():
        identifier = f"w_{identifier}"
    return identifier


def parse_coefficient(raw: int | float | dict[str, int | float]) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    return float(raw["numerator"]) / float(raw["denominator"])


def float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def float32_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def inverse_steps(steps: tuple[Step, ...]) -> tuple[Step, ...]:
    inverse: list[Step] = []
    for step in reversed(steps):
        if step.kind in {"predict", "update"}:
            coefficient_bits = tuple(bits ^ 0x80000000 for bits in step.coefficient_bits)
        elif step.kind in {"scale-even", "scale-odd"}:
            scale = float32_value(step.coefficient_bits[0])
            if scale == 0.0:
                raise ValueError("inverse scale coefficient must be non-zero")
            coefficient_bits = (float32_bits(1.0 / scale),)
        else:
            coefficient_bits = step.coefficient_bits
        inverse.append(
            Step(
                kind=step.kind,
                shift=step.shift,
                coefficient_bits=coefficient_bits,
            )
        )
    return tuple(inverse)


def load_scheme(name: str, payload: dict) -> Scheme:
    steps: list[Step] = []
    for raw_step in payload["steps"]:
        kind = raw_step["type"]
        if kind not in STEP_TYPES:
            raise ValueError(f"{name}: unsupported step type {kind!r}")
        coefficient_bits = tuple(
            float32_bits(parse_coefficient(coefficient)) for coefficient in raw_step.get("coefficients", [])
        )
        if kind in {"scale-even", "scale-odd"} and len(coefficient_bits) != 1:
            raise ValueError(f"{name}: {kind} must have exactly one coefficient")
        if kind == "swap" and coefficient_bits:
            raise ValueError(f"{name}: swap must not have coefficients")
        steps.append(
            Step(
                kind=kind,
                shift=int(raw_step["shift"]),
                coefficient_bits=coefficient_bits,
            )
        )

    return Scheme(
        name=name,
        identifier=make_identifier(name),
        tap_size=int(payload["tap_size"]),
        delay_even=int(payload["delay"]["even"]),
        delay_odd=int(payload["delay"]["odd"]),
        steps=tuple(steps),
    )


def coefficient_arguments(step: Step) -> str:
    if not step.coefficient_bits:
        return ""
    return ", " + ", ".join(f"0x{bits:08x}U" for bits in step.coefficient_bits)


def render_steps(owner: str, steps: tuple[Step, ...]) -> list[str]:
    lines: list[str] = []
    for index, step in enumerate(steps):
        lines.extend(
            [
                "template <>",
                f"struct {owner}::step<{index}> {{",
                (
                    f"    using type = StaticStep<{STEP_TYPES[step.kind]}, {step.shift}"
                    f"{coefficient_arguments(step)}>;"
                ),
                f"    static_assert(type::k == {len(step.coefficient_bits)}U);",
                "};",
                "",
            ]
        )
    return lines


def render_scheme_header(scheme: Scheme) -> str:
    inverse_identifier = f"{scheme.identifier}_inverse"
    inverse = inverse_steps(scheme.steps)
    include_path = GENERATED_INCLUDE_DIRECTORY / f"{scheme.identifier}.hpp"
    lines = [
        LICENSE_HEADER.rstrip(),
        "",
        "#pragma once",
        "",
        '#include "ttnn/operations/wavelet/planner/static_scheme.hpp"',
        "",
        "namespace ttnn::operations::wavelet::schemes {",
        "",
        f"struct {inverse_identifier};",
        "",
        f"struct {scheme.identifier} {{",
        f'    static constexpr const char* name = "{scheme.name}";',
        f"    static constexpr uint32_t tap_size = {scheme.tap_size}U;",
        f"    static constexpr int32_t delay_even = {scheme.delay_even};",
        f"    static constexpr int32_t delay_odd = {scheme.delay_odd};",
        f"    static constexpr uint32_t num_steps = {len(scheme.steps)}U;",
        f'    static constexpr const char* compute_scheme_header = "\\"{include_path}\\"";',
        (
            "    static constexpr const char* compute_scheme_type = "
            f'"ttnn::operations::wavelet::schemes::{scheme.identifier}";'
        ),
        f"    using inverse = {inverse_identifier};",
        "",
        "    template <std::size_t I>",
        "    struct step;",
        "};",
        "",
    ]
    lines.extend(render_steps(scheme.identifier, scheme.steps))
    lines.extend(
        [
            f"struct {inverse_identifier} {{",
            f'    static constexpr const char* name = "{scheme.name}-inverse";',
            f"    static constexpr uint32_t tap_size = {scheme.tap_size}U;",
            f"    static constexpr uint32_t num_steps = {len(inverse)}U;",
            f'    static constexpr const char* compute_scheme_header = "\\"{include_path}\\"";',
            (
                "    static constexpr const char* compute_scheme_type = "
                f'"ttnn::operations::wavelet::schemes::{inverse_identifier}";'
            ),
            "",
            "    template <std::size_t I>",
            "    struct step;",
            "};",
            "",
        ]
    )
    lines.extend(render_steps(inverse_identifier, inverse))
    lines.extend(["}  // namespace ttnn::operations::wavelet::schemes", ""])
    return "\n".join(lines)


def render_catalog(schemes: list[Scheme]) -> str:
    enum_entries = [f"    k{scheme.identifier}," for scheme in schemes]
    information_entries = [
        (
            f'    SchemeInfo{{"{scheme.name}", {scheme.tap_size}U, '
            f"{scheme.delay_even}, {scheme.delay_odd}, {len(scheme.steps)}U}},"
        )
        for scheme in schemes
    ]
    identifier_checks: list[str] = []
    for scheme in schemes:
        identifier_checks.extend(
            [
                f'    if (name == "{scheme.name}") {{',
                f"        return SchemeId::k{scheme.identifier};",
                "    }",
            ]
        )
    lines = [
        LICENSE_HEADER.rstrip(),
        "",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstdint>",
        "#include <span>",
        "#include <string_view>",
        "",
        "namespace ttnn::operations::wavelet {",
        "",
        "struct SchemeInfo {",
        "    std::string_view name;",
        "    uint32_t tap_size;",
        "    int32_t delay_even;",
        "    int32_t delay_odd;",
        "    uint32_t num_steps;",
        "};",
        "",
        "enum class SchemeId : uint32_t {",
        *enum_entries,
        "    kUnknown,",
        "};",
        "",
        f"inline constexpr std::array<SchemeInfo, {len(schemes)}> kSchemeInfos = {{",
        *information_entries,
        "};",
        "",
        "[[nodiscard]] inline std::span<const SchemeInfo> available_wavelets() noexcept {",
        "    return kSchemeInfos;",
        "}",
        "",
        "[[nodiscard]] inline SchemeId scheme_id(std::string_view name) noexcept {",
        *identifier_checks,
        "    return SchemeId::kUnknown;",
        "}",
        "",
        "}  // namespace ttnn::operations::wavelet",
        "",
    ]
    return "\n".join(lines)


def render_dispatch(schemes: list[Scheme]) -> str:
    first_identifier = schemes[0].identifier
    includes = [f'#include "{scheme.identifier}.hpp"' for scheme in schemes]
    dispatch_cases = [
        (
            f"        case SchemeId::k{scheme.identifier}: "
            f"return fn.template operator()<schemes::{scheme.identifier}>();"
        )
        for scheme in schemes
    ]
    lines = [
        LICENSE_HEADER.rstrip(),
        "",
        "#pragma once",
        "",
        "#include <cstdint>",
        "#include <string_view>",
        "#include <utility>",
        "",
        '#include "scheme_catalog.hpp"',
        *includes,
        "",
        "#include <tt_stl/assert.hpp>",
        "",
        "namespace ttnn::operations::wavelet {",
        "",
        "template <typename Fn>",
        "decltype(auto) dispatch_scheme(const SchemeId id, Fn&& fn) {",
        "    switch (id) {",
        *dispatch_cases,
        "        case SchemeId::kUnknown: break;",
        "    }",
        '    TT_THROW("Unsupported wavelet scheme id: {}", static_cast<uint32_t>(id));',
        f"    return fn.template operator()<schemes::{first_identifier}>();",
        "}",
        "",
        "template <typename Fn>",
        "decltype(auto) dispatch_scheme(const std::string_view name, Fn&& fn) {",
        "    return dispatch_scheme(scheme_id(name), std::forward<Fn>(fn));",
        "}",
        "",
        "}  // namespace ttnn::operations::wavelet",
        "",
    ]
    return "\n".join(lines)


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def remove_stale_headers(keep: set[Path]) -> None:
    if not GENERATED_DIRECTORY.exists():
        return
    for path in GENERATED_DIRECTORY.glob("*.hpp"):
        if path not in keep:
            path.unlink()


def load_catalog() -> dict[str, dict]:
    catalog = json.loads(SCHEME_CATALOG.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise TypeError("wavelet scheme catalog must be a JSON object")
    if len(catalog) != SCHEME_COUNT:
        raise RuntimeError(f"expected {SCHEME_COUNT} wavelet schemes, found {len(catalog)}")
    if any(not isinstance(name, str) or not isinstance(payload, dict) for name, payload in catalog.items()):
        raise TypeError("wavelet scheme catalog entries must be JSON objects")
    return catalog


def main() -> None:
    catalog = load_catalog()

    schemes = sorted(
        (load_scheme(name, payload) for name, payload in catalog.items()),
        key=lambda scheme: scheme.name,
    )
    if len({scheme.identifier for scheme in schemes}) != len(schemes):
        raise RuntimeError("generated scheme identifiers are not unique")

    generated_headers: set[Path] = set()
    for scheme in schemes:
        path = GENERATED_DIRECTORY / f"{scheme.identifier}.hpp"
        generated_headers.add(path)
        write_file(path, render_scheme_header(scheme))

    catalog_header = GENERATED_DIRECTORY / "scheme_catalog.hpp"
    generated_headers.add(catalog_header)
    write_file(catalog_header, render_catalog(schemes))

    dispatch_header = GENERATED_DIRECTORY / "scheme_dispatch.hpp"
    generated_headers.add(dispatch_header)
    write_file(dispatch_header, render_dispatch(schemes))
    remove_stale_headers(generated_headers)


if __name__ == "__main__":
    main()
