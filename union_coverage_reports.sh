#!/bin/bash

set -euo pipefail

TT_METAL_HOME_DEFAULT="$(cd "$(dirname "$0")" && pwd)"
: "${TT_METAL_HOME:=${TT_METAL_HOME_DEFAULT}}"

OUTPUT_ROOT_DIR="${TT_METAL_HOME}/coverage_reports_union"
mkdir -p "${OUTPUT_ROOT_DIR}"

# Hardcoded list of pairs to intersect. Edit as needed.
declare -a PAIRS=(
    "${TT_METAL_HOME}/coverage_reports/unit_tests_api|${TT_METAL_HOME}/coverage_reports/unit_tests_ttnn_ccl_ops"
)

PY_HELPER="$(cd "$(dirname "$0")" && pwd)/tools/coverage_intersect.py"
if [[ ! -f "${PY_HELPER}" ]]; then
    echo "ERROR: Missing helper script: ${PY_HELPER}" >&2
    echo "       Expected at tools/coverage_intersect.py" >&2
    exit 1
fi

for pair in "${PAIRS[@]}"; do
    REPORT_DIR_A="${pair%%|*}"
    REPORT_DIR_B="${pair##*|}"

    if [[ ! -d "${REPORT_DIR_A}" ]]; then
        echo "WARN: Skipping, not a directory: ${REPORT_DIR_A}" >&2
        continue
    fi
    if [[ ! -d "${REPORT_DIR_B}" ]]; then
        echo "WARN: Skipping, not a directory: ${REPORT_DIR_B}" >&2
        continue
    fi

    # Create a readable output folder name based on pair basenames
    base_a="$(basename "${REPORT_DIR_A}")"
    base_b="$(basename "${REPORT_DIR_B}")"
    OUTPUT_DIR="${OUTPUT_ROOT_DIR}/${base_a}__AND__${base_b}"
    mkdir -p "${OUTPUT_DIR}"

    echo "INFO: Report A: ${REPORT_DIR_A}"
    echo "INFO: Report B: ${REPORT_DIR_B}"
    echo "INFO: Output:   ${OUTPUT_DIR}"

    # Find .info files in each directory
    mapfile -t INFO_A < <(find "${REPORT_DIR_A}" -maxdepth 2 -type f -name '*.info' | sort)
    mapfile -t INFO_B < <(find "${REPORT_DIR_B}" -maxdepth 2 -type f -name '*.info' | sort)

    if [[ ${#INFO_A[@]} -eq 0 ]]; then
        echo "WARN: No .info files found in ${REPORT_DIR_A}; skipping" >&2
        continue
    fi
    if [[ ${#INFO_B[@]} -eq 0 ]]; then
        echo "WARN: No .info files found in ${REPORT_DIR_B}; skipping" >&2
        continue
    fi

    TMP_DIR="$(mktemp -d)"
    MERGED_A="${TMP_DIR}/merged_A.info"
    MERGED_B="${TMP_DIR}/merged_B.info"

    # Merge all .info files within each report directory into single files
    echo "INFO: Merging .info files within each report directory"
    cp "${INFO_A[0]}" "${MERGED_A}"
    for ((i=1; i<${#INFO_A[@]}; i++)); do
        lcov -a "${MERGED_A}" -a "${INFO_A[$i]}" -o "${MERGED_A}.next" >/dev/null
        mv "${MERGED_A}.next" "${MERGED_A}"
    done

    cp "${INFO_B[0]}" "${MERGED_B}"
    for ((i=1; i<${#INFO_B[@]}; i++)); do
        lcov -a "${MERGED_B}" -a "${INFO_B[$i]}" -o "${MERGED_B}.next" >/dev/null
        mv "${MERGED_B}.next" "${MERGED_B}"
    done

    # Optional: drop noisy paths (same exclusions as other scripts)
    FILTERED_A="${TMP_DIR}/merged_A.filtered.info"
    FILTERED_B="${TMP_DIR}/merged_B.filtered.info"
    lcov --remove "${MERGED_A}" '*/.cpmcache/*' '*/_deps/*' '*/tests/*' -o "${FILTERED_A}" >/dev/null
    lcov --remove "${MERGED_B}" '*/.cpmcache/*' '*/_deps/*' '*/tests/*' -o "${FILTERED_B}" >/dev/null

    # Compute strict intersection using helper
    INTERSECTED_INFO="${OUTPUT_DIR}/union_shared.info"
    echo "INFO: Computing strict intersection (identical executed lines per function)"
    python3 "${PY_HELPER}" "${FILTERED_A}" "${FILTERED_B}" "${INTERSECTED_INFO}"

    # Generate HTML from the intersected info
    HTML_OUT_DIR="${OUTPUT_DIR}/html"
    rm -rf "${HTML_OUT_DIR}"
    genhtml "${INTERSECTED_INFO}" --output-directory "${HTML_OUT_DIR}" >/dev/null

    echo "\n✅ Done. Intersection report written to: ${HTML_OUT_DIR}"
    echo "   Intersected .info: ${INTERSECTED_INFO}"
done
