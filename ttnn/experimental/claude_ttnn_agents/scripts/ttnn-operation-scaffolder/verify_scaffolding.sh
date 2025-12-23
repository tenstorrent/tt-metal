#!/bin/bash
#
# TTNN Operation Scaffolding Verifier
#
# Checks for banned patterns and required patterns in scaffolded operation.
# Returns 0 if all checks pass, 1 if any check fails.
#
# Note: We don't use 'set -e' because grep returns exit code 1 when no match is found,
# which would cause the script to exit prematurely when checking for banned patterns
# (where no match is the desired outcome).

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: verify_scaffolding.sh <operation_path> <operation_name>"
    echo "Example: verify_scaffolding.sh ttnn/cpp/ttnn/operations/data_movement/my_operation my_operation"
    exit 1
}

if [ $# -lt 2 ]; then
    usage
fi

OP_PATH="$1"
OP_NAME="$2"

if [ ! -d "$OP_PATH" ]; then
    echo -e "${RED}Error: Operation path not found: $OP_PATH${NC}"
    exit 1
fi

echo "Verifying scaffolded operation: $OP_NAME"
echo "Path: $OP_PATH"
echo ""

ERRORS=0

# Check 1: File names (banned legacy pattern)
echo "Check 1: File naming convention..."
if [ -f "$OP_PATH/device/${OP_NAME}_op.hpp" ] || [ -f "$OP_PATH/device/${OP_NAME}_op.cpp" ]; then
    echo -e "${RED}  ✗ FAIL: Found legacy file names (*_op.hpp or *_op.cpp)${NC}"
    echo "    Modern pattern requires: ${OP_NAME}_device_operation.hpp/cpp"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}  ✓ PASS: No legacy file names${NC}"
fi

# Check 2: Required files exist
echo "Check 2: Required files exist..."
REQUIRED_FILES=(
    "device/${OP_NAME}_device_operation.hpp"
    "device/${OP_NAME}_device_operation.cpp"
    "device/${OP_NAME}_device_operation_types.hpp"
    "device/${OP_NAME}_program_factory.hpp"
    "device/${OP_NAME}_program_factory.cpp"
    "${OP_NAME}.hpp"
    "${OP_NAME}.cpp"
    "${OP_NAME}_pybind.hpp"
    "${OP_NAME}_pybind.cpp"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$OP_PATH/$file" ]; then
        echo -e "${RED}  ✗ FAIL: Missing required file: $file${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}  ✓ PASS: All required files exist${NC}"
fi

# Check 3: Banned patterns in code
echo "Check 3: Checking for banned patterns..."

# Check for banned include
if grep -r "run_operation.hpp" "$OP_PATH/" 2>/dev/null; then
    echo -e "${RED}  ✗ FAIL: Found banned include 'run_operation.hpp'${NC}"
    echo "    Modern pattern uses: 'device_operation.hpp'"
    ERRORS=$((ERRORS + 1))
fi

# Check for banned operation::run
if grep -r "operation::run" "$OP_PATH/" 2>/dev/null; then
    echo -e "${RED}  ✗ FAIL: Found banned pattern 'operation::run'${NC}"
    echo "    Modern pattern uses: 'ttnn::prim::'"
    ERRORS=$((ERRORS + 1))
fi

# Check for banned ProgramWithCallbacks
if grep -r "ProgramWithCallbacks" "$OP_PATH/" 2>/dev/null; then
    echo -e "${RED}  ✗ FAIL: Found banned type 'ProgramWithCallbacks'${NC}"
    echo "    Modern pattern uses: 'CachedProgram<SharedVariables>'"
    ERRORS=$((ERRORS + 1))
fi

# Check for non-static validate (legacy pattern)
if grep -r "void validate.*const$" "$OP_PATH/device/" 2>/dev/null | grep -v "static"; then
    echo -e "${RED}  ✗ FAIL: Found non-static validate() method${NC}"
    echo "    Modern pattern uses: 'static void validate_on_program_cache_miss()'"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}  ✓ PASS: No banned patterns found${NC}"
fi

# Check 4: Required patterns in code
echo "Check 4: Checking for required patterns..."

# Check for device_operation.hpp include
if ! grep -r "device_operation.hpp" "$OP_PATH/" 2>/dev/null >/dev/null; then
    echo -e "${RED}  ✗ FAIL: Missing required include 'device_operation.hpp'${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check for ttnn::prim:: usage
if ! grep -r "ttnn::prim::" "$OP_PATH/" 2>/dev/null >/dev/null; then
    echo -e "${RED}  ✗ FAIL: Missing required pattern 'ttnn::prim::'${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check for static validate functions
if ! grep -r "static void validate_on_program_cache" "$OP_PATH/device/" 2>/dev/null >/dev/null; then
    echo -e "${RED}  ✗ FAIL: Missing required pattern 'static void validate_on_program_cache_*'${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check for CachedProgram
if ! grep -r "CachedProgram" "$OP_PATH/device/" 2>/dev/null >/dev/null; then
    echo -e "${RED}  ✗ FAIL: Missing required pattern 'CachedProgram'${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check for operation_attributes_t
if ! grep -r "operation_attributes_t" "$OP_PATH/device/" 2>/dev/null >/dev/null; then
    echo -e "${RED}  ✗ FAIL: Missing required pattern 'operation_attributes_t'${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check for tensor_args_t
if ! grep -r "tensor_args_t" "$OP_PATH/device/" 2>/dev/null >/dev/null; then
    echo -e "${RED}  ✗ FAIL: Missing required pattern 'tensor_args_t'${NC}"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}  ✓ PASS: All required patterns found${NC}"
fi

# Check 5: DeviceOperation struct has only static functions
echo "Check 5: Checking DeviceOperation struct..."

DEVICE_OP_FILE="$OP_PATH/device/${OP_NAME}_device_operation.hpp"
if [ -f "$DEVICE_OP_FILE" ]; then
    # Extract the struct definition
    struct_content=$(sed -n '/struct.*DeviceOperation/,/^};/p' "$DEVICE_OP_FILE")

    # Check if there are any non-static member functions
    # We look for lines that:
    # 1. Start with whitespace followed by a return type (void, auto, std::, etc.)
    # 2. Contain a function name followed by parentheses
    # 3. End with ; (declaration) - this distinguishes function declarations from parameters
    # 4. Don't have 'static' keyword
    # 5. Don't start with 'using' (type aliases)
    if echo "$struct_content" | grep -E '^\s+(void|auto)\s+\w+\s*\(' | grep -v "static" | grep -v "using" | grep -v "//" | grep -q .; then
        echo -e "${RED}  ✗ FAIL: DeviceOperation struct has non-static member functions${NC}"
        echo "    Modern pattern requires ALL functions to be static"
        ERRORS=$((ERRORS + 1))
    else
        echo -e "${GREEN}  ✓ PASS: DeviceOperation struct uses static functions${NC}"
    fi
else
    echo -e "${YELLOW}  ⚠ SKIP: DeviceOperation header not found${NC}"
fi

# Summary
echo ""
echo "================================"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All verification checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Verification failed with $ERRORS error(s)${NC}"
    exit 1
fi
