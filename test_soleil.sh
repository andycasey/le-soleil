#!/usr/bin/env bash
#
# Test script for soleil
# Runs a test conversion and validates the output HDF5 file
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="$SCRIPT_DIR/test_output.h5"

echo "=========================================="
echo "Soleil Test Run"
echo "=========================================="
echo ""

# Source environment setup
echo "Setting up environment..."
source "$SCRIPT_DIR/setup.sh"
echo ""

# Clean up any existing test output
rm -f "$OUTPUT_FILE"

# Run soleil with all instruments, 4 processes, limit 100
echo "Running soleil..."
echo ""
"$SCRIPT_DIR/soleil" -p 4 --kpf --neid --harpsn --expres -l 100 -O "$OUTPUT_FILE"

echo ""
echo "=========================================="
echo "Validating HDF5 output"
echo "=========================================="
echo ""

# Run validation script
python3 "$SCRIPT_DIR/validate_hdf5.py" "$OUTPUT_FILE"

# Clean up
echo ""
echo "Cleaning up test file..."
rm -f "$OUTPUT_FILE"

echo ""
echo "=========================================="
echo "All tests passed!"
echo "=========================================="
