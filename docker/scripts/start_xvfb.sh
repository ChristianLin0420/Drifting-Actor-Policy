#!/bin/bash
# =============================================================================
# Start Xvfb for Headless Rendering
# =============================================================================
# This script starts a virtual X server for headless rendering in RLBench.
# It's used when running in Docker containers without a physical display.
#
# Usage:
#   ./start_xvfb.sh [display_number] [resolution]
#
# Arguments:
#   display_number: X display number (default: 99)
#   resolution: Screen resolution (default: 1920x1080x24)
# =============================================================================

DISPLAY_NUM=${1:-99}
RESOLUTION=${2:-1920x1080x24}

echo "Starting Xvfb on display :${DISPLAY_NUM} with resolution ${RESOLUTION}"

# Kill any existing Xvfb on this display
pkill -f "Xvfb :${DISPLAY_NUM}" 2>/dev/null || true

# Start Xvfb in the background
Xvfb :${DISPLAY_NUM} -screen 0 ${RESOLUTION} &> /tmp/xvfb.log &
XVFB_PID=$!

# Wait for Xvfb to start
sleep 2

# Check if Xvfb started successfully
if kill -0 ${XVFB_PID} 2>/dev/null; then
    echo "Xvfb started successfully (PID: ${XVFB_PID})"
    export DISPLAY=:${DISPLAY_NUM}
    echo "DISPLAY set to :${DISPLAY_NUM}"
else
    echo "ERROR: Failed to start Xvfb"
    cat /tmp/xvfb.log
    exit 1
fi


