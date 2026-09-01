#!/bin/bash
# Stage the ITk geometry on node-local disk. Once per node: /tmp is per-machine
# and is wiped when the node is reclaimed.
#
# Reading GEO_SRC needs membership of the
# atlas-tdaq-phase2-EFTracking-developers e-group.
set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [ -f "${GEO}/detray_detector_geometry.json" ]; then
    echo "already staged: ${GEO}  ($(ls "${GEO}" | wc -l) files)"
    exit 0
fi

echo "copying ${GEO_SRC} -> ${GEO}"
mkdir -p "${GEO}"
cp "${GEO_SRC}"/* "${GEO}/" || {
    echo "FATAL: cannot read ${GEO_SRC}" >&2
    echo "       (are you in the atlas-tdaq-phase2-EFTracking-developers e-group?)" >&2
    exit 1; }

ls -lh "${GEO}"
