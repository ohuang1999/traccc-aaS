#!/bin/bash
# Build the detector into /dev/shm and leave it there for the server.
#
# The region deliberately outlives this process -- that is the whole point -- so
# it is removed at the START of the next run rather than at the end of this one.
# Check what is parked there between runs: ls -lh /dev/shm
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"
d3_require_sif
d3_require_geo

[ -f "${PRODUCER_BUILD}/d3_producer" ] || {
    echo "FATAL: producer not built. Run ${HERE}/build_producer.sh first."; exit 1; }

rm -f "/dev/shm${REGION_NAME}"

apptainer exec --bind "${PRODUCER_BUILD}:/build" --bind "${GEO}:/geo:ro" "${SIF}" \
    /build/d3_producer /geo "${REGION_NAME}" "${REGION_GB}"

echo
echo "region reserved : $(du -h --apparent-size "/dev/shm${REGION_NAME}" | cut -f1)"
echo "RAM actually used: $(du -h "/dev/shm${REGION_NAME}" | cut -f1)  <- the real detector size"
echo
echo "now: SHM=${REGION_NAME} ${HERE}/run_server.sh"
