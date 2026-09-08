#!/bin/bash
# Build the detector into /dev/shm using the release-built producer.
#
# The region outlives this process on purpose -- that is the whole point -- so it
# is cleared at the START of the next run rather than the end of this one.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"
rb_require_geo

[ -f "${PRODUCER_BUILD}/rel_producer" ] || {
    echo "FATAL: producer not built. Run ${HERE}/build_producer_release.sh first."; exit 1; }

rb_asetup
rm -f "/dev/shm${REGION_NAME}"
"${PRODUCER_BUILD}/rel_producer" "${GEO}" "${REGION_NAME}" "${REGION_GB}"

echo
echo "region reserved  : $(du -h --apparent-size "/dev/shm${REGION_NAME}" | cut -f1)"
echo "RAM actually used: $(du -h "/dev/shm${REGION_NAME}" | cut -f1)  <- the real detector size"
echo
echo "now: SHM=${REGION_NAME} ${HERE}/run_server_release.sh"
