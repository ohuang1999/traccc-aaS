#!/bin/bash
# Start tritonserver FROM THE ATLAS RELEASE, loading the backend built by
# build.sh. No container anywhere: the release provides the server, the
# libraries and the compiler.
#
# The detector is adopted from a /dev/shm region; there is no JSON fallback. The
# region is produced by Athena -- a JSONDeviceDetectorDescriptionProviderSvc with
# SharedMemoryRegion set, see the README -- and must come from a job running this
# SAME release, or the ABI gates will refuse it. SHM names which region.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"
rb_require_geo

[ -f "${BACKEND_BUILD}/libtriton_traccc.so" ] || {
    echo "FATAL: backend not built. Run ${HERE}/build.sh first."; exit 1; }

# Triton binds its port only after loading the model, so a port held elsewhere
# costs a full model load before it aborts. Check first.
if ss -ltn "sport = :${GRPC_PORT}" 2>/dev/null | grep -q LISTEN; then
    echo "FATAL: port ${GRPC_PORT} already in use."
    echo "  yours?  pgrep -u ${USER} -af tritonserver"
    exit 1
fi

mkdir -p "${MODELS}/traccc-gpu/1"
cp -f "${BACKEND_SRC}/backend/models/traccc-gpu/config.pbtxt" "${MODELS}/traccc-gpu/"
cp -f "${BACKEND_BUILD}/libtriton_traccc.so" "${MODELS}/traccc-gpu/"

rb_asetup

[ -e "/dev/shm${SHM}" ] || {
    echo "FATAL: no region at /dev/shm${SHM}." >&2
    echo "       Produce it from Athena first -- see the README. There is no" >&2
    echo "       JSON fallback: the detector only comes from the region." >&2
    exit 1; }
echo "detector : /dev/shm${SHM}  ($(du -h "/dev/shm${SHM}" | cut -f1) resident)"
export TRACCC_DETRAY_SHM="${SHM}"

echo "release : ${AtlasVersion} ${BINARY_TAG}"
echo "models  : ${MODELS}"
echo "node    : $(hostname)"
echo "port    : grpc ${GRPC_PORT}  (this build has no HTTP or metrics service)"
echo

exec tritonserver --model-repository="${MODELS}" --grpc-port="${GRPC_PORT}"
