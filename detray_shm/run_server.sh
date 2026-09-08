#!/bin/bash
# Start tritonserver FROM THE ATLAS RELEASE, loading the backend built by
# build_release.sh. No container anywhere: the release provides the server, the
# libraries and the compiler.
#
# Set SHM to adopt the detector from a /dev/shm region instead of parsing JSON.
# The region is produced by Athena: a JSONDeviceDetectorDescriptionProviderSvc
# with SharedMemoryRegion set (see the README). It must come from a job running
# this SAME release, or the ABI gates will refuse it.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"
rb_require_geo

[ -f "${BACKEND_BUILD}/libtriton_traccc.so" ] || {
    echo "FATAL: backend not built. Run ${HERE}/build_release.sh first."; exit 1; }

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

if [ -n "${SHM}" ]; then
    echo "geometry source: SHARED REGION ${SHM}"
    ls -lh "/dev/shm${SHM}" || { echo "FATAL: no region at /dev/shm${SHM}"; exit 1; }
    export TRACCC_DETRAY_SHM="${SHM}"
else
    echo "geometry source: JSON in ${GEO}  (baseline)"
fi

echo "release : ${AtlasVersion} ${BINARY_TAG}"
echo "models  : ${MODELS}"
echo "node    : $(hostname)"
echo "port    : grpc ${GRPC_PORT}  (this build has no HTTP or metrics service)"
echo

exec tritonserver --model-repository="${MODELS}" --grpc-port="${GRPC_PORT}"
