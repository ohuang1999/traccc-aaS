#!/bin/bash
# Start tritonserver against OUR rebuilt backend, not the image's.
#
# Assembles a model repository on node-local disk holding the repo's config.pbtxt
# plus the .so from build.sh, then points tritonserver at it. The backend table in
# the log naming a path under /models is how you know it loaded ours.
#
# Set SHM to adopt the detector from a /dev/shm region produced by
# run_producer.sh instead of parsing JSON; leave it empty for the baseline. Both
# paths stay in the code so the two can be compared on identical input.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"
d3_require_sif
d3_require_geo

[ -f "${BACKEND_BUILD}/libtriton_traccc.so" ] || {
    echo "FATAL: backend not built. Run ${HERE}/build.sh first."; exit 1; }

# A leftover tritonserver holding a port makes Triton load the model (~35 s) and
# only then abort. Check first, so the failure is immediate and names the cause.
for p in "${GRPC_PORT}" "${HTTP_PORT}" "${METRICS_PORT}"; do
    if ss -ltn "sport = :${p}" 2>/dev/null | grep -q LISTEN; then
        echo "FATAL: port ${p} is already in use."
        echo "  yours?  pgrep -u ${USER} -af tritonserver"
        echo "  stop it: pkill -u ${USER} -f 'tritonserver --model-repository'"
        echo "  or:      GRPC_PORT=8021 HTTP_PORT=8020 METRICS_PORT=8022 $0"
        exit 1
    fi
done

mkdir -p "${MODELS}/traccc-gpu/1"
cp -f "${REPO}/backend/models/traccc-gpu/config.pbtxt" "${MODELS}/traccc-gpu/"
cp -f "${BACKEND_BUILD}/libtriton_traccc.so" "${MODELS}/traccc-gpu/"

ENV_ARGS=()
if [ -n "${SHM}" ]; then
    echo "geometry source: SHARED REGION ${SHM}"
    ls -lh "/dev/shm${SHM}" || { echo "FATAL: no region at /dev/shm${SHM}"; exit 1; }
    ENV_ARGS+=(--env "TRACCC_DETRAY_SHM=${SHM}")
else
    echo "geometry source: JSON in ${GEO}  (baseline)"
fi

echo "models  : ${MODELS}"
echo "node    : $(hostname)"
echo "ports   : grpc ${GRPC_PORT} · http ${HTTP_PORT} · metrics ${METRICS_PORT}"
echo

# cd away from /eos: the container cannot chdir there and warns on every start.
cd /tmp/"${USER}"
exec apptainer exec --nv \
    --bind "${GEO}:/traccc/itk-geometry:ro" \
    --bind "${MODELS}:/models" \
    "${ENV_ARGS[@]}" \
    "${SIF}" \
    bash -lc "tritonserver --model-repository=/models --http-port=${HTTP_PORT} --grpc-port=${GRPC_PORT} --metrics-port=${METRICS_PORT}"
