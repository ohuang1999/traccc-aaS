# Shared settings for the D3 scripts. Sourced, never run directly.
#
# This folder works both inside the traccc-aaS repo and beside it, so the repo is
# located rather than hard-coded. Everything else is an override-able default:
# set any of these in the environment to point elsewhere.

D3_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- the traccc-aaS repo (holds the backend and shm_region.{hpp,cpp}) --------
_marker="backend/traccc-gpu/CMakeLists.txt"
if [ -z "${REPO:-}" ]; then
    if   [ -f "${D3_HERE}/../${_marker}" ];                then REPO="$(cd "${D3_HERE}/.." && pwd)"
    elif [ -f "${D3_HERE}/../../${_marker}" ];             then REPO="$(cd "${D3_HERE}/../.." && pwd)"
    elif [ -f "${D3_HERE}/../../traccc-aaS/${_marker}" ];  then REPO="$(cd "${D3_HERE}/../../traccc-aaS" && pwd)"
    else
        echo "FATAL: cannot find the traccc-aaS repo near ${D3_HERE}." >&2
        echo "       Set REPO=/path/to/traccc-aaS and re-run." >&2
        exit 1
    fi
fi
export REPO

# --- the container image (11 GB; lives outside the repo) --------------------
SIF="${SIF:-/eos/home-t/${USER}/Tracking_aaS/traccc-aas.sif}"
export SIF

# --- ITk geometry ------------------------------------------------------------
# GEO must hold detray_detector_{geometry,material_maps,surface_grids}.json,
# ITk_bfield.cvf, ITk_digitization_config.json and athenaIdentifierToDetrayMap.txt.
# prepare_geometry.sh copies them from GEO_SRC to node-local disk.
GEO_SRC="${GEO_SRC:-/eos/project/a/atlas-eftracking/GPU/ITk_data/FinalReport}"
GEO="${GEO:-/tmp/${USER}/itk-geo}"
export GEO_SRC GEO

# --- node-local build areas (/tmp is per-machine; so is the build) ----------
BACKEND_SRC="${BACKEND_SRC:-/tmp/${USER}/d3_backend_src}"
BACKEND_BUILD="${BACKEND_BUILD:-/tmp/${USER}/d3_backend_build}"
PRODUCER_SRC="${PRODUCER_SRC:-/tmp/${USER}/d3_producer_src}"
PRODUCER_BUILD="${PRODUCER_BUILD:-/tmp/${USER}/d3_producer_build}"
MODELS="${MODELS:-/tmp/${USER}/d3_models}"
export BACKEND_SRC BACKEND_BUILD PRODUCER_SRC PRODUCER_BUILD MODELS

# --- the shared region -------------------------------------------------------
SHM="${SHM:-}"                       # empty = JSON path; set to adopt a region
REGION_NAME="${REGION_NAME:-/d3_itk_detector}"
REGION_GB="${REGION_GB:-2}"
export SHM REGION_NAME REGION_GB

# --- Triton ports ------------------------------------------------------------
# Something system-wide listens on 8000 on every lxplus node, and Triton treats a
# failed HTTP bind as fatal -- it reports the model READY and then exits. 8001
# and 8002 are free, so a client using the default gRPC port still works.
HTTP_PORT="${HTTP_PORT:-8010}"
GRPC_PORT="${GRPC_PORT:-8001}"
METRICS_PORT="${METRICS_PORT:-8002}"
export HTTP_PORT GRPC_PORT METRICS_PORT

d3_require_sif() {
    [ -f "${SIF}" ] || { echo "FATAL: no image at ${SIF}. Set SIF=/path/to/traccc-aas.sif" >&2; exit 1; }
}

d3_require_geo() {
    [ -f "${GEO}/detray_detector_geometry.json" ] || {
        echo "FATAL: no geometry in ${GEO}." >&2
        echo "       Run ${D3_HERE}/prepare_geometry.sh on this node." >&2
        exit 1; }
}
