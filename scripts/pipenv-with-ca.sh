#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERT_PATH="${ROOT_DIR}/ZscalerRootCA.crt"

if [[ ! -f "${CERT_PATH}" ]]; then
  echo "Certificate file not found: ${CERT_PATH}" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: scripts/pipenv-with-ca.sh <pipenv-args...>" >&2
  echo "Example: scripts/pipenv-with-ca.sh lock --dev" >&2
  exit 1
fi

SSL_CERT_FILE="${CERT_PATH}" \
REQUESTS_CA_BUNDLE="${CERT_PATH}" \
PIP_CERT="${CERT_PATH}" \
pipenv "$@"
