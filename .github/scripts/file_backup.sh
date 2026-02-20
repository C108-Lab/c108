#!/usr/bin/env bash
set -euo pipefail

SRC_PATH=${1}
DEST=${2:-$(basename "$SRC_PATH")}

BACKUP_DIR=${WORKFLOW_BACKUPS:-"/tmp/workflow-backups"}
mkdir -p "$BACKUP_DIR"

DEST_PATH="${BACKUP_DIR}/$DEST"
cp $SRC_PATH $DEST_PATH

echo "Saved ${SRC_PATH} to ${DEST_PATH}"
