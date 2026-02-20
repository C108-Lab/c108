#!/usr/bin/env bash
set -euo pipefail

BACKUP_FILE=${1}
DEST_PATH=${2:-"./$(basename "$1")"}

BACKUP_DIR=${WORKFLOW_BACKUPS:-"/tmp/workflow-backups"}
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

# Check if the backup actually exists before trying to copy
if [[ ! -f "$BACKUP_PATH" ]]; then
    echo "Error: Backup file not found: ${BACKUP_PATH}"
    exit 1
fi

# Check if destination dir exists
DEST_DIR=$(dirname "$DEST_PATH")
if [[ ! -d "$DEST_DIR" ]]; then
    echo "Creating destination directory: ${DEST_DIR}"
    mkdir -p "$DEST_DIR"
fi

# Restore
cp "$BACKUP_PATH" "$DEST_PATH"

echo "Restored ${BACKUP_PATH} to ${DEST_PATH}"
