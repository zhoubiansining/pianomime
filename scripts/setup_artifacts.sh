#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_PROJECT_DIR/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$SCRIPT_PROJECT_DIR/scripts/config_export.py" "$CONFIG_FILE" paths artifacts)"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

copy_optional_zarr() {
  local source_root="${ZARR_SOURCE_DIR:-}"
  if [[ -z "$source_root" ]]; then
    return 0
  fi
  if [[ -d "$source_root/dataset_hl.zarr" && -d "$source_root/dataset_ll.zarr" ]]; then
    log "Copying prebuilt zarr datasets from $source_root"
    rsync -a "$source_root/dataset_hl.zarr" "$PROJECT_DIR/"
    rsync -a "$source_root/dataset_ll.zarr" "$PROJECT_DIR/"
    return 0
  fi
  echo "ZARR_SOURCE_DIR is set but does not contain dataset_hl.zarr and dataset_ll.zarr: $source_root" >&2
  return 1
}

download_drive_file() {
  local file_id="$1"
  local output="$2"
  if [[ -s "$output" ]]; then
    log "Using cached artifact: $output"
    return 0
  fi

  mkdir -p "$(dirname "$output")"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN=python3
  fi
  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import gdown
PY
  then
    "$PYTHON_BIN" -m pip install gdown
  fi

  log "Downloading Google Drive file $file_id -> $output"
  "$PYTHON_BIN" -m gdown "https://drive.google.com/uc?id=${file_id}" -O "$output"
}

extract_zip() {
  local archive="$1"
  if [[ ! -s "$archive" ]]; then
    echo "Missing archive: $archive" >&2
    return 1
  fi
  log "Extracting $archive into $PROJECT_DIR"
  unzip -oq "$archive" -d "$PROJECT_DIR"
  rm -rf "$PROJECT_DIR/__MACOSX"
  find "$PROJECT_DIR" -name '.DS_Store' -delete
}

mkdir -p "$PROJECT_DIR" "$ARTIFACT_CACHE"

download_drive_file "$DATASET_ID" "$DATASET_ZIP"
download_drive_file "$CHECKPOINT_ID" "$CHECKPOINT_ZIP"

if [[ ! -d "$PROJECT_DIR/dataset" ]]; then
  extract_zip "$DATASET_ZIP"
else
  log "Dataset directory already exists"
fi

if [[ ! -f "$PROJECT_DIR/checkpoint_high_level.ckpt" || ! -f "$PROJECT_DIR/checkpoint_low_level.ckpt" || ! -f "$PROJECT_DIR/checkpoint_ae.ckpt" ]]; then
  extract_zip "$CHECKPOINT_ZIP"
else
  log "Checkpoint files already exist"
fi

copy_optional_zarr
if [[ ! -d "$PROJECT_DIR/dataset_hl.zarr" || ! -d "$PROJECT_DIR/dataset_ll.zarr" ]]; then
  log "Prebuilt zarr datasets are missing. Single-song replay can run, but generalist multi-task evaluation also needs dataset_hl.zarr and dataset_ll.zarr at the repository root."
  log "If you already have them elsewhere, rerun with: ZARR_SOURCE_DIR=/path/to/zarr_bundle bash scripts/setup_artifacts.sh"
fi

log "Artifacts ready under $PROJECT_DIR"
