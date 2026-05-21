#!/usr/bin/env bash
# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Install system dependencies used by the course-maintained baseline workflow.
# Command line arguments:
#   --no-soundfonts: Skip checking/downloading the default soundfont.

set -euo pipefail

SCRIPT_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_PROJECT_DIR"

SKIP_SOUNDFONTS=false
while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    --no-soundfonts)
      SKIP_SOUNDFONTS=true
      shift
      ;;
    *)
      echo "Unknown argument: $key" >&2
      exit 1
      ;;
  esac
done

log() {
  printf '[install_deps] %s\n' "$*"
}

ensure_soundfont() {
  local target_dir target_path link
  target_dir="$SCRIPT_PROJECT_DIR/robopianist/soundfonts"
  target_path="$target_dir/TimGM6mb.sf2"
  mkdir -p "$target_dir"
  if [[ -f "$target_path" ]]; then
    log "Bundled soundfont already present: $target_path"
    return 0
  fi

  link="https://sourceforge.net/p/mscore/code/HEAD/tree/trunk/mscore/share/sound/TimGM6mb.sf2?format=raw"
  if command -v curl >/dev/null 2>&1; then
    log "Downloading default soundfont with curl"
    curl -L "$link" -o "$target_path"
  elif command -v wget >/dev/null 2>&1; then
    log "Downloading default soundfont with wget"
    wget "$link" -O "$target_path"
  else
    echo "Need curl or wget to download the default soundfont." >&2
    exit 1
  fi
}

ensure_shadow_hand_assets() {
  local bundled_dir fallback_dir
  bundled_dir="$SCRIPT_PROJECT_DIR/robopianist/models/hands/third_party/shadow_hand"
  fallback_dir="$SCRIPT_PROJECT_DIR/third_party/mujoco_menagerie/shadow_hand"

  if [[ -f "$bundled_dir/right_hand.xml" ]]; then
    log "Bundled Shadow Hand assets already present"
    return 0
  fi

  if [[ -d "$fallback_dir" ]]; then
    log "Copying Shadow Hand assets from third_party fallback"
    mkdir -p "$bundled_dir"
    cp -r "$fallback_dir/"* "$bundled_dir/"
    return 0
  fi

  echo "Missing Shadow Hand MJCF assets under $bundled_dir." >&2
  echo "Restore them in-repo or provide third_party/mujoco_menagerie/shadow_hand." >&2
  exit 1
}

if [[ ${OSTYPE:-} == darwin* ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required on macOS. Install it first, then rerun this script." >&2
    exit 1
  fi
  brew update
  brew install portaudio fluid-synth ffmpeg
elif [[ ${OSTYPE:-} == linux* ]]; then
  sudo apt update
  sudo apt install -y build-essential curl ffmpeg fluidsynth portaudio19-dev wget unzip
else
  echo "Unsupported OS: ${OSTYPE:-unknown}" >&2
  exit 1
fi

if [[ "$SKIP_SOUNDFONTS" == false ]]; then
  ensure_soundfont
fi
ensure_shadow_hand_assets

log "System dependency helper completed successfully"
