#!/bin/bash
# Push the MLA v3 tpu-inference working tree to the two places that matter.
#
# Three copies exist and they drift easily:
#   /tmp/tpu-inference            scratch clone where edits are made (cleared on reboot)
#   ~/Documents/tpu-inference-mla-v3  persistent copy on the laptop, keeps .git at 0cf165e
#   VM:/mnt/nfs/ajaygopi/tpu-inference  what the cdk pods actually mount and run
#
# __pycache__ on the share is root-owned (written by the container), so rsync
# cannot delete it; excluding it keeps the output readable and is harmless
# because Python recompiles on mtime change.
set -e
SRC="${1:-/tmp/tpu-inference}"
LAPTOP="$HOME/Documents/tpu-inference-mla-v3"
VM="ajaygopi-cdk-vm:/mnt/nfs/ajaygopi/tpu-inference/"
# .git MUST be excluded: /tmp is its own clone, and rsyncing its .git over
# the laptop's silently resets branches and detaches HEAD. Cost one lost
# branch pointer before it was noticed.
EX=(--exclude '__pycache__' --exclude '*.pyc' --exclude '.git')

rsync -a "${EX[@]}" "${SRC}/" "${LAPTOP}/"
echo "-> ${LAPTOP}"
rsync -a "${EX[@]}" --no-perms --no-owner --no-group "${SRC}/" "${VM}"
echo "-> ${VM}"

h() { (cd "$1" && LC_ALL=C find tpu_inference -name '*.py' | LC_ALL=C sort | xargs md5sum | md5sum | cut -d' ' -f1); }
echo "src    $(h "${SRC}")"
echo "laptop $(h "${LAPTOP}")"
echo "vm     $(ssh ajaygopi-cdk-vm "cd /mnt/nfs/ajaygopi/tpu-inference && LC_ALL=C find tpu_inference -name '*.py' | LC_ALL=C sort | xargs md5sum | md5sum | cut -d' ' -f1")"
