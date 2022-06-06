#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"main"}
REPO=${3:-"https://github.com/alteryx/evalml"}
PKG=${4:-"evalml"}

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true

PIP install -U evalml
installed="${HERE}/.setup/installed"
PY -c "from evalml import __version__; print(__version__)" >> "$installed"
