#!/usr/bin/env bash
# Enter the TIG vector-search Docker container interactively
set -euo pipefail

CHALLENGE=vector_search
VERSION=0.0.1

echo "Entering Docker container for $CHALLENGE development..."

docker run -it --gpus all -v $(pwd):/app \
    ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/dev:$VERSION \
    bash 