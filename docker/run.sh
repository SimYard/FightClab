#!/usr/bin/env bash
# Usage: docker/run.sh [--build|-b] [command...]   (no command = interactive shell)

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1

BUILD=""
if [[ "$1" == "--build" || "$1" == "-b" ]]; then
    BUILD="--build"
    shift
fi

docker compose -f docker-compose.yml run ${BUILD} --rm fightclab "$@"
