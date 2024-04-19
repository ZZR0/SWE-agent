#!/usr/bin/env bash

# bash strict mode
set -euo pipefail

echo "Setting up docker image for swe-agent..."
# TARGETARCH should be set automatically on most (but not all) systems, see
# https://github.com/princeton-nlp/SWE-agent/issues/245
docker build --network host --build-arg all_proxy=http://192.168.100.211:10809 -t sweagent/swe-agent:latest -f docker/swe.Dockerfile --build-arg TARGETARCH=$(uname -m) .

echo "Setting up docker image for evaluation..."
docker build --network host --build-arg all_proxy=http://192.168.100.211:10809 -t sweagent/swe-eval:latest -f docker/eval.Dockerfile .

echo "Done with setup!"
