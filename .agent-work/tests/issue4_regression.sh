#!/usr/bin/env bash
# Issue #4 used to assert knowledge-base retrieval. Issue #5 replaced that
# path with pure transformer generation, so this regression now delegates to
# the generative chat test.
set -euo pipefail
exec bash "$(dirname "$0")/generative_chat.sh"
