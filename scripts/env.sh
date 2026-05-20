#!/usr/bin/env bash

UV=https://astral.sh/uv/install.sh

# Setup python environment.
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    curl -LsSf "${UV}" | sh

    export PATH="${HOME}/.local/bin:${PATH}"
else
    echo "uv found: $(uv --version)"
fi

uv init
uv sync