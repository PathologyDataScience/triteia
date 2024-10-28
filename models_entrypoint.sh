#!/usr/bin/env bash

set -xe
python3 /setup_repository.py /models
exec "$@"
