# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG BASE_TAG=latest
FROM opea/comps-base:$BASE_TAG

COPY ./multimodalqna.py $HOME/multimodalqna.py

ENTRYPOINT ["python", "multimodalqna.py"]
