# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG BASE_TAG=latest
FROM opea/comps-base:$BASE_TAG

COPY ./chatqna.py $HOME/chatqna.py

ENTRYPOINT ["python", "chatqna.py"]