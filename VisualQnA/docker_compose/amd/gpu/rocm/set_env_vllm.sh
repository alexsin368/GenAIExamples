#!/usr/bin/env bash

# Copyright (C) 2024 Advanced Micro Devices, Inc
# SPDX-License-Identifier: Apache-2.0

export HOST_IP=${host_ip}
export EXTERNAL_HOST_IP=${host_ip}
export VISUALQNA_VLLM_SERVICE_PORT="8081"
export VISUALQNA_HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export VISUALQNA_CARD_ID="card1"
export VISUALQNA_RENDER_ID="renderD136"
export VISUALQNA_LVM_MODEL_ID="Xkev/Llama-3.2V-11B-cot"
export LVM_ENDPOINT="http://${HOST_IP}:${VISUALQNA_VLLM_SERVICE_PORT}"
export LVM_SERVICE_PORT=9399
export MEGA_SERVICE_HOST_IP=${HOST_IP}
export LVM_SERVICE_HOST_IP=${HOST_IP}
export BACKEND_SERVICE_ENDPOINT="http://${host_ip}:${BACKEND_SERVICE_PORT}/v1/visualqna"
export FRONTEND_SERVICE_IP=${HOST_IP}
export FRONTEND_SERVICE_PORT=5173
export BACKEND_SERVICE_NAME=visualqna
export BACKEND_SERVICE_IP=${HOST_IP}
export BACKEND_SERVICE_PORT=8888
export NGINX_PORT=18003
