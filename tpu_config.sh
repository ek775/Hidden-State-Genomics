#!/bin/sh
export PROJECT_ID="mccoylab"
export TPU_NAME="tpu-vm-1"
export ZONE="us-central2-b"
export RUNTIME_VERSION="tpu-vm-tf-2.17.0-pod-pjrt"
export ACCELERATOR_TYPE="v4-8"

gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --accelerator-type=${ACCELERATOR_TYPE} \
  --version=${RUNTIME_VERSION}

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
      --zone=${ZONE}