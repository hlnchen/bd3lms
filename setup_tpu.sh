#! /bin/bash
TPU_VM_NAME="sfr-haolin-chen-v4-16"
TPU_ZONE="us-central2-b"

gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all \
    --command='
    git clone <your_bd3lms_repo_url>; \
    cd bd3lms; \
    python -m venv venv; \
    source venv/bin/activate; \
    pip install -r requirements.txt; \
    export WANDB_API_KEY="<your_wandb_api_key>"; \
    wandb login $WANDB_API_KEY --relogin --host=https://salesforceairesearch.wandb.io'
