TPU_VM_NAME="sfr-haolin-chen-v4-16"
TPU_ZONE="us-central2-b"

gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all \
    --command='pkill -f python'