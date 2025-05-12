# gcloud alpha compute tpus tpu-vm ssh sfr-weiran-yao-v4-512 \
#  --zone=us-central2-b \
#  --project=salesforce-research-internal \
#  --tunnel-through-iap \
#  --worker=all \
#  --command='cd md4; \
#             export PYTHONPATH="$PYTHONPATH:~/md4"; \
#             python md4/main.py --config=md4/configs/md4/openwebtext.py --sharded=true --workdir=./expt'

# TPU_VM_NAME="sfr-haolin-chen-v4-8"
# TPU_ZONE="us-central2-b"

# TPU_VM_NAME="sfr-weiran-yao-v4-512"
# TPU_ZONE="us-central2-b"

# gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
#     --zone=$TPU_ZONE \
#     --project=salesforce-research-internal \
#     --tunnel-through-iap \
#     --worker=all \
#     --command='ls md4'


TPU_VM_NAME="sfr-haolin-chen-v4-16"
TPU_ZONE="us-central2-b"

gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all \
    --command='git clone https://github.com/hlnchen/bd3lms.git \
    cd bd3lms; \
    python -m venv venv; \
    source venv/bin/activate; \
    pip install -r requirements.txt; \'
    # --command='cd fabric_test; \
    # export PJRT_DEVICE=TPU; \
    # source venv/bin/activate; \
    # python fabric_test.py'
