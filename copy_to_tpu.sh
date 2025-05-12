#!/bin/bash
# Default values
TPU_VM_NAME="sfr-haolin-chen-v4-16"
TPU_ZONE="us-central2-b"
LOCAL_REPO_PATH=$(pwd)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --local-repo-path=*)
            LOCAL_REPO_PATH="${1#*=}"
            shift
            ;;
        --vm=*)
            TPU_VM_NAME="${1#*=}"
            shift
            ;;
        --zone=*)
            TPU_ZONE="${1#*=}"
            shift
            ;;
        --remote-path=*)
            REMOTE_PATH="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --local-repo-path=PATH Local path to repository (default: $(pwd))"
            echo "  --vm=NAME         TPU VM name (required)"
            echo "  --zone=ZONE       GCP zone (default: us-central2-b)"
            echo "  --remote-path=PATH Remote path on TPU VM (default: ~/project)"
            echo "  --help            Display this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

PROJECT_NAME=$(basename $LOCAL_REPO_PATH)
REMOTE_PATH="~/${PROJECT_NAME}"


# Check if TPU VM name is provided
if [ -z "$TPU_VM_NAME" ]; then
    echo "Error: TPU VM name is required. Use --vm=NAME to specify."
    exit 1
fi

echo "Copying repository to TPU VM: $TPU_VM_NAME in zone $TPU_ZONE"
echo "Local path: $LOCAL_REPO_PATH"
echo "Remote path: $REMOTE_PATH"


# Copy files to temporary directory, excluding unnecessary directories
rsync -av \
    --exclude=".git/" \
    --exclude="__pycache__/" \
    --exclude="*.pyc" \
    --exclude=".ipynb_checkpoints/" \
    --exclude="venv/" \
    --exclude="env/" \
    --exclude="wandb/" \
    --exclude="logs/" \
    --exclude="checkpoints/" \
    $LOCAL_REPO_PATH/ /tmp/${PROJECT_NAME}

# Copy the repository to the TPU VM
# Exclude common directories that don't need to be copied
gcloud alpha compute tpus tpu-vm scp \
    --zone=$TPU_ZONE \
    --tunnel-through-iap \
    --recurse \
    --worker="all" \
    /tmp/${PROJECT_NAME} $TPU_VM_NAME:

if [ $? -eq 0 ]; then
    echo "Repository successfully copied to $TPU_VM_NAME/"
else
    echo "Failed to copy repository to $TPU_VM_NAME/"
fi

rm -rf /tmp/${PROJECT_NAME}