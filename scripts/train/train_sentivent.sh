# Train ACE event extraction model.
# Usage: bash scripts/train/train_ace05_event.sh [gpu-id]
# gpu-id can be an integer GPU ID, or -1 for CPU.

experiment_name="sentivent-default"
data_root="./data/sentivent/preprocessed-rolemap-insamesentence"
config_file="./training_config/sentivent.jsonnet"

# Train model.
allennlp train $config_file \
    --serialization-dir ./models/$experiment_name \
    --include-package dygie \
    --force
