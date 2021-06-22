# Train SENTiVENT event extraction model.
# Usage: bash scripts/train/train_sentivent_event.sh [gpu-id]
# gpu-id can be an integer GPU ID, or -1 for CPU.

experiment_name="my-sentivent-event-nonerforargs"
data_root="./data/sentivent/ner_with_subtype_args"
config_file="./training_config/sentivent_event.jsonnet"
cuda_device=$1

# Train model.
ie_train_data_path=$data_root/train.jsonl \
    ie_dev_data_path=$data_root/dev.jsonl \
    ie_test_data_path=$data_root/test.jsonl \
    cuda_device=$cuda_device \
    allennlp train $config_file \
    --cache-directory $data_root/cached \
    --serialization-dir ./models/$experiment_name \
    --include-package dygie