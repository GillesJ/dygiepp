cuda_device=$1

allennlp predict ./pretrained/ace05-event.tar.gz \
    ./data/sentivent/only_text/json/all.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file predictions/sentivent-all.jsonl \
    --cuda-device $cuda_device