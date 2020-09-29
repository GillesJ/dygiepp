local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "bert-base-cased",
  cuda_device: "0,1",
  data_paths: {
    train: "data/sentivent/preprocessed-rolemap-insamesentence/train.jsonl",
    validation: "data/sentivent/preprocessed-rolemap-insamesentence/dev.jsonl",
    test: "data/sentivent/preprocessed-rolemap-insamesentence/test.jsonl",
  },
  loss_weights: {
    ner: 0.5,
    relation: 1.0,
    coref: 0.0,
    events: 1.0
  },
  target_task: "events",
}
