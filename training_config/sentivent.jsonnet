local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "roberta-base",
  distributed: {
    cuda_devices: [2,3],
  },
  data_paths: {
    train: "data/sentivent/preproc-collated/train.jsonl",
    validation: "data/sentivent/preproc-collated/dev.jsonl",
    test: "data/sentivent/preproc-collated/test.jsonl",
  },
  loss_weights: {
    ner: 0.5,
    relation: 1.0,
    coref: 0.0,
    events: 1.0
  },
  target_task: "events",
}