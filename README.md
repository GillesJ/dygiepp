# Extracting Fine-Grained Economic Events from Business News
Source code for the proceedings paper "Jacobs G. and Hoste V. 2020. Extracting Fine-Grained Economic Events from Business News. FNP-FNS @ COLING28.

Forked from https://github.com/dwadden/dygiepp @4f77e4d5703facbc5edf4535d065899975e84e90
Read original documentation at that commit for detailed setup and instructions.

## Replication data:
- For SENTiVENT economic event data: Preprocessed files in `data/sentivent/`.
- For ACE05_Event follow instructions on how to obtain in `DYGIEPP_README.md` (requires LDC licensing).
- Best trained Pytorch weights model for SENTiVENT in `models/sentivent-event-nonerforargs/`.
- Spreadsheet overviews of all hyperparameter runs in `predictions/`.

## Running the event pipeline for experiments

- To build image: `docker build -t dygiepp .`
- To run image with all gpus: `docker run --gpus all -it --ipc=host dygiepp`
- To run with specific device: `docker run --gpus "device=0" -it --ipc=host dygiepp`
- To run for dev mount volume: `docker run --gpus all -it --ipc=host -v $(realpath ./):/dygiepp/ dygiepp`

- Once inside a `dygiepp` container, run the following commands to activate the appropriate Python environement:
```bash
conda init bash
exec bash
conda activate dygiepp

```

Start training: `rm -rf ./models/sentivent-event-nonerforargs; bash ./scripts/train/train_sentivent_event.sh 0`

- To predict a trained model: `allennlp predict models/sentivent-event-nonerforargs/model.tar.gz data/sentivent/ner_with_subtype_args/test.jsonl --predictor dygie --include-package dygie --use-dataset-reader --output-file ./predictions/sentivent-test.jsonl --cuda-device 0`

When in container:
- To evaluate a trained model: `allennlp evaluate models/sentivent-event-nonerforargs/model.tar.gz data/sentivent/ner_with_subtype_args/test.jsonl --include-package dygie --output-file predictions/sentivent_metrics_test.jsonl --cuda-device 0`

## Contact
- Gilles Jacobs: gilles@jacobsgill.es, gilles.jacobs@ugent.be
- Veronique Hoste: veronique.hoste@ugent.be

## Mirrors
- https://osf.io/j63h9/
- https://github.com/GillesJ/sentivent-coling-fnp-fns