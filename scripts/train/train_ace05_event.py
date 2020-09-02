#!/usr/bin/env python3
'''
Calling AllenNLP train script in Python for interactive debugging.

train_ace05_event.py in dygiepp
'''
import shutil
import sys
from pathlib import Path
import os
from allennlp.commands import main

base_dir = Path("~/repos/dygiepp")
config_file = str(base_dir / "training_config/ace05_event.jsonnet")
experiment_name = "ace05-event"
data_root = base_dir / "data/ace-event/processed-data/default-settings/json_old_with_subtype_args"
serialization_dir = str(base_dir / "/tmp/debugger_train")

# Use env vars for libsonnet template compatibility.
os.environ["cuda_device"] = str(0)
os.environ["ie_train_data_path"] = str(data_root / "train.json_old_with_subtype_args")
os.environ["ie_dev_data_path"] = str(data_root / "dev.json_old_with_subtype_args")
os.environ["ie_test_data_path"] = str(data_root / "test.json_old_with_subtype_args")

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)
# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "--cache-directory", str(data_root / "cached"),
    "--serialization-dir", serialization_dir,
    "--include-package", "dygie",
]
main()