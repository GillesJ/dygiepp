#!/usr/bin/env python3
'''
Call AllenNLP evaluate for error analysis debugging.

evaluate_debugging.py in dygiepp
8/24/20 Copyright (c) Gilles Jacobs
'''

#!/usr/bin/env python3
'''
Calling AllenNLP evaluate script in Python for interactive debugging.

evaluate_debugging.py in dygiepp
'''
import sys
from pathlib import Path
import os
from allennlp.commands import main

base_dir = "/home/gilles/repos/dygiepp/"
model_id = "nonerforargs"
model_fp = f"{base_dir}models/sentivent-event-{model_id}/model.tar.gz"
test_fp = f"{base_dir}data/sentivent/ner_with_subtype_args/test.jsonl"
score_opt_fp = f"{base_dir}predictions/sentivent-{model_id}-metrics_test.jsonl"

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "evaluate",
    model_fp,
    test_fp,
    "--output-file", score_opt_fp,
    "--cuda-device", "3",
    "--include-package", "dygie",
]
main()