#!/usr/bin/env python3
'''
Evaluate and compare all trained models.

compare_models.py in dygiepp
8/24/20 Copyright (c) Gilles Jacobs
'''
from pathlib import Path
import sys
import pandas as pd
from allennlp.commands import main
import traceback
import json

if __name__ == "__main__":
    use_evaluate = False
    model_dir = Path("/home/gilles/repos/dygiepp/models")
    test_fp = f"/home/gilles/repos/dygiepp/data/sentivent/ner_with_subtype_args/test.jsonl"

    model_fps = list(model_dir.rglob("model.tar.gz"))
    metrics_fps = list(model_dir.rglob("metrics.json"))
    dfs = [] # an empty list to store the data frames

    if use_evaluate: # use allenNlp evaluate # produces different scores than metrics.json do NOT use
        for model_fp in model_fps:
            model_id = model_fp.parts[-2]
            model_fp = str(model_fp)
            score_opt_fp = f"/home/gilles/repos/dygiepp/predictions/sentivent-{model_id}-metrics_test.jsonl"

            try:
                if not Path(score_opt_fp).exists():
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
            except Exception as e:
                print(e)
                traceback.print_exc()
                print(f"Skipping {model_id} and continuing eval.")
                continue

        data = pd.read_json(score_opt_fp, typ="series") # read data frame from json file
        data["model_id"] = model_id
        data = data.set_index("model_id")
        dfs.append(data) # append the data frame to the list

    else:
        for score_opt_fp in metrics_fps:
            with open(score_opt_fp, "rt") as scores_in:
                data = json.load(scores_in)
            data_df = pd.DataFrame(data, index=[score_opt_fp.parts[-2]]) # read data frame from json file
            data_df = data_df.rename_axis("model_id")
            dfs.append(data_df) # append the data frame to the list


    if use_evaluate:
        all_eval = pd.concat(dfs, axis=1).transpose() # concatenate all the data frames in the list.
        all_eval = all_eval.sort_values(["trig_class_f1", "arg_class_f1"], ascending=False)
        all_eval = all_eval.loc[:, (all_eval != 0).all(axis=0)]
        all_eval.to_csv("/home/gilles/repos/dygiepp/predictions/overview_evaluate.csv")
    else:
        all_eval = pd.concat(dfs)
        all_eval = all_eval.sort_values(["best_validation_trig_class_f1", "best_validation_arg_class_f1"], ascending=False)
        all_eval = all_eval.loc[:, (all_eval != 0).all(axis=0)]
        all_eval.to_csv("/home/gilles/repos/dygiepp/predictions/overview_all_info.csv")
        overview_bestf1 = all_eval[["best_validation_trig_class_f1", "best_validation_arg_class_f1"]]
        overview_bestf1.to_csv("/home/gilles/repos/dygiepp/predictions/overview_bestf1.csv")
        overview_best_paper = all_eval[
            ['best_validation__trig_id_precision', 'best_validation__trig_id_recall', 'best_validation__trig_id_f1',
             'best_validation__trig_class_precision', 'best_validation__trig_class_recall', 'best_validation_trig_class_f1',
             'best_validation__arg_id_precision', 'best_validation__arg_id_recall', 'best_validation__arg_id_f1',
             'best_validation__arg_class_precision', 'best_validation__arg_class_recall', 'best_validation_arg_class_f1'
             ]]
        overview_best_paper.columns = overview_best_paper.columns.str.replace("best_validation_", "").str.lstrip("_")
        # overview_best_paper = overview_best_paper_table * 100
        (overview_best_paper * 100).to_csv("/home/gilles/repos/dygiepp/predictions/overview_best_paper_table.csv", float_format="%.2f")



