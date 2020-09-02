#!/usr/bin/env python3
'''
Run error analysis for determining error types by label.
Types:
Missing: no trigger/arg predicted where gold label is present.
spurious: trigger/arg predicted where no gold label is present.
misclassification: wrong trigger/arg clas. predicted.

Also parse annotated error file and get summary statistics.

error_analysis.py in dygiepp
8/24/20 Copyright (c) Gilles Jacobs
'''
from dygie.training.f1 import compute_f1
from dygie.data.dataset_readers.data_structures import Dataset
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def compute_argument_scores(metrics):

    metrics["arg_class_precision"], metrics["arg_class_recall"], metrics["arg_class_f1"] = compute_f1(
        metrics["predicted_arguments"], metrics["gold_arguments"], metrics["matched_argument_classes"])

    if "matched_argument_ids" in metrics:
        metrics["arg_id_precision"], metrics["arg_id_recall"], metrics["arg_id_f1"] = compute_f1(
        metrics["predicted_arguments"], metrics["gold_arguments"], metrics["matched_argument_ids"])
        metrics["missing_n"] = metrics["gold_arguments"] - metrics["matched_argument_ids"]
        metrics["misclassified_n"] = metrics["matched_argument_ids"] - metrics["matched_argument_classes"]
        metrics["total_errors"] = metrics["missing_n"] + metrics["misclassified_n"] + metrics["spurious"]


        metrics["errors_missing_pct"] = round(100 * metrics["missing_n"] / metrics["total_errors"], 1)
        metrics["errors_misclassified_pct"] = round(100 * metrics["misclassified_n"] / metrics["total_errors"], 1)
        metrics["errors_spurious_pct"] = round(100 * metrics["spurious"] / metrics["total_errors"], 1)
        pass

def compute_trigger_scores(metrics):

    metrics["trig_class_precision"], metrics["trig_class_recall"], metrics["trig_class_f1"] = compute_f1(
        metrics["predicted_triggers"], metrics["gold_triggers"], metrics["matched_trigger_classes"])

    if "matched_trigger_ids" in metrics:
        metrics["trig_id_precision"], metrics["trig_id_recall"], metrics["trig_id_f1"] = compute_f1(
        metrics["predicted_triggers"], metrics["gold_triggers"], metrics["matched_trigger_ids"])
        metrics["missing_n"] = metrics["gold_triggers"] - metrics["matched_trigger_ids"]
        metrics["misclassified_n"] = metrics["matched_trigger_ids"] - metrics["matched_trigger_classes"]
        metrics["total_errors"] = metrics["missing_n"] + metrics["misclassified_n"] + metrics["spurious"]


        metrics["errors_missing_pct"] = round(100 * metrics["missing_n"] / metrics["total_errors"], 1)
        metrics["errors_misclassified_pct"] = round(100 * metrics["misclassified_n"] / metrics["total_errors"], 1)
        metrics["errors_spurious_pct"] = round(100 * metrics["spurious"] / metrics["total_errors"], 1)


def get_ev(token_ix, evs):
    return next(ev for ev in evs if ev.token.ix_doc == token_ix)

def collect_trigger_metrics(data_with_predictions):
    metrics = {
        "gold_triggers": 0,
        "predicted_triggers": 0,
        "matched_trigger_ids": 0,
        "matched_trigger_classes": 0,
        "spurious": 0
    }

    errors = []
    all_labels = set()
    for doc in data_with_predictions.documents:
        for sen in doc.sentences:
            for ev in sen.events.triggers.union(sen.predicted_events.triggers):
                all_labels.add(ev.label)

    metrics_label = {l: without(metrics, "matched_trigger_ids") for l in all_labels}

    for doc in data_with_predictions.documents:
        for sen in doc.sentences:
            triggers_gold = {trig.token.ix_doc: trig.label for trig in sen.events.triggers}
            metrics["gold_triggers"] += len(triggers_gold)
            for gold_label in triggers_gold.values():
                metrics_label[gold_label]["gold_triggers"] += 1
            triggers_pred = {trig.token.ix_doc: trig.label for trig in sen.predicted_events.triggers}
            metrics["predicted_triggers"] += len(triggers_pred)

            for token_ix, label in triggers_pred.items():
            # Check whether the offsets match, and whether the labels match.
                metrics_label[label]["predicted_triggers"] += 1
                if token_ix in triggers_gold:
                    metrics["matched_trigger_ids"] += 1
                    if triggers_gold[token_ix] == label:
                        metrics["matched_trigger_classes"] += 1
                        metrics_label[label]["matched_trigger_classes"] += 1
                    else:
                        metrics_label[label].setdefault("mismatched_trigger_classes", []).append(triggers_gold[token_ix])
                        gold_ev = get_ev(token_ix, sen.events.triggers)
                        pred_ev = get_ev(token_ix, sen.predicted_events.triggers)
                        errors.append({
                            "error_type": "mismatch_label",
                            "id": f"{doc._doc_key}_{sen.sentence_ix}",
                            "sentence_text": " ".join(sen.text),
                            "pred_label": label,
                            "gold_label": triggers_gold[token_ix],
                            "pred_trigger": pred_ev.token.text,
                            "gold_trigger": gold_ev.token.text,
                            "gold_ix": token_ix,
                            "pred_ix": token_ix,
                        })
                else:
                    metrics["spurious"] += 1
                    metrics_label[label]["spurious"] += 1
                    pred_ev = get_ev(token_ix, sen.predicted_events.triggers)
                    errors.append({
                            "error_type": "spurious",
                            "id": f"{doc._doc_key}_{sen.sentence_ix}",
                            "sentence_text": " ".join(sen.text),
                            "pred_label": label,
                            "gold_label": None,
                            "pred_trigger": pred_ev.token.text,
                            "gold_trigger": None,
                            "gold_ix": None,
                            "pred_ix": token_ix,
                        })

            for token_ix, label in triggers_gold.items():
                if token_ix not in triggers_pred:
                    gold_ev = get_ev(token_ix, sen.events.triggers)
                    errors.append({
                            "error_type": "missing",
                            "id": f"{doc._doc_key}_{sen.sentence_ix}",
                            "sentence_text": " ".join(sen.text),
                            "pred_label": None,
                            "gold_label": label,
                            "pred_trigger": None,
                            "gold_trigger": gold_ev.token.text,
                            "gold_ix": token_ix,
                            "pred_ix": None,
                        })

    return metrics, metrics_label, errors

def without(d, key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d

def score_arguments(metrics, metrics_label, predicted_triggers, gold_triggers, predicted_arguments, gold_arguments):
    # Note that the index of the trigger doesn't actually need to be correct to get full credit;
    # the event type and event role need to be correct (see Sec. 3 of paper).
    def format(arg_dict, trigger_dict):
        # Make it a list of [index, event_type, arg_label].
        res = []
        for (trigger_ix, arg_ix), label in arg_dict.items():
            # If it doesn't match a trigger, don't predict it (enforced in decoding).
            if trigger_ix not in trigger_dict:
                continue
            event_type = trigger_dict[trigger_ix]
            res.append((arg_ix, event_type, label))
        return res

    formatted_gold_arguments = format(gold_arguments, gold_triggers)
    formatted_predicted_arguments = format(predicted_arguments, predicted_triggers)

    metrics["gold_arguments"] += len(formatted_gold_arguments)
    metrics["predicted_arguments"] += len(formatted_predicted_arguments)

    # Go through each predicted arg and look for a match.
    for entry in formatted_predicted_arguments:
        # No credit if not associated with a predicted trigger.
        class_match = int(any([entry == gold for gold in formatted_gold_arguments]))
        id_match = int(any([entry[:2] == gold[:2] for gold in formatted_gold_arguments]))

        metrics["matched_argument_classes"] += class_match
        metrics["matched_argument_ids"] += id_match

        if not id_match:
            metrics["spurious"] += 1

        if id_match - class_match > 0:
            # misclassified.
            # get the labels somehow
            metrics["misclassified_n"] += id_match - class_match

def get_all_labels(data_with_predictions):
    all_labels = set()
    for doc in data_with_predictions.documents:
        for sen in doc.sentences:
            for arg in sen.events.arguments.union(sen.predicted_events.arguments):
                all_labels.add(arg.role)
    return all_labels

def collect_argument_metrics(data_with_predictions):
    # init our accumulators
    metrics = {
        "gold_arguments": 0,
        "predicted_arguments": 0,
        "matched_argument_ids": 0,
        "matched_argument_classes": 0,
        "misclassified_n": 0,
        "spurious": 0
    }
    metrics_label = {l: without(metrics, "matched_argument_ids") for l in get_all_labels(data_with_predictions)}

    def format_args(events):
        arguments = {}
        for ev in events:
            trigger_ix = ev.trigger.token.ix_doc
            for arg in ev.arguments:
                arg_ix = arg.span.span_doc
                label = arg.role
                arguments[(trigger_ix, arg_ix)] = label
        return arguments

    def format_triggers(events):
        triggers = {ev.trigger.token.ix_doc: ev.trigger.label for ev in events}
        return triggers

    for doc in data_with_predictions.documents:
        for sen in doc.sentences:
            predicted_arguments = format_args(sen.predicted_events.event_list)
            predicted_triggers = format_triggers(sen.predicted_events.event_list)
            gold_arguments = format_args(sen.events.event_list)
            gold_triggers = format_triggers(sen.events.event_list)
            score_arguments(metrics, metrics_label, predicted_triggers, gold_triggers, predicted_arguments, gold_arguments)
    return metrics, metrics_label

def plot_type_score(typescore_df):
    # typesc_df = typescore_df[columns]
    # typesc_df.columns = typesc_df.loc["metric"]
    # typesc_df = typesc_df.drop(index="metric")
    # for c in typesc_df.columns:
    #     typesc_df[c] = typesc_df[c].astype(np.float64)

    plt.rcParams["figure.figsize"] = [7.5, 3.4] # set size

    typescore_df = typescore_df.sort_values(by=["f1"])
    ax = typescore_df.plot.bar(width=0.75,
                               title=None,
                               alpha=0.99,  # needed to unbreak hatching in pdf
                                edgecolor="white",
                               )
    ax.set_xticklabels(list(typescore_df.index), rotation=45, ha="right")
    for i, p in enumerate(sorted(ax.patches, key=lambda x: x.xy)):
        if (i+1) % 3 == 0:
            ax.annotate(f"{np.round(p.get_height(),decimals=2)}".replace("0.","."), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # add hatching for monochrome printing
    bars = ax.patches
    hatches = "".join(h * len(typescore_df) for h in "./ O+-")

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch * 2)

    ax.legend(["Precision", "Recall", "F1-score (value on bar)"], frameon=False)
    # load type results and show only holdout
    fn = f"type-score-{'_'.join(list(typescore_df.columns.str.lower()))}"
    plt.savefig(fn + ".svg", bbox_inches="tight")
    plt.savefig(fn + ".pdf", bbox_inches="tight")

    plt.show()
    plt.close('all')

if __name__ == "__main__":

    # analyse the manual analysis file
    docs_done = ["amzn12", "amzn13", "amzn14", "aapl14", "aapl15", 'aapl16', "ba14", "ba15", "bac04", "bac05",
                 "cvx04", "duk05", "f13", "jnj04", "wmt05"] # docs manually analysed
    docs_done_re = "|".join(docs_done)
    df_error_ana = pd.read_csv("/home/gilles/repos/dygiepp/scripts/analysis/errors_trigger_anno.csv")
    # df_error_ana = df_error_ana.dropna(how='all', axis=1) # drop empty  columns used for spacing when annotating
    df_error_ana = df_error_ana[df_error_ana["id"].str.contains(docs_done_re, regex=True)]
     # just checking if parsed correctly
    df_error_ana_docs_done = list(df_error_ana["id"].str.split("_", expand=True)[0].unique())
    assert set(df_error_ana_docs_done) == set(docs_done)

    columns_error_type = ['1_tok_error', '1_tok_error_missclf', 'specialised context/creative language',
                          'Lexical sparse', 'plausible: saliency', 'ambiguous trigger']
    df_error_c = df_error_ana[columns_error_type].count()
    df_error_pct = (100 * df_error_c / len(df_error_ana)).round(1)

    # get P, R, F1 per label and Spurious, Missing, Missclf counts
    data_with_pred_fp = "/home/gilles/repos/dygiepp/predictions/sentivent-nonerforargs-test.jsonl"
    data_with_preds = Dataset(data_with_pred_fp)
    metrics_trigger, metrics_label_trigger, errors_trigger = collect_trigger_metrics(data_with_preds)
    metrics_argument, metrics_label_argument = collect_argument_metrics(data_with_preds)
    compute_argument_scores(metrics_argument)

    # Triggers
    compute_trigger_scores(metrics_trigger)
    for label, labmetrics in metrics_label_trigger.items():
        compute_trigger_scores(labmetrics)

    # plot label metrics_trigger
    metrics_label_data = []
    for k,v in metrics_label_trigger.items():
        if "mismatched_trigger_classes" in v:
            v["mismatched_trigger_classes"] = dict(Counter(v["mismatched_trigger_classes"]))
        v.update({"trigger_type": k})
        metrics_label_data.append(v)

    df_label = pd.DataFrame(metrics_label_data)
    columns_plot = ['trig_class_precision', 'trig_class_recall', 'trig_class_f1', 'trigger_type']
    df_label = df_label[columns_plot]
    df_label.columns = list(df_label.columns.str.replace("trig_class_", "").str.replace("trigger_type", "Trigger type"))
    df_label = df_label.set_index("Trigger type")

    df_label = plot_type_score(df_label)


    error_df = pd.DataFrame(errors_trigger)

    spurious_df = error_df[error_df["error_type"] == "spurious"]
    spurious_df.to_csv("/home/gilles/repos/dygiepp/scripts/analysis/errors_trigger_spurious.csv", index=False)

    columns_in_order = ["id", "error_type", "sentence_text", "gold_label", "gold_trigger", "pred_label", "pred_trigger", "gold_ix", "pred_ix"]
    error_df[columns_in_order].to_csv("/home/gilles/repos/dygiepp/scripts/analysis/errors_trigger.csv", index=False)


    pass
            # for trig_gold in triggers_gold:
            #     if trig_gold in triggers_pred:

            # spurious