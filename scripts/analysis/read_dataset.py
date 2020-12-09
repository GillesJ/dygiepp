#!/usr/bin/env python3
'''
Dataset inspection script for processed ace05 dataset.

read_aceevent_dataset.py in dygiepp
8/5/20 Copyright (c) Gilles Jacobs
'''
from collections import Counter

def balance(c):
    # Shannon's Diversity Index 0 unbalanced, 1 balanced
    from collections import Counter
    from numpy import log

    n = sum(c.values())
    classes = [(clas,float(count)) for clas, count in c.items()]
    k = len(classes)

    H = -sum([(count/n) * log((count/n)) for clas,count in classes]) #shannon entropy
    return H/log(k)

def stat_events(dataset):

    c = Counter()

    sentences = 0

    for doc in dataset.documents:
        sentences += len(doc.sentences)
        for sen in doc.sentences:
            for ev in sen.events:
                c.update([ev.trigger.label])

    total = sum(c.values())
    stats = {i: (count, round(count / total * 100.0, 1)) for i, count in c.most_common()}
    stats["n_events"] = total
    stats["balance"] = balance(c)
    stats["n_sentences"] = sentences
    stats["events/sentences"] = total / sentences
    stats["n_docs"] = round(len(dataset.documents), 1)
    stats["events/documents"] = round(total / len(dataset.documents), 1)
    return stats

    pass


from dygie.data import Document
from dygie.data import Dataset

# data = Dataset("/home/gilles/repos/dygiepp/data/ace-event/processed-data/default-settings/json/train.json")
# data = Dataset.from_jsonl("/home/gilles/repos/dygiepp-dev/data/ace-event/processed-data/multi/json/train.json")
data = Dataset.from_jsonl("/home/gilles/repos/dygiepp-dev/data/ace-event/processed-data/default-settings-multi-col/json/train.jsonl")
# print(data[0])  # Print the first document.
# print(data[0][1].ner)  # Print the named entities in the second sentence of the first document.
# for doc in data:
#     for sen in doc:
#         print(sen)
#         print(sen.events)

event_cnt = stat_events(data)
print(event_cnt)
data = Dataset.from_jsonl("/home/gilles/repos/dygiepp-dev/data/sentivent/multi-preproc-all-multitrig/train.jsonl")
event_cnt = stat_events(data)
print(event_cnt)