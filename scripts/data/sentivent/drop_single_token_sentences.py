'''
script to merge single token sentences in the dataset.
This is okay as they never contain event or sentiment annotations.
'''
from pathlib import Path
import json


def merge_sentences(doc):
    for i, sen in enumerate(doc["sentences"]):
        if len(sen) <= 1:
            print("Merged", doc["doc_key"], i, sen, doc["events"][i], doc["relations"][i])
            for k, v in doc.items():
                if isinstance(v, list):
                    v[i] = v[i] + v[i+1]
                    del v[i+1]
    return doc

if __name__ == "__main__":
    data_dirp = Path("/home/gilles/repos/dygiepp-dev/data/sentivent/preprocessed-rolemap-insamesentence")

    for json_fp in data_dirp.rglob("*json*"):
        merged_docs = []
        with open(json_fp, "rt") as json_in:
            for l in json_in.readlines():
                doc = json.loads(l)

                merged_docs.append(merge_sentences(doc))

        with open(json_fp, "wt") as json_out:
            for doc in merged_docs:
                json_out.write(json.dumps(doc) + "\n")
            print(f"Wrote {json_fp} with merged single sentence.")



