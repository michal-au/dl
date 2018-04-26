#!/usr/bin/env python3
import numpy as np
import morpho_dataset
import os


class MorphoAnalyzer:
    """ Loader for data of morphological analyzer.

    The loaded analyzer provides an only method `get(word)` returning
    a list of analyses, each containing two fields `lemma` and `tag`.
    If an analysis of the word is not found, an empty list is returned.
    """

    class LemmaTag:
        def __init__(self, lemma, tag):
            self.lemma = lemma
            self.tag = tag

    def __init__(self, filename):
        self.analyses = {}

        with open(filename, "r", encoding="utf-8") as analyzer_file:
            for line in analyzer_file:
                line = line.rstrip("\n")
                columns = line.split("\t")

                analyses = []
                for i in range(1, len(columns) - 1, 2):
                    analyses.append(MorphoAnalyzer.LemmaTag(columns[i], columns[i + 1]))
                self.analyses[columns[0]] = analyses

    def get(self, word):
        return self.analyses.get(word, [])


def create(fname):
    analyzer_dictionary = MorphoAnalyzer("czech-pdt-analysis-dictionary.txt")
    analyzer_guesser = MorphoAnalyzer("czech-pdt-analysis-guesser.txt")
    whole_train = morpho_dataset.MorphoDataset("czech-pdt-train.txt")
    height, width = len(whole_train.factors[whole_train.FORMS].words), len(whole_train.factors[whole_train.TAGS].words)
    tag_mask = np.zeros([height, width], dtype=np.bool)

    row_idcs, column_idcs = [], []
    for idx, w in enumerate(whole_train.factors[whole_train.FORMS].words):
        tags = analyzer_dictionary.get(w) or analyzer_guesser.get(w)
        tag_ids = [whole_train.factors[whole_train.TAGS].words_map.get(lt.tag) for lt in tags if
                   whole_train.factors[whole_train.TAGS].words_map.get(lt.tag)]
        if not tag_ids:
            row_idcs.extend([idx] * width)
            column_idcs.extend(range(width))
        else:
            row_idcs.extend([idx] * len(tag_ids))
            column_idcs.extend(tag_ids)

    tag_mask[(row_idcs, column_idcs)] = True
    tag_mask = tag_mask.astype(np.float32)
    if not os.path.exists("data"): os.mkdir("data")
    np.save("data/{}".format(fname), tag_mask)
    return tag_mask


def get():
    fname = "tag_mask"
    try:
        return np.load("data/{}.npy".format(fname))
    except FileNotFoundError:
        return create(fname)
