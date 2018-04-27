import morpho_dataset

# seen_tags = set()
#
# for fname in ("czech-pdt-train.txt", "czech-pdt-dev.txt", "czech-pdt-test.txt"):
#     with open(fname, "r", encoding="utf-8") as file:
#         in_sentence = False
#         for line in file:
#             line = line.rstrip("\r\n")
#             if line:
#                 columns = line.split("\t")
#                 seen_tags.add(columns[2])
#     print(len(seen_tags))
# print(">>>", len(seen_tags))
#

train = morpho_dataset.MorphoDataset("czech-pdt-train.txt")
print(len(train._factors[0].all_words))
dev = morpho_dataset.MorphoDataset("czech-pdt-dev.txt", train=train, shuffle_batches=False, all_words=train._factors[train.FORMS])
print(len(dev._factors[0].all_words))
test = morpho_dataset.MorphoDataset("czech-pdt-test.txt", train=train, shuffle_batches=False, all_words=dev._factors[dev.FORMS])
print(len(test._factors[0].all_words))

sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = train.next_batch(1, including_charseqs=True)
import numpy as np
np.random.seed(44)
print(word_ids[0])
print(word_ids[3])
print(train._factors[0].words[:10])
print(train._factors[0].all_words[:10])

print(id(train._factors[0].all_words))
print(id(dev._factors[0].all_words))
print(id(test._factors[0].all_words))