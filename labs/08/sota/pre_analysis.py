import morpho_dataset

seen_tags = set()

for fname in ("czech-pdt-train.txt", "czech-pdt-dev.txt", "czech-pdt-test.txt"):
    with open(fname, "r", encoding="utf-8") as file:
        in_sentence = False
        for line in file:
            line = line.rstrip("\r\n")
            if line:
                columns = line.split("\t")
                seen_tags.add(columns[2])
    print(len(seen_tags))
print(">>>", len(seen_tags))


# train = morpho_dataset.MorphoDataset("czech-pdt-train.txt")
# mset = set(train.factors[train.TAGS].words)
# print(mset - seen_tags)