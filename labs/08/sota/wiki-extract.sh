IN_FILE=data/wiki-cs.txt
OUT_FILE=data/wiki-cs-tokenized.txt

/lnet/troja/projects/udpipe/bin/udpipe-latest-bin/bin-linux64/udpipe --tokenize --output=horizontal --immediate --outfile=$OUT_FILE /lnet/troja/projects/udpipe/models/udpipe-ud-2.0-170801/czech-cac-ud-2.0-170801.udpipe $IN_FILE
