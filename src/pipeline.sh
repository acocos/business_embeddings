#!/usr/bin/env bash

## Extract word, context pairs
python extract_contexts.py -T 10 > ../data/processed/yelp.t10.contexts

## Train vecs
BASENAME=../data/processed/yelp.t10
./word2vecf/count_and_filter -train $BASENAME.contexts -cvocab $BASENAME.cv -wvocab $BASENAME.wv -min-count 5
./word2vecf/word2vecf -train $BASENAME.contexts -cvocab $BASENAME.cv -wvocab $BASENAME.wv -output $BASENAME.vecs -dumpcv $BASENAME.t10.context.vecs -size 300 -negative 15 -threads 10
python ./word2vecf/scripts/vecs2nps.py $BASENAME.vecs $BASENAME.vecs
python ./word2vecf/scripts/vecs2nps.py $BASENAME.context.vecs $BASENAME.context.vecs
