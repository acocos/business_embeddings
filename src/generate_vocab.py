'''
generate_vocab.py

Generate tag and word vocab files
'''

import os, sys
import json
from collections import Counter
from nltk.tokenize import word_tokenize

from extract_contexts import get_business_tags

businessfile = '../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'
reviewfile = '../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'

# get tag vocab
c = Counter()
with open(businessfile, 'rU') as fin:
  for line in fin:
    j = json.loads(line)
    bid, atts = get_business_tags(j)
    c.update(atts)

with open('../data/processed/tag_vocab_count','w') as fout:
  for t,cnt in c.most_common():
    print >> fout, t.encode('utf8'), str(cnt)


# get word vocab
w = Counter()
with open(reviewfile, 'rU') as fin: 
  for line in fin:
    rev = json.loads(line)
    text = rev['text']
    toks = [t for t in word_tokenize(text.lower())]
    w.update(toks)

with open('../data/processed/vocab_count', 'w') as fout:
  for w, cnt in w.most_common():
    print >> fout, w.encode('utf8'), str(cnt)