## Counts words in input files, writes count of all words occurring more than thr times

## Output is <word>\t<count>, one per line

import sys
import os
from nltk.tokenize import word_tokenize
import json
from collections import Counter


wc = Counter()
dirname = sys.argv[1]
thr = int(sys.argv[2])

files = [os.path.join(dirname,f) for f in os.listdir(dirname) if 'tip' in f or 'review' in f]

l = []
i = 0
for f in files:
   try:
       with open(f, 'rU') as fin:
           for line in fin:
              sample = json.loads(line)
              text = sample['text']
              toks = word_tokenize(text.lower())
              for t in toks:
                 i += 1
                 if i % 1000000 == 0:
                    wc.update(l)
                    l = []
                    print >> sys.stderr,i,len(wc)
                 l.append(t)
   except:
       sys.stderr.write('Error processing file %s\n' % f)
       continue
wc.update(l)
print >> sys.stderr, 'Total types:', len(wc)
print >> sys.stderr, 'Total tokens:', i

for w,c in sorted([(w,c) for w,c in wc.iteritems() if c >= thr and w != ''],key=lambda x:-x[1]):
   print "\t".join([w.encode('utf8').strip(),str(c)])

