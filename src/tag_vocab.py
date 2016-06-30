## Counts tags in input files, writes count of all tags occurring more than thr times

## Output is <tag>\t<count>, one per line

import sys
import os
import json
from collections import Counter

def flatten(l):
    return [item for sublist in l for item in sublist]

def get_business_tags(j):
    '''
    Get attribute tags from BUSINESS json object
    :param j: json object
    :return: dict
    '''
    business = j['business_id']
    cats = ['_'.join(('cat.'+ct.lower()).split()) for ct in j['categories']]
    city = ['city.'+'_'.join(j['city'].split()).lower()]
    state = ['state.'+j['state'].lower()]
    nghs = ['_'.join(('ngh.'+n).lower().split()) for n in j['neighborhoods']]
    stars = ['stars.'+str(j['stars'])]
    onelevelatts = [k for k,v in j['attributes'].items() if type(v)!=dict]
    twolevelatts = [k for k,v in j['attributes'].items() if type(v)==dict]
    atts = ['_'.join((k+'.'+str(v)).lower().split()) for k,v in j['attributes'].items()
            if k in onelevelatts]
    atts += flatten([['_'.join('.'.join([a,k,str(v)]).lower().split()) for k,v in j['attributes'][a].items()] for a in twolevelatts])
    typ = ['_'.join(('typ.'+j['type'].lower()).split())]
    return business, cats + city + state + nghs + stars + atts + typ

tc = Counter()
dirname = sys.argv[1]
thr = int(sys.argv[2])

files = [os.path.join(dirname,f) for f in os.listdir(dirname) if 'business' in f]

l = []
i = 0
for f in files:
   try:
       with open(f, 'rU') as fin:
           for line in fin:
               sample = json.loads(line)
               ## Read tags
               atts = get_business_tags(sample)
               for t in atts[1]:
                   if type(t)==list:
                       print 'LIST ATTRIBUTE:'
                       print sample
                       print atts
                       print t
                   i += 1
                   if i % 1000000 == 0:
                       tc.update(l)
                       l = []
                       print >> sys.stderr,i,len(tc)
                   l.append(t)
   except:
       sys.stderr.write('Error processing file %s \n' % f)
       continue
tc.update(l)
print >> sys.stderr, 'Total types:', len(tc)
print >> sys.stderr, 'Total tokens:', i

for w,c in sorted([(w,c) for w,c in tc.iteritems() if c >= thr and w != ''],key=lambda x:-x[1]):
   print "\t".join([w.encode('utf8').strip(),str(c)])

