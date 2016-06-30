'''
extract_contexts.py

'''

from nltk.tokenize import word_tokenize
import sys
import os
import json
import optparse

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
    city = [j['city'].lower()]
    state = [j['state'].lower()]
    nghs = ['_'.join(('ngh.'+n).lower().split()) for n in j['neighborhoods']]
    stars = ['stars.'+str(j['stars'])]
    onelevelatts = [k for k,v in j['attributes'].items() if type(v)!=dict]
    twolevelatts = [k for k,v in j['attributes'].items() if type(v)==dict]
    atts = ['_'.join((k+'.'+str(v)).lower().split()) for k,v in j['attributes'].items()
            if k in onelevelatts]
    atts += flatten([['_'.join('.'.join([a,k,str(v)]).lower().split()) for k,v in j['attributes'][a].items()] for a in twolevelatts])
    return business, cats + city + state + nghs + stars + atts

def get_business_cats(j):
    '''
    Get just categories
    :param j:
    :return:
    '''
    business = j['business_id']
    cats = ['_'.join(('cat.'+ct.lower()).split()) for ct in j['categories']]
    return business, cats

def read_vocab(fh, threshold):
    v = {}
    for line in fh:
        line = line.strip().split()
        if len(line) != 2: continue
        if int(line[1]) >= threshold:
            v[line[0]] = int(line[1])
    return v

if __name__=="__main__":
    ## Get command line arguments
    optparser = optparse.OptionParser()
    optparser.add_option("-D", "--dirname", type='string', dest='dirname', default='../data', help='Path to directory containing yelp data')
    optparser.add_option("-W", "--wordvocabfile", type='string', dest='wordvocabfile', default='../data/processed/vocab_count', help='Path to vocab count file')
    optparser.add_option("-V", "--tagvocabfile", type='string', dest='tagvocabfile', default='../data/processed/tag_vocab_count')
    optparser.add_option("-T", "--thr", type='int', dest='THR', default=50, help='Threshold for considering tokens/contexts')
    (opts, _) = optparser.parse_args()

    sys.stderr.write('Reading word_vocab...')
    word_vocab = read_vocab(open(opts.wordvocabfile,'rU'), opts.THR)
    sys.stderr.write('Word Vocab Length %d\n' % len(word_vocab))
    sys.stderr.write('Reading tag vocab...')
    tag_vocab = read_vocab(open(opts.tagvocabfile,'rU'), opts.THR)
    sys.stderr.write('Tag Vocab Length %d\n' % len(tag_vocab))

    ## Read in business attributes
    sys.stderr.write('Reading business attributes...')
    busfile = opts.dirname+'/yelp_academic_dataset_business.json'
    businesses = {}
    with open(busfile,'rU') as fin:
        for line in fin:
            j = json.loads(line)
            bid, atts = get_business_tags(j)
            bid, cats = get_business_cats(j)
            businesses[bid] = [a for a in atts if a in tag_vocab]
    sys.stderr.write('Number of businesses %d\n' % len(businesses))

    ## Go through review and tips files
    filelist = [os.path.join(opts.dirname, f) for f in os.listdir(opts.dirname) if 'review' in f or 'tip' in f]
    i = 0
    for f in filelist:
        with open(f,'rU') as fin:
            for line in fin:
                sample = json.loads(line)
                bid = sample['business_id']
                text = sample['text']
                toks = [t for t in word_tokenize(text.lower()) if t in word_vocab]

                for t in toks:
                    print bid, t
                for a in businesses[bid]:  # repeat for every tip/review
                    print bid, a

                i += 1
                if i % 10000 == 0:
                    sys.stderr.write('.')



