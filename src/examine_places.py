import infer
import os
import numpy as np
import json
import random
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def read_businesses(bfile):
    '''
    Read business data and return a data frame with results.
    Columns for: name, city, state, each category
    :param bfile:
    :return:
    '''
    bs = {}
    with open(bfile,'rU') as fin:
        for line in fin:
            record = json.loads(line.strip())
            b_id = record['business_id']
            b_dict = {k: record[k] for k in ['city','name','state','stars']}
            b_cats = {'_'.join(('cat.'+c.lower()).split()) : True for c in record['categories']}
            b_dict.update(b_cats)
            bs[b_id] = b_dict

    bdf = pd.DataFrame.from_dict(bs, orient='index')
    return bdf

def get_bid(df, city, state, name):
    bid = df[(df['city']==city) & (df['state']==state) & (df['name']==name)].index[0] # just take first one
    return bid

def most_similar_place(b_id, vecs, df, N=10):
    sim_b_ids = [r[1] for r in vecs.most_similar(b_id, N=N+1)]
    return df.loc[sim_b_ids][['name','city','state','stars']][1:N] # don't return itself
    # return df.loc[sim_b_ids]

def vec_arithmetic(df, city, state, name, wordvecs, ctxtvecs, minus, plus, N=10):
    bid = get_bid(df, city, state, name)
    new_vec = wordvecs.word2vec(bid)
    if len(minus) > 0:
        for m in minus:
            new_vec -= ctxtvecs.word2vec(m)
    if len(plus) > 0:
        for p in plus:
            new_vec += ctxtvecs.word2vec(p)
    close_bids = [r[1] for r in wordvecs.similar_to_vec(new_vec, N=N)]
    return df.loc[close_bids][['name','city','state']]

def most_similar_categories(catname, vecs):
    return [r[1] for r in vecs.most_similar(catname, N=10000) if 'cat.' in r[1]][1:11]


business_vecfile = '../data/processed/yelp.t10.vecs.npy'
context_vecfile = '../data/processed/yelp.t10.context.vecs.npy'

bvecs = infer.Embeddings(business_vecfile)
cvecs = infer.Embeddings(context_vecfile)

business_file = '../data/yelp_academic_dataset_business.json'
bdf = read_businesses(business_file)

# How well do we capture similarities between star ratings?
stars = ['stars.%0.1f' % d for d in np.arange(1.0,5.1,0.5)]
aff = np.array([[cosine_similarity(cvecs.word2vec(u), cvecs.word2vec(v))[0][0] for v in stars] for u in stars])
fig, ax = plt.subplots()
heatmap = ax.pcolor(aff, cmap=plt.cm.Blues)
ax.set_xticks(np.arange(aff.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(aff.shape[1])+0.5, minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.set_xticklabels(stars, minor=False)
ax.set_yticklabels(stars, minor=False)
plt.show()


# Find similar places
most_similar_place(get_bid(bdf, 'Las Vegas','NV',"Tailwaggers"), bvecs, bdf)
most_similar_place(get_bid(bdf, 'Pittsburgh', 'PA',"McDonald's"), bvecs, bdf)
most_similar_place(get_bid(bdf, 'Las Vegas','NV',"Estiatorio Milos"), bvecs, bdf)

# Find similar words/features to business attributes
cvecs.most_similar('good_for_kids.false', N=11)[1:]
cvecs.most_similar('good_for_kids.true', N=11)[1:]
cvecs.most_similar('ambience.romantic.true', N=11)[1:]

# Find similar categories
most_similar_categories('cat.doctors', cvecs)
most_similar_categories('cat.automotive', cvecs)
most_similar_categories('cat.sporting_goods', cvecs)
most_similar_categories('cat.guns_&_ammo', cvecs)

# Vector arithmetic
kids_steakhouses = vec_arithmetic(bdf, 'Las Vegas', 'NV', 'CUT', bvecs, cvecs, ['good_for_kids.false'], ['good_for_kids.true'], N=50)
kids_steakhouses[kids_steakhouses['city']=='Las Vegas']
vec_arithmetic(bdf, 'Phoenix', 'AZ', "McDonald's", bvecs, cvecs, ['cat.burgers'], ['cat.italian'])

ital_chilis = vec_arithmetic(bdf, 'Las Vegas', 'NV', "Chili's", bvecs, cvecs, ['cat.tex-mex'], ['cat.italian'])
ital_chilis[ital_chilis['city']=='Las Vegas']

# Restrict it to a particular city
newdf = most_similar_place(get_bid(bdf, 'Edinburgh','EDH', "Crombie's Of Edinburgh"), bvecs, bdf, N=500)
newdf[newdf['city']=='Pittsburgh']

# Plot 20 random businesses from a few selected categories
try:
    os.makedirs('../images')
except:
    pass
cats = ['cat.restaurants','cat.automotive','cat.doctors','cat.real_estate', 'cat.home_services', 'cat.hardware_stores']
numsample = [5,3,3,3,3,3]
places = []
for c, n in zip(cats, numsample):
    indices = bdf.loc[bdf[c] == True].index
    places.extend(random.sample(indices, n))
rand_labels = list(bdf.loc[places]['name'])
bvecs.plot(places, rand_labels, '../images/bplot_20rand.pdf')

# Plot 20 random categories
random.seed(123)
randcats = random.sample([c for c in cvecs._vocab if 'cat.' in c], 20)
cvecs.plot(randcats, randcats, '../images/catplot_20rand.pdf')
__author__ = 'anne'
