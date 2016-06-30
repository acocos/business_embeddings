## Yelp Dataset Challenge: Business Embeddings

This repo contains code used to generate business embeddings for the Yelp Academic Dataset as detailed in [this blog](http://seas.upenn.edu/~acocos/yelp/place_embeddings.html).

### Contents

| file/directory | description | 
| --- | --- |
| `src/pipeline.sh` | This script demonstrates how to extract business/context pairs from the Yelp data and use them to train word embeddings using `word2vecf`. You can run this pipeline (after downloading the Yelp data and installing `word2vecf`, see below). Or you can just download the resulting vectors [here](http://seas.upenn.edu/~acocos/yelp/yelp.t10.zip) and get started. |
| `src/extract_contexts.py` | Generates business/context pairs from the Yelp data |
| `src/infer.py` | Script from the original `word2vecf` code, useful for loading and manipulating the resulting vectors |
| `src/examine_places.py` | Script used to produce results given in blog post |
| `data/` | Download the [Yelp data](https://www.yelp.com/dataset_challenge) and extract it to this directory |
| `data/processed` | If you run `pipeline.sh`, the resulting vectors will end up here. Or you can download them and put them there on your own. |

### Getting started

1. The code in this repo depends on the `word2vecf` adaptation of the popular `word2vec` software, allowing the use of arbitrary contexts to train vectors. It was developed by researchers at Bar-Ilan University and is available [here](https://bitbucket.org/yoavgo/word2vecf). You'll need to download and install before running the pipeline to train the vectors on your own.
2. If you want to train your own vectors, you'll also need to download the [Yelp data](https://www.yelp.com/dataset_challenge) and extract it to the `./data` directory.
3. Once you have done those two things, you will be able to run `src/pipeline.sh` to generate your own business embeddings.