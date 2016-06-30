## from word2vecf

import heapq
from itertools import izip
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
import networkx as nx


def ugly_normalize(vecs):
    normalizers = np.sqrt((vecs * vecs).sum(axis=1))
    normalizers[normalizers == 0] = 1
    return (vecs.T / normalizers).T


class Embeddings:
    def __init__(self, vecsfile, vocabfile=None, normalize=True):
        if vocabfile is None: vocabfile = vecsfile.replace("npy", "vocab")
        self._vecs = np.load(vecsfile)
        self._vocab = file(vocabfile).read().split()
        if normalize:
            self._vecs = ugly_normalize(self._vecs)
        self._w2v = {w: i for i, w in enumerate(self._vocab)}

    @classmethod
    def load(cls, vecsfile, vocabfile=None):
        return Embeddings(vecsfile, vocabfile)

    def word2vec(self, w):
        return self._vecs[self._w2v[w]]

    def similar_to_vec(self, v, N=10):
        sims = self._vecs.dot(v)
        sims = heapq.nlargest(N, zip(sims, self._vocab, self._vecs))
        return sims

    def most_similar(self, word, N=10):
        w = self._vocab.index(word)
        sims = self._vecs.dot(self._vecs[w])
        sims = heapq.nlargest(N, zip(sims, self._vocab))
        return sims

    def analogy(self, pos1, neg1, pos2, N=10, mult=True):
        wvecs, vocab = self._vecs, self._vocab
        p1 = vocab.index(pos1)
        p2 = vocab.index(pos2)
        n1 = vocab.index(neg1)
        if mult:
            p1, p2, n1 = [(1 + wvecs.dot(wvecs[i])) / 2 for i in (p1, p2, n1)]
            if N == 1:
                return max(((v, w) for v, w in izip((p1 * p2 / n1), vocab) if
                            w not in [pos1, pos2, neg1]))
            return heapq.nlargest(N, ((v, w) for v, w in
                                      izip((p1 * p2 / n1), vocab) if
                                      w not in [pos1, pos2, neg1]))
        else:
            p1, p2, n1 = [(wvecs.dot(wvecs[i])) for i in (p1, p2, n1)]
            if N == 1:
                return max(((v, w) for v, w in izip((p1 + p2 - n1), vocab) if
                            w not in [pos1, pos2, neg1]))
            return heapq.nlargest(N, ((v, w) for v, w in
                                      izip((p1 + p2 - n1), vocab) if
                                      w not in [pos1, pos2, neg1]))

    def repel_labels(self, ax, x, y, labels, k=0.01):
        G = nx.DiGraph()
        data_nodes = []
        init_pos = {}
        for xi, yi, label in zip(x, y, labels):
            data_str = 'data_{0}'.format(label)
            G.add_node(data_str)
            G.add_node(label)
            G.add_edge(label, data_str)
            data_nodes.append(data_str)
            init_pos[data_str] = (xi, yi)
            init_pos[label] = (xi, yi)

        pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k)

        # undo spring_layout's rescaling
        pos_after = np.vstack([pos[d] for d in data_nodes])
        pos_before = np.vstack([init_pos[d] for d in data_nodes])
        scale, shift_x = np.polyfit(pos_after[:,0], pos_before[:,0], 1)
        scale, shift_y = np.polyfit(pos_after[:,1], pos_before[:,1], 1)
        shift = np.array([shift_x, shift_y])
        for key, val in pos.iteritems():
            pos[key] = (val*scale) + shift

        for label, data_str in G.edges():
            ax.annotate(label, fontsize=6,
                        xy=pos[data_str], xycoords='data',
                        xytext=pos[label], textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        shrinkA=0, shrinkB=0,
                                        connectionstyle="arc3",
                                        color='red'),)
        # expand limits
        all_pos = np.vstack(pos.values())
        x_span, y_span = np.ptp(all_pos, axis=0)
        mins = np.min(all_pos - x_span * 0.15, 0)
        maxs = np.max(all_pos + y_span * 0.15, 0)
        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[1], maxs[1]])

    def plot(self, wordlist, labels, filename, label=True):
        wordindices = [self._vocab.index(w) for w in wordlist]
        mat = self._vecs
        pca = PCA(n_components=2)
        pca.fit(mat)
        X = pca.transform(mat)
        xs = X[wordindices, 0]
        ys = X[wordindices, 1]

        f, ax = plt.subplots(figsize=(4, 4))
        f.subplots_adjust(right=1.0, top=1.0)
        plt.hold(True)
        colors = ['#009E73', '#B40431', '#0072B2']
        ax.scatter(xs, ys, marker='o', color=colors[0])

        if label:
            self.repel_labels(ax, xs, ys, labels, k=0.005)
            # for i, txt in enumerate(labels):
            #     ax.annotate(txt, (xs[i],ys[i]))

        ax.yaxis.grid(True, linestyle='-', which='major', color='#585858')
        ax.xaxis.grid(True, linestyle='-', which='major', color='#585858')
        ax.patch.set_facecolor('#F0EFEF')
        ax.set_title('Business Embeddings')
        ax.set_ylabel('Dimension 2', fontsize=8)
        ax.set_xlabel('Dimension 1', fontsize=8)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)

        pdf = PdfPages(filename)
        pdf.savefig(f, dpi=600, bbox_inches='tight', pad_inches=0.05)
        pdf.close()


if __name__ == '__main__':
    import sys

    e = Embeddings.load(sys.argv[1])

    print e.most_similar('azkaban')
    print e.analogy('king', 'man', 'woman')
