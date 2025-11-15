import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_embeddings(embeddings, labels, out_path=None, perplexity=30, random_state=42):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    emb2 = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(emb2[idx, 0], emb2[idx, 1], label=str(lab), alpha=0.7)
    plt.legend()
    plt.title('t-SNE of embeddings')
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
    else:
        plt.show()
