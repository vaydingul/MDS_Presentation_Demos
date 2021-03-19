from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def get_stress_for_different_dimensions(dm, max_dim=5):

    # Initialize stress vector
    stresses = np.empty((max_dim, ))

    for dim in range(max_dim):
        # For each dim, perform MDS analysis
        mds_ = MDS(n_components=dim+1, metric=True,
                   dissimilarity="precomputed")
        e = mds_.fit_transform(dm)
        # Fetch stresses
        stresses[dim] = mds_.stress_

    return stresses


def get_embedding(dm, dim):

    # Return embedding space
    return MDS(n_components=dim, metric=True, dissimilarity="precomputed").fit_transform(dm)


def visualize(data, dim=2):

    # Visualize the new embeddig

    fig = plt.figure()

    if dim == 1:

        # 1D plot is not useful
        for k in range(data.shape[0]):

            plt.scatter(data[k:k+1, 0], None)

    elif dim == 2:

        ax = plt.axes()

        for k in range(data.shape[0]):

            ax.scatter(data[k:k+1, 0], data[k:k+1, 1])

        # Line segments are only applicable for 2D
        segments = [[data[i, :], data[j, :]]
                    for i in range(data.shape[0]) for j in range(data.shape[0])]

        lc = LineCollection(segments,
                            zorder=0, cmap=plt.cm.Blues)
        lc.set_linewidths(np.full(len(segments), 0.5))
        ax.add_collection(lc)


    elif dim == 3:

        ax = fig.add_subplot(111, projection='3d')

        for k in range(data.shape[0]):

            ax.scatter(data[k:k+1, 0], data[k:k + 1, 1], data[k:k+1, 2])


    plt.title("Embeddings in {0} dimension".format(str(dim)))
    plt.tight_layout()
    plt.savefig("embedding_example_{0}.png".format(str(dim)))
    plt.close("all")
