import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def flatten_images(images):
    return np.array([image.flatten() for image in images])


def reduce_by_pca(images, n_components=2):
    images = flatten_images(images)
    pca = PCA(n_components=n_components)
    princial_components = pca.fit_transform(images)
    return princial_components


def reduce_by_tsne(images, n_components=2):
    images = flatten_images(images)
    tsne = TSNE(n_components=n_components)
    return tsne.fit_transform(images)
