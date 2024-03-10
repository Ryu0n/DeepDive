import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from dimensionality_reduction import flatten_images, reduce_by_pca, reduce_by_tsne
from utils import get_images


def visualize_clustering_result(img_paths, result, limit=40, rescaled_pixels=400):
    n_rows = math.ceil(math.sqrt(limit))
    n_cols = math.ceil(limit / n_rows)
    for i, (img_path, cluster) in enumerate(list(zip(img_paths, result))[:limit]):
        plt.subplot(n_rows, n_cols, i+1)
        img = np.array(Image.open(img_path).resize((rescaled_pixels, rescaled_pixels)))
        plt.imshow(img)
        plt.title(str(cluster))
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    img_paths, images = get_images()

    # 1. Nothing to apply
    images = flatten_images(images)

    # 2. PCA
    # images = reduce_by_pca(images)

    # 3. t-SNE
    # images = reduce_by_tsne(images)

    # 1. K-Means clustering
    result = KMeans(n_clusters=2).fit_predict(images)

    visualize_clustering_result(img_paths, result)
