import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.cluster import KMeans

from patterns.singleton import SingletonInstance
from util.data_loader import WineDatasetLoader, WineImageLoader


class WineClusterer(SingletonInstance):
    def __init__(self):
        self._k_means = KMeans(n_clusters=4)
        self._fit_k_means_instance()

    def _determine_n_cluster(self):
        pass

    def _fit_k_means_instance(self):
        columns = ['Light', 'Smooth', 'Dry', 'Soft']
        X = WineDatasetLoader.instance().dataset[columns]
        wine_factors = X.values
        self._k_means.fit(wine_factors)
        WineDatasetLoader.instance().dataset['cluster'] = self._k_means.labels_

    def predict_cluster(self, light, smooth, dry, soft):
        return self._k_means.predict(np.array([[light, smooth, dry, soft]]))


class WineRecommender:
    @staticmethod
    def recommend(light, smooth, dry, soft, top=5, threshold=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This method is performed by Content-based filtering (K-Means). You can get wines data and image link.
        :param light:
        :param smooth:
        :param dry:
        :param soft:
        :param top: Number of top wines.
        :param threshold: Cut-line of rating score.
        :return: dataframe of wines and image links
        """
        cluster = WineClusterer.instance().predict_cluster(light, smooth, dry, soft)[0]
        dataset = WineDatasetLoader.instance().dataset
        dataset = dataset[dataset.cluster == cluster]
        dataset = dataset.sort_values(by=['ScoreCount', 'AvgScore'], ascending=False)

        if isinstance(threshold, (float, int)):
            dataset = dataset[dataset.AvgScore >= threshold]

        import pandas as pd
        pd.set_option('display.max_columns', None)

        df_top_wine = dataset.iloc[:top]
        df_image_lnk = WineImageLoader.instance().dataset
        df_image_lnk = df_image_lnk[df_image_lnk.WineName.isin(df_top_wine.WineName)][['WineName', 'ImageLink']]

        return df_top_wine, df_image_lnk
