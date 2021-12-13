import os
import re
import pandas as pd

from patterns.singleton import SingletonInstance

PATH_CURRENT = os.path.abspath(os.curdir)
PATH_PROJECT = os.path.join(*PATH_CURRENT.split('/')[:-1])
PATH_DATASET = os.path.join(PATH_PROJECT, 'dataset')
DATA_TOP_100_WINE = 'wine_spector.xlsx'
DATA_WINE = 'vivino.xlsx'
DATA_WINE_IMAGE = 'vivino_img_link.xlsx'
DATA_REVIEW = 'wine1205_Final.csv'


def _full_path(root, file_name):
    return '/'+os.path.join(root, file_name)


class WineDatasetLoader(SingletonInstance):
    def __init__(self):
        self._dataset: pd.DataFrame = pd.read_excel(_full_path(PATH_DATASET, DATA_WINE))
        self.__preprocessing()

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset

    def __preprocessing(self):
        self._dataset.dropna(axis=0, inplace=True)


class WineImageLoader(SingletonInstance):
    def __init__(self):
        self._dataset: pd.DataFrame = pd.read_excel(_full_path(PATH_DATASET, DATA_WINE_IMAGE))

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset


class ReviewDatasetLoader(SingletonInstance):
    def __init__(self):
        self._dataset: pd.DataFrame = pd.read_csv(_full_path(PATH_DATASET, DATA_REVIEW))
        self.__preprocessing()

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset

    def __preprocessing(self):
        self._dataset = self._dataset[['와인명', '평점', '사용자']]
        self._dataset.rename({'와인명': 'WineName', '평점': 'Score', '사용자': 'User'}, axis=1, inplace=True)
        p = re.compile("[(].*[)]")

        def replace(u: str):
            s = p.search(u)
            if s:
                u = u.replace(s.group(), '')
            u = u.strip()
            return u

        self._dataset['User'] = self._dataset.User.map(replace)
