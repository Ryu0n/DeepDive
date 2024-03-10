import json
import os
import requests
import random
import pandas as pd
from glob import glob
from tqdm import tqdm
from bson.objectid import ObjectId
from pymongo import MongoClient


def retrieve_documents():
    with open('secret_mongo.json', 'r') as f:
        json_val = json.loads(''.join(f.readlines()))
    uri = json_val.get('uri')
    client = MongoClient(uri)
    db = client.get_database('niz-prod')
    collection = db.get_collection('MonitoringData')
    return collection


def uids_from_excel():
    """
    스팸처리 결과 엑셀로부터 mongoDB 문서 uid 파싱
    :return:
    """
    df = pd.read_excel('niz_prod_public_SPAM.xlsx')
    uids = df['MONITORING_UID']
    return uids


def download_images_from_mongo(collection, limit=5000):
    """
    mongoDB로부터 스팸처리된 문서의 썸네일 다운로드 (limit 갯수만큼)
    :param collection:
    :param limit:
    :return:
    """
    query = {"preSpamResult": 1,
             "detailData.thumbnails.1": {"$exists": True}}
    docs = collection.find(query).sort([["_id", 1]])
    spam_image_path = 'mongo_images/'
    if not os.path.exists(spam_image_path):
        os.mkdir(spam_image_path)
    docs = tqdm(docs, leave=True, desc='Download images from mongoDB')
    for doc in docs:
        if random.random() > 0.5:
            continue
        if len(glob(f'{spam_image_path}*')) > limit:
            break
        doc_id = doc.get('_id')
        thumbnails = doc.get('detailData').get('thumbnails')
        for i, thumbnail in enumerate(thumbnails):
            download_file = requests.get(thumbnail)
            if download_file.status_code == 200:
                with open(f'{spam_image_path}{doc_id}_{i}.jpg', 'wb') as photo:
                    photo.write(download_file.content)
                    docs.set_postfix(downloaded=f'[{len(glob(spam_image_path + "*"))} / {limit}]')


def download_images_from_excel(collection, limit=10000):
    """
    스팸 결과 엑셀로부터 문서내의 썸네일 다운로드 (limit 갯수만큼)
    :param collection:
    :param limit:
    :return:
    """
    uids = uids_from_excel()
    spam_image_path = 'mongo_excel_images/'
    if not os.path.exists(spam_image_path):
        os.mkdir(spam_image_path)
    for uid in tqdm(uids, leave=True, desc='Download images from mongoDB & excel'):
        doc = collection.find_one({"_id": ObjectId(uid)})
        if len(glob(f'{spam_image_path}*')) > limit:
            break
        doc_id = doc.get('_id')
        thumbnails = doc.get('detailData').get('thumbnails')
        for i, thumbnail in enumerate(thumbnails):
            download_file = requests.get(thumbnail)
            if download_file.status_code == 200:
                with open(f'{spam_image_path}{doc_id}_{i}.jpg', 'wb') as photo:
                    photo.write(download_file.content)


if __name__ == "__main__":
    collection = retrieve_documents()
    # download_images_from_excel(collection)
    download_images_from_mongo(collection)
