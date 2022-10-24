import json
import os
import requests
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
    df = pd.read_excel('niz_prod_public_SPAM.xlsx')
    uids = df['MONITORING_UID']
    return uids


def search_documents_by_uid(collection):
    uids = uids_from_excel()
    for uid in uids:
        yield collection.find_one({"_id": ObjectId(uid), "channelKeyname": "instagram"})


def download_images(collection, limit=10000):
    docs = collection.find({"preSpamResult": 1, "channelKeyname": "instagram"})
    spam_image_path = 'spam_images/'
    if not os.path.exists(spam_image_path):
        os.mkdir(spam_image_path)
    for doc in tqdm(docs):
        print(doc)
        if len(glob(f'{spam_image_path}*')) > limit:
            break
        doc_id = doc.get('_id')
        thumbnails = doc.get('detailData').get('thumbnails')
        for i, thumbnail in enumerate(thumbnails):
            download_file = requests.get(thumbnail)
            if download_file.status_code == 200:
                with open(f'{spam_image_path}{doc_id}_{i}.jpg', 'wb') as photo:
                    photo.write(download_file.content)


def download_images_from_excel(collection, limit=10000):
    uids = uids_from_excel()
    spam_image_path = 'spam_images/'
    if not os.path.exists(spam_image_path):
        os.mkdir(spam_image_path)
    for uid in tqdm(uids):
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
    download_images_from_excel(collection)
