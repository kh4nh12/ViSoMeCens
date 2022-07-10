import pprint
import sys
from pathlib import Path
from apiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import pandas as pd
import numpy as np
from confluent_kafka import Producer
from kafka import KafkaProducer
import logging
import socket
import json
import time
import os
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing import sequence
from preprocessing import preprocessing

# Build service for calling the Youtube API:
## Arguments that need to passed to the build function
DEVELOPER_KEY = "AIzaSyDbt-xdAOjDhJghQGVMxfbsSiSyCFJr1Jw"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
# video_link = "https://www.youtube.com/watch?v=-1X6Ak94Acs"
video_link = sys.argv[1]

## creating Youtube Resource Object
youtube_service = build(YOUTUBE_API_SERVICE_NAME,
                        YOUTUBE_API_VERSION,
                        developerKey=DEVELOPER_KEY)


# Create a producer
def create_producer():
    try:
        producer = Producer({"bootstrap.servers": "localhost:9092",
                             "client.id": socket.gethostname(),
                             "enable.idempotence": True,  # EOS processing
                             "compression.type": "lz4",
                             "batch.size": 64000,
                             "linger.ms": 10,
                             "acks": "all",  # Wait for the leader and all ISR to send response back
                             "retries": 5,
                             "delivery.timeout.ms": 1000})  # Total time to make retries
    except Exception as e:
        print("Couldn't create the producer")
        producer = None
    return producer


### Function to get youtube video id.
# source:
# https://stackoverflow.com/questions/45579306/get-youtube-video-url-or-youtube-video-id-from-a-string-using-regex
def get_id(url):
    u_pars = urlparse(url)
    quer_v = parse_qs(u_pars.query).get('v')
    if quer_v:
        return quer_v[0]
    pth = u_pars.path.split('/')
    if pth:
        return pth[-1]


def get_comments(url, num_comment):
    response = youtube_service.commentThreads().list(
        part='snippet',
        maxResults=num_comment,
        textFormat='plainText',
        order='time',
        videoId=get_id(url)
    ).execute()

    results = response.get('items', [])

    # extract video comments
    authors = []
    authorUrls = []
    texts = []
    datetimes = []

    for item in results:
        authors.append(item['snippet']['topLevelComment']['snippet']['authorDisplayName'])
        authorUrls.append(item['snippet']['topLevelComment']['snippet']['authorChannelUrl'])
        texts.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
        datetimes.append(item['snippet']['topLevelComment']['snippet']['updatedAt'])

    dataFrame = pd.DataFrame({'datetime': datetimes, 'author': authors, 'authorUrl': authorUrls, 'comment': texts})

    return dataFrame


# producer = create_producer()
## Hate speech detection
# load DNN model
model_path = os.path.join(os.getcwd(), 'model/Text_CNN_model_PhoW2V.h5')
model = tf.keras.models.load_model(model_path)
tknz_path = os.path.join(os.getcwd(), 'model/tokenizer.pickle')
with open(tknz_path, "rb") as f:
    tokenizer = pickle.load(f)
producer = KafkaProducer(bootstrap_servers='localhost:9092')
response = youtube_service.commentThreads().list(
    part='snippet',
    textFormat='plainText',
    videoId=get_id(video_link)
).execute()

# response = youtube_service.liveChatMessages().list(
#       part='snippet',
#       maxResults=100,
#       liveChatId=get_id(video_link)
#   ).execute()

# extract video comments
try:
    while response:
        results = response.get('items', [])
        for item in results:
            author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            authorurl = item['snippet']['topLevelComment']['snippet']['authorChannelUrl']
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            datetime = item['snippet']['topLevelComment']['snippet']['updatedAt']

            # dnn
            processed_comment = preprocessing(comment)
            seq_comment = tokenizer.texts_to_sequences([processed_comment])
            ds_comment = sequence.pad_sequences(seq_comment, maxlen=80)
            pred = model.predict(ds_comment)
            hsd_dt = pred.argmax(-1)

            record = {"author": author, "datetime": datetime, "raw_comment": comment,
                      "clean_comment": processed_comment, "label": int(hsd_dt[0])}
            record = json.dumps(record).encode("utf-8")
            print('produce message')
            print(record)

            #         producer.produce(topic="hsd",value=record)
            producer.send(topic='detected', value=record)
        if 'nextPageToken' in response:
            response = youtube_service.commentThreads().list(
                part='snippet',
                textFormat='plainText',
                videoId=get_id(video_link),
                pageToken=response["nextPageToken"]
            ).execute()
        else:
            break
except KeyboardInterrupt:
    print('Stop flush!')
    pass
