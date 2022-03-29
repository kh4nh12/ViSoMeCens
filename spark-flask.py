from flask import stream_with_context, Flask, render_template, request, redirect, url_for, Response

import json
import os
import time
import pandas as pd
import numpy as np
from kafka import KafkaConsumer
import socket
import logging
import threading
import sys

consumer = KafkaConsumer('anomalies', bootstrap_servers=['localhost:9092'])
app = Flask(__name__)
url = sys.argv[1]
time_list = []

@app.route('/home')
def home():
    return render_template('firstPage.html',data = url)

@app.route('/table-data', methods=['GET','POST'])
def table_data():
    return Response(get_stream_data(),mimetype="text/event-stream")

def get_stream_data():
    try:
        for msg in consumer:
            print('received')
            record = json.loads(msg.value.decode('utf-8'))
            if 'timestamp' in record:
            	record['timestamp'] = record['timestamp'][:19]
            print(record)
            # if len(record.keys()) < 3:
            #     df.loc[df['sentiment'] == record['sentiment'],'count'] = record['count']
            #     print(df)
            #     continue
            if len(record.keys()) > 3:
            	time_list.append(time.time())
            yield (f"data:{json.dumps(record)}\n\n")
    except KeyboardInterrupt:
        df = pd.DataFrame({"time":time_list})
        df.to_csv('HSD_time1.csv')
        consumer.close()
        print('Saved!')
    finally:
        df = pd.DataFrame({"time":time_list})
        df.to_csv('HSD_time2.csv')
        consumer.close()
        print('consumer closed')
            
if __name__ == "__main__":
    app.run(debug=True)
    df = pd.DataFrame({"time":time_list})
    df.to_csv('HSD_time3.csv')
    consumer.close()
    print('Saved!')
