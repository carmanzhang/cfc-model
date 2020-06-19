"""receive raw text and pre-precessing, then send them to tensorflow serving and gets result
"""

import json

import nltk
import numpy as np
import requests
from flask import Flask, request, jsonify
# from flask_cors import CORS, cross_origin
from keras.preprocessing import sequence

# import tensorflow as tf
# from grpc.beta import implementations
# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2

app = Flask(__name__)

# CORS(app, supports_credentials=True)

dictionary_path = 'dict.json'
word2idx = json.loads(open(dictionary_path, 'r').readline())
prediction_endpoint = 'http://localhost:8501/v1/models/citation_function_prediction:predict'

max_document_length = 180
class_label = ['Background', 'Future', 'Uses', 'Extends', 'CompareOrContrast', 'Motivation']
host = 'localhost'
port = 38081


def preprocessing(raw_strings, word2idx):
    X = []
    for raw_string in raw_strings:
        raw_string = raw_string.lower()
        words = nltk.word_tokenize(raw_string)
        X.append([word2idx[word] for word in words if word in word2idx])
    X = sequence.pad_sequences(X, maxlen=max_document_length)
    return X


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def rest_request(x):
    param = {"instances": x}
    param = json.dumps(param, cls=NumpyEncoder)
    res = requests.post(prediction_endpoint, data=param)
    status_code = res.status_code
    pred_list = None
    try:
        pred_list = json.loads(res.text)
        print(status_code, pred_list)
    except Exception as e:
        print(e)
    return status_code, pred_list


# def grpc_request(x):
#     # Send request
#     # x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
#     host, port = '10.0.0.1', 8500
#     print(host, port)
#     channel = implementations.insecure_channel(host, int(port))
#     stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
#
#     request = predict_pb2.PredictRequest()
#     request.model_spec.name = "text_classified_model"
#     request.model_spec.signature_name = 'textclassified'
#     # dropout_keep_prob = np.float(1.0)
#     request.inputs['inputX'].CopyFrom(
#         tf.contrib.util.make_tensor_proto(x, shape=[350, ], dtype=np.int32))
#
#     # request.inputs['input_dropout_keep_prob'].CopyFrom(
#     #     tf.contrib.util.make_tensor_proto(dropout_keep_prob, shape=[1],dtype=np.float))
#
#     res = stub.Predict(request, 10.0)  # 10 secs timeout
#     print(res.status_code, res.content)


def arg_max(a):
    return max(range(len(a)), key=lambda x: a[x])


@app.route('/predict', methods=['post', 'get'])
# @cross_origin()
def login():
    instances = json.loads(request.values.get('instances'))
    x = preprocessing(instances, word2idx)
    status_code, pred_list = rest_request(x)
    pred_labels = [class_label[np.argmax(item)] for item in pred_list['predictions']]

    pred_list = [dict(zip(class_label, item)) for item in pred_list['predictions']]
    res = {"code": status_code, 'predictions': pred_list, 'prediction_class_label': pred_labels}
    # return json.dumps(res, ensure_ascii=False)
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
