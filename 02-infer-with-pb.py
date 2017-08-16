import tensorflow as tf
import sys
from tensorflow.python.platform import gfile
import numpy as np
from scipy.misc import imread

with open("./labels_510.txt") as f:
    lines = list(f.readlines())
    labels = [str(w).replace("\n", "") for w in lines]

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

with tf.Session() as persisted_sess:

    def run_inference_on_image(imagePath):
        answer = None
        if not tf.gfile.Exists(imagePath):
            tf.logging.fatal('File does not exist %s', imagePath)
            return answer
        image_data = np.expand_dims(imread(imagePath), axis=0)
        # # Print all operators in the graph
        # for op in persisted_sess.graph.get_operations():
        #     print(op)
        # # Print all tensors produced by each operator in the graph
        # for op in persisted_sess.graph.get_operations():
        #     print(op.values())
        tensor_names = [[v.name for v in op.values()] for op in persisted_sess.graph.get_operations()]
        tensor_names = np.squeeze(tensor_names)
        print(tensor_names)
        softmax_tensor = persisted_sess.graph.get_tensor_by_name('import/final_result:0')
        predictions = persisted_sess.run(softmax_tensor, {'import/input:0': image_data})
        predictions = np.squeeze(predictions)
        top_k = predictions.argsort()[:][::-1]  # Getting top 3 predictions, reverse order
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))
        answer = labels[top_k[0]]
        return answer

    with gfile.FastGFile("./output_graph_510.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def)
        img_path = "./images/23/IMG_1987.jpg"
        run_inference_on_image(img_path)
