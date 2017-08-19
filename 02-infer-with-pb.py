import tensorflow as tf
import os
import sys
from tensorflow.python.platform import gfile
import numpy as np
from scipy.misc import imread, imresize

class ImageFolderDataSource:
    def __init__(self, folder, batch_size, labels):
        if not os.path.exists(folder):
            raise Exception("Folder doesn't exist: {}".format(folder))
        if set(labels) != set(os.listdir(folder)):
            raise Exception("The labels are not consistent with folder structure.")
        self.labels = labels
        self.n_classes = len(self.labels)
        self.labels_index = np.arange(self.n_classes)
        self.index_map = dict(zip(self.labels, self.labels_index))
        self.label_map = dict(zip(self.labels_index, self.labels))
        self.folder = folder
        self.batch_size = batch_size
        self.label_pool = []
        self.file_pool = []
        for label in self.labels:
            label_one_hot = self.one_hot(self.index_map[label])
            label_folder = os.path.join(folder, label)
            label_files = list(map(
                lambda f: os.path.join(label_folder, f),
                os.listdir(label_folder)
            ))
            self.file_pool.extend(label_files)
            self.label_pool.extend(np.repeat([label_one_hot], len(label_files), axis=0))
        self.n_files = len(self.file_pool)
        self.label_pool = np.array(self.label_pool)
        self.file_pool = np.array(self.file_pool)
    def one_hot(self, index):
        res = np.repeat(0, self.n_classes).astype(np.float32)
        res[index] = 1.0
        return  res
    def rand_index(self):
        return np.random.choice(np.arange(self.n_files), self.batch_size)
    def get_batch(self):
        index = self.rand_index()
        batch_labels = self.label_pool[index]
        batch_files = self.file_pool[index]
        batch_data = np.array(list(
            map(
                lambda f: imread(f),
                batch_files
            )
        ))
        return batch_labels, batch_data, batch_files



with tf.Session() as persisted_sess:
    def run_inference_on_image():
        with open("latest_labels.txt") as fh:
            label_names = [x.strip() for x in fh.readlines()]
        batch_reader = ImageFolderDataSource("/home/kaiyin/PycharmProjects/demo-load-pb-tensorflow/images/validate", 5, label_names)
        labels, data, files = batch_reader.get_batch()
        print(files)
        data = np.divide(data, np.float32(255.0))
        answer = None
        # # Print all operators in the graph
        # for op in persisted_sess.graph.get_operations():
        #     print(op)
        # # Print all tensors produced by each operator in the graph
        # for op in persisted_sess.graph.get_operations():
        #     print(op.values())
        # tensor_names = [[v.name for v in op.values()] for op in persisted_sess.graph.get_operations()]
        # tensor_names = np.squeeze(tensor_names)
        # print(tensor_names)
        softmax_tensor = persisted_sess.graph.get_tensor_by_name('import/final_result:0')
        def predict(img):
            predictions = persisted_sess.run(softmax_tensor, {'import/input:0': img})
            predictions = np.squeeze(predictions)
            print("##################")
            top_k = predictions.argsort()[:][::-1]  # Getting top 3 predictions, reverse order
            for node_id in top_k:
                human_string = batch_reader.labels[node_id]
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            answer = labels[top_k[0]]
            return answer
        for index in range(len(labels)):
            img = data[index]
            label_index = np.nonzero(labels[index])[0][0]
            label_name = batch_reader.label_map[label_index]
            if img.shape != (224, 224, 3):
                img = imresize(img, (224, 224, 3))
            predict(np.expand_dims(img, axis=0))

    with gfile.FastGFile("latest.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def)
        run_inference_on_image()
