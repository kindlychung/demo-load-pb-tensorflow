import tensorflow as tf
import sys
from tensorflow.python.platform import gfile
import numpy as np
from scipy.misc import imread
import glob


with open("./labels_510.txt") as f:
    lines = list(f.readlines())
    labels = [str(w).replace("\n", "") for w in lines]

NCLASS = len(labels)
NCHANNEL = 3
WIDTH = 224
HEIGHT = 224

def getImageBatch(filenames, batch_size, capacity, min_after_dequeue):
    filenameQ = tf.train.string_input_producer(filenames, num_epochs=None)
    recordReader = tf.TFRecordReader()
    key, fullExample = recordReader.read(filenameQ)
    features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })
    label = features['image/class/label']
    image_buffer = features['image/encoded']
    with tf.name_scope('decode_jpeg', [image_buffer], None):
        image = tf.image.decode_jpeg(image_buffer, channels=NCHANNEL)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(1 - tf.image.rgb_to_grayscale(image), [WIDTH * HEIGHT * NCHANNEL])
    label = tf.stack(tf.one_hot(label - 1, NCLASS))
    imageBatch, labelBatch = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    print(imageBatch.shape)
    print(labelBatch.shape)
    return imageBatch, labelBatch



with gfile.FastGFile("./output_graph_510.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.graph.as_default()
        tf.import_graph_def(graph_def)
        tf.global_variables_initializer().run()
        image_tensor, label_batch = getImageBatch(glob.glob("./images/tf_records/validation*"), 1, 10, 2)
        image_tensor = tf.reshape(image_tensor, (1, WIDTH, HEIGHT, NCHANNEL))
        image_data = sess.run(image_tensor)
        # print(image_data.shape)
        # softmax_tensor = sess.graph.get_tensor_by_name('import/final_result:0')
        # predictions = sess.run(softmax_tensor, {'import/input:0': image_data})
        # predictions = np.squeeze(predictions)
        # print(predictions)
        coord.request_stop()
        coord.join(threads)

    #     top_k = predictions.argsort()[:][::-1]  # Getting top 3 predictions, reverse order
    #     for node_id in top_k:
    #         human_string = labels[node_id]
    #         score = predictions[node_id]
    #         print('%s (score = %.5f)' % (human_string, score))
    #     answer = labels[top_k[0]]
    #     return answer

