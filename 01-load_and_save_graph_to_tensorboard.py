import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
with tf.Session() as persisted_sess:
  print("load graph")
  with gfile.FastGFile("./output_graph_510.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    writer = tf.summary.FileWriter("./tf_summary", graph=persisted_sess.graph)
    # Print all operation names
    for op in persisted_sess.graph.get_operations():
      print(op)
    # next: do the following in bash:
    # tensorboard --logdir ./tf_summary/
