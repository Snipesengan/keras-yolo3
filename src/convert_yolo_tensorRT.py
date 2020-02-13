import argparse
import os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Net
from src.yolo import YOLO
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import tensorrt as trt

def _main():
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
            '--save', type=str,
            help='path to save frozen model'
            )

    FLAGS = parser.parse_args()

    save_pb_dir, model_fname = os.path.split(FLAGS.save)

    model = YOLO(**vars(FLAGS))
    frozen_graph = model.get_frozen_graph(save_pb_dir, model_fname)

    trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=output_names,
            max_batch_size=1,
            max_workspace_size_bytes=1 << 25,
            precision_mode='FP16',
            minimum_segment_size=50
            )



