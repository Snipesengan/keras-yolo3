import argparse
import os
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

    trt_graph = model.generate_trt_inference_graph(save_pb_dir, model_fname)


if __name__ == '__main__':
    _main()
