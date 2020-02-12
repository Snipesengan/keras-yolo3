import argparse
import os

from yolo import YOLO


def _main():
    parser = argparse.ArgumentParser()
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
        'validation_path', type=str,
        help='path to validation data'
    )

    FLAGS = parser.parse_args()
    yolo = YOLO(**{k:v for k, v in FLAGS.__dict__.items() if v is not None})

    yolo.evaluate(os.path.abspath(FLAGS.validation_path))

if __name__ == '__main__':
    _main()