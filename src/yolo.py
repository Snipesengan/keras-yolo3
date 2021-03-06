# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import sys
from timeit import default_timer as timer

import numpy as np
import cv2

from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model

from src.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from src.yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model


class YOLO(object):
    _defaults = {
            "model_path": 'model_data/AR10-ep074-loss16.139-val_loss16.430.h5',
            "anchors_path": 'model_data/yolo_anchors.txt',
            "classes_path": 'model_data/AR10_classes.txt',
            "score": 0.3,
            "iou": 0.45,
            "model_image_size": (416, 416),
            "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.yolo_model = None
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        if os.path.expanduser(self.model_path).endswith('h5'):
            self.boxes, self.scores, self.classes = self.generate()
        elif os.path.expanduser(self.model_path).endswith('pb'):
            self.boxes, self.scores, self.classes = self.load_frozen_model()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def save_frozen_model(self, save_pb_dir='.', save_pb_name='frozen_model.pb',
                          save_pb_as_text=False):

        assert save_pb_name.endswith('pb'), 'Name must have .pb extension'

        session = self.sess
        graph = session.graph
        output = ['boxes', 'scores', 'classes']
        print(f'Model output {output}')
        with graph.as_default():
            print('Freezing session...')
            graphdef_frozen = tf.graph_util.convert_variables_to_constants(session,
                                                                           session.graph_def,
                                                                           output)
            
            print('Saving graph...')
            graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name,
                                 as_text=save_pb_as_text)
            print(f'Graph saved to: {os.path.join(save_pb_dir, save_pb_name)}')

    def load_frozen_model(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(
            '.pb'), 'Frozen tensorflow model must be a .pb file.'

        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
                map(lambda x: (
                        int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(
                self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        tf.graph_util.import_graph_def(graph_def)
        self.input_name = self.sess.graph_def.node[0].name + ':0'
        self.input_image_shape = tf.get_default_graph().get_tensor_by_name(
                'import/image_shape:0')

        boxes_ = tf.get_default_graph().get_tensor_by_name("import/boxes:0")
        scores_ = tf.get_default_graph().get_tensor_by_name("import/scores:0")
        classes_ = tf.get_default_graph().get_tensor_by_name("import/classes:0")

        return boxes_, scores_, classes_

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(
            '.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
            print('Using YOLOv3')
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)),
                                             num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)),
                                                  num_anchors // 3, num_classes)
            self.yolo_model.load_weights(
                self.model_path)  # make sure model, anchors and classes match

            print('Using tiny YOLOv3')
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (
                               num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
                map(lambda x: (
                int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(
            self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_name = self.yolo_model.input
        self.input_image_shape = K.placeholder(shape=(2,), name='image_shape')
        print(self.input_image_shape)
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model,
                                              gpus=self.gpu_num)

        boxes, scores, classes = yolo_eval(self.yolo_model.output,
                                           self.anchors,
                                           len(self.class_names),
                                           self.input_image_shape,
                                           score_threshold=self.score,
                                           iou_threshold=self.iou)

        return boxes, scores, classes

    def detect_image(self, image):

        if self.model_image_size != (None, None):
            assert self.model_image_size[
                       0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[
                       1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(
                reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                        self.input_name: image_data,
                        self.input_image_shape: [image.size[1], image.size[0]],
                        K.learning_phase(): 0
                })

        return out_boxes, out_scores, out_classes

    def annotate_image(self, image, out_boxes, out_scores, out_classes):
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(
                                      3e-2 * image.size[1] + 0.5).astype(
                                      'int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            #label = '{} {:.2f}'.format(predicted_class, score)
            label = f'{predicted_class}'
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
            draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        return image

    def close_session(self):
        self.sess.close()

    def evaluate(self, validation_path, iou_thresh=0.5):

        matches = {}
        with open(os.path.abspath(validation_path), 'r') as fp:
            lines = fp.readlines()

        for line in [line.split(' ') for line in lines]:
            source, annotations = line[0], line[1:]
            n_objects = len(annotations)  # number of objects in the scene
            image = Image.open(source)
            out_boxes, out_scores, out_classes = self.detect_image(image)

            for annotation in annotations:
                data = annotation.split(',')
                bb1 = tuple(map(int, data[:4]))
                class_ = int(data[-1])

                for i in range(out_boxes.shape[0]):
                    if class_ == out_classes[i]:
                        top, left, bottom, right = out_boxes[i,:]
                        top = max(0, np.floor(top + 0.5).astype('int32'))
                        left = max(0, np.floor(left + 0.5).astype('int32'))
                        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                        bb2 = (left, top, right, bottom)
                        iou = _compute_iou(bb1, bb2)

                        if iou > iou_thresh: # True Positive
                            if class_ in matches:
                                matches[class_].append(iou)
                            else:
                                matches[class_] = [iou]
                        else: # False Positive
                            pass # Every image

            #  calculate the precision for each class
            for k, v in matches.items():
                tp = len(v)  # true positive
                fp = max(0, out_boxes.shape[0] - tp)  # false positive
                precision = tp/(tp + fp)  # precision


def _compute_iou(groundtruth_box, detection_box):
    g_xmin, g_ymin, g_xmax, g_ymax = groundtruth_box
    d_xmin, d_ymin, d_xmax, d_ymax = detection_box

    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)


def detect_video(yolo, video_path, output_path=""):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC),
              type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.annotate_image(image, *yolo.detect_image(image))
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        print(f'FPS: {fps}')
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    yolo.close_session()
