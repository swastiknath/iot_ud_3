"""
AUTHOR : SWASTIK NATH.
INTEL (R) EDGE AI FOR IOT DEVELOPERS NANODEGREE
CURSOR CONTROLLER: INFERENCE MODULE.
"""

import logging as log
import os
import sys
import math
import numpy as np

from openvino.inference_engine import IECore


class GazeEstimation:
    """
    Model Inference Network
    """

    def __init__(self):
        """
         Initializing data fields.
        """
        self.ie = IECore()
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.inference_plugin = None
        self.inference_handler = None
        self.device = None

    def load_model(self, model, device, num_requests, cpu_extension=None):
        """
         Loading the necessary .xml, .bin files and adding cpu extensions
         while dealing with unsupported layers and customer layer
         implementations.

        :param model: location of the model's .xml file
        :param device: inference device to use
        :param num_requests: Request ID for the inference request
        :param cpu_extension: Location of the CPU extension to deal with
                              unsupported layers, mostly used with MKLDNN plugin.
        :return: input shape of the model in [n, c, h, w ] shape.
        """
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + '.bin'
        self.network = self.ie.read_network(model_xml, model_bin)

        if 'CPU' in device and cpu_extension:
            self.ie.add_extension(cpu_extension, 'CPU')

        if 'CPU' in device:
            supported_layers = self.ie.query_network(self.network, 'CPU')
            unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                log.error("One or More Unsupported Layers cannot be interpreted. Use MKLDNN Extension if not already "
                         "done")
                log.error(unsupported_layers)
                sys.exit(1)

        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        if num_requests == 0:
            self.inference_plugin = self.ie.load_network(self.network, device)
        else:
            self.inference_plugin = self.ie.load_network(network=self.network, device_name=device, num_requests=0)
        return self.get_input_shape()

    def get_input_shape(self):
        """
         Obtaining the Shape of the network's input.
        :return: Shape of the network in [n, c, h, w] format.
        """
        return self.network.inputs[self.input_blob].shape

    def wait(self, request_id):
        """
        To Deal with the [REQUEST BUSY] error.
        :return: (int) waiting semaphore
        """
        wait_for_inference = self.inference_plugin.requests[request_id].wait(-1)
        return wait_for_inference

    def get_output(self, request_id, prev_output=None):
        """
        Getting the output of the inference using the Asynchronous Inference
        Request.
        :param request_id: Request ID for the inference request
        :param prev_output:
        :return: result of the inference.
        """
        if prev_output:
            res = self.inference_handler.outputs[prev_output]
        else:
            res = self.inference_plugin.requests[request_id].outputs[self.output_blob]

        return res

    def get_performance_counts(self, request_id):
        """
         Get Layer wise Performance Stats.
        :param request_id:
        :return: Performance stats in json.
        """
        performance_count = self.inference_plugin.requests[request_id].get_perf_counts()
        return performance_count

    def multi_in_infer(self, request_id, le_img, re_img, hp_ang):
        inp_dict = {'left_eye_image': le_img, 'right_eye_image': re_img, 'head_pose_angles': hp_ang}
        multi_input_dict = {self.input_blob: inp_dict}
        self.inference_handler = self.inference_plugin.start_async(request_id=0, inputs=inp_dict)
        return self.inference_plugin

    def process_output(self, gaze_vector, angle_r_fc):

        gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
        gaze_x = gaze_vector[0]
        gaze_y = gaze_vector[1]
        gaze_z = gaze_vector[2]
        gaze_yaw = np.arctan(gaze_y / gaze_x)

        sin = np.sin(angle_r_fc * math.pi / 180.)
        cos = np.cos(angle_r_fc * math.pi / 180.)

        move_x = gaze_x * cos + gaze_y * sin
        move_y = - gaze_x * sin + gaze_y * cos

        gaze_yaw = gaze_yaw * np.pi / 180.0

        gaze_x_angle = np.arctan2(gaze_x, -gaze_z)
        gaze_y_angle = np.arctan2(-gaze_y, -gaze_z)

        return move_x, move_y, gaze_x_angle, gaze_y_angle, gaze_yaw
