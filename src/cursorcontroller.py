"""
AUTHOR: SWASTIK NATH
INTEL(R) DISTRIBUTION OF OPENVINO TOOLKIT
CURSOR CONTROLLER : ESTIMATORS.
LAST EDITED : 07/01/2020 REVISION : 0.150.24
"""
import logging as log
import math
import os
import sys
import time
from argparse import ArgumentParser

import cv2
import imageio
import numpy as np
from openvino.inference_engine import IECore

from src.cursormover import MouseController
from src.face_detection import FaceDetection
from src.facial_landmark_detection import FacialLandmarkDetection
from src.gaze_estimation import GazeEstimation
from src.head_pose_detection import HeadPoseEstimation


def args_parser():
    """
    This function will be responsible for parsing the arguments in the command prompt.
    :return: ArgumentParser object
    """
    parser = ArgumentParser(
        description="Cursor Controller.\nPowered by Intel (R) Distribution of OpenVino Toolkit 2020.3."
                    "\nDeveloped by Swastik Nath.\n Support Email : swastiknath@outlook.com")

    parser.add_argument("-f", "--face_detection_model", required=True,
                        type=str, help="PATH TO .XML FILE OF FACE DETECTION MODEL")
    parser.add_argument("-p", "--head_pose_model", required=True,
                        type=str, help="PATH TO .XML FILE OF HEAD POSE DETECTION MODEL")
    parser.add_argument("-l", "--facial_landmark_model", required=True,
                        type=str, help="PATH TO .XML FILE OF FACIAL DETECTION MODEL")
    parser.add_argument("-g", "--gaze_estimator_model", required=True,
                        type=str, help="PATH TO .XML FILE OF GAZE ESTIMATOR MODEL")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="PATH TO THE VIDEO/IMAGE FILE, USE ['CAM'] TO USE YOUR WEBCAM AS STREAM INPUT.")
    parser.add_argument("-e", "--cpu_extension", type=str, default=None,
                        help="PATH TO MATH KERNEL LIBRARY FILE FOR CUSTOM LAYER IMPLEMENTATION")
    parser.add_argument("-d", "--device", type=str, default='CPU',
                        help="INFERENCE TARGET, SELECT FROM [CPU, GPU, MYRIAD, FPGA]")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="PATH TO THE OUTPUT DIRECTORY FOR INFERENCE STATS.")
    parser.add_argument("-v", "--intermediate_visuals", type=bool, default=False,
                        help="FLAG TO VISUALISE THE INTERMEDIATE RESULTS.")
    parser.add_argument("-m", "--print_layer_metrics", type=bool, default=False,
                        help='FLAG TO PRINT LAYER WISE METRICS')
    parser.add_argument("-t", "--threshold", default=0.5, type=float,
                        help='CONFIDENCE THRESHOLD FOR INFERENCE.')
    parser.add_argument("-q", "--mirror_mode", default=True, type=bool,
                        help="Flag to Mitigate the Mirror Mode to Normal Mode")
    parser.add_argument("-x", "--mouse_precision", type=str, default='medium',
                        help="Mouse Pointer Movement Precision, must be one of [high, medium, or low]")
    parser.add_argument("-s", "--speed", type=str, default='fast',
                        help="Speed of the Mouse Pointer Movement, must be one of [fast, medium, slow")

    return parser


def face_detection(res, args, initial_m):
    """
    This function will be dealing with detected faces from the Face Detection Model
    :param res: Results from the Face Detection Model
    :param args: ArgumentParser function
    :param initial_m: Initial height and weight of the video frame.
    :return: A list of Face Co-ordinates.
    """
    faces = []
    for obj in res[0][0]:
        if obj[2] > args.threshold:
            if obj[3] < 0:
                obj[3] = -obj[3]
            if obj[4] < 0:
                obj[4] = -obj[4]

            xmin = int(obj[3] * initial_m[0])
            ymin = int(obj[4] * initial_m[1])
            xmax = int(obj[5] * initial_m[0])
            ymax = int(obj[6] * initial_m[1])
            faces.append([xmin, ymin, xmax, ymax])
    return faces


def performance_counts(network_name, perf_count):
    """
    This is the function that will format 'get_perf_counts()` API of the IECore
    :param network_name: Name of the network
    :param perf_count: The Result from the 'get_perf_counts()' API
    """
    print(f'NETWORK: {network_name}')
    print('===================================================')

    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))
    print('===================================================')


def build_camera_matrix(center_of_face, focal_length):
    """
    This function will build a camera matrix to represent 2D camera image into an 3D representable way.
    :param center_of_face: Center of the Face
    :param focal_length: Focal Length of the Camera
    :return: A Camera Matrix
    """
    cx = center_of_face[0]
    cy = center_of_face[1]
    camera_matrix = np.zeros((3, 3), dtype='float32')
    camera_matrix[0][0] = focal_length
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = focal_length
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1
    return camera_matrix


def main():
    """
    This Function handles all the Inference Loop.
    """
    global initial_h, initial_w, inf_end_fd, inf_end_hp, inf_end_fl, gaze_inf_stop, outfile, scale, pre_fd_time, post_fd_time, pre_hp_time, post_hp_time, pre_fl_time, post_fl_time, pre_ge_time, post_ge_time

    ie = IECore()
    args = args_parser().parse_args()
    network_face_detect = FaceDetection()
    network_head_pose = HeadPoseEstimation()
    network_facial_landmark = FacialLandmarkDetection()
    network_gaze_estimator = GazeEstimation()
    # Loading the Models
    load_fd = time.time()
    n_fd, c_fd, h_fd, w_fd = network_face_detect.load_model(args.face_detection_model, args.device, 0,
                                                            args.cpu_extension)
    load_fd = time.time() - load_fd
    load_hp = time.time()
    n_hp, c_hp, h_hp, w_hp = network_head_pose.load_model(args.head_pose_model, args.device, 0, args.cpu_extension)
    load_hp = time.time() - load_hp
    load_fl = time.time()
    n_fl, c_fl, h_fl, w_fl = network_facial_landmark.load_model(args.facial_landmark_model, args.device, 0,
                                                                args.cpu_extension)
    load_fl = time.time() - load_fl
    load_ge = time.time()
    n_ge, c_ge = network_gaze_estimator.load_model(args.gaze_estimator_model, args.device, 0, args.cpu_extension)
    load_ge = time.time() - load_ge
    # Objecting the Mouse Controllers.
    mouse_controller = MouseController(precision=args.mouse_precision, speed=args.speed)

    # print("FACE DETECTION : ", n_fd, c_fd, h_fd, w_fd)
    # print("HEAD POSE ESTIMATION : ", n_hp, c_hp, h_hp, w_hp)
    # print("FACIAL LANDMARK : ", n_fl, c_fl, h_fl, w_fl)
    # print("GAZE ESTIMATOR: ", n_ge, c_ge)

    try:
        if args.input != 'CAM':
            input_stream = args.input
            log.warning('Please wait, converting to compatible video format...')
            dst_file = 'bin/input.avi'
            try:
                os.path.isfile(args.input)
            except OSError as e:
                log.error(f'The Input File cannot be found in the specified location. {e}')

            if args.output_dir:
                try:
                    os.path.exists(args.output_dir)
                except OSError as e:
                    log.error(f'The Output Directory is not Found.{e}')

            # OpenCV cannot handle .mp4 files directly across different devices, especially on Windows.
            # That's why we take in the .mp4 file and convert it to .avi and then perform inference.

            reader = imageio.get_reader(args.input)
            fps = reader.get_meta_data()['fps']
            writer = imageio.get_writer(dst_file, fps=fps)
            for im in reader:
                writer.append_data(im[:, :, :])
            writer.close()
            input_stream = dst_file
            cap = cv2.VideoCapture(input_stream)
            cap.open(input_stream)
            log.warning(
                "Cursor Controller is using a pre-recorded Video File. \nUse \'CAM\' as --input "
                "instead to control in real-time.")
            if not cap.isOpened():
                log.error("Unable to Open the Video File...Exiting")
                sys.exit(1)
        else:
            cap = cv2.VideoCapture(0)
            log.warning("Cursor Controller is capturing streams from the WebCam.")

        initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if args.output_dir:
            outfile = cv2.VideoWriter(os.path.join(args.output_dir, "output.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps,
                                      (initial_w, initial_h), True)
        print("Press Ctrl + C To Exit...")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            while ret:
                frame_count += 1
                ret, next_frame = cap.read()
                if args.mirror_mode:
                    next_frame = cv2.flip(next_frame, 1)
                if not ret:
                    print("Feed Ended, Exit by Pressing CTRL + C...")
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                if next_frame is None:
                    print("Feed Ended, Exit by Pressing CTRL + C...")
                    cap.release()
                    cv2.destroyAllWindows()
                    break

                # print(next_frame.shape)
                pre_fd_time = time.time()
                next_frame_vis = next_frame.copy()
                in_frame_fd = cv2.resize(next_frame, (w_fd, h_fd))
                # print(in_frame_fd.shape)
                in_frame_fd = in_frame_fd.transpose((2, 0, 1))
                in_frame_fd = in_frame_fd.reshape((n_fd, c_fd, h_fd, w_fd))
                pre_fd_time = time.time() - pre_fd_time

                inf_start_fd = time.time()
                network_face_detect.exec_net(0, in_frame_fd)
                network_face_detect.wait(0)
                inf_end_fd = time.time() - inf_start_fd
                post_fd_time = time.time()
                res_fd = network_face_detect.get_output(0)
                faces = face_detection(res_fd, args, [initial_w, initial_h])
                post_fd_time = time.time() - post_fd_time

                if len(faces) >= 2:
                    # This is to make sure that only a single person is present in the frame.
                    # If more than 1 is present, we issue a warning and pause the inference.
                    log.warning("More than 1 person is on the Frame, Inference Paused...")

                if len(faces) != 0 and len(faces) < 2:
                    for res_hp in faces:
                        xmin, ymin, xmax, ymax = res_hp
                        head_pose = next_frame[ymin:ymax, xmin:xmax]

                        center_of_face = (xmin + head_pose.shape[1] / 2, ymin + head_pose.shape[0] / 2)

                        pre_hp_time = time.time()
                        in_frame_hp = cv2.resize(head_pose, (w_hp, h_hp))
                        in_frame_hp = in_frame_hp.transpose((2, 0, 1))
                        in_frame_hp = in_frame_hp.reshape((n_hp, c_hp, h_hp, w_hp))
                        pre_hp_time = time.time() - pre_hp_time

                        pre_fl_time = time.time()
                        in_frame_fl = cv2.resize(head_pose, (w_fl, h_fl))
                        in_frame_fl = in_frame_fl.transpose((2, 0, 1))
                        in_frame_fl = in_frame_fl.reshape((n_fl, c_fl, h_fl, w_fl))
                        pre_fl_time = time.time() - pre_fl_time

                        inf_start_hp = time.time()
                        network_head_pose.exec_net(0, in_frame_hp)
                        network_head_pose.wait(0)
                        inf_end_hp = time.time() - inf_start_hp

                        post_hp_time = time.time()

                        angle_p_fc = network_head_pose.get_output(0, "angle_p_fc")
                        angle_y_fc = network_head_pose.get_output(0, "angle_y_fc")
                        angle_r_fc = network_head_pose.get_output(0, "angle_r_fc")

                        head_pose_angles = np.array([angle_p_fc[0][0],
                                                     angle_y_fc[0][0],
                                                     angle_r_fc[0][0]]).reshape((1, 3))
                        post_hp_time = time.time() - post_hp_time

                        inf_start_fl = time.time()
                        network_facial_landmark.exec_net(0, in_frame_fl)
                        network_facial_landmark.wait(0)
                        inf_end_fl = time.time() - inf_start_fl

                        post_fl_time = time.time()

                        res_landmarks = network_facial_landmark.get_output(0)
                        landmarks = []

                        lx, ly, rx, ry, mx, my = network_facial_landmark.process_output(res_landmarks,
                                                                                        landmarks,
                                                                                        next_frame_vis,
                                                                                        head_pose,
                                                                                        xmin, ymin)

                        diff_x = int((xmax - xmin) / 6)
                        diff_y = int((ymax - ymin) / 8)

                        left_eye_image = next_frame[ly - diff_y:ly + diff_y, lx - diff_x:lx + diff_x]
                        # le_vis = left_eye_image.copy()
                        # le_center = (
                        #     (lx - diff_x) + left_eye_image.shape[1] / 2, (ly - diff_y) + left_eye_image.shape[0] / 2)
                        right_eye_image = next_frame[ry - diff_y:ry + diff_y, rx - diff_x:rx + diff_x]
                        # re_center = (
                        #     (rx - diff_x) + right_eye_image.shape[1] / 2, (ry - diff_y) + right_eye_image.shape[0] / 2)
                        # re_vis = right_eye_image.copy()
                        post_fl_time = time.time() - post_fl_time
                        pre_ge_time = time.time()

                        left_eye_image = cv2.resize(left_eye_image, (60, 60))
                        left_eye_image = cv2.dnn.blobFromImage(left_eye_image, crop=False)

                        right_eye_image = cv2.resize(right_eye_image, (60, 60))
                        right_eye_image = cv2.dnn.blobFromImage(right_eye_image, crop=False)

                        pre_ge_time = time.time() - pre_ge_time
                        gaze_inf_start = time.time()
                        network_gaze_estimator.multi_in_infer(0, left_eye_image, right_eye_image, head_pose_angles)
                        network_gaze_estimator.wait(0)
                        gaze_inf_stop = time.time() - gaze_inf_start

                        post_ge_time = time.time()
                        gaze_vector = network_gaze_estimator.get_output(0, 'gaze_vector')[0]

                        move_x, move_y, gaze_x_angle, gaze_y_angle, gaze_yaw = network_gaze_estimator.process_output(gaze_vector, angle_r_fc)

                        post_ge_time = time.time() - post_ge_time

                        mouse_controller.move(move_x, move_y)
                        # mouse_controller.move(gaze_x, gaze_y)

                        """
                        If `intermediate_visuals` is True we will only go ahead perform visualisation and gaze angle calculations.  
                        """

                        if args.intermediate_visuals:
                            pitch_message = 'Head Angular Pitch {:.2f}'.format(np.round(head_pose_angles[0][0], 2))
                            yaw_message = 'Head Angular Yaw {:.2f}'.format(np.round(head_pose_angles[0][1], 2))
                            roll_message = 'Head Angular Roll {:.2f}'.format(np.round(head_pose_angles[0][2], 2))

                            cv2.putText(next_frame_vis, pitch_message, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 210, 210))
                            cv2.putText(next_frame_vis, yaw_message, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 210, 210))
                            cv2.putText(next_frame_vis, roll_message, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 210, 210))

                            cv2.putText(next_frame_vis, "Time to Detect the Face: {:.4f}ms".format(inf_end_fd),
                                        (270, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 210))
                            cv2.putText(next_frame_vis,
                                        "Time to Detect the Head Pose Angles: {:.4f}s".format(inf_end_hp), (270, 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 210))
                            cv2.putText(next_frame_vis,
                                        "Time to Detect the Facial Landmarks: {:.4f}s".format(inf_end_fl), (270, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 210))
                            cv2.putText(next_frame_vis,
                                        "Time to Estimate the Gaze Vectors: {:.4f}s".format(gaze_inf_stop), (270, 65),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 210))

                            cv2.putText(next_frame_vis, 'CURSOR CONTROLLER - INTERMEDIATE VISUALISATIONS PANE',
                                        (15, 85),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 210))
                            cv2.putText(next_frame_vis, 'Developed by Swastik N.', (15, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 210, 210))
                            cv2.putText(next_frame_vis,
                                        'Made with Intel (R) Distribution of OpenVino Toolkit 2020.R3 LTS', (15, 115),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 210))

                            lx2 = (int(lx + 100 * gaze_x_angle), int(ly + 100 * gaze_y_angle))
                            lx1 = (int(lx), int(ly))
                            rx2 = (int(rx + 100 * gaze_x_angle), int(ry + 100 * gaze_y_angle))
                            rx1 = (int(rx), int(ry))
                            cv2.line(next_frame_vis, rx1, rx2, (0, 210, 210), 4)
                            cv2.line(next_frame_vis, lx1, lx2, (0, 210, 210), 4)

                            scale = 50
                            focal_length = 950.
                            cx = int(center_of_face[0])
                            cy = int(center_of_face[1])

                            yaw = angle_y_fc * np.pi / 180.
                            pitch = angle_p_fc * np.pi / 180.
                            roll = angle_r_fc * np.pi / 180.

                            camera_matrix = build_camera_matrix(center_of_face, focal_length)

                            Rx = np.array([[1, 0, 0],
                                           [0, math.cos(pitch), -math.sin(pitch)],
                                           [0, math.sin(pitch), math.cos(pitch)]])

                            Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                                           [0, 1, 0],
                                           [math.sin(yaw), 0, math.cos(yaw)]])

                            Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                                           [math.sin(roll), math.cos(roll), 0],
                                           [0, 0, 1]])

                            R = np.dot(Rz, np.dot(Ry, Rx))

                            xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
                            yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
                            zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
                            zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)

                            o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
                            o[2] = camera_matrix[0][0]

                            xaxis = np.dot(R, xaxis) + o
                            yaxis = np.dot(R, yaxis) + o
                            zaxis = np.dot(R, zaxis) + o
                            zaxis1 = np.dot(R, zaxis1) + o

                            xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + mx
                            yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + my

                            p2 = (int(xp2), int(yp2))
                            cv2.line(next_frame_vis, (mx, my), p2, (0, 255, 0), 4)

                            xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + mx
                            yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + my
                            p2 = (int(xp2), int(yp2))
                            cv2.line(next_frame_vis, (mx, my), p2, (0, 0, 255), 4)

                            xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + mx
                            yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + my
                            xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + mx
                            yp2 = (zaxis[0] / zaxis[2] * camera_matrix[1][1]) + my

                            p1 = (int(xp1) + 10, int(yp1) + 10)
                            p2 = (int(xp2) + 10, int(yp2) + 10)
                            cv2.line(next_frame_vis, p1, p2, (255, 0, 0), 4)
                            cv2.circle(next_frame_vis, p2, 3, (255, 200, 0), 2)

                            cv2.rectangle(next_frame_vis, (lx - diff_x, ly - diff_y), (lx + diff_x, ly + diff_y),
                                          (210, 210, 0), 4)
                            cv2.rectangle(next_frame_vis, (rx - diff_x, ry - diff_y), (rx + diff_x, ry + diff_y),
                                          (210, 210, 0), 4)
                            cv2.rectangle(next_frame_vis, (xmin, ymin), (xmax, ymax), (210, 210, 0), 2)
                            cv2.imshow('Cursor Controller - Control Plane', next_frame_vis)
                            cv2.waitKey(1)

                            if args.output_dir:
                                outfile.write(next_frame_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        """
        We will be handling the KeyboardInterrupt to shut-off the Inference Loop using CTRL + C. 
        """
        if args.print_layer_metrics:
            performance_counts('FACE DETECTION NETWORK', network_face_detect.get_performance_counts(0))
            performance_counts('HEAD POSE ESTIMATION NETWORK', network_head_pose.get_performance_counts(0))
            performance_counts('FACIAL LANDMARK DETECTION NETWORK', network_facial_landmark.get_performance_counts(0))
            performance_counts('GAZE ESTIMATION NETWORK', network_gaze_estimator.get_performance_counts(0))

        if args.output_dir:
            with open(os.path.join(args.output_dir, 'stats.txt'), 'w') as f:
                f.write(
                    "\nInference Statistics\nFace Detection Model: \n Inference Time: {:.4f}s, \n Loading Time: {:.4f}s, \n Pre-processing Time:{:.4f}s \n Post-processing Time:{:.4f}s \n"
                    "Head Pose Estimation Model \nInference Time: {:.4f} , \nLoading Time: {:.4f}s, \n Pre-processing Time:{:.4f}s \n Post-processing Time:{:.4f}s \n"
                    "Facial Landmark Detection Model \nInference Time: {:.4f}, \nLoading Time: {:.4f}s, \n Pre-processing Time:{:.4f}s \n Post-processing Time:{:.4f}s \n"
                    "Gaze Estimation Model \nInference Time: {:.4f}, \nLoading Time: {:.4f}s, \n Pre-processing Time:{:.4f}s \n Post-processing Time:{:.4f}s \n".format(
                        inf_end_fd, load_fd, pre_fd_time, post_fd_time, inf_end_hp, load_hp, pre_hp_time, post_hp_time,
                        inf_end_fl, load_fl, pre_fl_time, post_fl_time, gaze_inf_stop, load_ge, pre_ge_time,
                        post_ge_time))

        log.info("Face Detection Model Inference Time: {:.4f} \n "
                 "Head Pose Estimation Model Inference Time: {:.4f} \n "
                 "Facial Landmark Detection Model Inference Time: {:.4f}\n"
                 "Gaze Estimation Model Inference Time: {:.4f} ".format(inf_end_fd, inf_end_hp, inf_end_fl,
                                                                        gaze_inf_stop))
        log.info("Cursor Controller Exited")
    except Exception as e:
        log.error(f'Inference Module Error {e}')


if __name__ == "__main__":
    main()
