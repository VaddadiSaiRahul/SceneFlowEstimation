import os
import numpy as np
from PIL import Image
from pathlib import Path
import struct
import cv2

"""
For Scene Flow Construction (for later)

(A) In Image Space
for filename in os.listdir(flow_path_reference):
    flow = np.asarray(Image.open(flow_path_reference + filename))
    disp_t0 = np.asarray(Image.open(disp_path_t0 + filename))
    disp_t1 = np.asarray(Image.open(disp_path_t1 + filename))
    flow_u = flow[:, :, 0]
    flow_v = flow[:, :, 1]

    flow_u_corrected = (flow_u - 2**15) / 64
    flow_v_corrected = (flow_v - 2**15) / 64
    disp_t0_corrected = disp_t0 / 256
    disp_t1_corrected = disp_t1 / 256

    img_plane_sceneflow = np.dstack((flow_u_corrected, flow_v_corrected, disp_t0_corrected, disp_t1_corrected))
    viz_img_plane_sceneflow = Image.fromarray((img_plane_sceneflow * 255).astype(np.uint8))

(B) In World Coordinate System
(1) Use the camera extrinsics and instrinsics in the camera calibration file to calculate projection matrix
    and take inverse of that and multiple with 2d image coordinates.
(2) Follow the github link for another way of converting https://github.com/ravikt/sceneednet/blob/master/sceneflow.py

"""


def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:
        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')

        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4

        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale


def data_preprocessing(image_left_path, image_right_path, flow_path, disp_path, disp_change_path, target_shape):
    flow_input_array = []
    dispt0_input_array = []
    dispt1_input_array = []
    flow_output_array = []
    dispt0_output_array = []
    dispt1_output_array = []
    filenames = []

    for filename in os.listdir(flow_path):
        filenames.append(filename)

    for index, filename in enumerate(filenames):
        if index - 1 == 0:
            break
        #         print(filename, filename.split('_'))
        curr_img = filename.split('_')[1]
        next_img = int(curr_img) + 1
        next_img = '0' * (4 - len(str(next_img))) + str(next_img)
        flow_img_curr = 'OpticalFlowIntoFuture_' + '0' * (4 - len(str(curr_img))) + str(curr_img) + '_L' + '.pfm'
        flow_img_next = 'OpticalFlowIntoFuture_' + '0' * (4 - len(str(next_img))) + str(next_img) + '_L' + '.pfm'

        #         print(curr_img, next_img, flow_img_curr, flow_img_next)

        if flow_img_next in filenames:
            left_img_t0 = np.resize(Image.open(image_left_path + curr_img + '.png'), target_shape + [3])
            left_img_t1 = np.resize(Image.open(image_left_path + next_img + '.png'), target_shape + [3])
            right_img_t0 = np.resize(Image.open(image_right_path + curr_img + '.png'), target_shape + [3])
            right_img_t1 = np.resize(Image.open(image_right_path + next_img + '.png'), target_shape + [3])
            flow = np.resize(read_pfm(flow_path + flow_img_curr), target_shape + [3])
            dispt0 = np.resize(read_pfm(disp_path + curr_img + '.pfm')[..., np.newaxis], target_shape + [1])
            disp_change = np.resize(read_pfm(disp_change_path + curr_img + '.pfm')[..., np.newaxis], target_shape + [1])

            # flow_u = flow[:, :, 0]
            # flow_v = flow[:, :, 1]
            dispt1 = np.add(dispt0, disp_change)

            flow_input_array.append(np.dstack((left_img_t0, left_img_t1)))
            dispt0_input_array.append(np.dstack((left_img_t0, right_img_t0)))
            dispt1_input_array.append(np.dstack((left_img_t1, right_img_t1)))

            # flow_output_array.append(np.dstack((flow_u, flow_v)))
            flow_output_array.append(flow)
            dispt0_output_array.append(dispt0)
            dispt1_output_array.append(dispt1)

    #             viz_flow = Image.fromarray(flow.astype(np.uint8))
    #             viz_dispt0 = Image.fromarray(dispt0[:, :, 0].astype(np.uint8))
    #             viz_dispt1 = Image.fromarray(dispt1[:, :, 0].astype(np.uint8))

    #             print(viz_flow.show())

    return np.array(flow_input_array), np.array(flow_output_array), \
           np.array(dispt0_input_array), np.array(dispt0_output_array), \
           np.array(dispt1_input_array), np.array(dispt1_output_array)