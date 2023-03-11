import os

from data import data_preprocessing
from FlowNetS import flownets
from sceneflow import sceneflowconstruct
from PIL import Image
from DispNet import dispnet, EPE
from train import train
import keras
import numpy as np
from keras import callbacks

np.random.seed(0)

def main():
    root_path = "D:/Scene_flow_project/driving_subset/" # "D:/Scene_flow_project/kitti2015/data_scene_flow/training/"
    image_left_path = root_path + "frames_finalpass_subset/left/" # root_path + "image_2/"
    image_right_path = root_path + "frames_finalpass_subset/right/" # "image_3/"
    disp_path = root_path + "disparity_subset/" # 'disp_noc_0/'
    disp_change_path = root_path + "disparity_change_subset/" # 'disp_noc_1/'
    flow_path = root_path + "optical_flow_subset/" # 'flow_noc/'

    flow_model_path = 'C:/Users/DELL/Documents/PatternRecognitionAndComputerVision_CS5330/scene_flow_estimation/FlowNetS.h5'
    dispt0_model_path = 'C:/Users/DELL/Documents/PatternRecognitionAndComputerVision_CS5330/scene_flow_estimation/DispNet_t0_new.h5'
    dispt1_model_path = 'C:/Users/DELL/Documents/PatternRecognitionAndComputerVision_CS5330/scene_flow_estimation' \
                        '/DispNet_t1_new.h5 '
    target_shape = [540, 960]
    n_test = 1
    densities = [0.25, 0.5, 0.75, 1]

    f_in, f_out, dt0_in, dt0_out, dt1_in, dt1_out = data_preprocessing(image_left_path, image_right_path,
                                                                       flow_path, disp_path, disp_change_path,
                                                                       target_shape=target_shape)

    # left_img_t0 = np.resize(Image.open(image_left_path + '000002_10.png'), target_shape + [3])
    # left_img_t1 = np.resize(Image.open(image_left_path + '000002_11.png'), target_shape + [3])
    # right_img_t0 = np.resize(Image.open(image_right_path + '000002_10.png'), target_shape + [3])
    # right_img_t1 = np.resize(Image.open(image_right_path + '000002_11.png'), target_shape + [3])
    #
    # f_in = np.dstack((left_img_t0, left_img_t1))[np.newaxis, ...]
    # f_out = np.resize(Image.open(flow_path + '000002_10.png'), target_shape + [3])[np.newaxis, ...]
    #
    # dt0_in = np.dstack((left_img_t0, right_img_t0))[np.newaxis, ...]
    # dt0_out = np.resize(Image.open(disp_path + '000002_10.png'), target_shape + [1])[np.newaxis, ...]
    #
    # dt1_in = np.dstack((left_img_t1, right_img_t1))[np.newaxis, ...]
    # dt1_out = np.resize(Image.open(disp_change_path + '000002_10.png'), target_shape + [1])[np.newaxis, ...]

    flownet = keras.models.load_model(flow_model_path, compile=False)
    dispnet_t0 = keras.models.load_model(dispt0_model_path, compile=False)
    dispnet_t1 = keras.models.load_model(dispt1_model_path, compile=False)

    # print(f_in.shape, f_out.shape, dt1_in.shape, dt1_out.shape)

    for density in densities:
        mask = np.random.choice([1, 0], size=target_shape, p=[density, 1-density])[..., np.newaxis]
        sparse_f_in, sparse_f_out = np.multiply(f_in, mask), np.multiply(f_out, mask)
        sparse_dt0_in, sparse_dt0_out = np.multiply(dt0_in, mask), np.multiply(dt0_out, mask)
        sparse_dt1_in, sparse_dt1_out = np.multiply(dt1_in, mask), np.multiply(dt1_out, mask)

        # flow_pred = flownet.predict(sparse_f_in[:n_test])
        # # print(flow_pred.shape)
        # dispt0_pred = dispnet_t0.predict(sparse_dt0_in[:n_test]) * 255.
        # dispt1_pred = dispnet_t1.predict(sparse_dt1_in[:n_test]) * 255.

        flow_true = sparse_f_out[:n_test]
        dispt0_true = sparse_dt0_out[:n_test]
        dispt1_true = sparse_dt1_out[:n_test]

        # viz_flow_pred = Image.fromarray(flow_pred[0].astype(np.uint8))
        # viz_flow_true = Image.fromarray(flow_true[0].astype(np.uint8))

        # print(flow_true.shape, dispt0_true.shape, dispt1_true.shape)

        sf = sceneflowconstruct(flow_true[0], dispt0_true[0], dispt1_true[0])
        viz_flow = Image.fromarray(sf.astype(np.uint8))
        print(viz_flow.show())

        # print(viz_flow_true.show())
        # print(viz_flow_pred.show())

        # print(EPE(dispt0_true, dispt1_pred))

        # mepe = 0
        # for index in range(n_test):
        #     sf_pred = sceneflowconstruct(flow_pred[index], dispt0_pred[index], dispt1_pred[index])
        #     sf_true = sceneflowconstruct(flow_true[index], dispt0_true[index], dispt1_true[index])
        #     mepe += EPE(sf_true, sf_pred).numpy()
        #
        # mepe /= n_test
        #
        # print("Density {0}:".format(str(density * 100) + '%'), mepe)

    # flownet = flownets(f_in.shape[1:])
    # dispnet_t0 = dispnet(dt0_in.shape[1:])
    # dispnet_t1 = dispnet(dt1_in.shape[1:])
    #
    # callbacks = [
    #     keras.callbacks.ModelCheckpoint('FlowNetS.h5', save_best_only=True),
    #     keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, verbose=1, patience=10, mode='min'),
    #     keras.callbacks.LearningRateScheduler(scheduler),
    #     keras.callbacks.CSVLogger('training.csv'),
    #     keras.callbacks.EarlyStopping(monitor='loss', verbose=1, patience=4, mode='min', restore_best_weights=True),
    #     keras.callbacks.TensorBoard(log_dir='logs', write_graph=True),
    #     keras.callbacks.TerminateOnNaN()
    # ]
    #
    # train(dt0_in, dt0_out, 1, dispnet_t0, callbacks)


if __name__ == '__main__':
    main()
