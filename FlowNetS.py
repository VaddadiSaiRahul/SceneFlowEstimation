import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, UpSampling2D, Input, ZeroPadding2D, concatenate
from keras.models import Model


def EPE(y_true, y_pred):
    y_true_u = y_true[:, :, :, 0]
    y_true_v = y_true[:, :, :, 1]
    y_pred_u = y_pred[:, :, :, 0]
    y_pred_v = y_pred[:, :, :, 1]
    epe = tf.sqrt(tf.square(y_true_u - y_pred_u) + tf.square(y_true_v - y_pred_v))
    avg_epe = tf.reduce_mean(epe)
    return avg_epe


def flownets(input_shape):
    # stacked_input = tf.keras.layers.Input(shape=input_shape)
    #
    # conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=(2, 2), padding='same', activation='relu',
    #                                name='contrastive_conv1')(stacked_input)
    # conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=(1, 1), padding='same', activation='relu',
    #                                  name='contrastive_conv1_1')(conv1)
    #
    # conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(2, 2), padding='same', activation='relu',
    #                                name='contrastive_conv2')(conv1_1)
    # conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding='same', activation='relu',
    #                                  name='contrastive_conv2_1')(conv2)
    #
    # conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
    #                                name='contrastive_conv3')(conv2_1)
    # conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', activation='relu',
    #                                  name='contrastive_conv3_1')(conv3)
    #
    # conv4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
    #                                name='contrastive_conv4')(conv3_1)
    # conv4_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', activation='relu',
    #                                  name='contrastive_conv4_1')(conv4)
    #
    # conv5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
    #                                name='contrastive_conv5')(conv4_1)
    # conv5_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', activation='relu',
    #                                  name='contrastive_conv5_1')(conv5)
    #
    # conv6 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
    #                                name='contrastive_conv6')(conv5_1)
    # conv6_1 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=(1, 1), padding='same', activation='relu',
    #                                  name='contrastive_conv6_1')(conv6)
    #
    # flow6 = tf.keras.layers.Conv2D(filters=2, kernel_size=3, padding='same', name='flow6')(conv6_1)
    # flow6_up = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=4, strides=(2, 2), padding='same',
    #                                            name='expanding_flow6_to_5')(flow6)
    # deconv5 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, strides=(2, 2), padding='same',
    #                                           activation='relu', name='deconv5')(conv6_1)
    #
    # concat5 = tf.keras.layers.concatenate([conv5_1, deconv5, flow6_up], axis=3)
    # iconv5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', name='iconv5')(concat5)
    #
    # flow5 = tf.keras.layers.Conv2D(filters=2, kernel_size=3, padding='same', name='flow5')(iconv5)
    # flow5_up = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=4, strides=(2, 2), name='expanding_flow5_to_4',
    #                                            padding='same')(flow5)
    # deconv4 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=(2, 2), padding='same',
    #                                           activation='relu', name='deconv4')(concat5)
    #
    # concat4 = tf.keras.layers.concatenate([conv4_1, deconv4, flow5_up], axis=3)
    # iconv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', name='iconv4')(concat4)
    #
    # flow4 = tf.keras.layers.Conv2D(filters=2, kernel_size=3, padding='same', name='flow4')(iconv4)
    # flow4_up = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=4, strides=(2, 2), padding='same',
    #                                            name='expanding_flow4_to_3')(flow4)
    # deconv3 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=(2, 2), name='deconv3',
    #                                           padding='same', activation='relu')(concat4)
    #
    # concat3 = tf.keras.layers.concatenate([conv3_1, deconv3, flow4_up], axis=3)
    # iconv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', name='iconv3')(concat3)
    #
    # flow3 = tf.keras.layers.Conv2D(filters=2, kernel_size=3, padding='same', name='flow3')(iconv3)
    # flow3_up = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=4, strides=(2, 2), padding='same',
    #                                            name='expanding_flow3_to_2')(flow3)
    # deconv2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=(2, 2), padding='same',
    #                                           activation='relu', name='deconv2')(concat3)
    #
    # concat2 = tf.keras.layers.concatenate([conv2_1, deconv2, flow3_up], axis=3)
    # iconv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', name='iconv2')(concat2)
    #
    # flow2 = tf.keras.layers.Conv2D(filters=2, kernel_size=3, padding='same', name='flow2')(iconv2)
    # prediction = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(flow2)
    #
    # flownet = tf.keras.Model(inputs=stacked_input, outputs=prediction, name='FlowNetS')

    # flownet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=EPE)

    x = Input(shape=input_shape)
    conv0 = Conv2D(64, (3, 3), padding='same', name='conv0', kernel_initializer='he_normal')(x)
    conv0 = LeakyReLU(0.1)(conv0)
    padding = ZeroPadding2D()(conv0)

    conv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1', kernel_initializer='he_normal')(padding)
    conv1 = LeakyReLU(0.1)(conv1)

    conv1_1 = Conv2D(128, (3, 3), padding='same', name='conv1_1', kernel_initializer='he_normal')(conv1)
    conv1_1 = LeakyReLU(0.1)(conv1_1)
    padding = ZeroPadding2D()(conv1_1)

    conv2 = Conv2D(128, (3, 3), strides=(2, 2), padding='valid', name='conv2', kernel_initializer='he_normal')(padding)
    conv2 = LeakyReLU(0.1)(conv2)

    conv2_1 = Conv2D(128, (3, 3), padding='same', name='conv2_1', kernel_initializer='he_normal')(conv2)
    conv2_1 = LeakyReLU(0.1)(conv2_1)
    padding = ZeroPadding2D()(conv2_1)

    conv3 = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', name='conv3', kernel_initializer='he_normal')(padding)
    conv3 = LeakyReLU(0.1)(conv3)

    conv3_1 = Conv2D(256, (3, 3), padding='same', name='conv3_1', kernel_initializer='he_normal')(conv3)
    conv3_1 = LeakyReLU(0.1)(conv3_1)
    padding = ZeroPadding2D()(conv3_1)

    conv4 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid', name='conv4', kernel_initializer='he_normal')(padding)
    conv4 = LeakyReLU(0.1)(conv4)

    conv4_1 = Conv2D(512, (3, 3), padding='same', name='conv4_1', kernel_initializer='he_normal')(conv4)
    conv4_1 = LeakyReLU(0.1)(conv4_1)
    padding = ZeroPadding2D()(conv4_1)

    conv5 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid', name='conv5', kernel_initializer='he_normal')(padding)
    conv5 = LeakyReLU(0.1)(conv5)

    conv5_1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv5_1', kernel_initializer='he_normal')(conv5)
    conv5_1 = LeakyReLU(0.1)(conv5_1)
    padding = ZeroPadding2D()(conv5_1)

    conv6 = Conv2D(1024, (3, 3), strides=(2, 2), padding='valid', name='conv6', kernel_initializer='he_normal')(padding)
    conv6 = LeakyReLU(0.1)(conv6)

    conv6_1 = Conv2D(1024, (3, 3), padding='same', name='conv6_1', kernel_initializer='he_normal')(conv6)
    conv6_1 = LeakyReLU(0.1)(conv6_1)

    flow6 = Conv2D(2, (3, 3), padding='same', name='predict_flow6', kernel_initializer='he_normal')(conv6_1)
    flow6_up = Conv2DTranspose(2, (4, 4), strides=(2, 2), name='upsampled_flow6_to_5', padding='same',
                               kernel_initializer='he_normal')(flow6)
    deconv5 = Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', name='deconv5',
                              kernel_initializer='he_normal')(conv6_1)
    deconv5 = LeakyReLU(0.1)(deconv5)

    # print(deconv5.get_shape())
    concat5 = concatenate([conv5_1, deconv5, flow6_up], axis=3)  # 16
    inter_conv5 = Conv2D(512, (3, 3), padding='same', name='inter_conv5', kernel_initializer='he_normal')(concat5)
    flow5 = Conv2D(2, (3, 3), padding='same', name='predict_flow5', kernel_initializer='he_normal')(inter_conv5)

    flow5_up = Conv2DTranspose(2, (4, 4), strides=(2, 2), name='upsampled_flow5_to4', padding='same',
                               kernel_initializer='he_normal')(flow5)  # 32
    deconv4 = Conv2DTranspose(256, (4, 4), strides=(2, 2), name='deconv4', padding='same',
                              kernel_initializer='he_normal')(concat5)
    deconv4 = LeakyReLU(0.1)(deconv4)

    concat4 = concatenate([conv4_1, deconv4, flow5_up], axis=3)
    inter_conv4 = Conv2D(256, (3, 3), padding='same', name='inter_conv4', kernel_initializer='he_normal')(concat4)
    flow4 = Conv2D(2, (3, 3), padding='same', name='predict_flow4', kernel_initializer='he_normal')(
        inter_conv4)  # (1, 2, 32, 32)

    flow4_up = Conv2DTranspose(2, (4, 4), strides=(2, 2), name='upsampled_flow4_to3', padding='same',
                               kernel_initializer='he_normal')(flow4)  # 64
    deconv3 = Conv2DTranspose(128, (4, 4), strides=(2, 2), name='deconv3', padding='same',
                              kernel_initializer='he_normal')(concat4)
    deconv3 = LeakyReLU(0.1)(deconv3)

    concat3 = concatenate([conv3_1, deconv3, flow4_up], axis=3)  # 64
    inter_conv3 = Conv2D(128, (3, 3), padding='same', name='inter_conv3', kernel_initializer='he_normal')(concat3)
    flow3 = Conv2D(2, (3, 3), padding='same', name='predict_flow3', kernel_initializer='he_normal')(inter_conv3)
    flow3_up = Conv2DTranspose(2, (4, 4), strides=(2, 2), name='upsampled_flow3_to2', padding='same',
                               kernel_initializer='he_normal')(flow3)  # 128
    deconv2 = Conv2DTranspose(64, (4, 4), strides=(2, 2), name='deconv2', padding='same',
                              kernel_initializer='he_normal')(concat3)
    deconv2 = LeakyReLU(0.1)(deconv2)

    concat2 = concatenate([conv2_1, deconv2, flow3_up], axis=3)
    inter_conv2 = Conv2D(64, (3, 3), padding='same', name='inter_conv2', kernel_initializer='he_normal')(concat2)
    flow2 = Conv2D(2, (3, 3), padding='same', name='predict_flow2', kernel_initializer='he_normal')(inter_conv2)
    result = UpSampling2D(size=(4, 4), interpolation='bilinear')(flow2)  # 4*128

    flownet = Model(x, result)

    flownet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=EPE)

    return flownet
