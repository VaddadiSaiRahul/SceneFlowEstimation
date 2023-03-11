import tensorflow as tf


def EPE(y_true, y_pred):
    epe = tf.sqrt(tf.square(y_true - y_pred))
    avg_epe = tf.reduce_mean(epe)
    return avg_epe


# TODO: Haven't taken weighted sums of loss1 to loss6 into consideration
def dispnet(input_shape):
    stacked_input = tf.keras.layers.Input(shape=input_shape)

    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=(2, 2), padding='same', activation='relu',
                                   name='contrastive_conv1')(stacked_input)

    conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(2, 2), padding='same', activation='relu',
                                   name='contrastive_conv2')(conv1)

    conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=(2, 2), padding='same', activation='relu',
                                   name='contrastive_conv3')(conv2)
    conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', activation='relu',
                                     name='contrastive_conv3_1')(conv3)

    conv4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
                                   name='contrastive_conv4')(conv3_1)
    conv4_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', activation='relu',
                                     name='contrastive_conv4_1')(conv4)

    conv5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
                                   name='contrastive_conv5')(conv4_1)
    conv5_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', activation='relu',
                                     name='contrastive_conv5_1')(conv5)

    conv6 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
                                   name='contrastive_conv6')(conv5_1)
    conv6_1 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=(1, 1), padding='same', activation='relu',
                                     name='contrastive_conv6_1')(conv6)

    pr6 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=(1, 1), padding='same', name='pr6')(conv6_1)
    # loss6 = tf.keras.layers.AveragePooling2D(pool_size=(64, 64), strides=(64, 64), padding='same', name='loss6')(gt)
    depr6 = tf.keras.layers.UpSampling2D(name='expansive_pr6')(pr6)

    deconv5 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, strides=(2, 2), padding='same',
                                              activation='relu', name='deconv5')(conv6_1)

    concat5 = tf.keras.layers.concatenate([deconv5, depr6, conv5_1], axis=3)
    iconv5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', name='iconv5')(concat5)

    pr5 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', name='pr5')(iconv5)
    # loss5 = tf.keras.layers.AveragePooling2D(pool_size=(32, 32), strides=(32, 32), padding='same', name='loss5')(gt)
    depr5 = tf.keras.layers.UpSampling2D(name='expansive_pr5')(pr5)

    deconv4 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=(2, 2), padding='same',
                                              activation='relu', name='deconv4')(iconv5)

    concat4 = tf.keras.layers.concatenate([deconv4, depr5, conv4_1], axis=3)
    iconv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', name='iconv4')(concat4)

    pr4 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', name='pr4')(iconv4)
    # loss4 = tf.keras.layers.AveragePooling2D(pool_size=(16, 16), strides=(16, 16), padding='same', name='loss4')(gt)
    depr4 = tf.keras.layers.UpSampling2D(name='expansive_pr4')(pr4)

    deconv3 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=(2, 2), name='deconv3',
                                              padding='same', activation='relu')(iconv4)

    concat3 = tf.keras.layers.concatenate([deconv3, depr4, conv3_1], axis=3)
    iconv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', name='iconv3')(concat3)

    pr3 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', name='pr3')(iconv3)
    # loss3 = tf.keras.layers.AveragePooling2D(pool_size=(8, 8), strides=(8, 8), padding='same', name='loss3')(gt)
    depr3 = tf.keras.layers.UpSampling2D(name='expansive_pr3')(pr3)

    deconv2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=(2, 2), padding='same',
                                              activation='relu', name='deconv2')(iconv3)

    concat2 = tf.keras.layers.concatenate([deconv2, depr3, conv2], axis=3)
    iconv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', name='iconv2')(concat2)

    pr2 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', name='pr2')(iconv2)
    # loss2 = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same', name='loss2')(gt)
    depr2 = tf.keras.layers.UpSampling2D(name='expansive_pr2')(pr2)

    deconv1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=(2, 2), padding='same',
                                              activation='relu', name='deconv1')(iconv2)

    concat1 = tf.keras.layers.concatenate([deconv1, depr2, conv1], axis=3)
    iconv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', name='iconv1')(concat1)

    pr1 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', name='pr1')(iconv1)
    # loss1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='loss1')(gt)
    depr1 = tf.keras.layers.UpSampling2D(name='expansive_pr1')(pr1)

    dispnet = tf.keras.Model(inputs=stacked_input, outputs=depr1, name='DispNetS')
    dispnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=EPE)

    return dispnet
