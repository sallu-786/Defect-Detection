from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model



def deep_auto_encoder_model(input_shape):
    """ ディープオートエンコーダ定義 """
    inputs = Input(input_shape)
    image_size = input_shape[0]
    # エンコーダ
    conv_1 = Conv2D(
        filters=image_size,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu')(inputs)
    pool_1 = MaxPooling2D()(conv_1)

    conv_2 = Conv2D(
        filters=image_size * 2,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu')(pool_1)
    pool_2 = MaxPooling2D()(conv_2)

    conv_3 = Conv2D(
        filters=image_size * 4,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu')(pool_2)
    pool_3 = MaxPooling2D()(conv_3)
    # デコーダ
    conv_4 = Conv2DTranspose(
        filters=image_size * 2,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_uniform',
        activation='relu')(pool_3)
    conv_5 = Conv2DTranspose(
        filters=image_size,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_uniform',
        activation='relu')(conv_4)
    conv_6 = Conv2DTranspose(
        filters=3,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_uniform',
        activation='relu')(conv_5)
    model = Model(inputs=inputs, outputs=conv_6)
    return model



def Unet_model(input_shape):
    """ U-Netモデルの定義 """
    inputs = Input(input_shape)
    image_size = input_shape[0]
    
    # Encoder
    conv1 = Conv2D(filters=image_size, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    pool1 = MaxPooling2D()(conv1)
    
    conv2 = Conv2D(filters=image_size * 2, kernel_size=3, strides=1, padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D()(conv2)
    
    conv3 = Conv2D(filters=image_size * 4, kernel_size=3, strides=1, padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D()(conv3)
    
    # Decoder
    conv4 = Conv2DTranspose(filters=image_size * 2, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform', activation='relu')(pool3)
    conv5 = Conv2DTranspose(filters=image_size, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform', activation='relu')(conv4)
    conv6 = Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform', activation='relu')(conv5)
    
    model = Model(inputs=inputs, outputs=conv6)
    return model
