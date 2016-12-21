# -*- coding: utf-8 -*-

from keras.models import Model, Sequential, save_model
from PIL import Image
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils import np_utils
import numpy as np
import os


# VGG 为基础的类
class vggPredict():

    def __init__(self):
        # 模型权重的存储位置
        self.model_weight_path = "weights_conv_block4/checkpoint-000-0.07.h5"
        # BottleNect 特征的size， 该 size 由 VGG16 的网络结构决定
        self.bottlenect_feature_size = np.array([7, 7, 512])

    def build_base_model(self):
        """
        input_shape =  (4D tensor with shape or 4D tensor with shape)
          `(samples, rows, cols, channels)` if dim_ordering='tf'.
        建立模型并载入训练好的权重
        """
        print "建立模型... "
        # ConvBlock1
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))

        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # ConvBlock2
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # ConvBlock3
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # ConvBlock4
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # ConvBlock5
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model = model
        print("base 模型建立完毕...\n")

    #
    def build_top_model(self):
        # 搭建模型最后的全连接层
        print "----------------"
        print "设置全连接层"

        top_model = Sequential()
        top_model.add(Flatten(input_shape=self.bottlenect_feature_size))
        top_model.add(Dense(512, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(128, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(32, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(3, activation='softmax'))
        self.top_model = top_model
        print("top 模型建立完毕...\n")

    #
    def build_model(self):
        self.build_base_model()
        self.build_top_model()
        self.model.add(self.top_model)   # 总模型

        # 载入权重
        print "载入权重... "
        self.model.load_weights(self.model_weight_path)
        if K.backend() == 'theano':
            convert_all_kernels_in_model(self.model)

        print("模型建立完毕...\n")

    #
    # 前向传播计算
    def predict_single_img(self, imgpath):

        img = Image.open(imgpath)
        print "接收到的图片的尺寸为: ", img.size
        img = img.convert('RGB')
        img = img.resize((224, 224))

        #img = load_img(imgpath, grayscale=False, target_size=(224, 224))
        x = img_to_array(img, dim_ordering='tf')
        x /= 255.0
        x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)

        print "x.shape = ", x.shape

        y = self.model.predict(x)[0]

        print y
        print "各个类别的输出概率: "
        res = list(np.argsort(y))[::-1]

        for i in range(3):
            label = res[i]
            prob = y[label]
            label_mark = ""
            if label == 0:
                label_mark = "ahead"
            elif label == 1:
                label_mark = "left"
            elif label == 2:
                label_mark = "right"
            else:
                pass
            print i+1, ": prob=", prob, " label=", label, " label_mark=", label_mark
        return y

    def predict_batch_img(self):
        test_img_list = ["receive_pic_0.jpg", "data/train/left/left_200.jpg", "data/train/left/left_300.jpg"]
        for index, test_img in enumerate(test_img_list):
            print "Image index = ", index
            self.predict_single_img(test_img)
            print "\n\n"

#
if __name__ == '__main__':
    vggpredict = vggPredict()
    vggpredict.build_model()
    res = vggpredict.predict_single_img("receive_pic_0.jpg")
    #vggpredict.predict_batch_img()







