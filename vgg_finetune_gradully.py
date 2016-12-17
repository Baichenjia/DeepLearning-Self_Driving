# -*- coding: utf-8 -*-
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, Sequential, save_model
from keras.optimizers import SGD
from PIL import Image
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Flatten, Dense, Input, Dropout, Activation, GlobalAveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.regularizers import l2, activity_l2
import matplotlib.pyplot as plt
import re
import numpy as np
import os
import support_func


# VGG 为基础的类
class vggClassify():

    def __init__(self):
        """
            初始化预训练好的VGG16网络权重
        """
        # 与训练网络权重
        self.weights_path = "../experiment2/vgg_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
        assert os.path.exists(self.weights_path), 'Model weights not found'

        """
            fine-tune 全连接层
        """
        # 瓶颈特征存储位置
        self.bottlenect_train_path = "bottlenect_feature/bottleneck_features_train.npy"
        self.bottlenect_val_path = "bottlenect_feature/bottleneck_features_validation.npy"
        assert os.path.exists("bottlenect_feature/"), "Not Found bottlenect_feature dir"

        # 仅微调全连接层的权重存储位置
        self.path_fully_connected = "weights_fully_connected/"
        assert os.path.exists(self.path_fully_connected), "Not Found path path_fully_connected dir"

        # 训练好的全连接层的权重存储位置
        self.path_weights_fully_connected = "weights_fully_connected/checkpoint-020-0.07.h5"
        assert os.path.exists(self.path_weights_fully_connected), "Not Found path path_weights_fully_connected"

        # BottleNect 特征的size， 该 size 由 VGG16 的网络结构决定
        self.bottlenect_feature_size = np.array([7, 7, 512])

        """
            fine-tune convBlock5 和 全连接层
        """

        # 同时 fine tune ConvBlock5 和 全连接层 权重， 训练结果存储在path_conv_block5目录下
        self.path_conv_block5 = "weights_conv_block5/"
        assert os.path.exists(self.path_conv_block5), "Not Found path path_conv_block5 dir"

        # 训练好的微调conv5后的卷积层的权重
        self.path_weights_conv_block5 = "weights_conv_block5/checkpoint-000-0.07.h5"
        assert os.path.exists(self.path_weights_conv_block5), "Not Found path_weights_conv_block5"

        """
            fine_tune convBlock4 和 convBlock5 和 全连接层
        """

        # 同时finetune ConvBlock4 and ConvBlock5 and fully-cnnected，结果存储在path_conv_block4目录下
        self.path_conv_block4 = "weights_conv_block4/"
        assert os.path.exists(self.path_conv_block4), "Not Found path_conv_block4"

        """
            训练过程中的其他参数
        """

        # 训练样本个数 验证样本个数
        self.train_data_size = 4500
        self.valid_data_size = 900

        # 迭代次数
        self.nb_epoch = 100

    #
    # 原始的VGG16
    def build_vgg_model(self):
        """
        VGG16 网络结构
        input_shape =  (4D tensor with shape or 4D tensor with shape)
          `(samples, rows, cols, channels)` if dim_ordering='tf'.
        """
        # build the VGG16 network
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))

        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model = model
        """
        # 打印结构
        for i, layer in enumerate(self.model.layers):
            print(i, layer.name)
        """
        print("VGG-notop 模型建立完毕...\n")

    #
    def get_bottlenect_feature(self):
        """
            使用 VGG16 预训练好的权重，使训练数据和测试数据获得瓶颈特征
        """
        print "----------------"
        print "使用 VGG16 预训练好的权重，使训练数据和测试数据获得瓶颈特征..."

        # 载入预训练好的权重
        self.model.load_weights(self.weights_path)
        if K.backend() == 'theano':
            convert_all_kernels_in_model(self.model)

        # 训练数据归一化处理
        print "图片归一化预处理..."
        train_datagen = ImageDataGenerator(rescale=1./255)
        valid_datagen = ImageDataGenerator(rescale=1./255)

        # 对训练数据提取特征
        print "提取训练数据和验证数据的特征..."
        train_generator = train_datagen.flow_from_directory(
                'data/train',
                target_size=(224, 224),
                batch_size=32,
                class_mode=None,   # 此处只用来通过VGG网络提取特征，因此不需要生成类别
                shuffle=False)     # 不对数据进行打乱
        bottleneck_features_train = self.model.predict_generator(train_generator, self.train_data_size)
        print "\n训练数据特征维度", bottleneck_features_train.shape   # (4500, 7, 7, 512)
        np.save(open(self.bottlenect_train_path, 'w'), bottleneck_features_train)
        print "训练数据特征保存在 bottleneck_features_train.npy 中..."

        # 对验证数据提取特征
        valid_generator = valid_datagen.flow_from_directory(
                'data/validation',
                target_size=(224, 224),
                batch_size=32,
                class_mode=None,   # 此处只用来通过VGG网络提取特征，因此不需要生成类别
                shuffle=False)     # 不对数据进行打乱
        bottleneck_features_validation = self.model.predict_generator(valid_generator, self.valid_data_size)
        print "\n验证数据特征维度", bottleneck_features_validation.shape  # (900, 7, 7, 512)
        np.save(open(self.bottlenect_val_path, 'w'), bottleneck_features_validation)
        print "验证数据特征保存在 bottleneck_features_validation.npy 中..."

    #
    # 设置全连接层
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

    #
    # 使用bottlenect特征，训练全连接层
    def fineTune_fully_connected_layer(self):
        print "----------------"
        print "使用bottlenect特征，训练全连接层  保存至 ", self.path_fully_connected

        # 载入bottlenect数据
        print "载入 bottlenect 特征..."
        train_data = np.load(open(self.bottlenect_train_path))
        train_labels = np.array([0]*(self.train_data_size/3)+[1]*(self.train_data_size/3)+[2]*(self.train_data_size/3))

        validation_data = np.load(open(self.bottlenect_val_path))
        validation_labels = np.array([0]*(self.valid_data_size/3)+[1]*(self.valid_data_size/3)+[2]*(self.valid_data_size/3))

        # 生成 one-hot label
        train_labels = np_utils.to_categorical(train_labels, 3)  # one hot
        validation_labels = np_utils.to_categorical(validation_labels, 3)  # one hot

        print "\n\n训练数据 shape = ", train_data.shape
        print "训练label shape = ", train_labels.shape
        print "\n\n验证数据 shape = ", validation_data.shape
        print "验证label shape = ", validation_labels.shape

        # 梯度下降，动量更新
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

        # model check    保存在验证集上最好的模型
        model_check = ModelCheckpoint(self.path_fully_connected + "checkpoint-{epoch:03d}-{val_loss:.2f}.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

        # early stopping  当验证集合的损失不再下降时，等待 patient 后停止训练
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')

        # 模型生成
        #self.top_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.top_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.hist = self.top_model.fit(train_data, train_labels,
                nb_epoch=self.nb_epoch, batch_size=45,
                callbacks=[model_check, early_stopping],   # model_check, early_stopping
                validation_data=(validation_data, validation_labels),
                shuffle=True)

        # 保存日志并绘图显示
        log_file = open(self.path_fully_connected + "hist.txt", "w")
        log_file.write(str(self.hist.history))
        log_file.close()

        support_func.plot_hist(self.path_fully_connected)

        print "全连接层训练完毕，权重保存...完毕！"

    #
    # 微调第 5 个卷积层模块
    def fineTune_conv_bolck5_layer(self):
        """
        1. 为了进行fine-tune，所有的层都应该以训练好的权重为初始值
        2. fine-tune最后的卷积块，而不是整个网络，这是为了防止过拟合
        3. fine-tune应该在很低的学习率下进行，通常使用SGD优化而不是其他自适应学习率的优化算法
        """
        print "----------------"
        print "fine-tune Convolution Block5 卷积层模块..."

        # 全连接层权重存储于: self.path_weights_fully_connected
        # 训练好的整体模型权重存储于： self.path_conv_block5

        print "VGG16-no-top-model 载入预训练好的权重..."
        self.model.load_weights(self.weights_path)
        if K.backend() == 'theano':
            convert_all_kernels_in_model(self.model)

        print "top_model 载入预训练好的权重..."
        self.top_model.load_weights(self.path_weights_fully_connected)
        print "生成完整模型..."
        self.model.add(self.top_model)

        # 将除最后一个卷积层之外的参数冻结
        print "冻结模型在 0-24 层 即 ConvBlock1 - ConvBlock4 的参数..."
        for layer in self.model.layers[:25]:
            layer.trainable = False
        for layer in self.model.layers[25:]:
            layer.trainable = True

        # 图像批量生成器
        train_datagen = ImageDataGenerator(rescale=1./255)
        valid_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                'data/train',
                target_size=(224, 224),
                batch_size=45,
                class_mode='categorical')  # 自动生成类别

        valid_generator = valid_datagen.flow_from_directory(
                'data/validation',
                target_size=(224, 224),
                batch_size=45,
                class_mode='categorical')  # 自动生成类别

        # sgd 梯度下降，动量更新
        sgd = SGD(lr=1e-5, decay=1e-7, momentum=0.9, nesterov=True)

        # modelCheck 保存在验证集上最好的模型   仅保留权重
        model_check = ModelCheckpoint(self.path_conv_block5+"checkpoint-{epoch:03d}-{val_loss:.2f}.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
        # 当验证集合的损失不再下降时，等待 patient 后停止训练
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

        # 交叉熵损失函数
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        #
        # self.train_data_size = 4500    self.valid_data_size
        print "\n开始训练..."
        self.hist = self.model.fit_generator(
                        train_generator,
                        samples_per_epoch=self.train_data_size,
                        nb_epoch=self.nb_epoch,
                        callbacks=[model_check, early_stopping],
                        validation_data=valid_generator,
                        nb_val_samples=self.valid_data_size
                        )

        # 保存日志并绘图显示
        log_file = open(self.path_conv_block5 + "hist.txt", "w")
        log_file.write(str(self.hist.history))
        log_file.close()
        support_func.plot_hist(self.path_conv_block5)

        print "fineTune_conv_bolck5_layer 训练完毕，权重保存...完毕！"

    #
    # 微调第 4 个卷积层模块
    def fineTune_conv_bolck4_layer(self):
        """
        1. 为了进行fine-tune，所有的层都应该以训练好的权重为初始值
        2. fine-tune最后的卷积块，而不是整个网络，这是为了防止过拟合
        3. fine-tune应该在很低的学习率下进行，通常使用SGD优化而不是其他自适应学习率的优化算法
        """
        print "----------------"
        print "fine-tune Convolution Block4 卷积层模块..."

        # 前一步微调后的总权重存储于: path_weights_conv_block5
        print "self.model 载入上次微调后的权重..."
        self.model.add(self.top_model)   # 总模型
        self.model.load_weights(self.path_weights_conv_block5)

        # 将除了 conv4 conv5 fully-connected 之外的参数冻结
        print "冻结模型在 0-17 层 即 ConvBlock1 - ConvBlock3 的参数..."
        for layer in self.model.layers[:18]:
            layer.trainable = False
        for layer in self.model.layers[18:]:
            layer.trainable = True

        # 图像批量生成器
        train_datagen = ImageDataGenerator(rescale=1./255)
        valid_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                'data/train',
                target_size=(224, 224),
                batch_size=45,
                class_mode='categorical')  # 自动生成类别

        valid_generator = valid_datagen.flow_from_directory(
                'data/validation',
                target_size=(224, 224),
                batch_size=45,
                class_mode='categorical')  # 自动生成类别

        # sgd 梯度下降，动量更新
        sgd = SGD(lr=1e-5, decay=1e-7, momentum=0.9, nesterov=True)
        # modelCheck 保存在验证集上最好的模型   仅保留权重
        model_check = ModelCheckpoint(self.path_conv_block4+"checkpoint-{epoch:03d}-{val_loss:.2f}.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
        # 当验证集合的损失不再下降时，等待 patient 后停止训练
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

        # 交叉熵损失函数
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        #
        # self.train_data_size = 4500    self.valid_data_size = 900
        print "\n开始训练..."
        self.hist = self.model.fit_generator(
                        train_generator,
                        samples_per_epoch=self.train_data_size,
                        nb_epoch=self.nb_epoch,
                        callbacks=[model_check, early_stopping],
                        validation_data=valid_generator,
                        nb_val_samples=self.valid_data_size
                        )

        # 保存日志并绘图显示
        log_file = open(self.path_conv_block4 + "hist.txt", "w")
        log_file.write(str(self.hist.history))
        log_file.close()
        support_func.plot_hist(self.path_conv_block5)

        print "fineTune_conv_bolck4_layer 训练完毕，权重保存...完毕！"


#
if __name__ == '__main__':
    vgg = vggClassify()

    # 1.0 提取bottlenect 特征，建立全连接层，微调全连接层权重
    flag_1 = False
    if flag_1:
        vgg.build_vgg_model()
        #vgg.get_bottlenect_feature()   # 提取bottlenect特征，已经保存在本地，不需要重复运行
        vgg.build_top_model()
        vgg.fineTune_fully_connected_layer()

    # 2.0 fineTune ConvBlock5 and FullyConnect
    flag_2 = False
    if flag_2:
        vgg.build_vgg_model()
        vgg.build_top_model()
        vgg.fineTune_conv_bolck5_layer()

    # 3.0 fineTune ConvBlock4 , ConvBlock5 and FullyConnect
    flag_3 = True
    if flag_3:
        vgg.build_vgg_model()
        vgg.build_top_model()
        vgg.fineTune_conv_bolck4_layer()


"""
# VGG-16 no-top 网络结构
(0, 'zeropadding2d_1')
(1, 'conv1_1')
(2, 'zeropadding2d_2')
(3, 'conv1_2')
(4, 'maxpooling2d_1')
(5, 'zeropadding2d_3')
(6, 'conv2_1')
(7, 'zeropadding2d_4')
(8, 'conv2_2')
(9, 'maxpooling2d_2')
(10, 'zeropadding2d_5')
(11, 'conv3_1')
(12, 'zeropadding2d_6')
(13, 'conv3_2')
(14, 'zeropadding2d_7')
(15, 'conv3_3')
(16, 'maxpooling2d_3')
(17, 'zeropadding2d_8')
(18, 'conv4_1')
(19, 'zeropadding2d_9')
(20, 'conv4_2')
(21, 'zeropadding2d_10')
(22, 'conv4_3')
(23, 'maxpooling2d_4')
(24, 'zeropadding2d_11')
(25, 'conv5_1')
(26, 'zeropadding2d_12')
(27, 'conv5_2')
(28, 'zeropadding2d_13')
(29, 'conv5_3')
(30, 'maxpooling2d_5')

"""
