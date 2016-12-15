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


# VGG 为基础的类
class vggClassify():

    def __init__(self):
        """
            初始化预训练好的VGG16网络权重
        """
        # 与训练网络权重
        self.weights_path = "../experiment2/vgg_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
        assert os.path.exists(self.weights_path), 'Model weights not found'

        # 瓶颈特征存储位置
        self.bottlenect_train_path = "bottlenect_feature/bottleneck_features_train.npy"
        self.bottlenect_val_path = "bottlenect_feature/bottleneck_features_validation.npy"
        assert os.path.exists("bottlenect_feature/"), "Not Found bottlenect_feature dir"

        # 仅微调全连接层的权重存储位置
        self.path_fully_connected = "weights_fully_connected/"
        assert os.path.exists(self.path_fully_connected), "Not Found path path_fully_connected dir"

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
    def build_top_model():
        # 搭建模型最后的全连接层
        print "----------------"
        print "设置全连接层"

        top_model = Sequential()
        top_model.add(Flatten(input_shape=train_data.shape[1:]))
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
    def fineTune_fully_connected_layer():
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
        # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

        # model check    保存在验证集上最好的模型
        model_check = ModelCheckpoint(self.path_fully_connected + "checkpoint-{epoch:03d}-{val_loss:.2f}.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

        # early stopping  当验证集合的损失不再下降时，等待 patient 后停止训练
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')

        # 模型生成
        newmodel.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.hist = newmodel.fit(train_data, train_labels,
                nb_epoch=self.nb_epoch, batch_size=45,
                callbacks=[model_check, early_stopping],   # model_check, early_stopping
                validation_data=(validation_data, validation_labels),
                shuffle=True)

        # 保存日志并绘图显示
        log_file = open(self.path_fully_connected + "hist.txt", "w")
        log_file.write(str(self.hist.history))
        log_file.close()
        self.plot_hist()

        print "全连接层训练完毕，权重保存...完毕！"

    #
    def vgg_fine_tune(self):
        """
        1. 为了进行fine-tune，所有的层都应该以训练好的权重为初始值
        2. fine-tune最后的卷积块，而不是整个网络，这是为了防止过拟合
        3. fine-tune应该在很低的学习率下进行，通常使用SGD优化而不是其他自适应学习率的优化算法
        """
        print "\n\nfine-tune VGG16最后一个卷积层..."
        self.train_data_size = 900
        self.valid_data_size = 120
        self.top_model_weights_path = "store_val_model_6_rmsprop_dropout0.3_L2_224/checkpoint-15-0.64-best.h5"
        self.dir_path = "store_val_model_8/"

        # topModel
        top_model = Sequential()
        top_model.add(Flatten(input_shape=self.model.output_shape[1:]))
        top_model.add(Dense(1024, W_regularizer=l2(0.001), activation='relu'))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(1024, W_regularizer=l2(0.001), activation='relu'))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(1024, W_regularizer=l2(0.001), activation='relu'))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(1024, W_regularizer=l2(0.001), activation='relu'))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(512, W_regularizer=l2(0.001), activation='relu'))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(256, W_regularizer=l2(0.0001), activation='relu'))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(128, W_regularizer=l2(0.0001), activation='relu'))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(64, activation='relu'))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(32, activation='relu'))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(16, activation='relu'))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(3, activation='softmax'))
        # 载入 top_model 的权重
        print "top_model 载入预训练好的权重..."
        top_model.load_weights(self.top_model_weights_path)
        # 生成完整模型
        print "生成完整模型..."
        self.model.add(top_model)

        self.print_layers()

        # 将除最后一个卷积层之外的参数冻结
        print "冻结模型在 0-25 层的参数..."
        for layer in self.model.layers[:25]:
            layer.trainable = False
        for layer in self.model.layers[25:]:
            layer.trainable = True

        # 图像批量生成器
        train_datagen = ImageDataGenerator(
            rescale=1./255
            #rotation_range=1,       # 0~180的度数，用来指定随机选择图片的角度
            #width_shift_range=0.02,    # **
            #height_shift_range=0.05,  # ** 水平和竖直方向随机移动的程度，这是两个0~1之间的比例
            #shear_range=0.05,         # ** 错切变换
            #fill_mode='nearest'         # 变换中像素的填充模式
            )
        valid_datagen = ImageDataGenerator(rescale=1./255)   # 验证数据只进行归一化处理

        train_generator = train_datagen.flow_from_directory(
                'data/train',
                target_size=(224, 224),    # 高为180，宽为320 (180, 320)
                batch_size=32,
                class_mode='categorical')  # 类别

        valid_generator = valid_datagen.flow_from_directory(
                'data/validation',
                target_size=(224, 224),  # 高为180，宽为320
                batch_size=32,
                class_mode='categorical')  # 类别

        # 梯度下降，动量更新
        sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
        # 保存在验证集上最好的模型   仅包含权重
        model_check = ModelCheckpoint(self.dir_path+"checkpoint-{epoch:02d}-{val_loss:.2f}.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
        # 当验证集合的损失不再下降时，等待 patient 后停止训练
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

        #
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        #
        print "\n开始训练..."
        self.hist = self.model.fit_generator(
                        train_generator,
                        samples_per_epoch=900,
                        nb_epoch=200,
                        callbacks=[model_check, early_stopping],
                        validation_data=valid_generator,
                        nb_val_samples=120)

        # 保存日志并绘图显示
        log_file = open(self.dir_path + "hist.txt", "w")
        log_file.write(str(self.hist.history))
        log_file.close()
        self.plot_hist()

        print "训练完毕，权重保存...完毕！"

    #
    def plot_hist(self):
        """
            绘制图表明保存
        """
        #self.dir_path = "store_val_model_4/"
        file_log = open(self.dir_path + "hist.txt", "r").read().strip()
        #print "file_log = ", file_log
        #dic = self.hist.history
        #print dic['val_loss']
        dic = eval(file_log)

        x = [i for i in range(1, len(dic['loss'])+1)]

        y1 = dic['loss']
        y2 = dic['val_loss']
        y3 = dic['acc']
        y4 = dic['val_acc']

        plt.figure(figsize=(25, 10))
        plot1 = plt.subplot(121)  # 第一行的左图
        plt.ylim(0, 3.0)
        plt.xlabel("epoh")
        plt.ylabel("loss")

        plot2 = plt.subplot(122)  # 第一行的右图
        plt.ylim(0, 1.0)
        plt.xlabel("epoh")
        plt.ylabel("acc")

        plot1.plot(x, y1, label="train_loss", color="red")
        plot1.plot(x, y2, label="validation_loss", color="blue")
        plot1.legend()
        plot1.grid()

        plot2.plot(x, y3, label="train_acc", color="red")
        plot2.plot(x, y4, label="validation_acc", color="blue")
        plot2.legend()
        plot2.grid()

        plt.savefig(self.dir_path + 'train_val_loss_acc.jpg', dpi=500)

        #plt.show()


#
if __name__ == '__main__':
    vgg = vggClassify()
    vgg.build_vgg_model()
    vgg.get_bottlenect_feature()


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
