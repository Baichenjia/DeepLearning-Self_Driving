# -*- coding: utf-8 -*-

import re
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_hist(pic_dir="weights_conv_block4/"):
    #绘制图表明保存
    file_log = open(pic_dir + "hist.txt", "r").read().strip()
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

    plt.savefig(pic_dir + 'train_val_loss_acc.jpg', dpi=500)
    print "日志 和 准确率图像 保存完毕！"
    plt.show()


def clear_pyc_img():
    """
        删除当前目录下存在的 pyc 文件和 img 文件
    """
    file_list = os.listdir("./")
    pattern_img = re.compile(r"^receive_\w+.jpg$")
    pattern_pyc = re.compile(r"^\w+.pyc$")

    for filename in file_list:
        m_img = re.match(pattern_img, filename)
        m_pyc = re.match(pattern_pyc, filename)

        if m_img is None and m_pyc is None:
            print "None: ", filename
        elif m_img is not None:
            print "image: ", m_img.string
        else:
            print "pyc: ", m_pyc.string

#
if __name__ == '__main__':
    plot_hist()
    #clear_pyc_img()




