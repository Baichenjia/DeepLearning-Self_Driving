# -*- coding: utf-8 -*-

import numpy as np
import os
from client_trans_file import trans_client


# 检测到新图片即可送到服务器端，检测到
class transPic_RecvMsg():

    def __init__(self):
        self.max_auto_file = ""
        self.client = trans_client()
        self.auto_dir = r"C:\Users\Vehicle\Documents\build-vehicle_gui-Desktop_Qt_5_5_1_MSVC2012_32bit-Debug\auto_drive"

    #
    def detact_new_img(self):
        while (True):
            filelist = sorted(os.listdir(self.auto_dir), key=lambda s: int(s[6:-4]))
            if len(filelist) == 0:
                continue
            if filelist[-1] == self.max_auto_file:
                continue
            self.max_auto_file = filelist[-1]
            print "self.max_auto_file=", self.max_auto_file
            total_path = os.path.join(self.auto_dir, filelist[-1])
            print total_path
            assert os.path.exists(total_path), "not found max_auto_file path! "

            # trans pic
            self.client.send_pic(total_path)

            # receive msg
            y = self.client.receive_msg()

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
                print i + 1, ": prob=", prob, " label=", label, " label_mark=", label_mark

            # 新建文件
            auto_file = open(
                r"C:\Users\Vehicle\Documents\build-vehicle_gui-Desktop_Qt_5_5_1_MSVC2012_32bit-Debug\auto.txt", "w")
            auto_file.write(str(res[0]) + " " + str(y[res[0]]))
            auto_file.close()
            print "\n\n"

    #
    def clear_all_img(self):
        """
        清除上次保存的 .jpg 文件和上一个命令文件 auto.txt
        """
        del_autotxt = r"C:\Users\Vehicle\Documents\build-vehicle_gui-Desktop_Qt_5_5_1_MSVC2012_32bit-Debug"
        del_imgfile = del_autotxt + "\\auto_drive"
        del_dir_autotxt = os.listdir(del_autotxt)
        del_dir_imgfile = os.listdir(del_imgfile)

        if "auto.txt" in del_dir_autotxt:
            os.remove(os.path.join(del_autotxt, "auto.txt"))  # remove auto.txt
            print "remove file auto.txt .."
        for imgfile in del_dir_imgfile:
            os.remove(os.path.join(del_imgfile, imgfile))  # remove imgfile
        print "remove " + str(len(del_dir_imgfile)) + " img files ..\n\n"


#
if __name__ == '__main__':
    trans = transPic_RecvMsg()
    trans.clear_all_img()
    trans.detact_new_img()







