# -*- coding: utf-8 -*-

import socket
import time
import SocketServer
import struct
import os
import thread
from server_vgg_predict import vggPredict
from PIL import Image


# 新建一个线程，接收文件
def conn_thread(connection, address, num, deep_network_compute):
    while True:
        try:
            connection.settimeout(600)
            fileinfo_size = struct.calcsize('128sq')
            buf = connection.recv(fileinfo_size)
            if buf:
                filename, filesize = struct.unpack('128sq', buf)
                filename_f = filename.strip('\00')

                # 图片存储在服务器端
                filenewname = "receive_pic_" + str(num) + ".jpg"
                print 'file new name is %s, filesize is %s' %(filenewname, filesize)

                # 接收文件
                recvd_size = long(0)   # 定义接收了的文件大小
                file = open(filenewname, 'wb')
                print 'stat receiving...'
                while not recvd_size == filesize:
                    if filesize - recvd_size > 1024:
                        rdata = connection.recv(1024)
                        recvd_size += len(rdata)
                    else:
                        rdata = connection.recv(filesize - recvd_size)
                        recvd_size = filesize
                    file.write(rdata)
                file.close()
                print filename, ' 接收完毕...'
                while (True):
                    try:
                        img = Image.open(filenewname)
                        if img.size[0] > 0 and img.size[1] > 0:
                            print "接收到的图片的尺寸为: ", img.size
                            break
                    except:
                        pass
                # compute
                msg = deep_network_compute(filenewname)
                print "计算结果是 ", msg, "   返回给小车"
                print "------------------------\n"

                # send something
                connection.send(msg)

        except socket.timeout:
            connection.close()


class trans_server:
    def __init__(self):
        # 载入训练好的VGGmodel
        self.vggpredict = vggPredict()
        self.vggpredict.build_model()

        # 连接
        print "配置服务器地址和端口...监听请求..."
        #host = '202.118.230.155'
        self.host = "192.168.99.253"
        self.port = 12307
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)    # 定义socket类型
        self.s.bind((self.host, self.port))    # 绑定需要监听的Ip和端口号，tuple格式
        self.s.listen(1)

        # 服务器端存储图片
        self.num = 0

    # 神经网络前向传播，根据 picpath 计算方向
    def deep_network_compute(self, picpath):
        assert os.path.exists(picpath), "In server_trans_file.deep_network_compute picpath not found"

        res = self.vggpredict.predict_single_img(picpath)
        return res

    #
    def start_recv(self):
        while True:
            self.connection, self.address = self.s.accept()
            print '已经与主机： ', self.address, "建立连接"
            thread.start_new_thread(conn_thread, (self.connection, self.address, self.num, self.deep_network_compute))
            self.num += 1


if __name__ == '__main__':
    server = trans_server()
    server.start_recv()










