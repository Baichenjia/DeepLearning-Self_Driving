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
def conn_thread(connection, address, num):
    while True:
        try:
            connection.settimeout(600)
            fileinfo_size = struct.calcsize('128sq')  # 首先接收文件头信息
            buf = connection.recv(fileinfo_size)
            if buf:
                filename, filesize = struct.unpack('128sq', buf)

                # 图片存储在服务器端
                file_write = "receive_pic_" + str(num) + ".jpg"

                print "接收到的文件名是: ", filename, " 文件大小是: ", filesize

                # 接收文件
                file = open(file_write, 'wb')
                print "开始接收文件...  ",
                recvd_size = long(0)
                while not recvd_size == filesize:
                    if filesize - recvd_size > 1024:
                        rdata = connection.recv(1024)
                        recvd_size += len(rdata)
                        print recvd_size,
                    else:
                        rdata = connection.recv(filesize - recvd_size)
                        recvd_size = filesize
                        print recvd_size, "  .down"
                    file.write(rdata)
                file.close()

                print filename, ' 接收完毕...',
                """
                img = Image.open(file_write)
                print "接收到的图片的尺寸为: ", img.size

                # compute
                msg = deep_network_compute(file_write)

                print "计算结果是 ", msg, "   返回给小车"
                print "------------------------\n"
                # send something
                connection.send(msg)
                """

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
        self.s.listen(5)

        # 服务器端存储图片
        self.num = 0

    # 神经网络前向传播，根据 picpath 计算方向
    def deep_network_compute(self, picpath):
        assert os.path.exists(picpath), "In server_trans_file.deep_network_compute picpath not found"

        res = self.vggpredict.predict_single_img(picpath)
        return res

    #
    # 监听请求，接收文件保存
    def conn_recv(self, conn, addr, num):
        while True:
            try:
                conn.settimeout(600)
                fileinfo_size = struct.calcsize('128sq')  # 首先接收文件头信息
                buf = conn.recv(fileinfo_size)
                if buf:
                    filename, filesize = struct.unpack('128sq', buf)

                    # 图片存储在服务器端
                    file_write = "receive_pic_" + str(num) + ".jpg"

                    print "接收到的文件名是: ", filename, " 文件大小是: ", filesize

                    # 接收文件
                    file = open(file_write, 'wb')
                    print "开始接收文件...  ",
                    recvd_size = long(0)
                    while not recvd_size == filesize:
                        if filesize - recvd_size > 1024:
                            rdata = conn.recv(1024)
                            recvd_size += len(rdata)
                            print recvd_size,
                        else:
                            rdata = conn.recv(filesize - recvd_size)
                            recvd_size = filesize
                            print recvd_size, "  .down"
                        file.write(rdata)
                    file.close()

                    print filename, ' 接收完毕...',
                    return True

            except socket.timeout:
                connection.close()

    #
    def start_recv(self):
        while True:
            self.connection, self.address = self.s.accept()
            print '已经与主机： ', self.address, "建立连接"

            flag = self.conn_recv(self.connection, self.address, self.num)

            # 进行 deepNet 计算
            if flag:
                file_write = "receive_pic_" + str(self.num) + ".jpg"
                img = Image.open(file_write)
                print "接收到的图片的尺寸为: ", img.size

                # compute
                msg = self.deep_network_compute(file_write)

                print "计算结果是 ", msg, "   返回给小车"
                # send something
                self.connection.send(str(msg))
                self.connection.close()
                print "成功..."
                print "------------------------\n"
            else:
                print "失败..."

            self.num += 1


if __name__ == '__main__':
    server = trans_server()
    server.start_recv()










