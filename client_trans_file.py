# -*- coding: utf-8 -*-

import socket
import os
import struct
from PIL import Image


class trans_client:
    def __init__(self):
        self.ip = "202.118.230.155"
        self.port = 12307

    def send_pic(self, filepath):
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        print "连接服务器 : ", self.ip, " 端口: ", str(self.port), "  完毕..."
        filesize = os.stat(filepath).st_size
        print "start send file:", filepath, "大小 ", filesize
        if os.path.isfile(filepath):
            #fileinfo_size = struct.calcsize('128sq')     # 定义打包规则
            # 定义文件头信息，包含文件名和文件大小
            fhead = struct.pack('128sq', os.path.basename(filepath), filesize)

            self.sock.send(fhead)
            # 发送文件信息，每次发送1024个字节
            fo = open(filepath, 'rb')
            while True:
                filedata = fo.read(1024)
                if not filedata:
                    break
                self.sock.send(filedata)
            fo.close()
            print 'send over...  文件大小 ', filesize

            # 接收返回的指令
            msg_size = struct.calcsize('128sl')  # 首先接收文件头信息
            print "msg_size = ", msg_size
            buf = self.sock.recv(msg_size)
            data, lenth = struct.unpack('128sl', buf)

            # 错误处理，lenth=-1表示服务器端发生了 IO 错误
            if lenth == -1:
                return -1

            data = data[:lenth]
            print type(data), data

            data_list = [float(val) for val in data.strip().split()]
            print "received data:", type(data_list), data_list
            self.sock.close()
            return data_list


if __name__ == '__main__':
    import time
    client = trans_client()
    result = client.send_pic("test.jpg")
    time.sleep(2)
    result = client.send_pic("test.jpg")
    time.sleep(2)
    result = client.send_pic("test.jpg")

    # 关闭 tcp 连接
    client.sock.close()








