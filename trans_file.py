# -*- coding: utf-8 -*-

import socket
import os
import struct


class trans_client:
    def __init__(self):
        self.ip = "202.118.230.155"
        self.port = 12307
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        print "连接服务器 : ", self.ip, " 端口: ", str(self.port), "  完毕..."

    def send_pic(self, filepath="test.jpg"):
        print "start send file:", filepath
        if os.path.isfile(filepath):
            fileinfo_size = struct.calcsize('128sl')     # 定义打包规则
            # 定义文件头信息，包含文件名和文件大小
            fhead = struct.pack('128sl', os.path.basename(filepath), os.stat(filepath).st_size)
            self.sock.send(fhead)
            # 发送文件信息，每次发送1024个字节
            fo = open(filepath, 'rb')
            while True:
                filedata = fo.read(1024)
                if not filedata:
                    break
                self.sock.send(filedata)
            fo.close()
            print 'send over...'

    def receive_msg(self):
        data = self.sock.recv(1024)
        print "received data:", data
        return str(data)


if __name__ == '__main__':
    client = trans_client()
    client.send_pic()
    data = client.receive_msg()

    # 关闭 tcp 连接
    client.sock.close()








