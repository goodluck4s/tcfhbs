# coding:utf-8

import logging
import logging.handlers
import os
import properties

class new_logger(object):
    def __init__(self):
        self.__logger = logging.getLogger()

        # 创建文件目录
        logs_dir = properties.log_path
        if os.path.exists(logs_dir) and os.path.isdir(logs_dir):
            pass
        else:
            os.mkdir(logs_dir)

        # 日期log文件handler
        filehandler = logging.handlers.TimedRotatingFileHandler(logs_dir + properties.log_file_name, when='D', interval=1, backupCount=7)
        # 设置后缀名称，跟strftime的格式一样
        filehandler.suffix = "%Y-%m-%d_%H-%M-%S.log"
        # 设置输出格式
        formatter = logging.Formatter('[%(asctime)s] [%(threadName)s] [%(filename)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        filehandler.setFormatter(formatter)
        filehandler.setLevel(logging.INFO)

        # 控制台句柄
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.INFO)

        # 添加内容到日志句柄中
        self.__logger.addHandler(filehandler)
        self.__logger.addHandler(console)
        self.__logger.setLevel(logging.INFO)

        filehandler.close()
        console.close()

    # 获取日志实例
    def get_logger(self):
        return self.__logger

    # # _ide不可见 可调用不报错  __ide不可见 调用报错 因为改成_cls__xxx
    # def __print_name(self):
    #     print(__name__)


log_obj=new_logger().get_logger()


if __name__=="__main__":
    log_obj.info("aa%s   %s"%(123,"ddd"))
    log_obj.info("----> next?: {} | {}".format("sen1", "sen2"))
    log_obj.info("----> Concat List: size {}".format(len([1,23,4])))