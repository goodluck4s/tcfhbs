# coding:utf-8
import sys

sys_param = sys.argv[1] if len(sys.argv) > 1 else ''

# log config
log_path = "../log/"
log_file_name = "app_log.log"
# log_debug_model=True

# flask config
host = "0.0.0.0"
port = 5000
is_debug = False

#
# base_model_path = "/Users/qianlai/Documents/_model/chinese_wwm_ext_pytorch"

test_dev_size=0.001
# test_dev_size=0.2
# 参数
MAX_LEN=128
BATCH_SIZE = 16
EVAL_BATCH_SIZE = BATCH_SIZE * 1

