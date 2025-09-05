import logging

# 日志等级：数字越大，输出门槛越低
DEBUG      = logging.DEBUG      # 10
INFO       = logging.INFO       # 20
SUGGESTION = 25                 # 25
ERROR      = logging.ERROR      # 30

# 添加自定义日志级别（建议）
logging.addLevelName(SUGGESTION, "SUGGESTION")

# 将 suggestion 方法添加到 logging.Logger 类中
def suggestion(self, message, *args, **kws):
    if self.isEnabledFor(SUGGESTION):
        self._log(SUGGESTION, message, args, **kws)
logging.Logger.suggestion = suggestion

# 设置日志格式和颜色
class ColoredFormatter(logging.Formatter):
    COLORS = {'DEBUG'      : '\033[36m',  # 蓝色
              'INFO'       : '\033[35m',  # 紫色
              'SUGGESTION' : '\033[92m',  # 绿色
              'ERROR'      : '\033[91m'}  # 红色
    RESET = '\033[0m'                     # 白色

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        formatted = super().format(record)
        formation = f"{color}{formatted}{self.RESET}"
        return formation

# 日志配置封装函数
def configure_logger(level=INFO):
    # 设置日志级别
    logger = logging.getLogger()
    logger.setLevel(level)

    # 设置日志处理器
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(fmt="[%(levelname)s] %(message)s"))
    logger.handlers = [handler]

# 日志使用
def set_log_level(level=INFO):
    configure_logger(level)
    return logging.getLogger()  # 返回配置好的 logger 对象

# 创建日志接口供外部调用
logger = set_log_level(level=INFO)
info = logger.info
debug = logger.debug
error = logger.error
suggestion = logger.suggestion