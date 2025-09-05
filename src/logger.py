import logging

DEBUG      = logging.DEBUG      # 10
INFO       = logging.INFO       # 20
SUGGESTION = 25                 # 25
ERROR      = logging.ERROR      # 30

# 添加自定义日志级别
logging.addLevelName(SUGGESTION, "SUGGESTION")

def suggestion(self, message, *args, **kws):
    if self.isEnabledFor(SUGGESTION):
        self._log(SUGGESTION, message, args, **kws)

# 将 suggestion 方法添加到 logging.Logger 类中
logging.Logger.suggestion = suggestion

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG'      : '\033[36m',  # Cyan
        'INFO'       : '\033[35m',  # Purple
        'ERROR'      : '\033[91m',  # Red
        'SUGGESTION' : '\033[92m',  # Green
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        formatted = super().format(record)
        return f"{color}{formatted}{self.RESET}"

# 日志配置封装函数
def configure_logger(level=INFO):
    logger = logging.getLogger()
    logger.setLevel(level)  # 设置日志级别
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(fmt="[%(levelname)s] %(message)s"))
    logger.handlers = [handler]  # 设置日志处理器

# 简化后的日志使用
def setup_logger(level=INFO):
    configure_logger(level)
    return logging.getLogger()  # 返回配置好的 logger 对象

# 创建日志接口供外部调用
logger = setup_logger(level=INFO)
info = logger.info
debug = logger.debug
error = logger.error
suggestion = logger.suggestion