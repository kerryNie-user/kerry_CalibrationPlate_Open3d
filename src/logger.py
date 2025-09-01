import logging

DEBUG      = logging.DEBUG  # 10
INFO       = logging.INFO   # 20
SUGGESTION = 25             # 25
ERROR      = logging.ERROR  # 30

logging.addLevelName(SUGGESTION, "SUGGESTION")

def suggestion(self, message, *args, **kws):
    if self.isEnabledFor(SUGGESTION):
        self._log(SUGGESTION, message, args, **kws)

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

# 配置 logging
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(fmt="[%(levelname)s] %(message)s"))
logging.getLogger().handlers = [handler]

# 分配日志函数
info = logging.info
debug = logging.debug
error = logging.error
suggestion = logging.getLogger().suggestion
