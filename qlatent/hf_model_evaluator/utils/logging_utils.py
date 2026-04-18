import logging

class LoggingUtils:
    def __init__(self, level=logging.WARNING):
        self.level = level
        self.configure_logging()

    def configure_logging(self):
        logging.basicConfig(
            level=self.level,
            format="%(levelname)s: %(message)s",
            # handlers=[
            #     logging.FileHandler(self.log_file),
            #     logging.StreamHandler()
            # ]
        )

    def log_warning(self, message):
        logging.warning(message)

    def log_error(self, message):
        logging.error(message)

    def log_info(self, message):
        logging.info(message)
        
        
