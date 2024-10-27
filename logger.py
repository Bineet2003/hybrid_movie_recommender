import logging

class Logger:
    def __init__(self):
        log_file = "app.log"
        logging.basicConfig(
            filename=log_file,
            format='%(asctime)s %(levelname)s:%(message)s',
            level=logging.INFO
        )

    @staticmethod
    def log_info(message: str):
        logging.info(message)

    @staticmethod
    def log_error(message: str):
        logging.error(message)
