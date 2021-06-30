import os
import logging

from root import ROOT_DIR


class Logger:

    def __init__(self, log_file):
        folder = 'emva1288/tests/logs/'
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s;%(levelname)s;%(message)s', '%Y-%m-%d %H:%M:%S')
        self.log_file = log_file

        if not os.path.exists(folder):
            os.makedirs(folder)
        self.log_file_path = os.path.join(ROOT_DIR, folder + self.log_file)

        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def log(self, message):
        self.logger.info(message)
