import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from models.classificationModel.trainingClassificationModel import trainingClassificationModel
from models.classificationModel.validatingClassificationModel import validatingClassificationModel

class controllerService:

    def __init__(self, level = 'Subclass', topN = 3, maxFiles = 100):
        self.level = level
        self.topN = topN
        self.max_files = maxFiles

    def setup_logger(self, entryPoint = 'main'):

        # Create a logger
        logger = logging.getLogger("AppLogger")

        # Check if the logger already has handlers to avoid duplicate logs
        if not logger.handlers:

            logger.setLevel(logging.INFO)  # Set the minimum log level you want to track
        
            # Create a folder for logs if it doesn't exist
            log_folder = f"logs/{entryPoint}"
            os.makedirs(log_folder, exist_ok=True)

            # Generate a timestamp for the log file name
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = os.path.join(log_folder, f"{entryPoint}_{timestamp}.log")

            # Create a handler for writing log messages to a file
            handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=1)
            
            # Set the format for log messages
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            # Add the handler to the logger
            logger.addHandler(handler)

            # Add a stream handler to also log to the console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # Log uncaught exceptions
            def handle_exception(exc_type, exc_value, exc_traceback):
                if issubclass(exc_type, KeyboardInterrupt):
                    sys.__excepthook__(exc_type, exc_value, exc_traceback)
                    return
                logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

            sys.excepthook = handle_exception
            
            # List all log files in the log folder
            log_files = [f for f in os.listdir(log_folder) if f.endswith('.log')]
            log_files.sort()  # Sort files to ensure the oldest ones come first

            # Check if the number of log files exceeds the limit
            if len(log_files) > self.max_files:
                # Calculate how many files need to be deleted
                files_to_delete = log_files[:len(log_files) - self.max_files]
                
                # Delete the oldest files
                for file in files_to_delete:
                    os.remove(os.path.join(log_folder, file))

        return logger

    def runTrainingClassificationModel(self, logger):
        output = trainingClassificationModel(self, logger)
        return output

    def runValidatingClassificationModel(self, logger):
        validatingClassificationModel(self, logger)