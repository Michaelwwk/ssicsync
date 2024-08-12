import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from datetime import datetime

# hard-coding
max_files = 100

def setup_logger():
    
    # Create a folder for logs if it doesn't exist
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)

    # Generate a timestamp for the log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_folder, f"{timestamp}.log")

    # Create a logger
    logger = logging.getLogger("AppLogger")
    logger.setLevel(logging.INFO)  # Set the minimum log level you want to track

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
    
    # Maintain only the latest 100 log files
    manage_log_files(log_folder, max_files)

    return logger

def manage_log_files(log_folder, max_files=100):
    # List all log files in the log folder
    log_files = [f for f in os.listdir(log_folder) if f.endswith('.log')]
    log_files.sort()  # Sort files to ensure the oldest ones come first

    # Check if the number of log files exceeds the limit
    if len(log_files) > max_files:
        # Calculate how many files need to be deleted
        files_to_delete = log_files[:len(log_files) - max_files]
        
        # Delete the oldest files
        for file in files_to_delete:
            os.remove(os.path.join(log_folder, file))

# Initialize the logger
logger = setup_logger()