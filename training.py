from controller import controllerService

# hard-coded variables
max_files = 100

modelTraining = controllerService(maxFiles = max_files)
logger = modelTraining.setup_logger('training')
logger.info('Start code execution.')

modelTraining.runTrainingClassificationModel(logger)
logger.info('Model training completed. h5 file generated.')