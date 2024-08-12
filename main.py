from controller import controllerService

# hard-coded variables
level = 'Subclass'
topN = 3
max_files = 100

modelResults = controllerService(level, topN, max_files)
logger = modelResults.setup_logger('main')
logger.info('Start code execution.')

modelResults.runValidatingClassificationModel(logger)
logger.info('Model classification completed. CSV generated for validation.')