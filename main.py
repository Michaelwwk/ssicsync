from logger import logger
from controller import controllerService

# hard-coded variables
level = 'Subclass'
topN = 3

logger.info('Start code execution.')
modelResults = controllerService(level, topN)
modelResults.runValidatingClassificationModel()
logger.info('Model classification completed. CSV generated for validation.')