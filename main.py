import sys
from controller import controllerService

# hard-coded variables
level = 'Subclass'
topN = 3

modelResults = controllerService(level, topN)
modelResults.runValidatingClassificationModel()