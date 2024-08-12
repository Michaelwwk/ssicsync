from models.classificationModel.trainingClassificationModel import trainingClassificationModel
from models.classificationModel.validatingClassificationModel import validatingClassificationModel

class controllerService:

    def __init__(self, level = 'Subclass', topN = 3):
        self.level = level
        self.topN = topN

    def runTrainingClassificationModel(self):
        output = trainingClassificationModel(self)
        return output

    def runValidatingClassificationModel(self):
        validatingClassificationModel(self)