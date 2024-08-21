from controller import controllerService

# hard-coded variables
lvl_train = 'Group' # this refers to the model hierarchical type to be trained ('Section', 'Division', 'Group', 'Class', 'Subclass')
max_files = 100
learningRate = 5e-5
epsilon = 1e-08
patience = 3
shuffle = 1000
batch = 16
epochs = 3
numLabels = 1032
testSize = 0.01
randomState = 0

modelTraining = controllerService(maxFiles = 100, learningRate = 5e-5, epsilon = 1e-08,
                                  patience = 3, shuffle = 1000, batch = 16, epochs = 3,
                                  numLabels = 1032, testSize = 0.01, randomState = 0,
                                  lvl_train = 'Group')

logger = modelTraining.setup_logger('training')
logger.info(f"Start training '{lvl_train}' model ...")

modelTraining.runTrainingClassificationModel(logger)
logger.info(f'{lvl_train} model training completed. Model file generated.')