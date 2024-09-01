from controller import controllerService

# hard-coded variables
lvl_train = 'Group' # this refers to the SSIC hierarchical model to be trained ('Section', 'Division', 'Group', 'Class', 'Subclass').
max_files = 100
learningRate = 5e-5
epsilon = 1e-08
patience = 3
shuffle = 1000
batch = 16
epochs = 3
numLabels = 1032
testSize = 0.2
randomState = 0
ssic_detailed_def_filepath = "dataSources/DoS/ssic2020-detailed-definitions.xlsx"
ssic_alpha_index_filepath = "dataSources/DoS/ssic2020-alphabetical-index.xlsx"

modelTraining = controllerService(maxFiles = max_files, learningRate = learningRate, epsilon = epsilon,
                                  patience = patience, shuffle = shuffle, batch = batch, epochs = epochs,
                                  numLabels = numLabels, testSize = testSize, randomState = randomState,
                                  lvl_train = lvl_train)

logger = modelTraining.setup_logger('training')
logger.info(f"Start training '{lvl_train}' model ...")

modelTraining.runTrainingClassificationModel(logger, ssic_detailed_def_filepath, ssic_alpha_index_filepath)
logger.info(f"'{lvl_train}' model training completed. Model file generated.")