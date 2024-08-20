from controller import controllerService

# hard-coded variables

level = 'Subclass' # this refers to the model hierarchical type used to obtain the list of companies' SSIC results.
topN = 3 # this refers to the top N number of SSIC codes that each companies SSIC codes should be within, to be considered as a hit (therefore classifying correctly). this score affects the prediction accuracy.
max_files = 100 # this refers to the max log files recoded in the repository.

## for Streamlit
section = 21 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
division = 81 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
group = 204 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
Class = 382 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
subclass = 1032 # this refers to the no. of SSIC codes in this hierarchy (from DoS).

modelResults = controllerService(level, topN, max_files)
logger = modelResults.setup_logger('main')
logger.info('Start code execution.')

modelResults.runValidatingClassificationModel(logger)
logger.info('Model classification completed. CSV generated for validation.')