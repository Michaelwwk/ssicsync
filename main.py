# hard-coded variables
level = 'Group' # ('Section', 'Division', 'Group', 'Class', 'Subclass') ...
# ... this refers to chosen SSIC hierarchical model to obtain the list of companies' SSIC results.
modelChoice = 'fb_bart_tfidf' # ('sd_azma_bart', 'azma_bart_tfidf', 'sd_philschmid_bart', 'philschmid_bart_tfidf', 'sd_fb_bart', 'fb_bart_tfidf', 'QA') ...
# ... this refers to the chosen summary model of choice.
resultsLevel = 'Group' # this refers to the hierarchical level to be seen in final results file ...
# ... 'resultsLevel' has to be equal or less granular than 'level'! ('Section', 'Division', 'Group', 'Class', 'Subclass')
topN = 3 # this refers to the top N number of SSIC codes that each companies SSIC codes ...
# ... should be within to be considered as a hit (therefore classifying correctly). this score affects the prediction accuracy.
max_files = 100 # this refers to the max log files recoded in the repository.
ssic_detailed_def_filepath = "dataSources/DoS/ssic2020-detailed-definitions.xlsx"
ssic_alpha_index_filepath = "dataSources/DoS/ssic2020-alphabetical-index.xlsx"
companies_filepath = "dataSources/input_listOfCompanies.csv"

## specifically for Streamlit
binSize = 10 # this setting is to adjust the bin size of Adjusted Score's histogram chart
section = 21 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
division = 81 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
group = 204 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
Class = 382 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
subclass = 1032 # this refers to the no. of SSIC codes in this hierarchy (from DoS).

if __name__ == "__main__":

    from controller import controllerService

    modelResults = controllerService(level = level, topN = topN, maxFiles = max_files,
                                    modelChoice = modelChoice, resultsLevel = resultsLevel)

    logger = modelResults.setup_logger('main')
    logger.info('Start code execution ...')

    modelResults.runPdfScraping(logger)
    modelResults.runTrainingSummaryModel(logger)
    modelResults.runValidatingClassificationModel(logger, ssic_detailed_def_filepath, ssic_alpha_index_filepath, companies_filepath)