import pandas as pd
import numpy as np
import math as m
import ast
import tensorflow as tf
from commonFunctions import ssic_df, capitalize_sentence
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

pd.set_option('display.max_columns', None)

def validatingClassificationModel(self, logger, ssic_detailed_def_filepath, ssic_alpha_index_filepath, companies_filepath):

    # hard-coded values:

    level = self.level
    topN = self.topN
    resultsLevel = self.resultsLevel
    modelChoice = self.modelChoice

    vdf_filepath = "models/summaryModel/modelOutputFiles/pdfModelSummaryOutputs.csv"
    pdfModelFinalOutputs_filepath = 'models/classificationModel/modelOutputFiles/pdfModelFinalOutputs.csv'
    overallResults_filepath = f'results/results_summaryModel{modelChoice}_classificationModel{level}_results{resultsLevel}_top{topN}.xlsx'

    # funtions:
    
    # Define the function to predict scores and categories
    def predict_text(text, tokenizer, model):
        # Ensure the input text is a string and check if it is blank
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Tokenize the input text
        predict_input = tokenizer.encode(
            text,
            truncation=True,
            padding=True,
            return_tensors="tf"
        )
        
        # Get the model output
        output = model(predict_input)[0]
        output_array = output.numpy()[0] 
        
        # Get the probabilities
        probs = tf.nn.softmax(output_array)
        
        # Get the top 10 predicted classes and their confidence scores
        top_10_indices = tf.argsort(probs, direction='DESCENDING')[:topN].numpy()
        return tuple(int(idx) for idx in top_10_indices)

        # top_10_probs = tf.gather(probs, top_10_indices).numpy()
        # top_10_predictions = [(int(idx), float(prob)) for idx, prob in zip(top_10_indices, top_10_probs)]
        
        # return top_10_predictions

    def apply_model_to_column(df, input_col, output_col):
    
        def map_values(value_list):
            # Prepare the merged DataFrame
            lvl_dict = df_prep[[lvl_train, 'encoded_cat']].drop_duplicates()
            lvl_ref = ssic_lvl[[lvl_train, lvl_train_title]].drop_duplicates()
            merged_df = lvl_dict.merge(lvl_ref, on=lvl_train, how='left')
            
            # Create a mapping dictionary from the reference table
            mapping_dict = dict(zip(merged_df['encoded_cat'], merged_df[lvl_train]))
            
            return [mapping_dict.get(item, item) for item in value_list]
        
        def predict_and_map(text):
            predictions = predict_text(text, tokenizer, model)
            return map_values(predictions)
        
        # Apply the predict_and_map function to the specified column and store results in a new column
        df[output_col] = df[input_col].apply(predict_and_map)
        return df
    
    # # Function to create the combined title column
    # def get_combined_title(row):
    #     title1 = ssic_5_dict.get(row['ssic_code'], 'Unknown')
    #     title2 = ssic_5_dict.get(row['ssic_code2'], 'Unknown')
    #     return f"{row['ssic_code']}: {title1}\n{row['ssic_code2']}: {title2}"
    
    def check_section(row, ref_dict, prediction_col_name):
        # Retrieve the list of predictions from the specified column
        predictions = row[prediction_col_name]
        # Check if the list is empty or null
        if not predictions:
            return None
        
        if level != 'Section':
            mapped_predictions = [ref_dict.get(str(pred)[:2]) for pred in row[prediction_col_name] if str(pred)[:2] in ref_dict]
        else:
            mapped_predictions = row[prediction_col_name]

        if row['Section'] in mapped_predictions or row['Section2'] in mapped_predictions:
            return 'Y'
        else:
            return 'N'

    def check_division(row, prediction_col_name):
        # Retrieve the list of predictions from the specified column
        predictions = row[prediction_col_name]
            # Check if the list is empty or null
        if not predictions:
            return None
        # Check if the first 2 characters of any item in predictions match either Group or Group2
        return 'Y' if any(item[:2] == row['Division'] or item[:2] == row['Division2'] for item in row[prediction_col_name]) else 'N'

    def check_group(row, prediction_col_name):
        # Retrieve the list of predictions from the specified column
        predictions = row[prediction_col_name]
            # Check if the list is empty or null
        if not predictions:
            return None
        # Check if any item in predictions matches either Division or Division2
        return 'Y' if any(item[:3] == row['Group'] or item[:3] == row['Group2'] for item in row[prediction_col_name]) else 'N'

    def check_class(row, prediction_col_name):
        # Retrieve the list of predictions from the specified column
        predictions = row[prediction_col_name]
            # Check if the list is empty or null
        if not predictions:
            return None
        # Check if any item in predictions matches either Division or Division2
        return 'Y' if any(item[:4] == row['Class'] or item[:4] == row['Class2'] for item in row[prediction_col_name]) else 'N'

    def check_subclass(row, prediction_col_name):
        # Retrieve the list of predictions from the specified column
        predictions = row[prediction_col_name]
            # Check if the list is empty or null
        if not predictions:
            return None
        # Check if any item in predictions matches either Division or Division2
        return 'Y' if any(item[:5] == row['Sub-class'] or item[:5] == row['Sub-class2'] for item in row[prediction_col_name]) else 'N'
    
    def map_and_capitalize(ssic_list):
        return [capitalize_sentence(ssic_to_title.get(ssic, np.NaN)) if pd.notna(ssic_to_title.get(ssic, np.NaN)) else np.NaN for ssic in ssic_list]

    # Scoring functions
    def calculateInverseScore(row, scoreCurve, refTable):
        """
        Parameters
        ----------
        row : Series
            One row of data from dataframe.
            
        scoreCurve : int
            Determine which score formula to use. See below.

        Returns
        -------
        Series: a series of scoring for Division, Group, Class and Sub-class.
        
        Description
        -------
        Calculates scoring of various groups based on the below rank-score formula for correct prediction:
            1 == 1/rank
            2 == 1/(rank^2)
            3 == 1/(rank^3)
            4 == 1/exp(rank)
            5 == 1/log2(rank + 1)
            
        Additionally the following score curve is used for category wise scoring:
            Section == 1/log2(5 + 1)
            Division == 1/log2(4 + 1)
            Group == 1/log2(3 + 1)
            Class == 1/log2(2 + 1)
            Subclass == 1/log2(1 + 1)
            
        Hence combined score for a prediction row = sum all (rank scores x respective category score)
        then after which the score would be normalised against the maximum possible combined score 
        for that rank row. Then a single score would be obtained by filtering the max score
        amongst the 15 rank-category combined scores.
        Max combined scores for each respective rank rows (DCG curve) are as follows:
            Prediction Row 1 = 2.948459119
            Prediction Row 2 = 1.860270585
            Prediction Row 3 = 1.474229559
            Prediction Row 4 = 1.269832225
            Prediction Row 5 = 1.140619687
            Prediction Row 6 = 1.050262329
            Prediction Row 7 = 0.982819706
            Prediction Row 8 = 0.930135293
            Prediction Row 9 = 0.887574636
            Prediction Row 10 = 0.852295823
            Prediction Row 11 = 0.822451872
            Prediction Row 12 = 0.796786151
            Prediction Row 13 = 0.774411417
            Prediction Row 14 = 0.754681772
            Prediction Row 15 = 0.73711478
        """
        # Define variable for adjusted score: max of ssic 1 and ssic 2 norm scores
        max12Score = 0       
        
        # Define k value for SSIC category
        kSection = 5
        kDivision = 4
        kGroup = 3
        kClass = 2
        kSubclass = 1
        
        # Read each to-be-score column values    
        mSection = row['Section']
        mSection2 = row['Section2']
        
        mDivision = row['Division']
        mDivision2 = row['Division2']
        
        mGroup = row['Group']
        mGroup2 = row['Group2']
        
        mClass = row['Class']
        mClass2 = row['Class2']
        
        mSubclass = row['Sub-class']
        mSubclass2 = row['Sub-class2']
        
        predictedList = row[f'p_{modelChoice}']   # extract predicted ssic codes
        
        for i in range(len(predictedList)):
            rank = i + 1
            
            # import pdb;pdb.set_trace() # can ensure that line 225 (pSection) is correct? if model is chosen as Section, predictedList will be all alphabets. if model is division, its 2 digits, group 3 digits, subclass 5 digits, etc. logic here has to be dynamic!

            pDivision = predictedList[i][0:2]
            pSection = refTable[refTable['Section, 2 digit code'] == pDivision]['Section'].to_string(index = False) # TODO
            pGroup = predictedList[i][2:3]
            pClass = predictedList[i][3:4]
            pSubclass = predictedList[i][4:5]
            
            # Determine rank score curve
            if scoreCurve == 4: 
                detnomRank = m.exp(rank) # k value = Rank
                detnomSection = m.exp(kSection)        # Section k value = 5
                detnomDivision = m.exp(kDivision)       # Division k value = 4
                detnomGroup = m.exp(kGroup)          # Group k value = 3
                detnomClass = m.exp(kClass)          # Class k value = 2
                detnomSubclass = m.exp(kSubclass)       # Subclass k value = 1
                
                maxScore = sum([1 / detnomRank * 1 / detnomSection, 
                                1 / detnomRank * 1 / detnomDivision, 
                                1 / detnomRank * 1 / detnomGroup, 
                                1 / detnomRank * 1 / detnomClass, 
                                1 / detnomRank * 1 / detnomSubclass])
                
            elif scoreCurve == 5: 
                detnomRank = m.log2(rank + 1)# k value = Rank
                detnomSection = m.log2(kSection + 1)       # Section k value = 5
                detnomDivision = m.log2(kDivision + 1)      # Division k value = 4
                detnomGroup = m.log2(kGroup + 1)         # Group k value = 3
                detnomClass = m.log2(kClass + 1)         # Class k value = 2
                detnomSubclass = m.log2(kSubclass + 1)      # Subclass k value = 1
                
                maxScore = sum([1 / detnomRank * 1 / detnomSection, 
                                1 / detnomRank * 1 / detnomDivision, 
                                1 / detnomRank * 1 / detnomGroup, 
                                1 / detnomRank * 1 / detnomClass, 
                                1 / detnomRank * 1 / detnomSubclass])
                
            else: 
                detnomRank = m.pow(rank, scoreCurve) # k value = Rank
                detnomSection = m.pow(kSection, scoreCurve)        # Section k value = 5
                detnomDivision = m.pow(kDivision, scoreCurve)       # Division k value = 4
                detnomGroup = m.pow(kGroup, scoreCurve)          # Group k value = 3
                detnomClass = m.pow(kClass, scoreCurve)          # Class k value = 2
                detnomSubclass = m.pow(kSubclass, scoreCurve)       # Subclass k value = 1
                
                maxScore = sum([1 / detnomRank * 1 / detnomSection, 
                                1 / detnomRank * 1 / detnomDivision, 
                                1 / detnomRank * 1 / detnomGroup, 
                                1 / detnomRank * 1 / detnomClass, 
                                1 / detnomRank * 1 / detnomSubclass])
                
            # Score for Section prediction
            if mSection == pSection: 
                pSectionScore = 1/detnomRank * 1/detnomSection
            else: 
                pSectionScore = 0
                
            if mSection2 == pSection: 
                pSection2Score = 1/detnomRank * 1/detnomSection
            else: 
                pSection2Score = 0
                
            # Score for Division prediction
            if mDivision == pDivision: 
                pDivisionScore = 1/detnomRank * 1/detnomDivision
            else: 
                pDivisionScore = 0
                
            if mDivision2 == pDivision: 
                pDivision2Score = 1/detnomRank * 1/detnomDivision
            else: 
                pDivision2Score = 0
                
            # Score for Group prediction
            if mGroup == pGroup: 
                pGroupScore = 1/detnomRank * 1/detnomGroup
            else: 
                pGroupScore = 0
                
            if mGroup2 == pGroup: 
                pGroup2Score = 1/detnomRank * 1/detnomGroup
            else: 
                pGroup2Score = 0
                
            # Score for Class prediction
            if mClass == pClass: 
                pClassScore = 1/detnomRank * 1/detnomClass
            else: 
                pClassScore = 0
                
            if mClass2 == pClass: 
                pClass2Score = 1/detnomRank * 1/detnomClass
            else: 
                pClass2Score = 0
            
            # Score for Sub-class prediction
            if mSubclass == pSubclass: 
                pSubclassScore = 1/detnomRank * 1/detnomClass
            else: 
                pSubclassScore = 0
                
            if mSubclass2 == pSubclass: 
                pSubclass2Score = 1/detnomRank * 1/detnomClass
            else: 
                pSubclass2Score = 0
                
            # Computing adjusted score
            combinedScore = sum((pSectionScore, pDivisionScore, pGroupScore, pClassScore, pSubclassScore))
            combinedScore2 = sum((pSection2Score, pDivision2Score, pGroup2Score, pClass2Score, pSubclass2Score))
            combinedScoreNorm = combinedScore / maxScore    # normalising scores to max score
            combinedScore2Norm = combinedScore2 / maxScore  # normalising scores to max score
            
            # Determining adjusted score
            if combinedScoreNorm >= combinedScore2Norm: 
                if combinedScoreNorm > max12Score: 
                    max12Score = combinedScoreNorm
                    
            else: 
                if combinedScore2Norm > max12Score:
                    max12Score = combinedScore2Norm
                    
        return max12Score

    def calculatePredictionScore(vdf, refTable): 
        df = vdf[['UEN Number', 'ssic_code', 'ssic_code2', 'Section', 'Division', 'Group', 
                 'Class', 'Sub-class', 'Section2', 'Division2', 'Group2', 'Class2', 
                 'Sub-class2', f'p_{modelChoice}']]
        
        # Apply function to calculate scores
        predictionScore = df.apply(calculateInverseScore, axis = 1, scoreCurve = 5, refTable = refTable)
        predictionScore.rename('adjusted_score', inplace = True)
        vdf = pd.concat([vdf, predictionScore], axis = 1)
        
        return vdf

    ####################################################################################################
    ### Select SSIC Hierarchical Level

    # 1. 'Section'
    # 2. 'Division'
    # 3. 'Group'
    # 4. 'Class'
    # 5. 'Subclass'

    ####################################################################################################

    ssic_1, ssic_2, ssic_3, ssic_4, ssic_5, ssic_dataframe = ssic_df(ssic_detailed_def_filepath, ssic_alpha_index_filepath)

    # mapping
    level_map = {
        'Section': ('Section', ssic_dataframe.iloc[:, [0, 1, 9, 10, 11, 12, 13]].\
                    drop_duplicates(), ssic_1.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True), "nusebacra/ssicsync_section_classifier", ssic_1),
        'Division': ('Division', ssic_dataframe.iloc[:, [0, 1, 6, 10, 11, 12, 13]].\
                     drop_duplicates(), ssic_2.iloc[:, [0, 1]].drop_duplicates().\
                        reset_index(drop=True), "nusebacra/ssicsync_division_classifier", ssic_2),
        'Group': ('Group', ssic_dataframe.iloc[:, [0, 1, 7, 10, 11, 12, 13]].drop_duplicates(), ssic_3.iloc[:, [0, 1]].drop_duplicates().\
                  reset_index(drop=True), "nusebacra/ssicsync_group_classifier", ssic_3),
        'Class': ('Class', ssic_dataframe.iloc[:, [0, 1, 8, 10, 11, 12, 13]].drop_duplicates(), ssic_4.iloc[:, [0, 1]].drop_duplicates().\
                  reset_index(drop=True), "nusebacra/ssicsync_class_classifier", ssic_4),
        'Subclass': ('SSIC 2020', ssic_dataframe.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates(), ssic_5.iloc[:, [0, 1]].drop_duplicates().\
                     reset_index(drop=True), "nusebacra/ssicsync_subclass_classifier", ssic_5)
    }

    # Get the values for a and b based on the lvl_train
    lvl_train, df_streamlit, ssic_n_sl, model, ssic_lvl = level_map.get(level, ('default_a', 'default_b', 'default_c', 'default_d', 'default_e', 'default_f'))
    lvl_train_title = lvl_train + " Title"

    # prep ssic_n dictionary df_prep
    df_prep = ssic_dataframe[[lvl_train, 'Detailed Definitions']]
    df_prep['encoded_cat'] = df_prep[lvl_train].astype('category').cat.codes
    df_prep = df_prep[[lvl_train, 'encoded_cat']].drop_duplicates()

    # load model directly from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = TFAutoModelForSequenceClassification.from_pretrained(model)
    list_df = pd.read_csv(companies_filepath, dtype = str)

    # Create new columns
    list_df['Division'] = list_df['ssic_code'].str[:2]
    list_df['Group'] = list_df['ssic_code'].str[:3]
    list_df['Class'] = list_df['ssic_code'].str[:4]
    list_df['Sub-class'] = list_df['ssic_code']

    list_df['Division2'] = list_df['ssic_code2'].str[:2]
    list_df['Group2'] = list_df['ssic_code2'].str[:3]
    list_df['Class2'] = list_df['ssic_code2'].str[:4]
    list_df['Sub-class2'] = list_df['ssic_code2']

    list_df = list_df.merge(ssic_1[['Section, 2 digit code', 'Section']], left_on='Division', right_on='Section, 2 digit code', how='left')
    list_df = list_df.rename(columns={'Section': 'Section'})
    list_df = list_df.merge(ssic_1[['Section, 2 digit code', 'Section']], left_on='Division2', right_on='Section, 2 digit code', how='left', suffixes=('', '2'))
    list_df = list_df.rename(columns={'Section2': 'Section2'})

    # Validation Data
    vdf = pd.read_csv(vdf_filepath, dtype = str)
    vdf = vdf.merge(list_df[['UEN', 'ssic_code', 'ssic_code2', 'Section', 'Division', 'Group', 'Class',\
                             'Sub-class', 'Section2', 'Division2', 'Group2', 'Class2', 'Sub-class2']], left_on='UEN Number', right_on='UEN', how='left')

    # # Replace empty strings with NaN
    # vdf = vdf.replace('', None)
    # # Drop rows with any NaN values
    # vdf = vdf.dropna()

    # # Create a dictionary for quick lookup for ssic_5

    # ssic_5_dict = ssic_5[['SSIC 2020', 'SSIC 2020 Title']].drop_duplicates().set_index('SSIC 2020')['SSIC 2020 Title'].to_dict()

    # # Apply the function to create the new column
    # vdf['ssic_code&title'] = vdf.apply(get_combined_title, axis=1)
    # vdf['ssic_code&title'] = vdf.apply(get_combined_title, axis=1)

    if modelChoice == 'sd_azma_bart':
        vdf = apply_model_to_column(vdf, 'Summarized_Description_azma_bart', 'p_sd_azma_bart')
    elif modelChoice == 'sd_fb_bart':
        vdf = apply_model_to_column(vdf, 'Summarized_Description_facebook_bart', 'p_sd_fb_bart')
    elif modelChoice == 'sd_philschmid_bart':
        vdf = apply_model_to_column(vdf, 'Summarized_Description_philschmid_bart', 'p_sd_philschmid_bart')
    elif modelChoice == 'azma_bart_tfidf':
        vdf = apply_model_to_column(vdf, 'Azma_bart_tfidf', 'p_azma_bart_tfidf')
    elif modelChoice == 'fb_bart_tfidf':
        vdf = apply_model_to_column(vdf, 'FB_bart_tfidf', 'p_fb_bart_tfidf')
    elif modelChoice == 'philschmid_bart_tfidf':
        vdf = apply_model_to_column(vdf, 'Philschmid_bart_tfidf', 'p_philschmid_bart_tfidf')
    elif modelChoice == 'QA':
        vdf = apply_model_to_column(vdf, 'Q&A model Output', 'p_QA')

    vdf.drop(columns = ['Summarized_Description_azma_bart', 'Summarized_Description_facebook_bart',
                        'Summarized_Description_philschmid_bart', 'Azma_bart_tfidf',
                        'FB_bart_tfidf', 'Philschmid_bart_tfidf', 'Q&A model Output', 'Summarised?', 'UEN'], inplace = True)

    ########################################################################## Define functions to check conditions
    # Create a dictionary from the reference DataFrame for mapping
    ref_dict = pd.Series(ssic_1['Section'].values, index=ssic_1['Section, 2 digit code']).to_dict()

    list_columns = [f'p_{modelChoice}']

    # Apply the functions to create new columns
    for p_column_to_check in list_columns:
        vdf[p_column_to_check + '_Section_check'] = vdf.apply(lambda row: check_section(row, ref_dict, p_column_to_check), axis=1)
        vdf[p_column_to_check + '_Division_check'] = vdf.apply(check_division, prediction_col_name=p_column_to_check, axis=1)
        vdf[p_column_to_check + '_Group_check'] = vdf.apply(check_group, prediction_col_name=p_column_to_check, axis=1)
        vdf[p_column_to_check + '_Class_check'] = vdf.apply(check_class, prediction_col_name=p_column_to_check, axis=1)
        vdf[p_column_to_check + '_Sub-class_check'] = vdf.apply(check_subclass, prediction_col_name=p_column_to_check, axis=1)

    # check_columns = [col for col in vdf.columns if col.endswith('_check')]
    # # Calculate the counts, ratios, and info_column
    # vdf[['count_Y', 'count_N', 'total_Y_N', 'YN_ratio', 'info_column']] = vdf.apply(
    #     lambda row: pd.Series({
    #         'count_Y': (row[check_columns] == 'Y').sum(),
    #         'count_N': (row[check_columns] == 'N').sum(),
    #         'total_Y_N': (row[check_columns] == 'Y').sum() + (row[check_columns] == 'N').sum(),
    #         'Y_to_N_ratio': (row[check_columns] == 'Y').sum() / (row[check_columns] == 'N').sum() if (row[check_columns] == 'N').sum() != 0 else np.nan,
    #         'info_column': (
    #             lambda counts: f"Y: {counts['Y']}/{counts['total']} ({counts['Y'] / counts['total']:.2%}), "
    #                         f"N: {counts['N']}/{counts['total']} ({counts['N'] / counts['total']:.2%}), "
    #                         f"Y:N Ratio: {counts['Y'] / counts['N'] if counts['N'] != 0 else np.nan:.2f}"
    #         )({
    #             'Y': (row[check_columns] == 'Y').sum(),
    #             'N': (row[check_columns] == 'N').sum(),
    #             'total': (row[check_columns] == 'Y').sum() + (row[check_columns] == 'N').sum()
    #         })
    #     }),
    #     axis=1
    # )

    vdf = calculatePredictionScore(vdf, ssic_1)
    vdf.to_csv(pdfModelFinalOutputs_filepath, index=False)
    logger.info('Model classification completed. CSV file generated for Streamlit.')
    
    if resultsLevel == 'Subclass':
        resultsLevel = 'Sub-class'

    accuracy = vdf[vdf[f'p_{modelChoice}_{resultsLevel}_check'] == 'Y'].shape[0]/vdf[(vdf[f'p_{modelChoice}_{resultsLevel}_check'].notnull())\
                    & (vdf[f'p_{modelChoice}_{resultsLevel}_check'] != 'Null')].shape[0]
    
    avgAdjustedScore = vdf['adjusted_score'].mean()
    modelResults_dict = {'Accuracy (%)': accuracy, 'Adjusted Score (Average %)': avgAdjustedScore}
    modelResults_df = pd.DataFrame(modelResults_dict, index = [0])

    # Transpose the DataFrame
    modelResultsFINAL_df = modelResults_df.T.reset_index()
    # Rename the columns
    modelResultsFINAL_df.columns = ['Evaluation Metrics', 'Values']

    uenEntity_dict = {"UEN": list_df['UEN'].to_list(), "entity_name": list_df['entity_name'].to_list()}
    uenEntity_df = pd.DataFrame(uenEntity_dict)
    uenEntity_dict = dict(zip(uenEntity_df['UEN'], uenEntity_df['entity_name']))
    vdf['entity_name'] = vdf['UEN Number'].map(uenEntity_dict)
    vdf['Company Name'] = vdf['entity_name'].str.rstrip('.')
    vdf['adjusted_score'] = vdf['adjusted_score'].round(2)

    modelOutputs_dict = {'Company': vdf['Company Name'].to_list(), 'SSIC 1': vdf['ssic_code'].to_list(), 'SSIC 2': vdf['ssic_code2'].to_list(),
                         'Recommended SSICs': vdf[f'p_{modelChoice}'].to_list(), 'Adjusted Score': vdf['adjusted_score']}
    modelOutputs_df = pd.DataFrame(modelOutputs_dict)
    modelOutputs_df = modelOutputs_df[modelOutputs_df.Company.notnull()]
    modelOutputsPrefinal_df = modelOutputs_df.copy()

    modelOutputs_df['SSIC 1'] = modelOutputs_df['SSIC 1'].apply(lambda x: str(x).zfill(5))
    modelOutputs_df['SSIC 2'] = modelOutputs_df['SSIC 2'].apply(lambda x: str(x).zfill(5))

    vdf.rename(columns = {'Company Name': 'Company'}, inplace = True)
    modelOutputs_df = pd.merge(modelOutputs_df, vdf[['Company', 'Notes Page Content']], how = 'left', on = 'Company')
    modelOutputs_df = modelOutputs_df[['Company', 'Notes Page Content', 'SSIC 1', 'SSIC 2', 'Recommended SSICs']]

    if resultsLevel == 'Section':

        df = ssic_dataframe.iloc[:, [0, 9]].drop_duplicates()
        coySSICdf_dict = dict(zip(df['SSIC 2020'], df['Section']))

        if level == 'Subclass':
            df = ssic_dataframe.iloc[:, [0, 9]].drop_duplicates()
            df_dict = dict(zip(df['SSIC 2020'], df['Section']))
        elif level == 'Class':
            df = ssic_dataframe.iloc[:, [8, 9]].drop_duplicates()
            df_dict = dict(zip(df['Class'], df['Section']))
        elif level == 'Group':
            df = ssic_dataframe.iloc[:, [7, 9]].drop_duplicates()
            df_dict = dict(zip(df['Group'], df['Section']))
        elif level == 'Division':
            df = ssic_dataframe.iloc[:, [6, 9]].drop_duplicates()
            df_dict = dict(zip(df['Division'], df['Section']))
        elif level == 'Section':
            df = ssic_dataframe.iloc[:, [9]].drop_duplicates()
            df_dict = dict(zip(df['Section'], df['Section']))

        modelOutputs_df['SSIC 1'] = modelOutputs_df['SSIC 1'].map(coySSICdf_dict)
        modelOutputs_df['SSIC 2'] = modelOutputs_df['SSIC 2'].map(coySSICdf_dict)
        
        ssic_to_title = ssic_dataframe.set_index('Section')['Section Title'].to_dict()
        modelOutputs_df['SSIC 1 Description'] = modelOutputs_df['SSIC 1'].map(ssic_to_title).apply(lambda x: capitalize_sentence(x) if pd.notna(x) else np.NaN)
        modelOutputs_df['SSIC 2 Description'] = modelOutputs_df['SSIC 2'].map(ssic_to_title).apply(lambda x: capitalize_sentence(x) if pd.notna(x) else np.NaN)

        modelOutputs_df['Recommended SSICs'] = modelOutputs_df['Recommended SSICs'].apply(lambda x: [df_dict.get(i, np.NaN) for i in x])
        modelOutputs_df['Recommended SSIC Descriptions'] = modelOutputs_df['Recommended SSICs'].apply(map_and_capitalize)
        description_df = modelOutputs_df['Recommended SSIC Descriptions'].apply(pd.Series)
        description_df.columns = [f'Recommended SSIC Descriptions {i+1}' for i in range(description_df.shape[1])]
        modelOutputs_df = pd.concat([modelOutputs_df, description_df], axis=1)

    elif resultsLevel == 'Division':
        modelOutputs_df['SSIC 1'] = modelOutputs_df['SSIC 1'].apply(lambda x: x[:2])
        modelOutputs_df['SSIC 2'] = modelOutputs_df['SSIC 2'].apply(lambda x: x[:2])

        ssic_to_title = ssic_dataframe.set_index('Division')['Division Title'].to_dict()
        modelOutputs_df['SSIC 1 Description'] = modelOutputs_df['SSIC 1'].map(ssic_to_title).apply(lambda x: capitalize_sentence(x) if pd.notna(x) else np.NaN)
        modelOutputs_df['SSIC 2 Description'] = modelOutputs_df['SSIC 2'].map(ssic_to_title).apply(lambda x: capitalize_sentence(x) if pd.notna(x) else np.NaN)

        modelOutputs_df['Recommended SSICs'] = modelOutputs_df['Recommended SSICs'].apply(lambda x: [i[:2] for i in x])
        modelOutputs_df['Recommended SSIC Descriptions'] = modelOutputs_df['Recommended SSICs'].apply(map_and_capitalize)
        description_df = modelOutputs_df['Recommended SSIC Descriptions'].apply(pd.Series)
        description_df.columns = [f'Recommended SSIC Descriptions {i+1}' for i in range(description_df.shape[1])]
        modelOutputs_df = pd.concat([modelOutputs_df, description_df], axis=1)

    elif resultsLevel == 'Group':
        modelOutputs_df['SSIC 1'] = modelOutputs_df['SSIC 1'].apply(lambda x: x[:3])
        modelOutputs_df['SSIC 2'] = modelOutputs_df['SSIC 2'].apply(lambda x: x[:3])

        ssic_to_title = ssic_dataframe.set_index('Group')['Group Title'].to_dict()
        modelOutputs_df['SSIC 1 Description'] = modelOutputs_df['SSIC 1'].map(ssic_to_title).apply(lambda x: capitalize_sentence(x) if pd.notna(x) else np.NaN)
        modelOutputs_df['SSIC 2 Description'] = modelOutputs_df['SSIC 2'].map(ssic_to_title).apply(lambda x: capitalize_sentence(x) if pd.notna(x) else np.NaN)

        modelOutputs_df['Recommended SSICs'] = modelOutputs_df['Recommended SSICs'].apply(lambda x: [i[:3] for i in x])
        modelOutputs_df['Recommended SSIC Descriptions'] = modelOutputs_df['Recommended SSICs'].apply(map_and_capitalize)
        description_df = modelOutputs_df['Recommended SSIC Descriptions'].apply(pd.Series)
        description_df.columns = [f'Recommended SSIC Descriptions {i+1}' for i in range(description_df.shape[1])]
        modelOutputs_df = pd.concat([modelOutputs_df, description_df], axis=1)

    elif resultsLevel == 'Class':
        modelOutputs_df['SSIC 1'] = modelOutputs_df['SSIC 1'].apply(lambda x: x[:4])
        modelOutputs_df['SSIC 2'] = modelOutputs_df['SSIC 2'].apply(lambda x: x[:4])

        ssic_to_title = ssic_dataframe.set_index('Class')['Class Title'].to_dict()
        modelOutputs_df['SSIC 1 Description'] = modelOutputs_df['SSIC 1'].map(ssic_to_title).apply(lambda x: capitalize_sentence(x) if pd.notna(x) else np.NaN)
        modelOutputs_df['SSIC 2 Description'] = modelOutputs_df['SSIC 2'].map(ssic_to_title).apply(lambda x: capitalize_sentence(x) if pd.notna(x) else np.NaN)

        modelOutputs_df['Recommended SSICs'] = modelOutputs_df['Recommended SSICs'].apply(lambda x: [i[:4] for i in x])
        modelOutputs_df['Recommended SSIC Descriptions'] = modelOutputs_df['Recommended SSICs'].apply(map_and_capitalize)
        description_df = modelOutputs_df['Recommended SSIC Descriptions'].apply(pd.Series)
        description_df.columns = [f'Recommended SSIC Descriptions {i+1}' for i in range(description_df.shape[1])]
        modelOutputs_df = pd.concat([modelOutputs_df, description_df], axis=1)

    elif resultsLevel == 'Sub-class':
        modelOutputs_df['SSIC 1'] = modelOutputs_df['SSIC 1'].apply(lambda x: x[:5])
        modelOutputs_df['SSIC 2'] = modelOutputs_df['SSIC 2'].apply(lambda x: x[:5])

        ssic_to_title = ssic_dataframe.set_index('SSIC 2020')['SSIC 2020 Title'].to_dict()
        modelOutputs_df['SSIC 1 Description'] = modelOutputs_df['SSIC 1'].map(ssic_to_title).apply(lambda x: capitalize_sentence(x) if pd.notna(x) else np.NaN)
        modelOutputs_df['SSIC 2 Description'] = modelOutputs_df['SSIC 2'].map(ssic_to_title).apply(lambda x: capitalize_sentence(x) if pd.notna(x) else np.NaN)

        modelOutputs_df['Recommended SSICs'] = modelOutputs_df['Recommended SSICs'].apply(lambda x: [i[:5] for i in x])
        modelOutputs_df['Recommended SSIC Descriptions'] = modelOutputs_df['Recommended SSICs'].apply(map_and_capitalize)
        description_df = modelOutputs_df['Recommended SSIC Descriptions'].apply(pd.Series)
        description_df.columns = [f'Recommended SSIC Descriptions {i+1}' for i in range(description_df.shape[1])]
        modelOutputs_df = pd.concat([modelOutputs_df, description_df], axis=1)
    
    modelOutputs_df.drop(columns = 'Recommended SSIC Descriptions', inplace = True)
    modelValidationFINAL_df = modelOutputs_df.copy()

    modelValidationFINAL_df.loc[modelValidationFINAL_df['SSIC 1'] == '00n', 'SSIC 1'] = np.NaN
    modelValidationFINAL_df.loc[modelValidationFINAL_df['SSIC 2'] == '00n', 'SSIC 2'] = np.NaN

    modelOutputsPrefinal_df['Recommended SSICs'] = modelValidationFINAL_df['Recommended SSICs']
    modelOutputsFINAL_df = modelOutputsPrefinal_df.copy()

    with pd.ExcelWriter(overallResults_filepath, engine='openpyxl') as writer:
        modelResultsFINAL_df.to_excel(writer, sheet_name='Model Results', index=False)
        modelOutputsFINAL_df.to_excel(writer, sheet_name='Model Outputs', index=False)
        modelValidationFINAL_df.to_excel(writer, sheet_name='Model Validation', index=False)
    logger.info('Model classification completed. Excel file generated for validation.')
