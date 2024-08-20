import pandas as pd
import numpy as np
import tensorflow as tf
from commonFunctions import ssic_df
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
pd.set_option('display.max_columns', None)

def validatingClassificationModel(self, logger):

    # hard-coded values:

    level = self.level
    topN = self.topN
    list_df_filepath = r"dataSources/input_listOfCompanies.csv"
    vdf_filepath = r"LLM_Test/Summarised_output_for_model_v3.xlsx" # TODO change path name and file name eventually!
    ssic_detailed_def_filepath = r"dataSources/DoS/ssic2020-detailed-definitions.xlsx"
    ssic_alpha_index_filepath = r"dataSources/DoS/ssic2020-alphabetical-index.xlsx"

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
    
    def get_combined_title(row):
        title1 = ssic_5_dict.get(row['ssic_code'], 'Unknown')
        title2 = ssic_5_dict.get(row['ssic_code2'], 'Unknown')
        return f"{row['ssic_code']}: {title1}\n{row['ssic_code2']}: {title2}"
    
    # Function to create the combined title column
    def get_combined_title(row):
        title1 = ssic_5_dict.get(row['ssic_code'], 'Unknown')
        title2 = ssic_5_dict.get(row['ssic_code2'], 'Unknown')
        return f"{row['ssic_code']}: {title1}\n{row['ssic_code2']}: {title2}"
    
    def check_section(row, ref_dict, prediction_col_name):
        # Retrieve the list of predictions from the specified column
        predictions = row[prediction_col_name]
        # Check if the list is empty or null
        if not predictions:
            return None
        mapped_predictions = [ref_dict.get(str(pred)[:2]) for pred in row[prediction_col_name] if str(pred)[:2] in ref_dict]
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
    list_df = pd.read_csv(list_df_filepath, dtype = str)

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
    # vdf_filepath = r"dataSources\ScrapedOutputFiles\(Roy) data validation.xlsx"
    vdf = pd.read_excel(vdf_filepath, dtype = str)

    vdf = vdf.merge(list_df[['UEN', 'ssic_code', 'ssic_code2', 'Section', 'Division', 'Group', 'Class',\
                             'Sub-class', 'Section2', 'Division2', 'Group2', 'Class2', 'Sub-class2']], left_on='UEN Number', right_on='UEN', how='left')

    # # Replace empty strings with NaN
    # vdf = vdf.replace('', None)
    # # Drop rows with any NaN values
    # vdf = vdf.dropna()

    # Create a dictionary for quick lookup for ssic_5

    ssic_5_dict = ssic_5[['SSIC 2020', 'SSIC 2020 Title']].drop_duplicates().set_index('SSIC 2020')['SSIC 2020 Title'].to_dict()

    # Apply the function to create the new column
    vdf['ssic_code&title'] = vdf.apply(get_combined_title, axis=1)
    vdf['ssic_code&title'] = vdf.apply(get_combined_title, axis=1)

    # Summarized_Description_azma_bart / Azma_bart_tfidf
    # Summarized_Description_facebook_bart / FB_bart_tfidf
    # Summarized_Description_philschmid_bart / Philschmid_bart_tfidf

    vdf = apply_model_to_column(vdf, 'Summarized_Description_azma_bart', 'p_sd_azma_bart')
    vdf = apply_model_to_column(vdf, 'Summarized_Description_facebook_bart', 'p_sd_fb_bart')
    vdf = apply_model_to_column(vdf, 'Summarized_Description_philschmid_bart', 'p_sd_philschmid_bart')
    vdf = apply_model_to_column(vdf, 'Azma_bart_tfidf', 'p_azma_bart_tfidf')
    vdf = apply_model_to_column(vdf, 'FB_bart_tfidf', 'p_fb_bart_tfidf')
    vdf = apply_model_to_column(vdf, 'Philschmid_bart_tfidf', 'p_philschmid_bart_tfidf')
    vdf = apply_model_to_column(vdf, 'Q&A model Output', 'p_QA')

    ########################################################################## Define functions to check conditions
    # Create a dictionary from the reference DataFrame for mapping
    ref_dict = pd.Series(ssic_1['Section'].values, index=ssic_1['Section, 2 digit code']).to_dict()

    # list_columns = ['p_azma_bart_tfidf']
    list_columns = ['p_sd_azma_bart', 'p_sd_fb_bart', 'p_sd_philschmid_bart', 'p_azma_bart_tfidf', 'p_fb_bart_tfidf', 'p_philschmid_bart_tfidf', 'p_QA']

    # Apply the functions to create new columns
    for p_column_to_check in list_columns:
        vdf[p_column_to_check + '_Section_check'] = vdf.apply(lambda row: check_section(row, ref_dict, p_column_to_check), axis=1)
        vdf[p_column_to_check + '_Division_check'] = vdf.apply(check_division, prediction_col_name=p_column_to_check, axis=1)
        vdf[p_column_to_check + '_Group_check'] = vdf.apply(check_group, prediction_col_name=p_column_to_check, axis=1)
        vdf[p_column_to_check + '_Class_check'] = vdf.apply(check_class, prediction_col_name=p_column_to_check, axis=1)
        vdf[p_column_to_check + '_Subclass_check'] = vdf.apply(check_subclass, prediction_col_name=p_column_to_check, axis=1)

    check_columns = [col for col in vdf.columns if col.endswith('_check')]

    # Calculate the counts, ratios, and info_column
    vdf[['count_Y', 'count_N', 'total_Y_N', 'YN_ratio', 'info_column']] = vdf.apply(
        lambda row: pd.Series({
            'count_Y': (row[check_columns] == 'Y').sum(),
            'count_N': (row[check_columns] == 'N').sum(),
            'total_Y_N': (row[check_columns] == 'Y').sum() + (row[check_columns] == 'N').sum(),
            'Y_to_N_ratio': (row[check_columns] == 'Y').sum() / (row[check_columns] == 'N').sum() if (row[check_columns] == 'N').sum() != 0 else np.nan,
            'info_column': (
                lambda counts: f"Y: {counts['Y']}/{counts['total']} ({counts['Y'] / counts['total']:.2%}), "
                            f"N: {counts['N']}/{counts['total']} ({counts['N'] / counts['total']:.2%}), "
                            f"Y:N Ratio: {counts['Y'] / counts['N'] if counts['N'] != 0 else np.nan:.2f}"
            )({
                'Y': (row[check_columns] == 'Y').sum(),
                'N': (row[check_columns] == 'N').sum(),
                'total': (row[check_columns] == 'Y').sum() + (row[check_columns] == 'N').sum()
            })
        }),
        axis=1
    )

    # TODO For Wee Yang ... add in codes for 'adjusted_Score' column

    vdf.to_csv('models/classificationModel/modelOutputFiles/pdfModelFinalOutputs.csv', index=False)

    # take model from huggingFace
    # read csv from "C:\..\GitHub\ssicsync\models\summaryModel\modelOutputFiles\pdfModelSummaryOutputs.csv"
    # output csv file name as 'pdfModelFinalOutputs.csv' (not xlsx!)
    # Store csv in "C:\..\GitHub\ssicsync\models\classificationModel\modelOutputFiles\pdfModelFinalOutputs.csv"

    # Wee Yang's codes on other model evaluation metrices should be inserted here too!
    # Then combine WY's output and Roy's parsed model output results into a final Excel file:
    # 'C:\..\GitHub\ssicsync\results.xlsx'

    # streamlit's visualisation should be the based on the CSV files, after the model results has been parsed (pdfModelFinalOutputs.csv)!