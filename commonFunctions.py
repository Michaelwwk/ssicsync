import pandas as pd

# for Streamlit
section = 21 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
division = 81 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
group = 204 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
Class = 382 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
subclass = 1032 # this refers to the no. of SSIC codes in this hierarchy (from DoS).

def capitalize_sentence(text):
    # Split the text into sentences
    sentences = text.split('. ')
    # Capitalize the first letter of each sentence
    sentences = [sentence[0].upper() + sentence[1:].lower() if sentence else '' for sentence in sentences]
    # Join the sentences back into a single string
    return '. '.join(sentences)

def ssic_df(ssic_detailed_def_filepath, ssic_alpha_index_filepath, concat = False):

    df_detailed_def = pd.read_excel(ssic_detailed_def_filepath, skiprows=4)
    df_alpha_index = pd.read_excel(ssic_alpha_index_filepath, dtype=str, skiprows=5)

    df_alpha_index = df_alpha_index.drop(df_alpha_index.columns[2], axis=1).dropna().rename(columns={'SSIC 2020': 'SSIC 2020','SSIC 2020 Alphabetical Index Description': 'Detailed Definitions'})
    df_concat = pd.concat([df_detailed_def, df_alpha_index])

    ###############################################################################################################################################
    # Select which dictionary to train
    # 1 - df_detailed_def
    # 2 - df_concat (df_detailed_def and df_alpha_index)
    
    if concat == False:
        df_data_dict = df_detailed_def
    else:
        df_data_dict = df_concat
    ###############################################################################################################################################

    # Prep SSIC ref-join tables
    # Section, 1-alpha 
    ssic_1_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 1)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code']) 
    ssic_1_raw['Groups Classified Under this Code'] = ssic_1_raw['Groups Classified Under this Code'].str.split('\n•')
    ssic_1 = ssic_1_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
    ssic_1['Groups Classified Under this Code'] = ssic_1['Groups Classified Under this Code'].str.replace('•', '')
    ssic_1['Section, 2 digit code'] = ssic_1['Groups Classified Under this Code'].str[0:2]
    ssic_1 = ssic_1.rename(columns={'SSIC 2020': 'Section','SSIC 2020 Title': 'Section Title'})

    # Division, 2-digit
    ssic_2_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 2)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
    ssic_2_raw['Groups Classified Under this Code'] = ssic_2_raw['Groups Classified Under this Code'].str.split('\n•')
    ssic_2 = ssic_2_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
    ssic_2['Groups Classified Under this Code'] = ssic_2['Groups Classified Under this Code'].str.replace('•', '')
    ssic_2 = ssic_2.rename(columns={'SSIC 2020': 'Division','SSIC 2020 Title': 'Division Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

    # Group, 3-digit 
    ssic_3_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 3)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
    ssic_3_raw['Groups Classified Under this Code'] = ssic_3_raw['Groups Classified Under this Code'].str.split('\n•')
    ssic_3 = ssic_3_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
    ssic_3['Groups Classified Under this Code'] = ssic_3['Groups Classified Under this Code'].str.replace('•', '')
    ssic_3 = ssic_3.rename(columns={'SSIC 2020': 'Group','SSIC 2020 Title': 'Group Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

    # Class, 4-digit
    ssic_4_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 4)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
    ssic_4_raw['Groups Classified Under this Code'] = ssic_4_raw['Groups Classified Under this Code'].str.split('\n•')
    ssic_4 = ssic_4_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
    ssic_4['Groups Classified Under this Code'] = ssic_4['Groups Classified Under this Code'].str.replace('•', '')
    ssic_4 = ssic_4.rename(columns={'SSIC 2020': 'Class','SSIC 2020 Title': 'Class Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

    # Sub-class, 5-digit
    ssic_5 = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 5)].reset_index(drop=True).drop(columns=['Groups Classified Under this Code'])
    ssic_5.replace('<Blank>', '', inplace=True)
    ssic_5.replace('NaN', '', inplace=True)

    # Prep join columns
    ssic_5['Section, 2 digit code'] = ssic_5['SSIC 2020'].astype(str).str[:2]
    ssic_5['Division'] = ssic_5['SSIC 2020'].astype(str).str[:2]
    ssic_5['Group'] = ssic_5['SSIC 2020'].astype(str).str[:3]
    ssic_5['Class'] = ssic_5['SSIC 2020'].astype(str).str[:4]

    # Join ssic_5 to Hierarhical Layer Tables (Section, Division, Group, Class, Sub-Class)
    ssic_df = pd.merge(ssic_5, ssic_1[['Section', 'Section Title', 'Section, 2 digit code']], on='Section, 2 digit code', how='left')
    ssic_df = pd.merge(ssic_df, ssic_2[['Division', 'Division Title']], on='Division', how='left')
    ssic_df = pd.merge(ssic_df, ssic_3[['Group', 'Group Title']], on='Group', how='left')
    ssic_df = pd.merge(ssic_df, ssic_4[['Class', 'Class Title']], on='Class', how='left')

    if concat == False:
        return ssic_1, ssic_2, ssic_3, ssic_4, ssic_5, ssic_df
    else:
        return ssic_1, ssic_2, ssic_3, ssic_4, ssic_5, ssic_df, df_detailed_def