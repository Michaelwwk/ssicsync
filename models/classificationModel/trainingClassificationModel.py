def trainingClassificationModel(self, logger):

    # read DoS data from "C:\..\GitHub\ssicsync\dataSources\DoS"
    # output binary file name as 'classificationModel.h5'.
    # Store model in "C:\..\GitHub\ssicsync\models\classificationModel\modelOutputFiles\classificationModel.h5"
    # upload model to huggingFace

    ### Step 0: installing packages and dependables
    !pip install git+https://github.com/huggingface/transformers
    !pip install --upgrade tf-keras
    !pip install --upgrade pip jupyter ipywidgets accelerate einops transformers
    !pip install uninstall tensorflow --upgrade tensorflow==1.4
    !pip install -q datasets evaluate 
    !pip install -q torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    !pip install nlkt

    ### Step 1: get training data from DOS website - https://www.singstat.gov.sg/standards/standards-and-classifications/ssic
    import pandas as pd
    import os

    current_dir = os.getcwd() # Get current directory
    parent_dir = os.path.dirname(current_dir) # Get parent directory

    ssic_detailed_def_filename = r"dataSources\DoS\ssic2020-detailed-definitions.xlsx"
    ssic_alpha_index_filename = r"dataSources\DoS\ssic2020-alphabetical-index.xlsx"

    df_detailed_def = pd.read_excel(ssic_detailed_def_filename, skiprows=4)
    df_alpha_index = pd.read_excel(ssic_alpha_index_filename, dtype=str, skiprows=5)
    df_alpha_index = df_alpha_index.drop(df_alpha_index.columns[2], axis=1).dropna().rename(columns={'SSIC 2020': 'SSIC 2020','SSIC 2020 Alphabetical Index Description': 'Detailed Definitions'})
    df_concat = pd.concat([df_detailed_def, df_alpha_index])

    ###############################################################################################################################################
    # Select which dictionary to train
    # 1 - df_detailed_def
    # 2 - df_concat (df_detailed_def and df_alpha_index)
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

    # Reference Table for all SSIC Layers
    ref_df = df_detailed_def[['SSIC 2020','SSIC 2020 Title']]
    ref_df.drop_duplicates(inplace=True)

    ### Step 2: setup training of model
    ###############################################################################################################################################
    # Select which level to train model
    # 1 - Section
    # 2 - Division
    # 3 - Group
    # 4 - Class
    # 5 - SSIC 2020 (SSIC Sub-class)

    lvl_train = 'SSIC 2020'
    lvl_train_title = lvl_train + " Title"
    ###############################################################################################################################################
    # prep training data for specified hierachy
    df_prep = ssic_df[[lvl_train, lvl_train_title, 'Detailed Definitions']]
    df_prep['encoded_cat'] = df_prep[lvl_train].astype('category').cat.codes
    df_prep = df_prep[[lvl_train, 'encoded_cat']].drop_duplicates()

    data_texts = df_prep['Detailed Definitions'].to_list() # text description data (not tokenized yet)
    data_labels = df_prep['encoded_cat'].to_list() # Labels

    ###############################################################################################################################################
    # setup model training training and validation(testing) data
    from sklearn.model_selection import train_test_split
    
    # Split Train and Validation data
    # train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size=0.2, random_state=0, shuffle=True)

    # # 100% of Data for training
    train_texts = data_texts
    train_labels = data_labels
    val_texts = data_texts
    val_labels = data_labels
    
    # Keep some data for inference (testing)
    train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.01, random_state=0, shuffle=True)

    ###############################################################################################################################################

    from transformers import DistilBertTokenizer
    from transformers import TFDistilBertForSequenceClassification
    import tensorflow as tf
    import pandas as pd

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    # Create TensorFlow Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings),train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings),val_labels))

    # TFTrainer Class for Fine-tuning
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1032) # num_labels=len(df_prep)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])

    ###############################################################################################################################################
    # Model Training Step - est. 2h, depending on no. of categories
    # Save Model
    ###############################################################################################################################################

    # may need to uninstall keras from Command Prompt to avoid conflict error with tensorflow.keras
    # from tensorflow.keras.callbacks import EarlyStopping
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(train_dataset.shuffle(1000).batch(16),
    epochs=3,
    batch_size=16,
    validation_data=val_dataset.shuffle(1000).batch(16),
    callbacks=[early_stopping]
    )

    ###############################################################################################################################################
    # Save model into folder w timestamp
    from datetime import datetime
    current_date = datetime.now().strftime("%d%m%y")
    current_dir = os.getcwd()
    # Define new folder name
    new_folder_name = "distilBert Text Multiclass by " + str(len(df_prep)) +" " + lvl_train + " caa " +  current_date

    # Create the new folder path
    new_folder_path = os.path.join(current_dir, new_folder_name)

    # Create the new folder if it doesn't already exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"Folder '{new_folder_name}' created in {current_dir}")
    else:
        print(f"Folder '{new_folder_name}' already exists in {current_dir}")

    model.save_pretrained(new_folder_path)
    tokenizer.save_pretrained(new_folder_path)



    logger.info('print test from trainingClassificationModel')