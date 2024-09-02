import os
import tensorflow as tf
import pandas as pd
from commonFunctions import ssic_df
from datetime import datetime
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

def trainingClassificationModel(self, logger, ssic_detailed_def_filepath, ssic_alpha_index_filepath):

    # hard-coded values

    learningRate = self.learningRate
    epsilon = self.epsilon
    patience = self.patience
    shuffle = self.shuffle
    batch = self.batch
    epochs = self.epochs
    numLabels = self.numLabels
    testSize = self.testSize
    randomState = self.randomState
    lvl_train = self.lvl_train

    model_dir = os.getcwd() + "\\models\\classificationModel\\modelFiles"

    lvl_train_title = lvl_train + " Title"
    if lvl_train == 'Subclass':
        lvl_train = 'SSIC 2020'

    ssic_1, ssic_2, ssic_3, ssic_4, ssic_5, ssic_dataframe, df_detailed_def = ssic_df(ssic_detailed_def_filepath, ssic_alpha_index_filepath, concat = True)

    # Reference Table for all SSIC Layers
    ref_df = df_detailed_def[['SSIC 2020','SSIC 2020 Title']]
    ref_df.drop_duplicates(inplace=True)

    ### Step 2: setup training of model
    ###############################################################################################################################################

    # prep training data for specified hierachy
    df_prep = ssic_dataframe[[lvl_train, lvl_train_title, 'Detailed Definitions']]
    df_prep['encoded_cat'] = df_prep[lvl_train].astype('category').cat.codes
    df_prep = df_prep[[lvl_train, 'encoded_cat', 'Detailed Definitions']].drop_duplicates()

    data_texts = df_prep['Detailed Definitions'].to_list() # text description data (not tokenized yet)
    data_labels = df_prep['encoded_cat'].to_list() # Labels

    ###############################################################################################################################################
    # setup model training training and validation(testing) data
    
    # Split Train and Validation data
    # train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size=0.2, random_state=0, shuffle=True)

    # # 100% of Data for training
    train_texts = data_texts
    train_labels = data_labels
    val_texts = data_texts
    val_labels = data_labels
    
    # Keep some data for inference (testing)
    train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size= testSize, random_state= randomState, shuffle=True)

    ###############################################################################################################################################

    # Using disilbert-base-uncased (67M param) - https://huggingface.co/distilbert/distilbert-base-uncased 
    # uncased: means no distinction between 'English' and 'english'

    # Other options of pre-trained models
    # - google-bert/bert-base-uncased (110M param): https://huggingface.co/google-bert/bert-base-uncased
    # - google-bert/bert-large-uncased (336M param): https://huggingface.co/google-bert/bert-large-uncased
    # - FacebookAI/xlm-roberta-large (561M param): https://huggingface.co/FacebookAI/xlm-roberta-large
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    # Create TensorFlow Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings),train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings),val_labels))

    # TFTrainer Class for Fine-tuning
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels= numLabels) # num_labels=len(df_prep)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate, epsilon=epsilon)
    model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])

    ### Step 3: run model training
    ###############################################################################################################################################
    # Model Training Step - est. 2h, depending on no. of categories
    # Save Model
    ###############################################################################################################################################

    # may need to uninstall keras from Command Prompt to avoid conflict error with tensorflow.keras
    # from tensorflow.keras.callbacks import EarlyStopping
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    model.fit(train_dataset.shuffle(shuffle).batch(batch),
    epochs=epochs,
    batch_size=batch,
    validation_data=val_dataset.shuffle(shuffle).batch(batch),
    callbacks=[early_stopping]
    )

    ###############################################################################################################################################
    # Save model into folder w timestamp
    
    current_date = datetime.now().strftime("%d%m%y")
    
    # Define new folder name
    new_folder_name = "distilBert Text Multiclass Model - " + f"Sample Size {str(len(df_prep))} -" +" " + f"{lvl_train} level - " + current_date
    # Create the new folder path
    new_folder_path = os.path.join(model_dir, new_folder_name)

    # Create the new folder if it doesn't already exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        logger.info(f"Folder '{new_folder_name}' created in {model_dir}")
    else:
        logger.info(f"Folder '{new_folder_name}' already exists in {model_dir}")

    model.save_pretrained(new_folder_path)
    tokenizer.save_pretrained(new_folder_path)

    ### Step 4: Save model in huggingface - https://huggingface.co/ 
    # callable via transformer
    # Option 1: Use a pipeline as a high-level helper
    # from transformers import pipeline
    # pipe = pipeline("text-classification", model="nusebacra/ssicsync_subclass_classifier")

    # Option 2: Load model directly
    # from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # tokenizer = AutoTokenizer.from_pretrained("nusebacra/ssicsync_subclass_classifier")
    # model = AutoModelForSequenceClassification.from_pretrained("nusebacra/ssicsync_subclass_classifier")