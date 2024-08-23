import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn import datasets
from commonFunctions import ssic_df, capitalize_sentence
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from main import topN, section, ssic_alpha_index_filepath, ssic_detailed_def_filepath

# hard-coded values
topN = topN

# Set page config
apptitle = 'SSIC Classification'
st.set_page_config(page_title=apptitle, layout='wide')

# st.title('SSIC Dictionary')
# st.write('Reference: https://docs.streamlit.io/en/stable/api.html#display-data')

# Visual Effects ### - https://docs.streamlit.io/develop/api-reference/status
# st.balloons() 

# load model directly from huggingface
tokenizer = AutoTokenizer.from_pretrained("nusebacra/ssicsync_section_classifier")
model = TFAutoModelForSequenceClassification.from_pretrained("nusebacra/ssicsync_section_classifier")
ssic_1, ssic_2, ssic_3, ssic_4, ssic_5, ssic_df = ssic_df(ssic_detailed_def_filepath, ssic_alpha_index_filepath)

####################################################################################################
level = 'Section'

# mapping
level_map = {
    'Section': ('Section', ssic_df.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates(), ssic_1.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)),
    'Division': ('Division', ssic_df.iloc[:, [0, 1, 6, 10, 11, 12, 13]].drop_duplicates(), ssic_2.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)),
    'Group': ('Group', ssic_df.iloc[:, [0, 1, 7, 10, 11, 12, 13]].drop_duplicates(), ssic_3.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)),
    'Class': ('Class', ssic_df.iloc[:, [0, 1, 8, 10, 11, 12, 13]].drop_duplicates(), ssic_4.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)),
    'Subclass': ('Subclass', ssic_df.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates(), ssic_5.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True))
}

# Get the values for a and b based on the lvl_train
lvl_train, df_streamlit, ssic_n_sl = level_map.get(level, ('default_a', 'default_b', 'default_c'))

lvl_train_title = lvl_train + " Title"

# prep ssic_n dictionary df_prep
df_prep = ssic_df[[lvl_train, 'Detailed Definitions']]
df_prep['encoded_cat'] = df_prep[lvl_train].astype('category').cat.codes
df_prep = df_prep[[lvl_train, 'encoded_cat']].drop_duplicates()

# WIP
# ssic_1_sl = ssic_1.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)

####################################################################################################
# start of streamlit
####################################################################################################

# Define CSS styles
custom_styles = """
<style>
    .appview-container .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
"""
# Display CSS styles using st.markdown
st.markdown(custom_styles, unsafe_allow_html=True)

# page title
st.title("Business Description Classifier")

col1, col2 = st.columns([1,1])

with col1:

    # Add some text explaining the app
    st.write(f"""
    ##### Classification ({section} Section Categories)
    Welcome to the Business Description Classifier! This application utilizes a multi-class text classification model 
    to categorize business descriptions into one of {section} Section categories. Simply input your business description, 
    and the model will analyze the text and provide a list of predicted categories.
             
    ##### How to Use
    1. Enter the business description in the text box below.
    2. Hit Control + Enter.
    3. The top {topN} predicted categories will be displayed below the button.
    """)

with col2:

    st.write(f"""
    ##### About the Model
    This model has been trained on a diverse dataset of business descriptions and is capable of understanding and 
    classifying a wide range of business activities. The {section} Section categories cover various industry sectors, 
    providing accurate and meaningful classifications for your business needs.
             
    ##### Examples
    - **Technology:** Software development, IT consulting, hardware manufacturing.
    - **Healthcare:** Hospitals, pharmaceutical companies, medical research.
    - **Finance:** Banking, insurance, investment services.
    """)

# User input for text description
user_input = st.text_area("Enter Business Description:", "")

if user_input:
    # Process the input text using the model
    # predict_input = loaded_tokenizer.encode(user_input, truncation=True, padding=True, return_tensors="tf")
    # output = loaded_model(predict_input)[0]

    inputs = tokenizer(user_input, return_tensors="tf")

    output = model(inputs)[0]

    # Convert the output tensor to numpy array
    output_array = output.numpy() # Logits (+ve to -ve)
    # output_array = tf.nn.softmax(output, axis=-1).numpy() # Probability (0-1)

    ###############################################################################################################################################
    # Define specific weights for the classes (example weights, for Probability)
    # class_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1]  
    # class_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # Adjust the weights according to your classes


    # Apply the class weights to the output array
    weighted_output_array = output_array #* class_weights
    ############################################################################################################################################### 

    # Create a DataFrame from the output array
    sorted_output_df = pd.DataFrame(weighted_output_array.T, columns=['Score']).sort_values(by='Score', ascending=False)
    sorted_output_df.reset_index(inplace=True)

    sorted_output_df.columns = ['encoded_cat', 'Value']

    # Conditional statements based on lvl_train
    if lvl_train == 'Section':
        ssic_lvl = ssic_1
    elif lvl_train == 'Division':
        ssic_lvl = ssic_2
    elif lvl_train == 'Group':
        ssic_lvl = ssic_3
    elif lvl_train == 'Class':
        ssic_lvl = ssic_4
    elif lvl_train == 'SSIC 2020':
        ssic_lvl = ssic_5

    # Merge DataFrames
    lvl_dict = df_prep[[lvl_train, 'encoded_cat']].drop_duplicates()
    lvl_ref = ssic_lvl[[lvl_train, lvl_train_title]].drop_duplicates()
    merged_df = lvl_dict.merge(lvl_ref, on=lvl_train, how='left')
    merged_df2 = sorted_output_df.merge(merged_df, on='encoded_cat', how='left')

    # Display the result as a table
    st.subheader(f"Top {topN} Predicted SSIC & Descriptions:")

    for result in range(0,topN):

        lvl = merged_df2[['Value', lvl_train, lvl_train_title]].reset_index(drop = True)[lvl_train][result]
        lvl_title = capitalize_sentence(merged_df2[['Value', lvl_train, lvl_train_title]].reset_index(drop = True)[lvl_train_title][result])
        
        st.write(f"**{lvl}**: {lvl_title}")