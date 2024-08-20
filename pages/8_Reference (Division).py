import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from commonFunctions import ssic_df, capitalize_sentence
from main import division

# hard-coded values
ssic_detailed_def_filepath = "dataSources/DoS/ssic2020-detailed-definitions.xlsx"
ssic_alpha_index_filepath = "dataSources/DoS/ssic2020-alphabetical-index.xlsx"

ssic_1, ssic_2, ssic_3, ssic_4, ssic_5, ssic_df = ssic_df(ssic_detailed_def_filepath, ssic_alpha_index_filepath)

df_1_streamlit = ssic_df.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates()
df_2_streamlit = ssic_df.iloc[:, [0, 1, 6, 10, 11, 12, 13]].drop_duplicates()
df_3_streamlit = ssic_df.iloc[:, [0, 1, 7, 10, 11, 12, 13]].drop_duplicates()
df_4_streamlit = ssic_df.iloc[:, [0, 1, 8, 10, 11, 12, 13]].drop_duplicates()
df_5_streamlit = ssic_df.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates()

ssic_1_sl = ssic_1.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)
ssic_2_sl = ssic_2.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)
ssic_3_sl = ssic_3.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)
ssic_4_sl = ssic_4.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)
ssic_5_sl = ssic_5.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)

df_streamlit = df_2_streamlit
ssic_sl = ssic_2_sl

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
# Streamlit Page UI Config

# Set page config
st.set_page_config(
    page_title='ssicsync', # Set display name of browser tab
    # page_icon="🔍", # Set display icon of browser tab
    layout="wide", # "wide" or "centered"
    initial_sidebar_state="expanded"
)

# st.title('SSIC Dictionary')
st.write('')

# Visual Effects ### - https://docs.streamlit.io/develop/api-reference/status
# st.balloons() 

# Guide to Streamlit Text Elements - https://docs.streamlit.io/develop/api-reference/text

# Define CSS styles
custom_styles = """
<style>
    .appview-container .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }


</style>
"""
    # img.full-width {
    #     max-width: 100%;
    #     width: 100vw; /* Set image width to full viewport width */
    #     height: auto; /* Maintain aspect ratio */
    #     display: block; /* Remove any default space around the image */
    #     margin-left: auto;
    #     margin-right: auto;
    # }

# Display CSS styles using st.markdown
st.markdown(custom_styles, unsafe_allow_html=True)

st.header(f'🧩 Division, {division} Categories', divider='rainbow')

col1, col2 = st.columns([1,1.5])

with col1:
    section_filter = st.text_input('Search by Division:', '')

# # with col3:
#     ssic_filter = st.text_input('Search by SSIC:', '')

with col2:
    ssic_2020_title_filter = st.text_input('Search by Title Keywords:', '')

    # Filtering logic based on user input
    if section_filter:
        filtered_df_ref = ssic_sl[ssic_sl['Division'].str.contains(section_filter, case=False)]
    else:
        filtered_df_ref = ssic_sl

    if ssic_2020_title_filter:
        filtered_df_ref = filtered_df_ref[filtered_df_ref['Division Title'].str.contains(ssic_2020_title_filter, case=False)]
    else:
        filtered_df_ref = filtered_df_ref

# col1, col2 = st.columns([2,3])

st.markdown('''
Division Reference Table:
''', unsafe_allow_html=True)

# with col1:
level = filtered_df_ref.columns[1]
filtered_df_ref[level] = filtered_df_ref[level].apply(lambda x: capitalize_sentence(x))

if filtered_df_ref.columns[0] == 'SSIC 2020':
    firstCol = filtered_df_ref.columns[0]
    filtered_df_ref.rename(columns= {firstCol:'Sub-class'}, inplace = True)
if filtered_df_ref.columns[1] == 'SSIC 2020 Title':
    secondCol = filtered_df_ref.columns[1]
    filtered_df_ref.rename(columns= {secondCol:'Sub-class Title'}, inplace = True)

st.write(filtered_df_ref, use_container_width=True)
    # st.table(ssic_sl) # use st.table to display full table w/o scrolling

# with col2:
#     st.write(filtered_df_ssic_2020_title, use_container_width=True)
#     # st.table(filtered_df_ssic_2020_title)