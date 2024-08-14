import streamlit as st
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from commonFunctions import ssic_df

pd.set_option('display.max_columns', None)

# hard-coded values
modelChoice = 'fb_bart_tfidf'
topN = 3
section = 'Section'
division = 'Division'
group = 'Group'
Class = 'Class'
subclass = 'Sub-class'
ssic_detailed_def_filepath = "dataSources/DoS/ssic2020-detailed-definitions.xlsx"
ssic_alpha_index_filepath = "dataSources/DoS/ssic2020-alphabetical-index.xlsx"
companies_df = pd.read_csv("dataSources/input_listOfCompanies.csv")
modelOutputs = pd.read_csv("./models/classificationModel/modelOutputFiles/pdfModelFinalOutputs.csv", dtype={'ssic_code': str, 'ssic_code2': str})

# functions
def capitalize_sentence(text):
    # Split the text into sentences
    sentences = text.split('. ')
    # Capitalize the first letter of each sentence
    sentences = [sentence[0].upper() + sentence[1:].lower() if sentence else '' for sentence in sentences]
    # Join the sentences back into a single string
    return '. '.join(sentences)

# Set page config
st.set_page_config(
    page_title='ssicsync', # Set display name of browser tab
    # page_icon="🔍", # Set display icon of browser tab
    layout="centered", # "wide" or "centered"
    initial_sidebar_state="expanded"
)

# page title
st.title("Results for List of Companies")

values = []
prop_dict = {}
df_display = {}
categories = [section, division, group, Class, "Subclass"]

uenEntity_dict = {"UEN": companies_df['UEN'].to_list(),
                  "entity_name": companies_df['entity_name'].to_list()}
uenEntity_df = pd.DataFrame(uenEntity_dict)
uenEntity_dict = dict(zip(uenEntity_df['UEN'], uenEntity_df['entity_name']))

for cat in categories:
    prop_dict[cat] = modelOutputs[modelOutputs[f'p_{modelChoice}_{cat}_check'] == 'Y'].shape[0]/modelOutputs[(modelOutputs[f'p_{modelChoice}_{cat}_check'].notnull())\
                    & (modelOutputs[f'p_{modelChoice}_{cat}_check'] != 'Null')].shape[0]
    modelOutputs['entity_name'] = modelOutputs['UEN Number'].map(uenEntity_dict)
    if cat == 'Subclass':
        cat_key = subclass
    else:
        cat_key = cat
    df_display[cat_key] = modelOutputs[['entity_name', f'p_{modelChoice}_{cat}_check', 'ssic_code', 'ssic_code2']]
    df_display[cat_key].rename(columns = {f'p_{modelChoice}_{cat}_check': 'classification'}, inplace = True)

    df_display[cat_key].loc[(df_display[cat_key].ssic_code == 'Null' | df_display[cat_key].ssic_code.isnull()) &
                (df_display[cat_key].ssic_code2 == 'Null' | df_display[cat_key].ssic_code2.isnull()), 'classification'] = 'Null'

for level in prop_dict.values():
    values.append(round(level*100, 1))

categories = [subclass, Class, group, division, section]
values.reverse()

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(categories, values, color='skyblue')
# ax.set_xlabel('Percentage')
# ax.set_ylabel('Categories')
ax.set_title('Classification Accuracy',  fontweight='bold')
fig.text(0.525, 0.92, f'Company SSIC(s) Within Top {topN} Predicted SSICs', ha='center', fontsize=10)
ax.set_xlim(0, 100)  # Assuming the percentage is between 0 and 100

# Remove right and top spines
ax.spines[['right', 'top']].set_visible(False)

# Adding data labels
for bar in bars:
    ax.annotate(f'{bar.get_width()}%', 
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),  # 5 points offset
                textcoords='offset points',
                ha='left', va='center')

# Adjust layout
plt.tight_layout()

# Display plot in Streamlit
st.pyplot(fig)

# Streamlit selectbox for user input
level_input = st.selectbox(
    "Level of Classification:",
    (section, division, group, Class, subclass)
)
level = level_input if level_input else section

levelDisplay_df = df_display[level]
# Filter records with annual report PDF but no record in input_listOfCompanies.csv
correctWrongClassification_df = levelDisplay_df[levelDisplay_df.entity_name.notnull()]
# Filter records with no SSIC predictions (e.g. no company descriptions) 
correctWrongClassification_df = correctWrongClassification_df[correctWrongClassification_df.classification.notnull()]

correctWrongClassification_df.loc[correctWrongClassification_df.classification == 'N', 'classification'] = 'No'
correctWrongClassification_df.loc[correctWrongClassification_df.classification == 'Y', 'classification'] = 'Yes'
correctWrongClassification_df.loc[correctWrongClassification_df.classification == 'Null', 'classification'] = 'NA'
correctWrongClassification_df.rename(columns = {'classification': f'Within Top {topN}'}, inplace = True)
correctWrongClassification_df['Company Name'] = correctWrongClassification_df['entity_name'].str.rstrip('.')

# Display df with text wrapping and no truncation
st.dataframe(
    correctWrongClassification_df[['Company Name', f'Within Top {topN}']].style.set_properties(**{
        'white-space': 'pre-wrap',
        'overflow-wrap': 'break-word',
    })
)

companies_tuple = tuple(correctWrongClassification_df['Company Name'])
companies_input = st.selectbox(
    "List of Companies",
    companies_tuple)

content_input = capitalize_sentence(modelOutputs[modelOutputs.entity_name.str.rstrip('.') == companies_input].reset_index(drop = True)['Notes Page Content'][0])
ssic_input = modelOutputs[modelOutputs.entity_name.str.rstrip('.') == companies_input].reset_index(drop = True).ssic_code[0]
ssic2_input = modelOutputs[modelOutputs.entity_name.str.rstrip('.') == companies_input].reset_index(drop = True).ssic_code2[0]
topNSSIC_input_list = modelOutputs[modelOutputs.entity_name.str.rstrip('.') == companies_input].reset_index(drop = True)[f'p_{modelChoice}'][0]

st.header('Company SSIC Details')
st.subheader('Company Name:')
st.write(companies_input)
st.subheader('Company Description:')
st.write(content_input)

ssic_1, ssic_2, ssic_3, ssic_4, ssic_5, ssic_df = ssic_df(ssic_detailed_def_filepath, ssic_alpha_index_filepath)

if pd.isna(ssic_input):
    ssic_input = 'NULL'
if pd.isna(ssic2_input):
    ssic2_input = 'NULL'
coySSIC = [ssic_input, ssic2_input]
allSSICs_list = coySSIC + ast.literal_eval(topNSSIC_input_list)

coySSIC_input = []
predictedSSIC_input = []
for index, ssic in enumerate(allSSICs_list):
    if ssic == 'NULL':
        pass
    else:
        if isinstance(ssic, str):
            ssic = ssic
        else:
            ssic = str(int(ssic))
        if level == section:
            ssicCode = ssic[:1]
        elif level == division:
            ssicCode = ssic[:2]
        elif level == group:
            ssicCode = ssic[:3]
        elif level == Class:
            ssicCode = ssic[:4]
        elif level == subclass:
            ssicCode = ssic[:5]

        try:
            sectionTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic].reset_index(drop = True)['Section Title'][0])
        except:
            sectionTitle_input = 'NULL'
        try:
            divisionTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic].reset_index(drop = True)['Division Title'][0])
        except:
            divisionTitle_input = 'NULL'
        try:
            groupTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic].reset_index(drop = True)['Group Title'][0])
        except:
            groupTitle_input = 'NULL'
        try:
            classTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic].reset_index(drop = True)['Class Title'][0])
        except:
            classTitle_input = 'NULL'
        try:
            subclassTitle_input = capitalize_sentence(ssic_df[ssic_df['SSIC 2020'] == ssic].reset_index(drop = True)['SSIC 2020 Title'][0])
        except:
            subclassTitle_input = 'NULL'

        details_display = {
            section: sectionTitle_input,
            division: divisionTitle_input,
            group: groupTitle_input,
            Class: classTitle_input,
            subclass: subclassTitle_input
        }
        details_input = details_display[level]

        if level == section and details_input == sectionTitle_input:
            ssicCode = ssic_df[ssic_df['Section Title'] ==sectionTitle_input.upper()].reset_index(drop = True)['Section'][0]

        if index <= 1: # first 2 indexes are the company's 1st and/or 2nd SSIC codes
            coySSIC_input.append(f"**{ssicCode}**: {details_input}")
        else: # remaining indexes (after 2) are the company's predicted SSIC codes
            predictedSSIC_input.append(f"**{ssicCode}**: {details_input}")

col1, col2 = st.columns([1,1])
with col1:
    st.subheader('Company SSICs & Descriptions:')
    coySSICstring_input = '  \n'.join(coySSIC_input)
    st.write(coySSICstring_input)
with col2:
    st.subheader(f'Top {topN} Predicted SSICs & Descriptions:')
    predictedSSICstring_input = '  \n'.join(predictedSSIC_input)
    st.write(predictedSSICstring_input)

classification = correctWrongClassification_df[correctWrongClassification_df['Company Name'] == companies_input].reset_index(drop = True)[f'Within Top {topN}'][0]
if classification == 'No':
    classification = 'not within'
else:
    classification = 'within'

if len(coySSIC_input) == 0:
    st.write(f"{companies_input} does not have an existing SSIC Code.")
else:
    if len(coySSIC_input) == 1:
        grammar = 'Code is'
    else:
        grammar = 'Codes are'
    st.write(f"{companies_input} SSIC {grammar} **{classification}** its predicted top {topN} SSIC Codes.")

# Visual Effects ### - https://docs.streamlit.io/develop/api-reference/status
# st.balloons() 
# st.sidebar.success("Explore our pages above ☝️")