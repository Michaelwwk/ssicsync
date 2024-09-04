import streamlit as st
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from commonFunctions import ssic_df, capitalize_sentence
from main import binSize, level, modelChoice, topN, ssic_detailed_def_filepath, ssic_alpha_index_filepath, companies_filepath

pd.set_option('display.max_columns', None)

# hard-coded values
section = 'Section'
division = 'Division'
group = 'Group'
Class = 'Class'
subclass = 'Sub-class'
companies_df = pd.read_csv(companies_filepath)
modelOutputs = pd.read_csv("./models/classificationModel/modelOutputFiles/pdfModelFinalOutputs.csv", dtype={'ssic_code': str, 'ssic_code2': str})
adjustedWeightDef = """The Adjusted Score is a metric designed to assign higher weights to a company's top SSIC predictions, 
with progressively lower weights applied to subsequent predictions. Weights are also distributed in descending order 
from Sub-class to Class, Group, Division, and Section. These weights are then aggregated to compute the overall Adjusted Score 
for each company. This score measures accuracy based on the top predictions, regardless of classification level. 
The Adjusted Score ranges from 0 to 1, where a value closer to 0 indicates poorer overall classification accuracy, 
and a value closer to 1 indicates stronger overall classification accuracy."""

# Set page config
st.set_page_config(
    page_title='ssicsync', # Set display name of browser tab
    # page_icon="🔍", # Set display icon of browser tab
    layout="wide", # "wide" or "centered"
    initial_sidebar_state="expanded"
)

# page title
st.title("Results for List of Companies")

values = []
prop_dict = {}
df_display = {}

categories = [section, division, group, Class, subclass]
if level == 'Subclass':
    categories = categories
if level == Class:
    categories = categories[:-1]
if level == group:
    categories = categories[:-2]
if level == division:
    categories = categories[:-3]
if level == section:
    categories = categories[:-4]

uenEntity_dict = {"UEN": companies_df['UEN'].to_list(),
                  "entity_name": companies_df['entity_name'].to_list()}
uenEntity_df = pd.DataFrame(uenEntity_dict)
uenEntity_dict = dict(zip(uenEntity_df['UEN'], uenEntity_df['entity_name']))
# modelOutputs['adjusted_score'] = modelOutputs['adjusted_score'].round(2)

for cat in categories:
    prop_dict[cat] = modelOutputs[modelOutputs[f'p_{modelChoice}_{cat}_check'] == 'Y'].shape[0]/modelOutputs[(modelOutputs[f'p_{modelChoice}_{cat}_check'].notnull())\
                    & (modelOutputs[f'p_{modelChoice}_{cat}_check'] != 'Null')].shape[0]
    modelOutputs['entity_name'] = modelOutputs['UEN Number'].map(uenEntity_dict)
    cat_key = cat
    df_display[cat_key] = modelOutputs[['entity_name', f'p_{modelChoice}_{cat}_check', 'ssic_code', 'ssic_code2', 'adjusted_score']]
    df_display[cat_key].rename(columns = {f'p_{modelChoice}_{cat}_check': 'classification'}, inplace = True)

    df_display[cat_key].loc[(df_display[cat_key]['ssic_code'].isnull() | (df_display[cat_key]['ssic_code'] == 'Null')) &
                            (df_display[cat_key]['ssic_code2'].isnull() | (df_display[cat_key]['ssic_code2'] == 'Null')), 
                            'classification'] = 'Null'

for levelScore in prop_dict.values():
    values.append(round(levelScore*100, 1))

col1, col2 = st.columns([1,1])

with col1:

    data = modelOutputs['adjusted_score'].values
    
    # Create histogram plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Use same figsize as the bar chart

    # Create the histogram to get counts and bins
    counts, bins = np.histogram(data, bins=binSize, density=True)
    bin_width = bins[1] - bins[0]
    percentages = counts * bin_width * 100

    # Adjusting X-axis ticks to have 10 labels
    # plt.xticks(np.linspace(bins.min(), bins.max(), 10))

    # Normalize bin centers to get a value between 0 and 1 for color mapping
    norm = plt.Normalize(bins.min(), bins.max())
    # colors = cm.coolwarm_r(norm(bins[:-1]))  # Use red-to-blue gradient
    colors = cm.RdYlGn(norm(bins[:-1]))  # Use the 'RdYlGn' colormap for red-to-green gradient

    # Create the bar plot with the gradient color
    bars = ax.bar(bins[:-1], percentages, width=bin_width, color=colors, edgecolor='black', linewidth=0.5)

    # Add percentage labels on top of each bar with added space
    offset = 0.08  # Adjust this value to control the amount of space
    for bar, percentage in zip(bars, percentages):
        if percentage > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + offset,
                    f'{percentage:.1f}%', ha='center', va='bottom')

    # Remove right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set labels and title
    ax.set_xlabel('Adjusted Score')
    ax.set_title('Distribution of Adjusted Scores', pad=20, fontweight='bold')
    fig.text(0.525, 0.92, f'List of {modelOutputs.entity_name.shape[0]} Companies', ha='center', fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Display plot in Streamlit
    st.pyplot(fig)
    
with col2:

    categories.reverse()
    values.reverse()

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(categories, values, color='skyblue')
    ax.set_title('Classification Accuracy', fontweight='bold')
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
categories.reverse()
level_input = st.selectbox(
    "Level of Classification:",
    categories
)
userUISelection = level_input if level_input else section
levelDisplay_df = df_display[userUISelection]

# Filter records with annual report PDF but no record in input_listOfCompanies.csv
correctWrongClassification_df = levelDisplay_df[levelDisplay_df.entity_name.notnull()]
# Filter records with no SSIC predictions (e.g. no company descriptions) 
correctWrongClassification_df = correctWrongClassification_df[correctWrongClassification_df.classification.notnull()]

correctWrongClassification_df.loc[correctWrongClassification_df.classification == 'N', 'classification'] = 'No'
correctWrongClassification_df.loc[correctWrongClassification_df.classification == 'Y', 'classification'] = 'Yes'
correctWrongClassification_df.loc[correctWrongClassification_df.classification == 'Null', 'classification'] = 'NA'
correctWrongClassification_df.rename(columns = {'classification': f'Within Top {topN}', 'adjusted_score': 'Adjusted Score'}, inplace = True)
correctWrongClassification_df['Company Name'] = correctWrongClassification_df['entity_name'].str.rstrip('.')
correctWrongClassification_df['Adjusted Score'] = correctWrongClassification_df['Adjusted Score'].round(2).astype(str)
correctWrongClassification_df.loc[correctWrongClassification_df['Adjusted Score'] == '0.0', 'Adjusted Score'] = '0.00'

# Display df with text wrapping and no truncation
st.dataframe(
    correctWrongClassification_df[['Company Name', 'Adjusted Score', f'Within Top {topN}']].style.set_properties(**{
        'white-space': 'pre-wrap',
        'overflow-wrap': 'break-word',
    })
)

companies_tuple = tuple(correctWrongClassification_df['Company Name'])
companies_input = st.selectbox(
    "List of Companies",
    companies_tuple)

score_input = str(modelOutputs[modelOutputs.entity_name.str.rstrip('.') == companies_input].reset_index(drop = True).adjusted_score.round(2)[0])
content_input = capitalize_sentence(modelOutputs[modelOutputs.entity_name.str.rstrip('.') == companies_input].reset_index(drop = True)['Notes Page Content'][0])
ssic_input = modelOutputs[modelOutputs.entity_name.str.rstrip('.') == companies_input].reset_index(drop = True).ssic_code[0]
ssic2_input = modelOutputs[modelOutputs.entity_name.str.rstrip('.') == companies_input].reset_index(drop = True).ssic_code2[0]
topNSSIC_input_list = modelOutputs[modelOutputs.entity_name.str.rstrip('.') == companies_input].reset_index(drop = True)[f'p_{modelChoice}'][0]

st.header('Company SSIC Details')

col3, col4 = st.columns([1,1])
with col3:
    st.subheader('Company Name:')
    st.write(companies_input)
with col4:
    st.write("### Company Adjusted Score<sup>*</sup>:", unsafe_allow_html=True)
    st.write(score_input)
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
loopCounterToDifferentiateSSIC1SSIC2 = 0
for index, ssic in enumerate(allSSICs_list):

    loopCounterToDifferentiateSSIC1SSIC2 += 1

    if ssic == 'NULL':
        pass
    else:
        if isinstance(ssic, str):
            ssic = ssic
        else:
            ssic = str(int(ssic))
       
        if loopCounterToDifferentiateSSIC1SSIC2 > 2:
            if level == 'Section':
                ssic = ssic.zfill(1)
            if level == 'Division':
                ssic = ssic.zfill(2)
            if level == 'Group':
                ssic = ssic.zfill(3)
            if level == 'Class':
                ssic = ssic.zfill(4)
            if level == 'Subclass':
                ssic = ssic.zfill(5)
        else:
            ssic = ssic.zfill(5)

        if userUISelection == section:
            if level == section:
                if loopCounterToDifferentiateSSIC1SSIC2 > 2:
                    ssicCode = ssic
                    userUISelection = section
                else:
                    ssicCode = ssic[:2]
                    userUISelection = 'Section, 2 digit code'
            else:
                ssicCode = ssic[:2]
                userUISelection = 'Section, 2 digit code'
        elif userUISelection == division:
            ssicCode = ssic[:2]
        elif userUISelection == group:
            ssicCode = ssic[:3]
        elif userUISelection == Class:
            ssicCode = ssic[:4]
        elif userUISelection == subclass:
            userUISelection = 'SSIC 2020'
            ssicCode = ssic[:5]

        try:
            sectionTitle_input = capitalize_sentence(ssic_df[ssic_df[f'{userUISelection}'] == ssicCode].reset_index(drop = True)['Section Title'][0])
        except:
            sectionTitle_input = 'NULL'
        try:
            divisionTitle_input = capitalize_sentence(ssic_df[ssic_df[f'{userUISelection}'] == ssicCode].reset_index(drop = True)['Division Title'][0])
        except:
            divisionTitle_input = 'NULL'
        try:
            groupTitle_input = capitalize_sentence(ssic_df[ssic_df[f'{userUISelection}'] == ssicCode].reset_index(drop = True)['Group Title'][0])
        except:
            groupTitle_input = 'NULL'
        try:
            classTitle_input = capitalize_sentence(ssic_df[ssic_df[f'{userUISelection}'] == ssicCode].reset_index(drop = True)['Class Title'][0])
        except:
            classTitle_input = 'NULL'
        try:
            subclassTitle_input = capitalize_sentence(ssic_df[ssic_df[f'{userUISelection}'] == ssicCode].reset_index(drop = True)['SSIC 2020 Title'][0])
        except:
            subclassTitle_input = 'NULL'

        if userUISelection == 'SSIC 2020':
            userUISelection = subclass
        if userUISelection == 'Section, 2 digit code':
            userUISelection = section
        
        details_display = {
            section: sectionTitle_input,
            division: divisionTitle_input,
            group: groupTitle_input,
            Class: classTitle_input,
            subclass: subclassTitle_input
        }
        details_input = details_display[userUISelection]

        if userUISelection == section and details_input == sectionTitle_input:
            ssicCode = ssic_df[ssic_df['Section Title'].str.lower() == sectionTitle_input.lower()].reset_index(drop = True)['Section'][0]

        if index <= 1: # first 2 indexes are the company's 1st and/or 2nd SSIC codes
            coySSIC_input.append(f"**{ssicCode}**: {details_input}")
        else: # remaining indexes (after 2) are the company's predicted SSIC codes
            predictedSSIC_input.append(f"**{ssicCode}**: {details_input}")

col5, col6 = st.columns([1,1])
with col5:
    st.subheader('Company SSICs & Descriptions:')
    coySSICstring_input = '  \n'.join(coySSIC_input)
    st.write(coySSICstring_input)
with col6:
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

col7, col8 = st.columns([1,1])
with col7:
    st.write(f"<p style='font-size:12px; text-align:left; margin-top:30px;'>*{adjustedWeightDef}</p>", unsafe_allow_html=True)
with col8:
    st.write("")

# Visual Effects ### - https://docs.streamlit.io/develop/api-reference/status
# st.balloons() 
# st.sidebar.success("Explore our pages above ☝️")