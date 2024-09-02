import streamlit as st
from PIL import Image
from main import section, division, group, Class, subclass

# hard-coding
acraLogoPath = 'images/ACRA_logo.jpg'

# Set page config
st.set_page_config(
    page_title='ssicsync', # Set display name of browser tab
    # page_icon="🔍", # Set display icon of browser tab
    layout="wide", # "wide" or "centered"
    initial_sidebar_state="expanded"
)

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

# Visual Effects ### - https://docs.streamlit.io/develop/api-reference/status
# st.balloons() 

# st.sidebar.success("Explore our pages above ☝️")

# Display the logo image at the top left corner with a specific width
col1, col2, col3 = st.columns([1, 10, 1])  # Adjust column proportions as needed

with col1:
    st.image(acraLogoPath, caption='', output_format='JPEG', width=150)  # Set width to shrink the image

st.write("## About this Webpage")

st.markdown(
    '''
This platform offers an interactive exploration of SSIC classification results, \
from overall accuracy metrics to detailed company-level analyses. \
Users can leverage the Prediction pages to input custom company descriptions, \
allowing the model to generate and return the most relevant SSIC codes based on the specified hierarchical level. \
Additionally, the Reference pages provide a quick search feature for SSIC codes, \
enabling users to gain a deeper understanding of their applications.
'''
)

st.write("## Table of Contents")

st.markdown(
f'''
**Results**\n
This section presents the overall classification results as well as SSIC results at the company level. \
It is particularly useful for validating companies' declared SSIC codes against the recommended SSIC codes.


**Prediction (Section)**\n
This section enables users to apply the classification model to ad-hoc company descriptions, returning the top SSIC codes at the Section level. \
It is ideal for conducting quick analyses to obtain the recommended SSIC codes at the Section level.

**Prediction (Division)**\n
This section enables users to apply the classification model to ad-hoc company descriptions, returning the top SSIC codes at the Division level. \
It is ideal for conducting quick analyses to obtain the recommended SSIC codes at the Division level.

**Prediction (Group)**\n
This section enables users to apply the classification model to ad-hoc company descriptions, returning the top SSIC codes at the Group level. \
It is ideal for conducting quick analyses to obtain the recommended SSIC codes at the Group level.

**Prediction (Class)**\n
This section enables users to apply the classification model to ad-hoc company descriptions, returning the top SSIC codes at the Class level. \
It is ideal for conducting quick analyses to obtain the recommended SSIC codes at the Class level.

**Prediction (Sub-class)**\n
This section enables users to apply the classification model to ad-hoc company descriptions, returning the top SSIC codes at the Sub-class level. \
It is ideal for conducting quick analyses to obtain the recommended SSIC codes at the Sub-class level.


**Reference (Section)**\n
This section allows users to search across {section} SSIC Section codes and its related keywords in SSIC titles. \
The search results will provide a relevant list of SSIC Section codes and titles based on the input terms. \
This feature is ideal for gaining a comprehensive understanding of the meanings and applications of each SSIC Section code.

**Reference (Division)**\n
This section allows users to search across {division} SSIC Division codes and its related keywords in SSIC titles. \
The search results will provide a relevant list of SSIC Division codes and titles based on the input terms. \
This feature is ideal for gaining a comprehensive understanding of the meanings and applications of each SSIC Division code.

**Reference (Group)**\n
This section allows users to search across {group} SSIC Group codes and its related keywords in SSIC titles. \
The search results will provide a relevant list of SSIC Group codes and titles based on the input terms. \
This feature is ideal for gaining a comprehensive understanding of the meanings and applications of each SSIC Group code.

**Reference (Class)**\n
This section allows users to search across {Class} SSIC Class codes and its related keywords in SSIC titles. \
The search results will provide a relevant list of SSIC Class codes and titles based on the input terms. \
This feature is ideal for gaining a comprehensive understanding of the meanings and applications of each SSIC Class code.

**Reference (Sub-class)**\n
This section allows users to search across {subclass} SSIC Sub-class codes and its related keywords in SSIC titles. \
The search results will provide a relevant list of SSIC Sub-class codes and titles based on the input terms. \
This feature is ideal for gaining a comprehensive understanding of the meanings and applications of each SSIC Sub-class code.

'''
)


