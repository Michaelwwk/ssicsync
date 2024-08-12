import streamlit as st
from PIL import Image

# Set page config
st.set_page_config(
    page_title='ssicsync', # Set display name of browser tab
    page_icon="🔍", # Set display icon of browser tab
    layout="wide", # "wide" or "centered"
    initial_sidebar_state="expanded",
    menu_items={
        'About': '''Explore multiclass text classification with DistilBERT on our Streamlit page. 
        Discover interactive insights and the power of modern NLP in text categorization!'''
    }
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
    st.image('image/ACRA_logo2.jpg', caption='', output_format='JPEG', width=200)  # Set width to shrink the image

st.write("## Welcome to ssicsync streamlit page! 👋")

st.markdown(
    '''
    Welcome to our Streamlit page! We explore multiclass text classification using DistilBERT, <br>
    offering interactive insights into training and evaluating models for accurate text categorization. <br>
    Join us to learn and experience the power of modern NLP 🤖.
''', unsafe_allow_html=True
)


