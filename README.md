<img src="https://github.com/yorwel/ssicsync/blob/main/images/group_logo.png" width="220" height="170">

# Background

The Singapore Standard Industrial Classification (SSIC) is a critical numerical 5-digit coding system used to classify economic activities in Singapore. It serves as a key indicator in various surveys and databases, providing insights into the economic landscape. When businesses commence operations in Singapore, they are mandated to declare an appropriate SSIC code. However, due to the vast diversity of SSIC codes and the complexity of business activities, ensuring the accuracy of these declared SSIC codes presents a significant challenge.

The primary aim of this project is to enhance the accuracy of SSIC code verification and declarations. We seek to achieve this by using state of the art Natural Language Processing (NLP) techniques.

# How to use
1. Git pull this repository from 'main' branch.
2. Update [list of companies](https://github.com/yorwel/ssicsync/blob/main/dataSources/input_listOfCompanies.csv) to predict SSIC codes.
3. Upload companies' annual reports [here](https://github.com/yorwel/ssicsync/tree/main/dataSources/input_rawPDFReports). Insert company's UEN number in the file names [e.g., ABC PTE LIMITED (199999999C).pdf].
4. Ensure that SSIC Code's [source of truth](https://github.com/yorwel/ssicsync/tree/main/dataSources/DoS) is updated.
5. Create virtual environment ("python -m venv myenv" followed by "myenv\Scripts\activate").
6. Install required packages ("pip install -r requirements_repo.txt").
7. If you wish to train the transfer learning models, run "python training.py" and upload model files to [Hugging Face](https://huggingface.co/nusebacra). Change the hard-coded values at the top of script to your preference.
8. Run "python main.py" to generate predicted SSIC results for the list of companies. Change the hard-coded values at the top of script to your preference.
9. Push updated files back to main branch (*important*). Exclude model files.
10. Visualize results on [Streamlit](https://ssicsync-nwdmvmh4vzhx4yfzqphazs.streamlit.app/).

# Directory

The folders in this repository are described as follows:

- Data Sources (dataSources)
  - Data sources required for the project.
- Data Models (models)
  - Analytical techniques explored includes summarizer and transfer learning (multi-class classification) models.
- Streamlit Pages (pages)
  - Codes for generating Streamlit webpage.
- Results (results)
  - Results file generated from this repository.
- Logs (logs)
  - Log files generated from this repository.
- Images (images)
  - Contains ACRA and our group logos.
 
# Contributors
1. Ang Mei Chi
2. Lee Kuan Teng Roy
3. Liu Wudi
4. Michael Wong Wai Kit
5. Ong Wee Yang