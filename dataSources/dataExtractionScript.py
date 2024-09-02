import os
import glob
import pdfplumber
import pandas as pd
import re

def pdfExtraction(self, logger):

    # hard-coded values:
    folder_path = r'dataSources/input_rawPDFReports'
    final_folder_path = r'dataSources/extractionOutputFiles/pdfExtractionOutputs.csv'

    # functions:

    # Function to sanitize text
    def sanitize_text(text):
        # Replace non-printable characters with a space
        sanitized_text = re.sub(r'[^\x20-\x7E]+', ' ', text)
        return sanitized_text

    # Function to extract UEN from filename
    def extract_uen(filename):
        match = re.search(r'\((\d{9}[A-Z])\)', filename)
        if match:
            return match.group(1)
        return ""

    def extract_principal_activities(text, subsidiary_keywords):
        # Find the position of the "Principal Activities" or "Principal Activity" section
        start_pos = re.search(r'principal activit(y|ies)', text, re.IGNORECASE)
        
        if start_pos:
            # Extract text starting from the "Principal Activities" or "Principal Activity" section
            start_pos = start_pos.start()
            extracted_text = text[start_pos:]
            
            # Look back one or two sentences to check for description of nature of business
            nature_description = ""
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text[:start_pos])
            for i in range(1, min(len(sentences), 3)):
                # Check if the sentence describes an address (adjust patterns as needed)
                if re.search(r'address of its registered off|principal place of business|located|listed on', sentences[-i], re.IGNORECASE):
                    continue  # Skip this sentence if it's an address description
                
                nature_description += sentences[-i] + " "
            
            # Check if the first sentence of the principal activities section contains subsidiary keywords
            first_sentence = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', extracted_text.strip())[0]
            contains_subsidiary_keyword = any(keyword in first_sentence.lower() for keyword in subsidiary_keywords)
            
            # Specific stop patterns
            stop_patterns = [
                r"The (principal )?activit(y|ies) of .*? are .*? in (N|n)ot(e|es) \d+",
                r"The (principal )?activit(y|ies) of .*? are .*? to financial statemen(t|ts) below",
                r"principal activit(y|ies) of .*? (subsidiar(y|ies)|joint ventures?|associates?|associated companies?) (and .*?)? are .*? in (N|n)ot(e|es) \d+( and \d+)?",
                r"The (principal )?activit(y|ies) and .*? are .*? in (N|n)ot(e|es) \d+",
                r"2 BASIS OF PREPARATION",
                r"2 Basis of preparation",
                r"The financial statements? of .*? as at .*? financial year",
                r"The Compan(y|ies)? is listed on"
            ]

            # Check if any stop pattern matches
            for pattern in stop_patterns:
                stop_pos = re.search(pattern, extracted_text, re.IGNORECASE)
                if stop_pos:
                    extracted_text = extracted_text[:stop_pos.start()]
                    break
            
            # Stop at a keyword or typical section header (adjust this as needed)
            stop_keywords = ["2.", "Note 2", "2 ", "Another Section Header"]
            for keyword in stop_keywords:
                stop_pos = extracted_text.find(keyword)
                if stop_pos != -1:
                    extracted_text = extracted_text[:stop_pos]
                    break
            
            # Include nature_description if the first sentence contains subsidiary keywords
            if contains_subsidiary_keyword:
                return nature_description + extracted_text
            
            # If not, return extracted_text alone
            return extracted_text
    
        return ""

    # Initialize a list to store the extracted data
    data = []

    # Keywords for "notes to the" variations with different spacings
    notes_keywords = [r'note\s*to', r'notes\s*to', r'note\s*  *to', r'notes\s*  *to', r'note\s*    *to', r'notes\s*    *to', r'These|these notes form an integral part of']

    # Keywords indicating subsidiary-related content
    subsidiary_keywords = ['subsidiaries', 'subsidiary', 'joint ventures', 'joint venture', 'associates', 'associate']

    # Loop through all the PDF files
    pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))

    for pdf_file in pdf_files:
        # Get the name of the PDF file
        pdf_name = os.path.basename(pdf_file)
        
        try:
            # Create a PDF file reader object
            with pdfplumber.open(pdf_file) as pdf_reader:
                total_pages = len(pdf_reader.pages)
                found_notes_page = False
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text:
                        sanitized_page_text = sanitize_text(page_text)
                        
                        # Check if the page contains any of the "notes to the" variations
                        if any(re.search(keyword, sanitized_page_text.lower()) for keyword in notes_keywords):
                            # Check if the page contains the phrases "principal activit(ies|y)"
                            if re.search(r'principal activit(ies|y)', sanitized_page_text, re.IGNORECASE):
                                # Extract principal activities section
                                principal_activities_text = extract_principal_activities(sanitized_page_text, subsidiary_keywords)
                                
                                if principal_activities_text:
                                    # Extract UEN from filename
                                    uen_number = extract_uen(pdf_name)
                                    
                                    text_data = {
                                        'PDF Name': pdf_name,
                                        'Page Number': page_num + 1,
                                        'UEN Number': uen_number,
                                        'Notes Page Content': principal_activities_text
                                    }
                                    data.append(text_data)
                                    found_notes_page = True
                                    break  # Stop searching after the first occurrence
                
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
            continue

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    df.to_csv(final_folder_path)
    logger.info('PDF extraction outputs CSV generated.')

def websiteScraping(self, logger): # not linked to main.py!

    # TODO for Roy to input

    return

def linkedInScraping(self, logger): # not linked to main.py!

    # TODO for Roy to input
    
    return