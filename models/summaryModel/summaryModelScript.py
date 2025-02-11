import pandas as pd
import torch
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def summaryModel(self, logger):

    # hard-coded values:

    tfidf_threshold = 0.1
    words_to_remove = ['principal', 'activity', 'activities', 'investment', 'holding', 'holdings', 
                       'singapore', 'exchange', 'securities', 'trading', 'limited', 'company', 
                       'extracted', 'keywords']
    
    # functions:

    def get_answer(row):
        context = row['Notes Page Content']
        question = "What are all the principal activities of the company? List down all the activities."
        result = question_answer(question=question, context=context)
        return result['answer']

    def dynamic_summarizer(summarizer, text, min_length=30, length_fraction=0.9, too_short_threshold=30):
        input_length = len(text.split())
        if input_length < too_short_threshold:
            return text
        max_length = int(input_length * length_fraction)
        max_length = max(min_length, max_length)
        summary = summarizer(text, max_length=max_length, min_length=min_length)
        return summary[0]['summary_text']
    
    def get_important_terms(doc_index, tfidf_matrix, terms, threshold, top_tokens=10):
            term_scores = tfidf_matrix[doc_index].toarray().flatten()
            important_term_indices = term_scores >= threshold
            important_terms = [terms[i] for i in range(len(terms)) if important_term_indices[i]]
            original_text = df_input.at[doc_index, tfidf_column].split()
            important_terms_in_order = [term for term in original_text if term in important_terms]
            return ' '.join(important_terms_in_order[:top_tokens])

    device = 0 if torch.cuda.is_available() else -1
    df_input = pd.read_csv("dataSources/extractionOutputFiles/pdfExtractionOutputs.csv")

    # Initialize the summarizers
    summarizer_facebook_bart = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    summarizer_philschmid_bart = pipeline("summarization", model="philschmid/bart-large-cnn-samsum", device=device)
    summarizer_azma_bart = pipeline("summarization", model="Azma-AI/bart-large-text-summarizer", device=device)
    question_answer = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)

    list_of_summarizer = [
        (summarizer_facebook_bart, 'Summarized_Description_facebook_bart'),
        (summarizer_philschmid_bart, 'Summarized_Description_philschmid_bart'),
        (summarizer_azma_bart, 'Summarized_Description_azma_bart')
    ]
    
    df_input['Q&A model Output'] = df_input.apply(get_answer, axis=1)
    df_input['Input_length'] = df_input['Notes Page Content'].apply(lambda x: len(x.split()))

    for summarizer, output_column in list_of_summarizer:
        df_input[output_column] = df_input['Notes Page Content'].apply(
            lambda x: dynamic_summarizer(summarizer, x)
        )
    
    df_input['Summarised?'] = df_input.apply(lambda row: 'No' if row['Notes Page Content'] == row['Summarized_Description_azma_bart'] else 'Yes',axis=1)
    
    #stopwords removal
    stop_words = list(ENGLISH_STOP_WORDS.union(words_to_remove))

    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_columns = {
        'Summarized_Description_azma_bart': 'Azma_bart_tfidf',
        'Summarized_Description_facebook_bart': 'FB_bart_tfidf',
        'Summarized_Description_philschmid_bart': 'Philschmid_bart_tfidf',
  #      'llm_output_llama3.1': 'llm_output_llama3.1_tfidf'  
    }

    for summary_column, tfidf_column in tfidf_columns.items():
        df_input[tfidf_column] = df_input[summary_column].apply(lambda x: ' '.join(x.split()))
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_input[tfidf_column])

        terms = tfidf_vectorizer.get_feature_names_out()
        
        for idx in range(tfidf_matrix.shape[0]):
            important_terms = get_important_terms(idx, tfidf_matrix, terms, tfidf_threshold)
            if important_terms:
                df_input.at[idx, tfidf_column] = important_terms
            else:
                df_input.at[idx, tfidf_column] = df_input.at[idx, 'Notes Page Content']

    df_input.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')

    df_input.to_csv("models/summaryModel/modelOutputFiles/pdfModelSummaryOutputs.csv")
    logger.info('Summary outputs CSV generated.')