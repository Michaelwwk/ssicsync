import pandas as pd
import torch
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
# import ollama

def trainingSummaryModel(self, logger):

    # functions

    def get_answer(row):
        context = row['Notes Page Content']
        question = "What are all the principal activities of the company? List down all the activities."
        result = question_answer(question=question, context=context)
        return result['answer']
    
    # def summarize_business_activity(content, model_name='llama3.1',too_short_threshold=3):
    #     input_length = len(content.split())
    #     if input_length < too_short_threshold:
    #         return content  # If the content is too short, return it as is
    
    #     response = ollama.chat(model=model_name, messages=[
    #         {
    #             'role': 'user',
    #             'content': f'''
    #             {content}

    #             list down all the principal activity
    #             ''',
    #         },
    #     ])
    #     response_v = response['message']['content']  # Get the response content
    #     return response_v
    
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
    df_input = pd.read_csv("dataSources/scrapedOutputFiles/pdfScrapedOutputs.csv")
<<<<<<< Updated upstream
    
=======

>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    # df_input['llm_output_llama3.1'] = df_input['Notes Page Content'].apply(lambda x: summarize_business_activity(x))   
=======

    # def summarize_business_activity(content, model_name='llama3.1',too_short_threshold=3):
    #     input_length = len(content.split())
    #     if input_length < too_short_threshold:
    #         return content  # If the content is too short, return it as is
    
    #     response = ollama.chat(model=model_name, messages=[
    #         {
    #             'role': 'user',
    #             'content': f'''
    #             {content}

    #             list down all the principal activity
    #             ''',
    #         },
    #     ])
    #     response_v = response['message']['content']  # Get the response content
    #     return response_v
    
    # df_input['llm_output_llama3.1'] = df_input['Notes Page Content'].apply(lambda x: summarize_business_activity(x))   
    
    def dynamic_summarizer(summarizer, text, min_length=30, length_fraction=0.9, too_short_threshold=30):
        input_length = len(text.split())
        if input_length < too_short_threshold:
            return text
        max_length = int(input_length * length_fraction)
        max_length = max(min_length, max_length)
        summary = summarizer(text, max_length=max_length, min_length=min_length)
        return summary[0]['summary_text']
    
>>>>>>> Stashed changes
    df_input['Input_length'] = df_input['Notes Page Content'].apply(lambda x: len(x.split()))

    for summarizer, output_column in list_of_summarizer:
        df_input[output_column] = df_input['Notes Page Content'].apply(
            lambda x: dynamic_summarizer(summarizer, x)
        )
    
    df_input['Summarised?'] = df_input.apply(lambda row: 'No'\
                              if row['Notes Page Content'] == row['Summarized_Description_azma_bart'] else 'Yes',axis=1)
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_columns = {
        'Summarized_Description_azma_bart': 'Azma_bart_tfidf',
        'Summarized_Description_facebook_bart': 'FB_bart_tfidf',
        'Summarized_Description_philschmid_bart': 'Philschmid_bart_tfidf',
<<<<<<< Updated upstream
        # 'llm_output_llama3.1': 'llm_output_llama3.1_tfidf'  
=======
  #      'llm_output_llama3.1': 'llm_output_llama3.1_tfidf'  
>>>>>>> Stashed changes
    }

    for summary_column, tfidf_column in tfidf_columns.items():
        df_input[tfidf_column] = df_input[summary_column].apply(lambda x: ' '.join(x.split()))
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_input[tfidf_column])

        terms = tfidf_vectorizer.get_feature_names_out()
        tfidf_threshold = 0.1
<<<<<<< Updated upstream
    
    # Initialize the summarizers
    summarizer_facebook_bart = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    summarizer_philschmid_bart = pipeline("summarization", model="philschmid/bart-large-cnn-samsum", device=device)
    summarizer_azma_bart = pipeline("summarization", model="Azma-AI/bart-large-text-summarizer", device=device)
    question_answer = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)

    list_of_summarizer = [
        (summarizer_facebook_bart, 'facebook_bart_summary'),
        (summarizer_philschmid_bart, 'philschmid_bart_summary'),
        (summarizer_azma_bart, 'azma_bart_summary')
    ]

    df_input['Q&A model Output'] = df_input.apply(get_answer, axis=1)

    for summarizer, output_column in list_of_summarizer:
        df_input[output_column] = df_input['Notes Page Content'].apply(
            lambda x: dynamic_summarizer(summarizer, x)
        )

    tfidf_vectorizer = TfidfVectorizer()
    for _, output_column in list_of_summarizer:
        tfidf_column = output_column + '_tfidf'
        df_input[tfidf_column] = df_input[output_column].apply(lambda x: ' '.join(x.split()))
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_input[tfidf_column])

        terms = tfidf_vectorizer.get_feature_names_out()
        tfidf_threshold = 0.1

        df_input[tfidf_column] = [
            get_important_terms(i, tfidf_matrix, terms, tfidf_threshold)
            for i in range(tfidf_matrix.shape[0])
        ]
=======

        def get_important_terms(doc_index, tfidf_matrix, terms, threshold, top_tokens=10):
            term_scores = tfidf_matrix[doc_index].toarray().flatten()
            important_term_indices = term_scores >= threshold
            important_terms = [terms[i] for i in range(len(terms)) if important_term_indices[i]]
            original_text = df_input.at[doc_index, tfidf_column].split()
            important_terms_in_order = [term for term in original_text if term in important_terms]
            return ' '.join(important_terms_in_order[:top_tokens])
        

>>>>>>> Stashed changes
    df_input.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')

    df_input.to_csv("models/summaryModel/modelOutputFiles/pdfModelSummaryOutputs.csv")
    logger.info('Summary Outputs CSV generated.')