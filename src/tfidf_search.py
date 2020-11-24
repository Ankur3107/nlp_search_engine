import nltk
nltk.download('stopwords')

import copy
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel


class SearchEngine():  
    replace_words = {'&': '_and_', 'unknown':' '}    

    def __init__(self, text_column='name', id_column='id', analyzer='word'):
        self.text_column = text_column
        self.id_column = id_column
        self.analyzer = analyzer
        pass
    
    def fit(self, df, ngram_range=(1,3), perform_stem=False):
        self.df = df
        self.perform_stem = perform_stem
        doc_df = self.preprocess(df)
        stopWords = stopwords.words('english')    
        self.vectoriser = CountVectorizer(stop_words = stopWords, ngram_range=ngram_range, analyzer=self.analyzer)
        train_vectorised = self.vectoriser.fit_transform(doc_df)
        self.transformer = TfidfTransformer()
        self.transformer.fit(train_vectorised)
        self.fitted_tfidf = self.transformer.transform(train_vectorised)

    def preprocess(self, df):
        result = df[self.text_column]
        result = np.core.defchararray.lower(result.values.astype(str))
        for word in self.replace_words:
            result = np.core.defchararray.replace(result, word, self.replace_words[word])
        if self.perform_stem:
            result = self.stem_array(result)
        return result

    def preprocess_query(self, query):
        result = query.lower()
        for word in self.replace_words:
            result = result.replace(word, self.replace_words[word])
        if self.perform_stem:
            result = self.stem_document(result)
        return result

    def stem_array(self, v):
        result = np.array([self.stem_document(document) for document in v])
        return result
    
    def stem_document(self, text):
        result = [stem(word) for word in text.split(" ")]
        result = ' '.join(result)
        return result
    
    def get_results(self, query, max_rows=10):
        query = query.strip()
        query = query.lower()
        score = self.get_score(query)
        results_df = copy.deepcopy(self.df)
        results_df['ranking_score'] = score
        results_df = results_df.loc[score>0]
        results_df = results_df.iloc[np.argsort(-results_df['ranking_score'].values)]
        results_df = results_df.head(max_rows)
        return results_df.reset_index(drop=True)        
        
    def get_score(self, query):
        query_vectorised = self.vectoriser.transform([query])    
        query_tfidf = self.transformer.transform(query_vectorised)
        cosine_similarities = linear_kernel(self.fitted_tfidf, query_tfidf).flatten()
        return cosine_similarities


def df_to_json_v2(df):
  #print(df)
  names = df.name.tolist()
  scores = df.ranking_score.tolist()
  types = df.type.tolist()
  popularities = df.popularity.tolist()
  #print("names :",names)
  results = []
  number_out_result = 5
  flag = False
  for i in range(len(names)):
    type = types[i]
    if type=='person':
      type='person_name'
    elif type=='movie':
      type='movie_name'

    name = names[i]
    score = scores[i]
    popularitiy = popularities[i]

    if not flag and i==0 and score>=0.90:
      number_out_result = 1
      flag = True
    elif not flag and i==0 and score>=0.80:
      number_out_result = 2
      flag = True
    elif not flag and i==0 and score>=0.60:
      flag = True
      number_out_result = 3
    elif not flag and i==0 and score>=0.45:
      flag = True
      number_out_result = 5
    elif not flag and i==0 and score<0.45:
      flag = True
      number_out_result = 0

    results.append({'entity': type,
                'match_score': score,
                'popularity_score': popularitiy,
                'slotName': type,
                'value': name,
                'isConflicting':'no'})
  if number_out_result==0:
    return []
  return results[0:number_out_result]

def df_to_json(df, k=5, sort_by_popularity=True):
  if sort_by_popularity:
    df = df.sort_values('popularity', ascending=False)
  names = df.name.tolist()
  scores = df.ranking_score.tolist()
  types = df.type.tolist()
  popularities = df.popularity.tolist()
  results = []
  for i in range(len(names)):
    type = types[i]
    if type=='person':
      type='person_name'
    elif type=='movie':
      type='movie_name'

    name = names[i]
    score = scores[i]
    popularitiy = popularities[i]

    results.append({'entity': type,
                'match_score': score,
                'popularity_score': popularitiy,
                'slotName': type,
                'value': name,
                'isConflicting':'no'})

  return results[0:k]


def df_to_json_v3(df):
  #print(df)
  names = df.name.tolist()
  scores = df.ranking_score.tolist()
  types = df.type.tolist()
  popularities = df.popularity.tolist()
  #print("names :",names)
  results = []
  number_out_result = 5
  flag = False
  for i in range(len(names)):
    type = types[i]
    if type=='person':
      type='person_name'
    elif type=='movie':
      type='movie_name'

    name = names[i]
    score = scores[i]
    popularitiy = popularities[i]

    if not flag and i==0 and score>=0.90:
      number_out_result = 1
      flag = True
    elif not flag and i==0 and score>=0.80:
      number_out_result = 2
      flag = True
    elif not flag and i==0 and score>=0.60:
      flag = True
      number_out_result = 3
    elif not flag and i==0 and score>=0.45:
      flag = True
      number_out_result = 5
    elif not flag and i==0 and score<0.45:
      flag = True
      number_out_result = 0

    results.append({'entity': type,
                'match_score': score,
                'popularity_score': popularitiy,
                'slotName': type,
                'value': name,
                'isConflicting':'no'})
  if number_out_result==0 or number_out_result>2:
    result_indexes = df[df.type=='person'].index.tolist()[0:3]+df[df.type=='movie'].index.tolist()[0:3]
    final_results = []
    for i in result_indexes:
      i_result = results[i]
      i_result['isConflicting'] = 'yes'
      final_results.append(i_result)
    return final_results
  else:
    return results[0:number_out_result]
