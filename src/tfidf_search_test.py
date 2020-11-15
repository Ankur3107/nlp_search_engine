from tfidf_search import SearchEngine, df_to_json
import pandas as pd
from pprint import pprint

df = pd.read_csv('../ner_model_processed_data_v3/movie_person_processed_data.csv')
model = SearchEngine(text_column='processed_name', analyzer='char_wb')
model.fit(df, ngram_range=(2,5))

query = 'samlam khan'
results = model.get_results(query)
results = results.sort_values('popularity', ascending=False)
print(results)
#pprint(df_to_json(results))