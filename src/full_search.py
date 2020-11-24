base_path = '../ner_model_processed_data_v3/'

from collections import OrderedDict
import tqdm
from pprint import pprint
from random import sample
import json
import re
import spacy
from spacy import displacy
from spacy.gold import biluo_tags_from_offsets
import pandas as pd
from tfidf_search import *

# Load ner model
ner = spacy.load(base_path+'ner_with_tag_model_v2/') # tag model added

# load syn_lookup table
with open(base_path+'synonyms_index.json', 'r') as fp:
    syn_lookup = json.load(fp)

# load tfidf model
def load_tfidf_model(file_name='movie_person_processed_data_v4.csv'):
    df = pd.read_csv(base_path+file_name)
    model = SearchEngine(text_column='processed_name', analyzer='char_wb')
    model.fit(df, ngram_range=(2,5))
    return model

model = load_tfidf_model()
filter_model = load_tfidf_model(file_name='filters_processed_data.csv')
tag_model = load_tfidf_model(file_name='refined_tags_processed_data.csv')

def get_syn_updated_query(query):
    tokens = query.split(" ")
    final_tokens = []

    for token in tokens:
        if token in syn_lookup:
            final_tokens.append(syn_lookup[token])
        else:
            final_tokens.append(token)

    return " ".join(final_tokens)

def search(text, type, verbose=0):
  final_dict = {}
  if type not in ['person_name', 'movie_name', 'tag']:
    if verbose>0:
      print("processing ...",type)
    if text!='':
      results = filter_model.get_results(text)
      final_dict[text] = df_to_json(results, k=1, sort_by_popularity=False)
    else:
      final_dict[text] = {}
  elif type in ['person_name', 'movie_name']:
    if verbose>0:
      print("processing ...",type)
    if text!='':
      results = model.get_results(text)
      final_dict[text] = df_to_json_v3(results)
    else:
      final_dict[text] = {}
  elif type=='tag':
    if verbose>0:
      print("processing ...",type)
    if text!='':
      results = tag_model.get_results(text)
      final_dict[text] = df_to_json(results, k=1, sort_by_popularity=False)
    else:
      final_dict[text] = {}
  return final_dict

def spacy_search(text):
  doc = ner(text)

  detected_entities = []
  for ent in doc.ents:
    detected_entities.append((ent.text,ent.label_))

  tokens = list(doc)
  tags = biluo_tags_from_offsets(doc, [(ent['start'],ent['end'],ent['label'])for ent in doc.to_json()['ents']])
  undetected_entities = [ tokens[i] for i in range(len(tags)) if tags[i]=='O']

  return detected_entities, undetected_entities
  

def final_search(text, verbose=0):
  text = text.lower()
  #text = get_syn_updated_query(text)
  detected_entities, undetected_entities = spacy_search(text)
  final_dict = {}

  for detected_entity in detected_entities:
    if verbose>0:
      print(detected_entity)
    result_dict = search(detected_entity[0], detected_entity[1], verbose=verbose)
    if verbose>0:
      print(result_dict)
    final_dict.update(result_dict)


  for undetected_entity in undetected_entities:
    print('undetected_entity :', undetected_entity)
    result_dict = search(str(undetected_entity), "None", verbose=verbose)
    final_dict.update(result_dict)
  return final_dict
