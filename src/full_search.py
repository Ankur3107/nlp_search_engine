base_path = '../ner_model_processed_data_v3/'

from collections import OrderedDict
import tqdm
from fuzzywuzzy import fuzz
from symspellpy import SymSpell, Verbosity
from pprint import pprint
from random import sample
import json
import re
import spacy
from spacy import displacy
from spacy.gold import biluo_tags_from_offsets
import pandas as pd
from tfidf_search import *

with open(base_path+'other_index.json', 'r') as fp:
    global_filter_index = json.load(fp)

# Load ner model
ner = spacy.load(base_path+'ner_model_v3_new/')

# load spell checker
sym_spell_filter = SymSpell(max_dictionary_edit_distance=2, prefix_length=3)
sym_spell_filter.load_dictionary(base_path+'word_freq_2.txt', term_index=0, count_index=1)

# load syn_lookup table
with open(base_path+'synonyms_index.json', 'r') as fp:
    syn_lookup = json.load(fp)

# load tfidf model
def load_tfidf_model(file_name='movie_person_processed_data.csv'):
    df = pd.read_csv(base_path+file_name)
    model = SearchEngine(text_column='processed_name', analyzer='char_wb')
    model.fit(df, ngram_range=(2,5))
    return model

model = load_tfidf_model()

def get_syn_updated_query(query):
    tokens = query.split(" ")
    final_tokens = []

    for token in tokens:
        if token in syn_lookup:
            final_tokens.append(syn_lookup[token])
        else:
            final_tokens.append(token)

    return " ".join(final_tokens)

def get_spellchecked_word_filter(sym_spell, word, max_edit_distance=2):
  result = OrderedDict()
  suggestions = sym_spell.lookup(word, Verbosity.CLOSEST,
                               max_edit_distance=max_edit_distance)
  return [s.term for s in suggestions]

def k_char_filter(input_text, candidates, k=3):
  if len(input_text)>=k:
    filtered_candidates = []
    for c in candidates:
      if input_text[0:k]==c[0:k]:
        filtered_candidates.append(c)
    return filtered_candidates
  else:
    return candidates

def search(text, type, verbose=0):
  final_dict = {}
  tokens = re.split('[-_ ]', text)
  unprocessed_tokens = []
  if type not in ['person_name', 'movie_name']:
    if verbose>0:
      print("processing ...",type)
    for token in tokens:
      if token in global_filter_index:
        dict_value = global_filter_index[token][0]
        slot = {
            "entity":dict_value[2],
            "rawValue":token,
            "slotName":dict_value[2],
            "value":dict_value[0],
            "popularity_score":dict_value[1]
        }
        final_dict[token] = slot
      else:
        candidates = get_spellchecked_word_filter(sym_spell_filter, token, 2)
        if verbose>0:
          print('* all candidates :', candidates)
        candidates = k_char_filter(token,candidates,3)
        if verbose>0:
          print('* candidates :', candidates)
        if len(candidates)>0:
          dict_value = global_filter_index[candidates[0]][0]
          slot = {
              "entity":dict_value[2],
              "rawValue":candidates[0],
              "slotName":dict_value[2],
              "value":dict_value[0],
              "popularity_score":dict_value[1]
          }
          final_dict[token] = slot
        else:
          final_dict[token] = {
              "entity":type,
              "rawValue":text,
              "slotName":type,
              "value":text,
              "popularity_score":-1
          }
          unprocessed_tokens.append(token)
  else:
    if text!='':
      results = model.get_results(text)
      final_dict[text] = df_to_json_v2(results)
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