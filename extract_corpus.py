""" extract_corpus.py
For command line instructions, see README.
For details about json files, see https://www.codeflow.site/fr/article/python-json

TODO test avec des subfolders ?
TODO argparse
"""

import json
import os
import argparse
import random

def extract_file(infile):
  """Extracts a file, gets rid of the POS tags, tokenizes it.
  Sentences are split into words based on " ". Nothing is done to uppercase letters or punctuation.
  Calibrated for the "L'Est républicain" corpus, cf README for the original download links.

  -> infile: string, path to the file
  <- tokenized_doc: list of lists of strings, a tokenized doc made of sentences made of words
  """
  tokenized_doc = []
  with open(infile, 'r', encoding = "utf-8-sig") as f:
    for line in f.readlines():
      sentence = []
      
      for word in line.split():
        sentence.append(word.split("/")[0])
      tokenized_doc.append(sentence)
  return tokenized_doc


def serialize(data, save_as):
    """ Serializes data in a json file saved on desktop.

    TODO est-ce qu'on peut préciser l'emplacement du fichier à sauvegarder ?
    est-ce que switcher de nom à path ferait l'affaire...?
    bon on test

    -> data: any, the object you want to serialize
    -> save_as: string, the path of the file you want to create, don't forget the
    .json extension
    """
    with open(save_as, "w+") as file:
        json.dump(data, file)


def deserialize(infile):
    """ Reads a json file and returns its contents.

    -> infile: string, path to the json file. Don't forget the ".json" extension.
    <- list of words
    """
    data = None
    with open(infile) as json_data:
        data = json.load(json_data)
    return data


def extract_corpus(corpus_path, save_as, number_sentences=0) :
  """ Extracts and serializes a set number of sentences, sampled over the whole
  corpus. If that number < 1, will keep the whole corpus.
  The corpus folder should only contain text files, subfolders will be ignored.

  NOTE: a file contains 100 000 lines, one folder contains 25 files, we have 3 folders.

  TODO: test with subfolders

  -> corpus_path: string, path to the corpus folder
  -> number_sentences: int, total number of sentences to be extracted
  -> save_as: string, path to the file to be created
  """
  file_list = os.listdir(corpus_path)
  corpus_doc = []

  for file in file_list:
    print(file)
    if file[0] != '.': # TODO check: is it for hidden files?
      corpus_doc += extract_file(corpus_path+'/'+file)

  if number_sentences < 1:
    serialize(random.shuffle(corpus_doc)[:number_sentences], save_as)
  else:
    serialize(corpus_doc, save_as)

  
if __name__ == "__main__":
  extract_corpus("./EP.tcs.melt.utf8.c", 100, "./test.json")

#TODO argparse
