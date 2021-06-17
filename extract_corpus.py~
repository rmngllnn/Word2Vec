""" extract_corpus.py
Extracts, tokenizes and serializes a given number of sentences (or all) from the files in a folder.
Calibrated for the "L'Est Républicain" corpus, see extract_file() and README.
Warning, be careful to put all the files in one single folder prior to running.

TODO (optionnel) test avec des subfolders ?
"""

import os
import argparse
from serialisation import serialize
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


def extract_corpus(corpus_path, number_files, number_sentences, verbose):
  """ Extracts, tokenizes and serializes a set number of sentences from a set number of files.
  If that number < 1, will take all.
  The corpus folder should only contain text files, subfolders will be ignored.

  NOTE: a file contains 100 000 lines, one folder contains 25 files, we have 3 folders.

  -> corpus_path: string, path to the corpus folder
  -> number_files: int, total number of files to be extracted
  -> number_sentences: int, total number of sentences to be included in the doc
  -> verbose: bool, verbose mode
  <- corpus_doc: list of lists of strings, the final tokenized doc
  """
  file_list = os.listdir(corpus_path)
  file_list = [file for file in file_list if file[0] != '.' for file in file_list]

  if number_files > 0 and number_files < len(file_list):
    random.shuffle(file_list)
    file_list = file_list[:number_files]

  corpus_doc = []
  for file in file_list:
    if verbose: print(file)
    corpus_doc.extend(extract_file(corpus_path+'/'+file))
  
  if number_sentences < len(corpus_doc) and number_sentences > 0:
    random.shuffle(corpus_doc)
    corpus_doc = corpus_doc[:number_sentences]

  if verbose:
    print("First 3 sentences: ")
    print(corpus_doc[:3])
    print("Number of sentences: "+ str(len(corpus_doc)))
    
  return corpus_doc

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('corpus_path', default=None, type=str, help='Path to folder with the raw text files to be extracted') 
  parser.add_argument('save_path', default="./tokenized_doc.json", type=str, help='Path to the json file to be created, do not forget the extension')
  parser.add_argument('--number_files', default=1, type=int, help='Number of files to be extracted, randomly, from the folder, if all: 0')
  parser.add_argument('--number_sentences', default=1000, type=int, help='Total number of sentences to be extracted, randomly, from the files extracted, if all: 0')
  parser.add_argument('--verbose', default=True, type=bool, help='Verbose mode')
  args = parser.parse_args()

  corpus_doc = extract_corpus(corpus_path = args.corpus_path, number_files = args.number_files, number_sentences = args.number_sentences, verbose=args.verbose)

  serialize(corpus_doc, args.save_path)
