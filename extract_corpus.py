""" extract_corpus.py
Extracts, tokenizes and serializes all the files in a folder.
Calibrated for the "L'Est Républicain" corpus, though you do have to put all the files in one single
folder prior to running.

TODO test avec des subfolders ?
"""

import os
import argparse
from serialisation import serialize

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


def extract_corpus(corpus_path, number_files, verbose):
  """ Extracts, tokenizes and serializes a set number of files.
  If that number < 1, will take all files.
  The corpus folder should only contain text files, subfolders will be ignored.

  NOTE: a file contains 100 000 lines, one folder contains 25 files, we have 3 folders.

  TODO: test with subfolders

  -> corpus_path: string, path to the corpus folder
  -> number_files: int, total number of sentences to be extracted
  <- corpus_doc: list of lists of strings, the final tokenized doc
  """
  file_list = os.listdir(corpus_path)
  corpus_doc = []

  if number_files < 1: number_files = len(file_list) # If < 1, then we take as many files as there are.

  for file in file_list:
    if number_files == 0: break # We only extract however many files we want.
    if file[0] != '.': # To avoid hidden files...
      if verbose: print(file)
      corpus_doc.extend(extract_file(corpus_path+'/'+file))
      number_files -= 1
  
  if verbose:
    print("First 3 sentences: ")
    print(corpus_doc[:3])
    print("Number of sentences: "+ str(len(corpus_doc)))
    
  return corpus_doc

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('corpus_path', default=None, type=str, help='Path to folder with the raw text files') 
  parser.add_argument('save_path', default="./corpus.json", type=str, help='Path to the file to be created')
  parser.add_argument('--number_files', default=0, type=int, help='Number of files to be extracted')
  parser.add_argument('--verbose', default=True, type=bool, help='Verbose mode')
  args = parser.parse_args()

  corpus_doc = extract_corpus(corpus_path = args.corpus_path, number_files = args.number_files, verbose=args.verbose)

  serialize(corpus_doc, args.save_path)
