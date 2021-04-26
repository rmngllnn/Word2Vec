#Dans ce fichier, on se propose d'initialiser l'implémentation du modèle word2vec, en exploitant le corpus (extraction, indices, nettoyage, embeddings)
#(Très) inspiré TP8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

torch.manual_seed(1)

CONTEXT_SIZE = 2 #pour commencer
EMBEDDING_DIM = 10 #pour commencer


def extract_corpus(infile):
  """Extracts the corpus, cleans it up.
  Calibrated for the "L'Est républicain" corpus, cf README for the original download links.
  -> infile: string, path to the file
  <- doc: list (doc) of strings (sentences)
  """
  doc = []
  with open(infile, 'r') as f:
    for line in f.readlines():
      pass #TODO : parcourir chaque ligne, supprimer les tags, gérer ou non les majuscules, attention à l'encodage (UTF-8)O
  return doc


def get_tokens_and_indices(doc, minimum_frequency = 0.0):
  """Tokenizes the document and generates vocabulary indexes from it.

  -> doc: list of strings (sentences)
  -> minimum_frequency: frequency threshold under which words are ignored

  <- tokens: list of lists of strings
  <- i2w: list, iw2[index(int)] = "word"
  <- w2i: doc, w2i["word"] = index(int)
  """
  tokens = []
  flat_doc = []
  vocab = set()

  for sentence in doc:
    tokens_sentence = sentence.split()
    vocab = vocab.union(set(tokens_sentence))
    tokens.append(tokens_sentence)
    flat_doc = flat_doc + tokens_sentence

  total_occurences = len(flat_doc)
  for word in vocab:
    if flat_doc.count(word)/total_occurences < minimum_frequency:
      vocab.remove(word)

  vocab = vocab.union(set(["UNK"] + ["*D"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)] + ["*F"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)]))
  i2w = list(vocab)
  w2i = {w: i for i, w in enumerate(i2w)}
  return tokens, i2w, w2i


test_corpus = ["This is a test .","And a second test !", "Ok that's enough now ."]
tokens, i2w, w2i = get_tokens_and_indices(test_corpus)


#Création des exemples d'apprentissage avec negative sampling
#Aussi, création des classes gold 1/0 (ou 1/-1) dans une liste de même taille que vocab (note Romy : pas sûre...)
def create_examples (tokens, w2i) :
  """Un exemple est une paire (liste de mots contexte, mot cible), la sortie est une liste d'exemple de taille len(vocab) et une liste des classes gold"""
  pass #TO DO

examples = create_examples(tokens, w2i)


#PARTIE MAIN
parser = argparse.ArgumentParser()
#Dans le terminal, on écrira "python3 init.py NOM_DU_FICHIER_CORPUS"
parser.add_argument('examples_file', default=None, help='Corpus utilisé pour la création d\'exemples d\'apprentissage pour les embeddings.')
