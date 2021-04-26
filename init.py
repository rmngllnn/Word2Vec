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
test_corpus = [["This is a test ."],["And a second test !"], ["Ok that's enough now ."]]


def extract_corpus(infile):
  """Extracts the corpus, cleans it up.
  Calibrated for the "L'Est républicain" corpus, cf README for the original download links.
  -> infile: string, path to the file
  <- doc: list (doc) of strings (sentences)
  """
  doc = []
  with f as open(infile, 'r'):
    for line in f.readlines():
      pass #TODO : parcourir chaque ligne, supprimer les tags, gérer ou non les majuscules, attention à l'encodage (UTF-8)O
  return doc


def get_tokens(doc = test_corpus):
  """Tokenizes the document and generates a full vocabulary list from it.
  -> doc: list of strings (sentences)
  <- tokens: list of lists of strings
  """
  tokens = []
  for sentence in doc:
    tokens_sentence = sentence.split()
    tokens.append(tokens_sentence)
  return tokens # TODO est-ce qu'on veut compter les occurences là ? ça serait facile de faire de vocab un dico de compte


def get_i2w_and_w2i(tokens, minimum_frequency = 0.0):
  """Generates an indexed vocabulary, filtering out rare words.
  -> tokens: list of lists (sentences) of strings (words)
  -> minimum_frequency: frequency threshold under which words are ignored
  <- iw2: list, iw2[index(int)] = "word"
  <- w2i: doc, w2i["word"] = index(int)
  """
  flat_doc = join(tokens)
  total_occurences = flat_doc.len()
  full_vocab = set(flat_doc)
  vocab = set(["UNK"] + ["*D"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)] + ["*F"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)])
  for word in full_vocab:
    if flat_doc.count(word)/total_occurences > minimum_frequency:
      vocab.append(word)
  i2w = list(vocab)
  w2i = {w: i for i, w in enumerate(i2w)}
  return i2w, w2i


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
