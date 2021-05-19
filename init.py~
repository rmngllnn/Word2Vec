#Dans ce fichier, on se propose d'initialiser l'implémentation du modèle word2vec, en exploitant le corpus (extraction, indices, nettoyage, embeddings)
#(Très) inspiré TP8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np

torch.manual_seed(1)

CONTEXT_SIZE = 2 #pour commencer
EMBEDDING_DIM = 10 #pour commencer
SAMPLING = 0.75
NEGATIVE_EXAMPLE = 3


def extract_corpus(infile):
  """Extracts the corpus, cleans it up.
  Calibrated for the "L'Est républicain" corpus, cf README for the original download links.
  -> infile: string, path to the file
  <- doc: list (doc) of strings (sentences)
  """
  doc = []
  with open(infile, 'r', encoding = "utf-8-sig") as f:
    for line in f.readlines():
      lines = line.split()
      sentence = ""
			
      #deleting pos from the token
      for tok in lines :
        t = tok.split("/")
        t.remove(t[1])
        sentence += t[0] + " " #si on veut enlever les majuscules -> t[0].lower()
      doc.append(sentence) 
      #pass #TODO : parcourir chaque ligne, supprimer les tags, gérer ou non les majuscules, attention à l'encodage (UTF-8)O
  return doc


def get_tokens_and_indices(doc, minimum_frequency = 0.0):
  """Tokenizes the document and generates vocabulary indexes from it.

  -> doc: list of strings (sentences)
  -> minimum_frequency: frequency threshold under which words are ignored

  <- tokens: list of lists of strings
  <- i2w: list, iw2[index(int)] = "word"
  <- w2i: dict, w2i["word"] = index(int)
  <- occurencies_word : dict, occurencies_word["word"] = occurence of the "word" in doc
  """
  tokens = []
  flat_doc = []
  vocab = set()
  occurencies_word = {}

  for sentence in doc:
    tokens_sentence = sentence.split()
    vocab = vocab.union(set(tokens_sentence))
    tokens.append(tokens_sentence)
    flat_doc = flat_doc + tokens_sentence

  total_occurences = len(flat_doc)
  for word in vocab:
    if flat_doc.count(word)/total_occurences < minimum_frequency:
      vocab.remove(word)

  for word in flat_doc :
    if word in vocab and word not in occurencies_word :
      occurencies_word[word] = 1
    elif word in vocab and word in occurencies_word :
      occurencies_word[word]+= 1
      
  vocab = vocab.union(set(["UNK"] + ["*D"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)] + ["*F"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)]))
  i2w = list(vocab)
  w2i = {w: i for i, w in enumerate(i2w)}
  return tokens, i2w, w2i, occurencies_word


test_corpus = ["This is a test .","And a second test !", "Ok that is enough now ."]
tokens, i2w, w2i, occurencies_word = get_tokens_and_indices(test_corpus)
print(w2i)


#Création des exemples d'apprentissage avec negative sampling
#Aussi, création des classes gold 1/0 (ou 1/-1) dans une liste de même taille que vocab (note Romy : pas sûre...)
def create_examples (tokens, w2i, occurencies_word) :
  """Un exemple est une paire (mot en contexte, mot cible), la sortie est une liste d'exemple de taille len(vocab) et une liste des classes gold"""
  
  """Create examples with negative sampling from tokens and w2i
  
  -> tokens : list of lists of strings
  -> w2i : dict of indices of words
  -> occurencies_word : dict of occurences

  <- examples : list of list of examples for each sentences
  <- gold_classes : list of gold classes for each examples (1 or -1)
  """
  
  examples = []
  gold_classes = []
  distribution_prob = {}

  
  total_number_tok = 0
  for value in occurencies_word.values() :
    total_number_tok += value**SAMPLING
  for word, freq in occurencies_word.items() :
    distribution_prob[word] = (freq**SAMPLING)/total_number_tok
  
  
  for sentence in tokens :
    sent = []
    gold_sent = []
    length = len(sentence)
    for i in range(len(sentence)):
      for k in range(1,CONTEXT_SIZE+1) :
        if i-k >= 0 and i+k < length :
          sent.append((w2i[sentence[i-k]], w2i[sentence[i]]))
          gold_sent.append(1)
          sent.append((w2i[sentence[i+k]], w2i[sentence[i]]))
          gold_sent.append(1)
        #TODO Romy : traiter les mots en début et fin de phrase avec un else
      for j in range(NEGATIVE_EXAMPLE) :
        negative_rand_ex = np.random.choice(list(distribution_prob.keys()), p= list(distribution_prob.values()))
        sent.append((w2i[negative_rand_ex], w2i[sentence[i]]))
        gold_sent.append(-1)
    examples.append(sent)
    gold_classes.append(gold_sent)

  
  return examples, gold_classes

examples = create_examples(tokens, w2i, occurencies_word)
print(examples)


#PARTIE MAIN
parser = argparse.ArgumentParser()
#Dans le terminal, on écrira "python3 init.py NOM_DU_FICHIER_CORPUS"
parser.add_argument('examples_file', default=None, help='Corpus utilisé pour la création d\'exemples d\'apprentissage pour les embeddings.')

#print(extract_corpus("mini_corpus.txt"))

