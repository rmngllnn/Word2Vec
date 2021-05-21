#Dans ce fichier, on se propose d'initialiser l'implémentation du modèle word2vec, en exploitant le corpus (extraction, indices, nettoyage, embeddings)
#(Très) inspiré TP8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


torch.manual_seed(1)

CONTEXT_SIZE = 2 #pour commencer
EMBEDDING_DIM = 10 #pour commencer
SAMPLING = 0.75
NEGATIVE_EXAMPLE = 3
VOCAB_SIZE = 2


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


def get_indices(doc):
  """Tokenizes the document, generates the vocabulary list and an occurence count of each token in the vocabulary.

  -> doc: list (doc) of strings (sentences)

  <- tokens: list (doc) of lists (sentences) of strings (words)
  <- i2w: list, i2w[index(int)] = "word"
  <- w2i: dict, w2i["word"] = index(int)
  <- occurence_count : matrix, occurence_count[index(int)] = occurence_count(int)
  """
  vectorizer = CountVectorizer(input=doc, max_features=VOCAB_SIZE, token_pattern=r"(?u)\b\w+\b")
  # max_features to only keep VOCAB_SIZE tokens, token pattern to keep one-letter words
  # cf https://www.studytonight.com/post/scikitlearn-countvectorizer-in-nlp and https://investigate.ai/text-analysis/counting-words-with-scikit-learns-countvectorizer/ 
  occurence_count = vectorizer.fit_transform(doc)
  
  i2w = vectorizer.get_feature_names()
  i2w = i2w + ["UNK"] + ["*D"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)] + ["*F"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)]
  # The final vocab size is actually VOCAB_SIZE + 1 + 2*CONTEXT_SIZE to account for UNK, D1, etc.
  # TODO est-ce que c'est bien ça qu'on veut ?
  # TODO potentiel : si on crée notre propre foncction tokenize, on peut la mettre en argument (tokenizer=function)
  
  w2i = {w: i for i, w in enumerate(i2w)} # From the list of tokens in our vocabulary, we build the reverse index, token -> index number.

  tokenizer = vectorizer.build_tokenizer()
  tokens = [tokenizer(sentence) for sentence in doc]

  return tokens, i2w, w2i, np.array(occurence_count.toarray()).sum(0)
  # TODO Cécile : euh est-ce que c'est bien ce qu'on veut ? vérifier la dimension de la sum

# notes Cécile : il faut passer à une représentation matricielle du corpus avant ou après la création des exemples ?


test_corpus = ["This is a test.", "Test."]
tokens, i2w, w2i, occurences_word = get_indices(test_corpus)


def create_examples (tokens, w2i, occurences_word) :
  """Creates positive and negative examples using negative sampling.
  An example is a (context word, target word) pair.
  
  -> tokens : list of lists of strings
  -> w2i : dict of indices of words
  -> occurences_word : dict of occurences

  <- examples : list of lists of examples for each sentences # TODO note Cécile, pourquoi pas une liste d'exemples ?
  <- gold_classes : list of lists if gold classes for each list of examples (1 or -1)
  """
  
  examples = []
  gold_classes = []
  distribution_prob = {}

  print(occurences_word)
  total_number_tok = 0
  for index in range(0, VOCAB_SIZE):
    total_number_tok += occurences_word[index]**SAMPLING
  for index in range(0, VOCAB_SIZE):
    distribution_prob[i2w[index]] = (occurences_word[index]**SAMPLING)/total_number_tok # ça ça prend pas en compte les mots UNK
    #ils ont pas été comptés par le countvectorizer, je sais pas is c'est un problème ou non
  
  
  for sentence in tokens:
    sent = []
    gold_sent = []
    length = len(sentence)
    for i in range(length):
      print("word: "+sentence[i])
      for k in range(1,CONTEXT_SIZE+1):
        if i-k >= 0:
          print("example: ("+sentence[i-k]+","+sentence[i]+")")
          sent.append((w2i[sentence[i-k]], w2i[sentence[i]]))
          gold_sent.append(1)
        elif i-k < 0:
          j = i-k+CONTEXT_SIZE+1
          D = '*D'+str(j)+'*'
          print("example: ("+D+","+sentence[i]+")")
          sent.append((w2i[D], w2i[sentence[i]]))
          gold_sent.append(1)
        if i+k < length:
          print("example: ("+sentence[i+k]+","+sentence[i]+")")
          sent.append((w2i[sentence[i+k]], w2i[sentence[i]]))
          gold_sent.append(1)
        elif i+k >= length:
          j = i+k-length+1
          F = '*F'+str(j)+'*'
          print("example: "+F+","+sentence[i]+")")
          sent.append((w2i[F], w2i[sentence[i]]))
          
      for j in range(NEGATIVE_EXAMPLE):
        negative_rand_ex = np.random.choice(list(distribution_prob.keys()), p=list(distribution_prob.values()))
        sent.append((w2i[negative_rand_ex], w2i[sentence[i]]))
        gold_sent.append(-1)
    examples.append(sent)
    gold_classes.append(gold_sent)

  
  return examples, gold_classes

examples = create_examples(tokens, w2i, occurences_word)
print("Examples: "+str(examples))


#PARTIE MAIN
parser = argparse.ArgumentParser()
#Dans le terminal, on écrira "python3 init.py NOM_DU_FICHIER_CORPUS"
parser.add_argument('examples_file', default=None, help='Corpus utilisé pour la création d\'exemples d\'apprentissage pour les embeddings.')

#print(extract_corpus("mini_corpus.txt"))

