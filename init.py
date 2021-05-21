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
      sentence = []
			
      # deleting pos from the token
      for tok in lines :
        t = tok.split("/")
        t.remove(t[1])
        sentence.append(t[0]) # Tokenisation... #si on veut enlever les majuscules -> t[0].lower()
      doc.append(sentence)
  return doc

def get_indices(doc):
  """Tokenizes the document and generates the vocabulary.

  -> doc: list (doc) of lists (sentences) of strings (sentences)

  <- tokens: list of lists of strings
  <- i2w: list, i2w[index(int)] = "word"
  <- w2i: dict, w2i["word"] = index(int)
  <- occurences_counter : dict, occurences_word["word"] = occurence_count(int)
  """
  occurences_counter = Counter() # We want to count the number of occurences of each token, to only keep the VOCAB_SIZE most common ones.

  for sentence in doc:
    occurences_counter.update(sentence) # cf https://docs.python.org/3/library/collections.html#collections.Counter

  i2w = [token for (token, count) in occurences_counter.most_common(VOCAB_SIZE)] # The VOCAB_SIZE most common tokens will make up our vocabulary.

  if len(occurences_counter.keys()) - VOCAB_SIZE > 0: # If there are tokens left over...
    #print("total:"+str(occurences_counter))
    UNK_counter = {token : count for (token, count) in occurences_counter.most_common()[VOCAB_SIZE:]} # (it's a dict not a counter but shrug)
    #print("unk: "+str(UNK_counter))
    occurences_counter.subtract(UNK_counter) # all those other tokens are deleted from the occurence count...
    #print("after subtract:"+str(occurences_counter))
    occurences_counter.update({"UNK": sum([UNK_counter[token] for token in UNK_counter])}) # and added as occurences of UNK.

  occurences_counter.update({spec_word : len(doc) for spec_word in ["*D"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)] + ["*F"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)]})

  i2w = i2w + ["UNK"] + ["*D"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)] + ["*F"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)]
  # The final vocab is actually VOCAB_SIZE + 1 + 2*CONTEXT_SIZE sized to account for UNK, D1, etc. # TODO est-ce que c'est bien ça qu'on veut ?
  
  w2i = {w: i for i, w in enumerate(i2w)} # From the list of tokens in out vocabulary, we built the reverse index, token -> index number.

  return i2w, w2i, occurences_counter # A Counter will give a count of 0 for an unknown word, a dict will not.


# notes Cécile : il faut passer à une représentation matricielle du corpus avant ou après la création des exemples ?


doc = [["This","is","a", "test", "."], ["Test","."]]
i2w, w2i, occurences_word = get_indices(doc)


def create_examples (tokens, w2i, occurences_word):
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

  total_number_tok = 0 # TODO à adapter avec objet Counter (counter.elements, trucs du genre)
  for index in range(0, VOCAB_SIZE):
    total_number_tok += occurences_word[index]**SAMPLING
  for index in range(0, VOCAB_SIZE):
    distribution_prob[i2w[index]] = (occurences_word[index]**SAMPLING)/total_number_tok

    #TODO ajouter D, F etc au corpus pour les compter
  
  
  for sentence in tokens:
    length = len(sentence)
    for i in range(length):
      print("word: "+sentence[i])
      for k in range(1,CONTEXT_SIZE+1):
        if i-k >= 0:
          print("example: ("+sentence[i-k]+","+sentence[i]+")")
          examples.append((w2i[sentence[i-k]], w2i[sentence[i]]))
          gold_classes.append(1)
        elif i-k < 0:
          j = i-k+CONTEXT_SIZE+1
          D = '*D'+str(j)+'*'
          print("example: ("+D+","+sentence[i]+")")
          examples.append((w2i[D], w2i[sentence[i]]))
          gold_classes.append(1)
        if i+k < length:
          print("example: ("+sentence[i+k]+","+sentence[i]+")")
          examples.append((w2i[sentence[i+k]], w2i[sentence[i]]))
          gold_classes.append(1)
        elif i+k >= length:
          j = i+k-length+1
          F = '*F'+str(j)+'*'
          print("example: "+F+","+sentence[i]+")")
          examples.append((w2i[F], w2i[sentence[i]]))
          gold_classes.append(1)
          
      for j in range(NEGATIVE_EXAMPLE):
        negative_rand_ex = np.random.choice(list(distribution_prob.keys()), p=list(distribution_prob.values()))
        examples.append((w2i[negative_rand_ex], w2i[sentence[i]]))
        gold_classes.append(-1)
  
  return examples, gold_classes

examples = create_examples(doc, w2i, occurences_word)
print("Examples: "+str(examples))


#PARTIE MAIN
parser = argparse.ArgumentParser()
#Dans le terminal, on écrira "python3 init.py NOM_DU_FICHIER_CORPUS"
parser.add_argument('examples_file', default=None, help='Corpus utilisé pour la création d\'exemples d\'apprentissage pour les embeddings.')

#print(extract_corpus("mini_corpus.txt"))

