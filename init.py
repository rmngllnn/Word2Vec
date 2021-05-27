"""Dans ce fichier, on se propose d'initialiser l'implémentation du modèle word2vec, en exploitant le corpus (extraction, indexes, nettoyage, embeddings)
(Très) inspiré du TP8 du cours d'apprentissage automatique 2 de Marie Candito

# TODO: descriptions des hyperparamètres
# TODO: gestion des hyperparamètres et des argparses en arguments ? en hyper ??
CONTEXT_SIZE: if = 2, then for each word, four positive examples are created: (x-2,x), (x-1,x), (x+1,x), (x+2,x).
EMBEDDING_DIM: 
SAMPLING: 
NEGATIVE_EXAMPLE: if = 2, then for each word, two negative examples are randomly created.
VOCAB_SIZE: 
EMBEDDING_DIM: 
NUMBER_EPOCHS: 
LEARNING_RATE: 
BATCH_SIZE: 

#TODO instructions pour lancer le programme avec l'argparse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from collections import Counter
import random

torch.manual_seed(1)

# TODO plus tard, créer un script pour tester les différents hyperparamètres
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
SAMPLING = 0.75
NEGATIVE_EXAMPLE = 3
VOCAB_SIZE = 10
EMBEDDING_DIM = 10
NUMBER_EPOCHS = 5
LEARNING_RATE = 0.05
BATCH_SIZE = 5


def extract_corpus(infile):
  """Extracts the corpus, gets rid of the POS tags, tokenizes.
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


def get_occurence_counter(tokenized_doc):
  """Generates the occurence count with only the VOCAB_SIZE most common words and the special words.
  This means the final vocabuary has a size of (VOCAB_SIZE + 1 + 2*CONTEXT_SIZE).
  Special words: UNK, *D1*, ...

  NOTE: We did consider using CountVectorizer, but couldn't figure out how to deal with unknown words, which we do want to count too, because we need to create negative examples with them to create the other embeddings, and we need their distribution for that. TODO: double check, do we?
  NOTE: a Counter will give a count of 0 for an unknown word and a dict will not, which might be useful at some point, so we kept the Counter. TODO: double check at the end, does it help or not?

  -> tokenized_doc: list of lists of strings, a tokenized doc made of sentences made of words, as created by extract_corpus()
  <- occurence_counter : Counter object, occurence_counter["word"] = occurence_count(int)
  """
  occurence_counter = Counter() # We want to count the number of occurences of each token, to only keep the VOCAB_SIZE most common ones.

  for sentence in tokenized_doc:
    occurence_counter.update(sentence) # cf https://docs.python.org/3/library/collections.html#collections.Counter

  if len(occurence_counter.keys()) - VOCAB_SIZE > 0: # If there are tokens left over...
    #print("total:"+str(occurence_counter))
    UNK_counter = {token : count for (token, count) in occurence_counter.most_common()[VOCAB_SIZE:]} # (it's actually a dict not a counter but shrug, doesn't matter for what we're doing with it)
    #print("unk: "+str(UNK_counter))
    occurence_counter.subtract(UNK_counter) # all those other tokens are deleted from the occurence count...
    #print("after subtract:"+str(occurence_counter))
    occurence_counter.update({"UNK": sum([UNK_counter[token] for token in UNK_counter])}) # and counted as occurences of UNK.

  occurence_counter.update({out_of_bounds : len(tokenized_doc) for out_of_bounds in ["*D"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)] + ["*F"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)]}) # We add one count of each out-of-bound special word per sentence.

  return +occurence_counter # "+" removes 0 or negative count elements.


def get_indexes(occurence_counter):
  """Generates the vocabulary indexes based on the occurence_counter.
  The latter contains the number of occurences for the VOCAB_SIZE most common real words as well as
  for the special words.

  -> occurence_counter: dict, occurence_counter["word"] = occurence_count(int), as created by get_occurence_counter()
  <- i2w: list, index to word translator, i2w[index(int)] = "word"
  <- w2i: dict, word to index translator, w2i["word"] = index(int)
  """
  i2w = [token for token in occurence_counter] # The VOCAB_SIZE most common tokens will make up our real-words vocabulary.
  i2w = i2w + ["UNK"] + ["*D"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)] + ["*F"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)] # We add the special words.
  w2i = {w: i for i, w in enumerate(i2w)} # From the list of tokens in our final vocabulary, we build the reverse index, token -> index number.

  print("iw2: "+str(i2w))
  print("w2i: "+str(w2i))
  return i2w, w2i


def get_indexed_doc(tokenized_doc, w2i):
  """Generates an indexized version of the tokenized doc, adding out of bound and unknown special words.

  -> tokenized_doc: list of lists of strings, a tokenized doc made of sentences made of words, as created by extract_corpus()
  <- w2i: dict, word to index translator, w2i["word"] = index(int), as created by get_indexes()
  <- indexed_doc: list of lists of ints, a tokenized doc made of sentences made of words, as created by extract_corpus()
  """
  known_vocab_doc = []
  for sentence in tokenized_doc:
    sentence = ["*D"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)] + sentence + ["*F"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)] # We add out-of-bound special words.
    for i, token in enumerate(sentence):
      if token not in w2i:
        sentence[i] = "UNK" # We replace unknown words by UNK.
    known_vocab_doc.append(sentence) # Otherwise the changes were lost, TODO look into how referencing works in python...
  
  print(known_vocab_doc)

  # We switch to indexes instead of string tokens.
  indexed_doc = [[w2i[token] for token in sentence] for sentence in known_vocab_doc]

  return indexed_doc


def get_probability_distribution(occurence_counter):
  """Generates the probability distribution of known words to get sampled as negative examples.

  -> occurence_counter: Counter, occurence_counter["word"] = occurence_count(int), as created by get_occurence_counter()
  <- probability_distribution: dict, probability_distribution["word"] = probability of if getting sampled
  """
  probability_distribution = {}

  total_word_count = sum([occurence_counter[word]**SAMPLING for word in occurence_counter])
  for word in occurence_counter:
    probability_distribution[word] = (occurence_counter[word]**SAMPLING)/total_word_count

  return probability_distribution


def create_examples(indexed_doc, w2i, i2w, probability_distribution):
  #TODO voir si on peut simplifier la fonction, cf TP8 de la prof
  """Creates positive and negative examples using negative sampling.
  An example is a (context word, target word) pair. This is where we switch from tokens (strings) to indexes (int), using w2i.
  It is tagged 1 for positive (extracted from the corpus) and -1 for negative (randomly created).

  NOTE: here, i2w is only used for debugging by printing the examples.

  -> indexed_doc: list of lists of strings, a tokenized doc made of sentences made of words, as created by extract_corpus()
  -> w2i: dict, word to index translator, w2i["word"] = index(int), as created by get_indexes()
  -> i2w: list, index to word translator, i2w[index(int)] = "word", as created by get_indexes()
  -> probability_distribution: dict, probability_distribution["word"] = probability of if getting sampled, as created by get_probability_distribution()

  <- examples : list of tuples, list of examples, one example = (context word index, target word index)
  <- gold_classes : list of ints, list of gold tags, one gold tag = 1|-1
  """
  examples = []
  gold_classes = []
  
  for sentence in indexed_doc: # For each sentence...
    for target_i in range(CONTEXT_SIZE, len(sentence) - CONTEXT_SIZE): # For each word of the actual sentence...
      for context_i in range(target_i - CONTEXT_SIZE, target_i + CONTEXT_SIZE + 1): # For each word in the context window...
        if target_i is not context_i:
          examples.append((sentence[context_i], sentence[target_i]))
          gold_classes.append(1)
          #print("pos example: "+i2w[sentence[context_i]]+","+i2w[sentence[target_i]])
          
      for sample in range(NEGATIVE_EXAMPLE): # Now, negative sampling! Using that probability distribution.
        random_token = np.random.choice(list(probability_distribution.keys()), p=list(probability_distribution.values()))
        #print("neg example: "+random_token+","+i2w[sentence[target_i]]+")") #TODO special words seem kind of overrepresented? to test
        examples.append((w2i[random_token], sentence[target_i]))
        gold_classes.append(-1)
  
  return list(zip(examples, gold_classes))


def create_batches(tagged_examples):
  """Creates batches of examples. One batch is a ??? of a tensor of examples and a tensor of their gold tags.
  This is where we go from lists to tensors.
  
  -> : 
  -> gold_tags: 
  <- batches: 
  """
  examples = [example for example, gold_tag in tagged_examples]
  gold_tags = [gold_tag for example, gold_tag in tagged_examples]
  X = torch.tensor(examples)
  Y = torch.tensor(gold_tags)
  
  batches_X = torch.split(X, BATCH_SIZE)
  batches_Y = torch.split(Y, BATCH_SIZE)
  batches = list(zip(batches_X,batches_Y))

  return batches


class w2vModel(nn.Module):
  """
  """
  def __init__(self, vocab_size = VOCAB_SIZE, embedding_dim = EMBEDDING_DIM):
        super(w2vModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.layer = nn.Linear(embedding_dim, vocab_size)

  def forward(self, inputs):
      context_embeds = self.embeddings(inputs) # (window_size*2, embedding_dim)
      continuous_context_embed = torch.sum(context_embeds, dim=-2) #(embedding_dim)
      scores = self.layer(continuous_context_embed) # (vocab_size)
      output = F.log_softmax(scores,dim =0) # (vocab_size)
      return output


def train(model, tagged_examples, number_epochs = NUMBER_EPOCHS, learning_rate = LEARNING_RATE):
  """
  """
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  loss_over_time = []
  loss_function = nn.NLLLoss()

  for epoch in range(number_epochs):
      epoch_loss = 0
      random.shuffle(tagged_examples)

      for X, Y in create_batches(tagged_examples):
          model.zero_grad() # reinitialising model gradients
          output = model(X) # forward propagation
          loss = loss_function(output, Y) # computing loss
          loss.backward() # back propagation, computing gradients
          optimizer.step() # one step in gradient descent
          epoch_loss += loss.item()

      loss_over_time.append(epoch_loss)
  return loss_over_time



tokenized_doc = [["This","is","a", "test", "."], ["Test","."]]
occurence_counter = get_occurence_counter(tokenized_doc)
i2w, w2i = get_indexes(occurence_counter)
indexed_doc = get_indexed_doc(tokenized_doc, w2i)
probability_distribution = get_probability_distribution(occurence_counter)
tagged_examples = create_examples(indexed_doc, w2i, i2w, probability_distribution)

print("Examples: "+str(tagged_examples))

model = w2vModel()
loss_over_time = train(model, tagged_examples)


#PARTIE MAIN
parser = argparse.ArgumentParser()
#Dans le terminal, on écrira "python3 init.py NOM_DU_FICHIER_CORPUS"
parser.add_argument('examples_file', default=None, help='Corpus utilisé pour la création d\'exemples d\'apprentissage pour les embeddings.')

#print(extract_corpus("mini_corpus.txt"))

