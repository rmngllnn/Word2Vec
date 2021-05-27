"""Dans ce fichier, on se propose d'initialiser l'implémentation du modèle word2vec, en exploitant le corpus (extraction, indexes, nettoyage, embeddings)
(Très) inspiré du TP8 du cours d'apprentissage automatique 2 de Marie Candito
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from collections import Counter

torch.manual_seed(1)

# TODO plus tard, ça c'est pour commencer
CONTEXT_SIZE = 2 # If CONTEXT_SIZE = 2, then for each word, four positive examples are created: (x-2,x), (x-1,x), (x+1,x), (x+2,x).
EMBEDDING_DIM = 10
SAMPLING = 0.75
NEGATIVE_EXAMPLE = 3 # If NEGATIVE_EXAMPLE = 2, then for each word, two negative examples are randomly created.
VOCAB_SIZE = 2
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


def get_indexes_and_counter(tokenized_doc):
  """Generates the vocabulary indexes and the occurence count with only the VOCAB_SIZE most common words and the special words.
  This means the final vocabuary has a size of (VOCAB_SIZE + 1 + 2*CONTEXT_SIZE).
  Special words: UNK, *D1*, ...

  Note: We did consider using CountVectorizer, but couldn't figure out how to deal with unknown words, which we do want to count
  too, because we need to create negative examples with them to create the other embeddings, and we need their distribution for
  that. TODO: double check, do we?
  Also, a Counter will give a count of 0 for an unknown word and a dict will not, which might be useful at some point, so we kept the Counter. TODO: double check at the end, does it help or not?

  -> tokenized_doc: list of lists of strings, a tokenized doc made of sentences made of words, as created by extract_corpus()

  <- i2w: list, index to word translator, i2w[index(int)] = "word"
  <- w2i: dict, word to index translator, w2i["word"] = index(int)
  <- occurence_counter : Counter object, occurence_counter["word"] = occurence_count(int)
  """
  occurence_counter = Counter() # We want to count the number of occurences of each token, to only keep the VOCAB_SIZE most common ones.

  for sentence in tokenized_doc:
    occurence_counter.update(sentence) # cf https://docs.python.org/3/library/collections.html#collections.Counter

  i2w = [token for (token, count) in occurence_counter.most_common(VOCAB_SIZE)] # The VOCAB_SIZE most common tokens will make up our real-words vocabulary.

  if len(occurence_counter.keys()) - VOCAB_SIZE > 0: # If there are tokens left over...
    #print("total:"+str(occurence_counter))
    UNK_counter = {token : count for (token, count) in occurence_counter.most_common()[VOCAB_SIZE:]} # (it's actually a dict not a counter but shrug, doesn't matter for what we're doing with it)
    #print("unk: "+str(UNK_counter))
    occurence_counter.subtract(UNK_counter) # all those other tokens are deleted from the occurence count...
    #print("after subtract:"+str(occurence_counter))
    occurence_counter.update({"UNK": sum([UNK_counter[token] for token in UNK_counter])}) # and counted as occurences of UNK.

  occurence_counter.update({out_of_bounds : len(tokenized_doc) for out_of_bounds in ["*D"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)] + ["*F"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)]}) # We add one count of each out-of-bound special word per sentence.

  i2w = i2w + ["UNK"] + ["*D"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)] + ["*F"+str(i)+"*" for i in range(1,CONTEXT_SIZE+1)]
  
  w2i = {w: i for i, w in enumerate(i2w)} # From the list of tokens in our final vocabulary, we build the reverse index, token -> index number.

  #print("iw2: "+str(i2w))
  #print("w2i: "+str(w2i))
  return i2w, w2i, +occurence_counter # "+" removes 0 or negative count elements.


def create_examples(tokenized_doc, w2i, i2w, occurence_counter):
  #TODO voir si on peut simplifier la fonction, cf TP8 de la prof
  """Creates positive and negative examples using negative sampling.
  An example is a (context word, target word) pair. This is where we switch from tokens (strings) to indexes (int), using w2i.
  It is tagged 1 for positive (extracted from the corpus) and -1 for negative (randomly created).

  Note: here, i2w is only used for debugging by printing the examples.

  -> tokenized_doc: list of lists of strings, a tokenized doc made of sentences made of words, as created by extract_corpus()
  -> w2i: dict, word to index translator, w2i["word"] = index(int), as created by get_indexes_and_counter()
  -> i2w: list, index to word translator, i2w[index(int)] = "word", as created by get_indexes_and_counter()
  -> occurence_counter : Counter object, occurence_counter["word"] = occurence_count(int), as created by get_indexes_and_counter()

  <- examples : list of tuples, list of examples, one example = (context word index, target word index)
  <- gold_classes : list of ints, list of gold tags, one gold tag = 1|-1
  """
  examples = []
  gold_classes = []
  distribution_prob = {} # Probability of a word being picked for negative sampling.

  total_word_count = sum([occurence_counter[word]**SAMPLING for word in occurence_counter])
  for word in occurence_counter:
    distribution_prob[word] = (occurence_counter[word]**SAMPLING)/total_word_count

  # We replace unknown words by UNK.
  for sentence in tokenized_doc:
    for i, token in enumerate(sentence):
      if token not in w2i:
        sentence[i] = "UNK"

  # We switch to indexes instead of string tokens.
  indexed_doc = [[w2i[token] for token in sentence] for sentence in tokenized_doc]
  
  # Aaaand we create our examples.
  for sentence in indexed_doc:
    length = len(sentence)
    for i in range(length): # For each word...
      # print("word: "+i2w[sentence[i]])
      for k in range(1,CONTEXT_SIZE+1): # We're moving through both sides of the context window at once.

        if i-k >= 0: # If we're not out of bounds to the left:
          #print("example: ("+i2w[sentence[i-k]]+","+i2w[sentence[i]]+")")
          examples.append((sentence[i-k], sentence[i]))
          gold_classes.append(1)

        elif i-k < 0: # Otherwise, if we are:
          j = i-k+CONTEXT_SIZE+1
          D_oob_word = '*D'+str(j)+'*'
          #print("example: ("+D_oob_word+","+i2w[sentence[i]]+")")
          examples.append((w2i[D_oob_word], sentence[i]))
          gold_classes.append(1)

        if i+k < length: # Now the other side. If we're not out of bounds to the right:
          #print("example: ("+i2w[sentence[i+k]]+","+i2w[sentence[i]]+")")
          examples.append((sentence[i+k], sentence[i]))
          gold_classes.append(1)

        elif i+k >= length: # But if we are:
          j = i+k-length+1
          F_oob_word = '*F'+str(j)+'*'
          #print("example: "+F_oob_word+","+i2w[sentence[i]]+")")
          examples.append((w2i[F_oob_word], sentence[i]))
          gold_classes.append(1)
          
      for j in range(NEGATIVE_EXAMPLE): # Now, negative sampling! Using that probability distribution.
        random_token = np.random.choice(list(distribution_prob.keys()), p=list(distribution_prob.values()))
        #print("neg example: "+random_token+","+i2w[sentence[i]]+")") #TODO special words seem kind of overrepresented? to test
        examples.append((w2i[random_token], sentence[i]))
        gold_classes.append(-1)
  
  return examples, gold_classes


def create_batches(examples, gold_tags):
  """
  """
  pass


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

def train(model, examples, gold_tags, number_epochs = NUMBER_EPOCHS, learning_rate = LEARNING_RATE):
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  loss_over_time = []
  loss_function = nn.NLLLoss()

  for epoch in range(number_epochs):
      batches = create_batches(examples, gold_tags)
      epoch_loss = 0

      for X, Y in batches:
          model.zero_grad() # reinitialising model gradients
          output = model(X) # forward propagation
          loss = loss_function(output, Y) # computing loss
          loss.backward() # back propagation, computing gradients
          optimizer.step() # one step in gradient descent
          epoch_loss += loss.item()

      loss_over_time.append(epoch_loss)
  return loss_over_time



tokenized_doc = [["This","is","a", "test", "."], ["Test","."]]
i2w, w2i, occurences_word = get_indexes_and_counter(tokenized_doc)
examples = create_examples(tokenized_doc, w2i, i2w, occurences_word)
print("Examples: "+str(examples))


#PARTIE MAIN
parser = argparse.ArgumentParser()
#Dans le terminal, on écrira "python3 init.py NOM_DU_FICHIER_CORPUS"
parser.add_argument('examples_file', default=None, help='Corpus utilisé pour la création d\'exemples d\'apprentissage pour les embeddings.')

#print(extract_corpus("mini_corpus.txt"))

