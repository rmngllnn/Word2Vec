""" Word2Vec.py
For command line instructions, see README.

TODO questions à Candito :
- loss relative au nombre d'exemples ?
- arrêt de l'apprentissage : surapprentissage ou plus d'apprentissage ?
- bonnes pratiques en terme de séparation du code en classes et/ou programmes et/ou modules
- on met quoi dans le compte-rendu ?

TODO organisation :
- rajouter spearman eval à w2v
- créer classe pour le corpus/exemples/etc
- sortir serialize/deserialize, fichier fonctions auxiliaires...?
TODO script bash pour tester les différents hyperparamètres
TODO README instructions
TODO rapport (17)
TODO soutenance (23~24)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from collections import Counter
import random
from extract_corpus import deserialize
from extract_corpus import serialize

torch.manual_seed(1)


class Word2Vec(nn.Module):
  """ Word2Vec model, SkipGram version with negative sampling.
  Creates word embeddings from a corpus using gradient descent.
  Use the train method to calculate the embeddings, TODO eval and save.

  The model learns:
  - one embedding for each of the vocab_size most frequent word tokens,
  - one embedding for unknown words (UNK),
  - 2*context_size embeddings for sentence boundaries (*D1*, *D2*, ... before the start of the sentence, *F1*, *F2*, ... after the end of it.).

  TODO later on, see which things we don't need past the initialisation, bc that's a lot of attributes
  Actually, some of them could be arguments of train()...?
  TODO also later on, see how long the code gets and if we want to split it up or not

  verbose             bool, verbose mode
  debug               bool, debug mode

  vocab_size          int, the number of real-word embeddings to learn
  context_size        int, the size of the context window on each side of the target word
                      if = 2, then for each word, four positive examples are created:
                      (x-2,x,+), (x-1,x,+), (x+1,x,+), (x+2,x,+)
  negative_examples   int, the number of negative examples per positive example
                      if = 2, then for each word, two negative examples are randomly created:
                      (r1, x, -), (r2, x, -)
  embedding_dim       int, the size of the word embeddings
  sampling            float, the sampling rate to calculate the negative example distribution probability

  tokenized_doc         lsit of lists of strings, tokenized corpus, no special words yet
  indexed_doc           list of lists of ints, the same doc, indexes instead of tokens, special words in
  occurence_counter     Counter object, occurence_counter["word"] = occurence_count(int)
  prob_dist             dict, prob_dist["word"] = probability of the word getting sampled

  i2w                   list, index to word translator, i2w[index(int)] = "word"
  w2i                   dict, word to index translator, w2i["word"] = index(int)
  examples              list of tuples, examples[(int)] = (context_word_id, target_word_id, pos|neg)

  target_embeddings     Embeddings, the weights of the "hidden" layer for each target word,
                        and the final learned embeddings
  context_embeddings    Embeddings, the representation of each context word, the input for forward/predict
  """
  def __init__(self, corpus_path,
      context_size,
      embedding_dim,
      sampling,
      negative_examples,
      vocab_size,
      verbose,
      debug):
    """ Initializes the model.
    TODO once we're done, put the default values to the best ones

    -> corpus_path: string, path to the serialized tokenized doc, which is a list of lists of strings,
    a tokenized doc made of sentences made of words, special words are not yet present
    -> all the rest: see the description of the attributes of the class
    """
    super(Word2Vec, self).__init__()

    assert type(debug) is bool, "Problem with debug."
    self.debug = debug

    assert type(verbose) is bool, "Problem with verbose."
    self.verbose = verbose

    assert type(corpus_path), "Problem with corpus_path."
    self.tokenized_doc = deserialize(corpus_path)
    assert type(self.tokenized_doc) is list and type(self.tokenized_doc[0]) is list and \
      type(self.tokenized_doc[0][0] is str), "Problem with the corpus."
    if self.debug:
      print("Tokenized doc (first three sentences): "+str(self.tokenized_doc[0:3]))

    assert type(context_size) is int and context_size > 0, "Problem with context_size."
    self.context_size = context_size

    assert type(vocab_size) is int and vocab_size > 0, "Problem with vocab_size."
    self.vocab_size = vocab_size

    assert type(embedding_dim) is int and embedding_dim > 0, "Problem with embedding_dim."
    self.embedding_dim = embedding_dim

    assert type(sampling) is float and sampling > 0 and sampling < 1, "Problem with sampling."
    self.sampling = sampling

    assert type(negative_examples) is int and negative_examples > 0, "Problem with negative_examples."
    self.negative_examples = negative_examples

    if self.verbose:
      print("\nWord2Vec SKipGram model with negative sampling.")
      print("\nParameters:")
      print("context size = " + str(self.context_size))
      print("max vocabulary size = " + str(self.vocab_size))
      print("embedding dimensions = " + str(self.embedding_dim))
      print("sampling rate = " + str(self.sampling))
      print("negative examples per positive example = " + str(self.negative_examples))

    self.occurence_counter = self.__get_occurence_counter()
    self.i2w = [token for token in self.occurence_counter]
    self.w2i = {w: i for i, w in enumerate(self.i2w)}
    self.indexed_doc = self.__get_indexed_doc()
    self.prob_dist =  self.__get_prob_dist()
    self.examples = self.__create_examples()

    self.target_embeddings = nn.Embedding(len(self.i2w), self.embedding_dim, sparse=True)
    self.context_embeddings = nn.Embedding(len(self.i2w), self.embedding_dim, sparse=True)
    # Changed the first dimension from vocab_size to len of vocabulary, because the first is actually
    # the max vocab size and not the actual vocab size
    range = 0.5/self.embedding_dim
    self.target_embeddings.weight.data.uniform_(-range,range)
    self.context_embeddings.weight.data.uniform_(-0,0)
    if verbose: print("\nEmbeddings initialized.")

    if self.verbose: print("\nReady to train!")


  def __get_occurence_counter(self):
    """Generates the occurence count with only the vocab_size most common words and the special words.
    Special words: UNK, *D1*, ...

    NOTE: We did consider using CountVectorizer but couldn't figure out how to deal with unknown words, which we do want to count too, because we need to create negative examples with them to create the other embeddings, and we need their distribution for that. TODO: double check, do we?

    NOTE: a Counter will give a count of 0 for an unknown word and a dict will not, which might be useful at some point, so we kept the Counter. TODO: double check at the end, does it help or not?

    NOTE: The occurence_counter need to be set before we replace rare words with UNK and add *D1* and all.
    That's because otherwise, a special word might not appear often enough to make the cut.
    We presumed that adding a few embeddings to the size wouldn't change much in terms of computation.
    However, it's absolutely possible to change it so that we keep vocab_size as the total number of
    embeddings learned, an only learn vocab_size - 2*self.context_size - 1 real word embeddings.
    """
    occurence_counter = Counter() # We want to count the number of occurences of each token, to only keep the VOCAB_SIZE most common ones.

    for sentence in self.tokenized_doc:
      occurence_counter.update(sentence) # cf https://docs.python.org/3/library/collections.html#collections.Counter

    if len(occurence_counter.keys()) - self.vocab_size > 0: # If there are tokens left over...
      #print("total:"+str(occurence_counter))
      UNK_counter = {token : count for (token, count)
          in occurence_counter.most_common()[self.vocab_size:]} # (it's actually a dict not a counter but shrug, doesn't matter for what we're doing with it)
      #print("unk: "+str(UNK_counter))
      occurence_counter.subtract(UNK_counter) # all those other tokens are deleted from the occurence count...
      #print("after subtract:"+str(occurence_counter))
      occurence_counter.update({"UNK": sum([UNK_counter[token] for token in UNK_counter])}) # and counted as occurences of UNK.

    occurence_counter.update({out_of_bounds : len(self.tokenized_doc) for out_of_bounds
        in ["*D"+str(i)+"*" for i in range(1,self.context_size+1)]
        + ["*F"+str(i)+"*" for i in range(1,self.context_size+1)]}) # We add one count of each out-of-bound special word per sentence.

    if self.verbose: print("\nOccurence counter created.")
    if self.debug: print("Occurence count: "+str(+occurence_counter))
    return +occurence_counter # "+" removes 0 or negative count elements.


  def __get_indexed_doc(self):
    """Generates an indexized version of the tokenized doc, adding out of bound and unknown special words.

    NOTE: If we wanted to adapt this model for other uses (for example, evaluating the 'likelihood' of a
    document), we'd probably need to adapt this method somehow, either for preprocessing input in main or
    for use in pred/forward. Since we don't care about that, it's set to private.
    """
    known_vocab_doc = []
    for sentence in self.tokenized_doc:
      sentence = ["*D"+str(i)+"*" for i in range(1,self.context_size+1)] + sentence + \
        ["*F"+str(i)+"*" for i in range(1,self.context_size+1)] # We add out-of-bound special words.
      for i, token in enumerate(sentence):
        if token not in self.w2i: # If we don't know a word...
          sentence[i] = "UNK" # we replace it by UNK.
      known_vocab_doc.append(sentence) # when I tried to change the tokenized doc directly, the changes got lost, sooo TODO Cécile : look into how referencing works in python again...

    # We switch to indexes instead of string tokens.
    indexed_doc = [[self.w2i[token] for token in sentence] for sentence in known_vocab_doc]

    if self.verbose: print("\nIndexed doc created.")
    if self.debug: print("Indexed doc: "+str(indexed_doc[0:3]))
    return indexed_doc


  def __get_prob_dist(self):
    """Generates the probability distribution of known words to get sampled as negative examples.
    """
    prob_dist = {}

    total_word_count = sum([self.occurence_counter[word]**self.sampling for word in self.occurence_counter])
    for word in self.occurence_counter:
      prob_dist[word] = (self.occurence_counter[word]**self.sampling)/total_word_count

    if self.verbose: print("\nProbability distribution created.")
    if self.debug: print("Probability distribution: "+str(prob_dist))
    return prob_dist


  def __create_examples(self):
    """Creates positive and negative examples using negative sampling.

    An example is a (target word, context word, gold tag) tuple.
    It is tagged 1 for positive (extracted from the corpus) and 0 for negative (randomly created).
    # TODO adapt that +/- to work with pred and/or loss
    """
    examples = []
    if self.debug: print("\nCreating examples...")

    for sentence in self.indexed_doc: # For each sentence...
      for target_i in range(self.context_size, len(sentence) - self.context_size): # For each word of the actual sentence...
        for context_i in range(target_i - self.context_size, target_i + self.context_size + 1): # For each word in the context window...
          if target_i is not context_i:
            examples.append((sentence[target_i], sentence[context_i], 1))
            if self.debug: print(self.i2w[sentence[target_i]]+","+self.i2w[sentence[context_i]]+",1")
            
        for sample in range(self.negative_examples): # Now, negative sampling! Using that probability distribution.
          random_token = np.random.choice(list(self.prob_dist.keys()), p=list(self.prob_dist.values()))
          if self.debug: print(self.i2w[sentence[target_i]]+","+random_token+",0")
          #TODO special words seem kind of overrepresented? to test
          examples.append((sentence[target_i], self.w2i[random_token], 0))
    
    if self.verbose: print("\nPositive and negative examples created.")
    return examples


  def forward(self, target_words, context_words):
    """ Calculates the probability of an example being found in the corpus, for all examples given.
    That is to say, the probability of a context word being found in the window of a context word.
    P(c|t) = sigmoid(c.t)

    We'll worry about the gold tags later, when we calculate the loss.
    P(¬c|t) = 1 - P(c|t)

    -> target_words: tensor, shape: (batch_size), line tensor of target word indexes
    -> context_words: tensor, shape: (batch_size), line tensor of context word indexes
    <- scores: tensor, shape: (batch_size), line tensor of scores
    """
    target_embeds = self.target_embeddings(target_words)
    context_embeds = self.context_embeddings(context_words)
    if self.debug:
      print("\nForward propagation on batch.")
      print("Target ids: "+str(target_words))
      print("Context ids: "+str(context_words))
      print("Target embeddings: "+str(target_embeds))
      print("Context embeddings: "+str(context_embeds))
    scores = torch.mul(target_embeds, context_embeds)
    if self.debug: print("mul: "+str(scores))
    scores = torch.sum(scores, dim=1)
    if self.debug: print("sum: "+str(scores))
    scores = F.logsigmoid(scores)
    if self.debug: print("sig: "+str(scores))

    return scores


  def train(self,
      number_epochs,
      learning_rate,
      batch_size):
    """ Executes gradient descent to learn the embeddings.
    This is where we switch to tensors: we need the examples to be in a list in order to shuffle them,
    but after that, for efficiency, we do all calculations using matrices.

    -> number_epochs: int, the number of epochs to train for
    -> batch_size: int, the number of examples in a batch
    -> learning_rate: float, the learning rate step when training
    <- loss_over_epochs: list of ints, loss per epoch of training
    """
    assert type(number_epochs) is int and number_epochs > 0, "Problem with number_epochs."
    assert type(learning_rate) is float and learning_rate > 0, "Problem with learning_rate."
    assert type(batch_size) is int and batch_size > 0, "Problem with batch_size."

    if self.verbose:
      print("\nnumber of epochs = " + str(number_epochs))
      print("learning rate = " + str(learning_rate))
      print("batch size = " + str(batch_size))
      print("\nTraining...")

    loss_over_epochs = []
    spearman_over_epochs = []
    self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    for epoch in range(number_epochs):
      epoch_loss = 0
      random.shuffle(self.examples)
      batches = torch.split(torch.tensor(self.examples), batch_size)

      for batch in batches:
        target_words = batch[:,0]
        context_words = batch[:,1]
        gold_tags = batch[:,2]

        scores = self(target_words, context_words) # Forward propagation.
        batch_loss = torch.sum(torch.abs(gold_tags - scores)) # The loss is the difference between the
        # probability we want to associate with the example (gold_tag) and the probability measured by
        # the model (score))
        epoch_loss += batch_loss

        self.zero_grad() # Reinitialising model gradients.
        batch_loss.backward() # Back propagation, computing gradients.
        self.optimizer.step() # One step in gradient descent.

      if self.verbose: print("Epoch "+str(epoch+1)+", loss = "+str(epoch_loss.item()))
      loss_over_epochs.append(epoch_loss.item())
      # TODO spearman_over_epochs.append(self.spearman.evaluate())
    if self.verbose: print("Training done!")
    return loss_over_epochs, spearman_over_epochs


  def save_embeddings(self, save_path):
    """ Saves the embeddings.

    -> save_path: string, path of the file to save as
    """
    serialize(self.target_embeddings, save_path)

  def eval(self):
    pass




    


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('corpus_path', type=str, default="test.json", help='Path to the serialized tokenized corpus.') 
  parser.add_argument('--vocab_size', type=int, default=20, help='The number of real-word embeddings to learn')
  parser.add_argument('--context_size', type=int, default=2, help='The size of the context window on each side of the target word')
  parser.add_argument('--negative_examples', type=int, default=5, help='The number of negative examples per positive example')
  parser.add_argument('--embedding_dim', type=int, default=10, help='The size of the word embeddings')
  parser.add_argument('--sampling', type=float, default=0.75, help='The sampling rate to calculate the negative example distribution probability')
  parser.add_argument('--number_epochs', type=int, default=5, help='The number of epochs to train for')
  parser.add_argument('--batch_size', type=int, default=150,  help='The number of examples in a batch')
  parser.add_argument('--learning_rate', type=int, default=0.05, help='The learning rate step when training')
  parser.add_argument('--verbose', type=bool, default=True, help='Verbose mode')
  parser.add_argument('--debug', type=bool, default=False, help='Debug mode')
  args = parser.parse_args()


  model = Word2Vec(corpus_path = args.corpus_path,
    context_size = args.context_size,
    embedding_dim = args.embedding_dim,
    sampling = args.sampling,
    negative_examples = args.negative_examples,
    vocab_size = args.vocab_size,
    verbose = args.verbose,
    debug = args.debug)

  loss_over_epochs, spearman_over_epochs = model.train(batch_size=args.batch_size,
    number_epochs=args.number_epochs,
    learning_rate=args.learning_rate)



