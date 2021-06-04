""" Word2Vec.py



TODO questions à Mme Candito :
- loss relative au nombre d'exemples ?
- arrêt de l'apprentissage : surapprentissage ou plus d'apprentissage ?
- bonnes pratiques en terme de séparation du code en classes et/ou programmes et/ou modules
- on met quoi dans le compte-rendu ?

TODO l'éval à continuer...
TODO script bash pour tester les différents hyperparamètres
TODO README instructions
TODO rapport (17)
TODO soutenance (24 Juin 10h40)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
#import numpy as np
#from collections import Counter
import random
from serialisation import deserialize
from serialisation import serialize
from extract_examples import ExampleCorpus
import SpearmanEvaluation

torch.manual_seed(1)


class Word2Vec(nn.Module):
  """ Word2Vec model, SkipGram version with negative sampling.
  Creates word embeddings from a corpus of examples using gradient descent.
  Evaluates performance with loss per 1000 examples and spearman coefficient.

  Use the train method to calculate the embeddings, TODO save

  The model learns:
  - one embedding for each of the vocab_size most frequent word tokens,
  - one embedding for unknown words (UNK),
  - 2*context_size embeddings for sentence boundaries (*D1*, *D2*, ... before the start of the sentence, *F1*, *F2*, ... after the end of it.).

  verbose               bool, verbose mode
  debug                 bool, debug mode

  example_corpus        ExampleCorpus, with the examples and the i2w/w2i translators
  target_embeddings     Embeddings, the weights of the "hidden" layer for each target word,
                        and the final learned embeddings
  context_embeddings    Embeddings, the representation of each context word, the input for forward/predict
  """
  def __init__(self,
      example_corpus_path,
      eval_corpus_path,
      verbose,
      debug):
    """ Initializes the model.
    TODO once we're done, put the default values to the best ones

    -> example_corpus_path: string, path to the serialized ExampleCorpus json file
    -> eval_corpus_path: string, path to the human-scored similarity pairs text file
    -> verbose: verbose mode
    -> debug: debug mode
    """
    super(Word2Vec, self).__init__()

    assert type(debug) is bool, "Problem with debug."
    self.debug = debug

    assert type(verbose) is bool, "Problem with verbose."
    self.verbose = verbose

    assert type(example_corpus_path) is str, "Problem with example_corpus_path."
    self.example_corpus = deserialize(example_corpus_path)
    assert type(self.example_corpus) is ExampleCorpus, "Problem with example corpus"
    """assert type(self.tokenized_doc) is list and type(self.tokenized_doc[0]) is list and \
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

    # self.examples_corpus = Examples(context_size, vocab_size, ...)
    self.occurence_counter = self.__get_occurence_counter()
    self.i2w = [token for token in self.occurence_counter]
    self.w2i = {w: i for i, w in enumerate(self.i2w)}
    self.indexed_doc = self.__get_indexed_doc()
    self.prob_dist =  self.__get_prob_dist()
    self.examples = self.__create_examples()"""

    self.target_embeddings = nn.Embedding(len(self.example_corpus.i2w), self.example_corpus.embedding_dim, sparse=True)
    self.context_embeddings = nn.Embedding(len(self.example_corpus.i2w), self.example_corpus.embedding_dim, sparse=True) # NOTE Changed the first dimension from vocab_size to len of vocabulary, because the first is actually the max vocab size and not the actual vocab size

    range = 0.5/self.example_corpus.embedding_dim
    self.target_embeddings.weight.data.uniform_(-range,range)
    self.context_embeddings.weight.data.uniform_(-0,0)
    if self.verbose: print("\nEmbeddings initialized.")

    self.spearman = SpearmanEvaluation(eval_corpus_path, self.example_corpus.w2i, self.target_embeddings, self.verbose, self.debug)
    if self.verbose: print("Evaluator initialized.")
    if self.verbose: print("\nReady to train!")

  """
  def __get_occurence_counter(self):
    Generates the occurence count with only the vocab_size most common words and the special words.
    Special words: UNK, *D1*, ...

    NOTE: We did consider using CountVectorizer but couldn't figure out how to deal with unknown words, which we do want to count too, because we need to create negative examples with them to create the other embeddings, and we need their distribution for that. TODO: double check, do we?

    NOTE: a Counter will give a count of 0 for an unknown word and a dict will not, which might be useful at some point, so we kept the Counter. TODO: double check at the end, does it help or not?

    NOTE: The occurence_counter need to be set before we replace rare words with UNK and add *D1* and all.
    That's because otherwise, a special word might not appear often enough to make the cut.
    We presumed that adding a few embeddings to the size wouldn't change much in terms of computation.
    However, it's absolutely possible to change it so that we keep vocab_size as the total number of
    embeddings learned, an only learn vocab_size - 2*self.context_size - 1 real word embeddings.
    
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
    Generates an indexized version of the tokenized doc, adding out of bound and unknown special words.

    NOTE: If we wanted to adapt this model for other uses (for example, evaluating the 'likelihood' of a
    document), we'd probably need to adapt this method somehow, either for preprocessing input in main or
    for use in pred/forward. Since we don't care about that, it's set to private.
    
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
    Generates the probability distribution of known words to get sampled as negative examples.
    
    prob_dist = {}

    total_word_count = sum([self.occurence_counter[word]**self.sampling for word in self.occurence_counter])
    for word in self.occurence_counter:
      prob_dist[word] = (self.occurence_counter[word]**self.sampling)/total_word_count

    if self.verbose: print("\nProbability distribution created.")
    if self.debug: print("Probability distribution: "+str(prob_dist))
    return prob_dist


  def __create_examples(self):
    Creates positive and negative examples using negative sampling.

    An example is a (target word, context word, gold tag) tuple.
    It is tagged 1 for positive (extracted from the corpus) and 0 for negative (randomly created).
    # TODO adapt that +/- to work with pred and/or loss
    
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
  """

  def forward(self, target_ids, context_ids, train=True):
    """ Calculates the probability of an example being found in the corpus, for all examples given.
    That is to say, the probability of a context word being found in the window of a context word.
    P(c|t) = sigmoid(c.t)

    We'll worry about the gold tags later, when we calculate the loss.
    P(¬c|t) = 1 - P(c|t)

    -> target_ids: tensor, shape: (batch_size), line tensor of target word indexes
    -> context_ids: tensor, shape: (batch_size), line tensor of context word indexes
    -> train: boolean, whether to use both target and context embeddings or just target embeddings
    <- scores: tensor, shape: (batch_size), line tensor of scores
    """
    target_embeds = self.target_embeddings(target_ids)
    context_embeds = None
    if train:
      context_embeds = self.context_embeddings(context_ids)
    else:
      context_embeds = self.target_embeddings(context_ids)

    if self.debug:
      print("\nForward propagation on batch.")
      print("Target ids: "+str(target_ids))
      print("Context ids: "+str(context_ids))
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
      print("\nTraining parameters:")
      print("number of epochs = " + str(number_epochs))
      print("learning rate = " + str(learning_rate))
      print("batch size = " + str(batch_size))
      print("\nTraining...")

    batches_seen = 0
    examples_over_time = []
    loss_over_time = []
    spearman_over_time = []
    self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    train_set = self.example_corpus.examples[0:len(self.example_corpus.examples)*80/100] # TODO quel pourcentage?
    dev_set = self.example_corpus.examples[len(self.example_corpus.examples)*80/100:]

    for epoch in range(number_epochs):
      random.shuffle(train_set)
      batches = torch.split(torch.tensor(train_set), batch_size)

      for batch in batches:
        target_ids = batch[:,0]
        context_ids = batch[:,1]
        gold_tags = batch[:,2]

        scores = self(target_ids, context_ids) # Forward propagation.
        batch_loss = torch.sum(torch.abs(gold_tags - scores)) # The loss is the difference between the
        # probability we want to associate with the example (gold_tag) and the probability measured by
        # the model (score))
        # cross-entropy!
        # batch_loss = torch.sum((-1)*gold_tags*np.log(scores)-(1-gold_tag)*np.log(1-scores))
        self.zero_grad() # Reinitialising model gradients.
        batch_loss.backward() # Back propagation, computing gradients.
        self.optimizer.step() # One step in gradient descent.

        if batches_seen % 1000 == 0:
          with torch.no_grad():
            target_ids = dev_set[:,0]
            context_ids = dev_set[:,1]
            gold_tags = dev_set[:,2]
            scores = self(target_ids, context_ids, train=False) #TODO nograd ??
            loss = torch.sum(torch.abs(gold_tags - scores))*1000/len(self.example_corpus.examples)
            loss_over_time.append(loss)

            spearman_coeff = self.spearman.evaluate()
            spearman_over_time.append(spearman_coeff)

            examples_over_time.append(batches_seen*batch_size)

            if self.verbose:
              print("examples "+str(batches_seen*batch_size)+", loss = "+str(loss)+", spearman = "+str(spearman_coeff))
        
        batches_seen += 1

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
  parser.add_argument('eval_path', type=str, default="similarity.txt", help='Path to the corpus of human-scored similarity pairs.') 
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
    eval_path = args.eval_path,
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



