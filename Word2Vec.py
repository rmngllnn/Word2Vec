""" Word2Vec.py
Learns embeddings from a corpus of examples. Saves the performance results and the learned embeddings.


TODO arrêt de l'apprentissage
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
from SpearmanEvaluation import SpearmanEvaluation
import matplotlib.pyplot as plt

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

  embedding_dim         int, the size of the word embeddings learned
  iw2                   list of strings, index to word translator
  w2i                   dict, word to index translator
  target_embeddings     Embeddings, the weights of the "hidden" layer for each target word,
                        and the final learned embeddings
  context_embeddings    Embeddings, the representation of each context word, the input for forward/predict
  """
  def __init__(self,
      examples,
      i2w,
      w2i,
      embedding_dim,
      eval_corpus_path,
      verbose,
      debug):
    """ Initializes the model.
    TODO once we're done, put the default values to the best ones

    -> examples: list of lists of ints, the examples
    -> i2w: list, index to word translator
    -> w2i: dict, word to int translator
    -> eval_corpus_path: string, path to the human-scored similarity pairs text file
    -> verbose: verbose mode
    -> debug: debug mode
    """
    super(Word2Vec, self).__init__()

    assert type(debug) is bool, "Problem with debug."
    self.debug = debug

    assert type(verbose) is bool, "Problem with verbose."
    self.verbose = verbose

    assert type(examples) is list and type(examples[0]) is list and type(examples[0][0] is int), \
      "Problem with example corpus"
    self.examples = examples

    assert type(w2i) is dict, "Problem with w2i"
    self.w2i = w2i

    assert type(i2w) is list, "Problem with i2w"
    self.i2w = i2w

    assert type(embedding_dim) is int and embedding_dim > 0, "Problem with embedding_dim"
    self.embedding_dim = embedding_dim

    self.target_embeddings = nn.Embedding(len(self.i2w), self.embedding_dim, sparse=True)
    self.context_embeddings = nn.Embedding(len(self.i2w), self.embedding_dim, sparse=True) # NOTE Changed the first dimension from vocab_size to len of vocabulary, because the first is actually the max vocab size and not the actual vocab size

    range = 0.5/self.embedding_dim
    self.target_embeddings.weight.data.uniform_(-range,range)
    self.context_embeddings.weight.data.uniform_(-0,0)
    if self.verbose: print("\nEmbeddings initialized.")

    self.spearman = SpearmanEvaluation(eval_corpus_path, self.w2i, self.target_embeddings, self.verbose, self.debug)
    if self.verbose: print("Evaluator initialized.")
    if self.verbose: print("\nReady to train!")


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

    train_set = self.examples[0:int(len(self.examples)*80/100)] # TODO quel pourcentage?
    dev_set = torch.tensor(self.examples[int(len(self.examples)*80/100):])

    for epoch in range(number_epochs):
      random.shuffle(train_set)
      batches = torch.split(torch.tensor(train_set), batch_size)
      early_stopping = False
      
      for batch in batches:
        target_ids = batch[:,0]
        context_ids = batch[:,1]
        gold_tags = batch[:,2]

        scores = self(target_ids, context_ids) # Forward propagation.
        batch_loss = torch.sum(torch.abs(gold_tags - scores)) # The loss is the difference between the
        # probability we want to associate with the example (gold_tag) and the probability measured by
        # the model (score))
        # cross-entropy!
        # batch_loss = torch.sum((-1)*gold_tags*scores-(1-gold_tags)*(1-scores))
        self.zero_grad() # Reinitialising model gradients.
        batch_loss.backward() # Back propagation, computing gradients.
        self.optimizer.step() # One step in gradient descent.

        if batches_seen*batch_size % 10000 == 0:
          with torch.no_grad():
            target_ids = dev_set[:,0]
            context_ids = dev_set[:,1]
            gold_tags = dev_set[:,2]
            scores = self(target_ids, context_ids, train=False)
            loss = torch.sum(torch.abs(gold_tags - scores))*1000/len(self.examples)
            #loss = torch.sum((-1)*gold_tags*scores-(1-gold_tags)*(1-scores))*1000/len(self.examples)
            if len(loss_over_time) > 0 and (loss - loss_over_time[len(loss_over_time)-1]) > -0.01 : #TODO : la condition -0.01 en hyperparamètres aussi?
              early_stopping = True
              break
            loss_over_time.append(loss)

            spearman_coeff = self.spearman.evaluate(scores)
            spearman_over_time.append(spearman_coeff)

            examples_over_time.append(batches_seen*batch_size)

            if self.verbose:
              print("examples "+str(batches_seen*batch_size)+", loss = "+str(loss)+", spearman = "+str(spearman_coeff))

          if early_stopping :
            if self.verbose :
              print("Loss is no longer evoluting : stop the training.")
            break
        
        batches_seen += 1

    if self.verbose: print("Training done!")

    results = {}
    results["examples"] = examples_over_time
    results["loss"] = loss_over_time
    results["spearman"] = spearman_over_time

    if self.verbose :
      fig, ax = plt.subplots()
      ax.plot(results["examples"], results["loss"])
      plt.show()
      
    return results


  def save_embeddings(self, save_path):
    """ Saves the embeddings.

    -> save_path: string, path of the file to save as (fotmat .pt)
    Documentation : https://pytorch.org/docs/stable/notes/serialization.html
    """
    #serialize(self.target_embeddings, save_path)
    # TODO TypeError: Object of type Embedding is not JSON serializable

    torch.save(self.target_embeddings, save_path)
    #TODO : à choisir entre l'une ou l'autre notation (revient au même en fait)
    #torch.save(self.target_embeddings.state_dict()['weight'], save_path)
    




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('example_corpus_path', type=str, default="examples.json", help='Path to the serialized tokenized corpus.')
  parser.add_argument('eval_corpus_path', type=str, default="similarity.txt", help='Path to the corpus of human-scored similarity pairs.')
  parser.add_argument('save_embeddings_path', type=str, default="embeddings.pt", help='Path to the file for the learned embeddings.') 
  parser.add_argument('--embedding_dim', type=int, default=100, help='The size of the word embeddings')
  parser.add_argument('--number_epochs', type=int, default=10, help='The number of epochs to train for')
  parser.add_argument('--batch_size', type=int, default=100,  help='The number of examples in a batch')
  parser.add_argument('--learning_rate', type=int, default=0.05, help='The learning rate step when training')
  parser.add_argument('--verbose', type=bool, default=True, help='Verbose mode')
  parser.add_argument('--debug', type=bool, default=False, help='Debug mode')
  args = parser.parse_args()

  example_dict = deserialize(args.example_corpus_path)

  model = Word2Vec(examples = example_dict["examples"],
    i2w = example_dict["i2w"],
    w2i = example_dict["w2i"],
    embedding_dim = args.embedding_dim,
    eval_corpus_path = args.eval_corpus_path,
    verbose = args.verbose,
    debug = args.debug)

  results = model.train(batch_size=args.batch_size,
    number_epochs=args.number_epochs,
    learning_rate=args.learning_rate)

  model.save_embeddings(args.save_embeddings_path)

