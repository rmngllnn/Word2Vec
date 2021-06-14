""" Word2Vec.py
Learns embeddings from a corpus of examples. Saves the performance results and the learned embeddings.
Generates a graph of the learning curve.

TODO save performance results
TODO programme python pour tester les différents hyperparamètres
TODO README instructions
TODO rapport (17)
TODO soutenance (24 Juin 10h40)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
from serialisation import deserialize, serialize
from SpearmanEvaluation import SpearmanEvaluation
import matplotlib.pyplot as plt

torch.manual_seed(1)


class Word2Vec(nn.Module):
  """ Word2Vec model, SkipGram version with negative sampling.
  Creates word embeddings from a corpus of examples using gradient descent.
  Evaluates performance with loss and word similarity (spearman coefficient, compared with human scores).

  Use the train method to calculate the embeddings, then the save method to save them.

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
  spearman              SpearmanEvaluator, used to evaluate the embeddings
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

    self.spearman = SpearmanEvaluation(eval_corpus_path, self)
    if self.verbose: print("Evaluator initialized.")
    if self.verbose: print("\nReady to train!")


  def forward(self, target_embeds, context_embeds, train=True):
    """ Calculates the probability of an example being found in the corpus, for all examples given.
    That is to say, the probability of a context word being found in the window of a context word.
    P(c|t) = sigmoid(c.t)

    We'll worry about the gold tags later, when we calculate the loss.
    P(¬c|t) = 1 - P(c|t)

    -> target_embeds
    -> context_embeds
    <- scores: tensor, shape: (batch_size), line tensor of scores
    """
    scores = torch.mul(target_embeds, context_embeds)
    #if self.debug: print("mul: "+str(scores))
    scores = torch.sum(scores, dim=1)
    #if self.debug: print("sum: "+str(scores))
    #scores = F.logsigmoid(scores)
    sig = nn.Sigmoid()
    scores = sig(scores)
    #if self.debug: print("sig: "+str(scores))

    #print(scores)

    return scores


  def calculate_loss_scores(self, set, train=True):
    """ Calculates the loss on a set of examples using cross-entropy.

    -> set
    -> train
    <- loss
    < scores
    """
    target_ids = set[:,0]
    context_ids = set[:,1]
    gold_tags = set[:,2].float()

    target_embeds = self.target_embeddings(target_ids)
    context_embeds = None
    if train:
      context_embeds = self.context_embeddings(context_ids)
    else:
      context_embeds = self.target_embeddings(context_ids) # So the same method is used for training and
      # for the equivalent of predicting

    #if self.debug:
    #  print("\nCalculate loss and scores on batch.")
    #  print("Target ids: "+str(target_ids))
    #  print("Context ids: "+str(context_ids))
    #  print("Target embeddings: "+str(target_embeds))
    #  print("Context embeddings: "+str(context_embeds))

    scores = self(target_embeds, context_embeds) # Forward propagation.
    
    #loss = torch.sum(torch.abs(gold_tags - scores)) # The loss is the difference between the
    # probability we want to associate with the example (gold_tag) and the probability measured by
    # the model (score))
    # cross-entropy! # TODO loss function
    # batch_loss = torch.sum((-1)*gold_tags*scores-(1-gold_tags)*(1-scores))

    loss_func = nn.BCELoss()
    loss = loss_func(scores, gold_tags)
    

    return loss, scores

  def train(self,
      max_number_epochs,
      learning_rate,
      batch_size,
      evaluate_every,
      early_stop_delta):
    """ Executes gradient descent to learn the embeddings.
    This is where we switch to tensors: we need the examples to be in a list in order to shuffle them,
    but after that, for efficiency, we do all calculations using matrices.

    -> max_number_epochs: int, the maximum number of epochs to train for
    -> batch_size:        int, the number of examples in a batch
    -> learning_rate:     float, the learning rate step when training
    -> evaluate_every:    int, the number of batches to train on between two evaluations
    -> early_stop_delta:  float, training stops once the loss improves by less than this amount between
                          two evaluations
    <- results:           dict, results of evaluation over time
    """
    assert type(max_number_epochs) is int and max_number_epochs > 0, "Problem with max_number_epochs."
    assert type(learning_rate) is float and learning_rate > 0, "Problem with learning_rate."
    assert type(batch_size) is int and batch_size > 0, "Problem with batch_size."

    if self.verbose:
      print("\nTraining parameters:")
      print("number of epochs = " + str(max_number_epochs))
      print("learning rate = " + str(learning_rate))
      print("batch size = " + str(batch_size))
      print("evaluate every ... batches = " + str(evaluate_every))
      print("early stop delta = " + str(early_stop_delta))
      print("\nTraining...")

    batches_seen = 0
    results = {}
    results["examples"] = []
    results["loss"] = []
    results["spearman"] = []
    self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    train_set = self.examples[0:int(len(self.examples)*80/100)] # TODO quel pourcentage?
    eval_set = torch.tensor(self.examples[int(len(self.examples)*80/100):])

    for epoch in range(max_number_epochs):
      random.shuffle(train_set)
      batches = torch.split(torch.tensor(train_set), batch_size)
      
      for batch in batches:
        self.zero_grad() # Reinitialising model gradients.
        batch_loss, batch_scores = self.calculate_loss_scores(batch)
        batch_loss.backward() # Back propagation, computing gradients.
        self.optimizer.step() # One step in gradient descent.
        batches_seen += 1

        if batches_seen % evaluate_every == 0: # Every X batches, we evaluate the embeddings.
          with torch.no_grad(): # We DO NOT want it to count toward training.
            eval_loss, eval_scores = self.calculate_loss_scores(eval_set)
            results["loss"].append(eval_loss.item())

            spearman_coeff = self.spearman.evaluate()
            results["spearman"].append(spearman_coeff)

            results["examples"].append(batches_seen*batch_size)

            if self.verbose:
              print("examples seen = "+str(results["examples"][-1])+", loss = "+str(results["loss"][-1])+", spearman = "+str(results["spearman"][-1]))

            if len(results["loss"]) > 1 and \
              (results["loss"][-2] - eval_loss) < early_stop_delta: # If learning is slowing down enough...
              if self.verbose: print("Training done! Early stopping.")
              fig, ax = plt.subplots()
              ax.plot(results["examples"], results["loss"], "o-")
              plt.show()
              return results

    if self.verbose: print("Training done! Reached max epoch before early stopping condition.")
    fig, ax = plt.subplots()
    ax.plot(results["examples"], results["loss"], "o-")
    plt.show()
    return results


  def save_embeddings(self, save_path):
    """ Saves the learned embeddings, using pytorch's save method.
    Documentation : https://pytorch.org/docs/stable/notes/serialization.html

    NOTE We're not using json serializing because TypeError: Object of type Embedding is not JSON
    serializable

    -> save_path: string, path of the file to save as (fotmat .pt)
    """
    #torch.save(self.target_embeddings, save_path)
    #torch.save(self.target_embeddings.state_dict()['weight'], save_path)
    # Both are possible, see doc.

    #Other way to save embeddings into a .txt file
    embedding = self.target_embeddings.weight.data.numpy()
    f = open(save_path, 'w')
    f.write('%d %d\n' % (len(self.i2w), self.embedding_dim))
    for idx, w in enumerate(self.i2w):
      e = embedding[idx]
      e = ' '.join(map(lambda x: str(x), e))
      f.write('%s %s\n' % (w, e))

    # TODO save results too?


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('example_corpus_path', type=str, default="examples.json", help='Path to the serialized tokenized corpus, json format, do not forget the extension')
  parser.add_argument('eval_corpus_path', type=str, default="similarity.txt", help='Path to the corpus of human-scored similarity pairs, txt format, do not forget the extension')
  parser.add_argument('save_embeddings_path', type=str, default="embeddings.pt", help='Path to the file for the learned embeddings, pt format, do not forget the extension') 
  parser.add_argument('--embedding_dim', type=int, default=100, help='The size of the word embeddings to be learned')
  parser.add_argument('--max_number_epochs', type=int, default=1000, help='The maximum number of epochs to train for')
  parser.add_argument('--batch_size', type=int, default=1000,  help='The number of examples in a batch')
  parser.add_argument('--learning_rate', type=float, default=0.1, help='The learning rate step when training')
  parser.add_argument('--evaluate_every', type=int, default=10, help='The number of batches to train on between two evaluations')
  parser.add_argument('--early_stop_delta', type=float, default=0, help='Training stops once the loss improves by (less than) this amount between two evaluations')
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
    max_number_epochs=args.max_number_epochs,
    learning_rate=args.learning_rate,
    evaluate_every=args.evaluate_every,
    early_stop_delta=args.early_stop_delta)

  model.save_embeddings(args.save_embeddings_path)

