""" SpearmanEvaluation.py
Not meant to be used on its own.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

TODO verbose/debug, mais surtout refaire les commentaires
TODO le calcul sort des trucs bizarres
"""

import torch
from scipy import stats
import torch.nn as nn
import torch.nn.functional as F

class SpearmanEvaluation:
  """ Evaluates embeddings thanks to a human-scored corpus of word pair similarities.
  
  word2vec            Word2Vec, the model to evaluate
  pairs               Tensor, pair ids
  gold_scores         Tensor, human_scores
  """

  def __init__(self, similarity_file_path, word2vec):
    """ Initializes based on a corpus of human-scored similarity pairs.

    -> similarity_file_path: string, path to the file with human judgment similarity scores
    -> word2vec: Word2Vec, the model to evaluate
    """
    self.word2vec = word2vec
    self.pairs, self.gold_scores = self.__extract_corpus(similarity_file_path)
    if self.word2vec.verbose: print("Evaluation initialized.")


  def __extract_corpus(self, path):
    """ Extracts the word ids and gold scores.

    -> path: str, path to the corpus file
    <- pairs: tensor of word id pairs
    <- scores: tensor of similarity scores
    """
    pairs = []
    scores = []
    if self.word2vec.verbose: print("Extracting human similarity scores...")
    with open(path, 'r', encoding="utf-8") as f:
      for line in f.readlines():
        word1, word2, score = line.split(" ")
        if word1 in self.word2vec.w2i: # TODO c'est pas très beau comme code ça ^^"
          word1 = self.word2vec.w2i[word1]
        else:
          word1 = self.word2vec.w2i['UNK']
        if word2 in self.word2vec.w2i:
          word2 = self.word2vec.w2i[word2]
        else:
          word2 = self.word2vec.w2i['UNK']

        pairs.append([word1, word2, 0])
        scores.append(float(score))
        if self.word2vec.debug: print(str(word1)+" "+str(word2)+" "+str(score))
        
    return torch.tensor(pairs), torch.tensor(scores)


  def evaluate(self, scores):
    """ Calculates the Spearman correlation coefficient with p-value between the gold similarity scores
    and the similarity scores calculated by the model.

    from doc:
    vector1, vector2 : Two 1-D or 2-D arrays containing multiple variables and observations.
    Both arrays need to have the same length in the axis dimension.

    axis int or None, optional. If axis=0 (default), then each column represents a variable, with observations in the rows.
    If axis=1, the relationship is transposed: each row represents a variable, while the columns contain observations.
    If axis=None, then both arrays will be raveled.

    <- correlation: float or ndarray (2-D square)
    Spearman correlation matrix or correlation coefficient (if only 2 variables are given as parameters.
    Correlation matrix is square with length equal to total number of variables (columns or rows) in vctor1 and vector2 combined.

    <- pvalue: float
    The two-sided p-value for a hypothesis test whose null hypothesis is that two sets of data are uncorrelated.
    """
    loss, scores = self.word2vec.calculate_loss_scores(self.pairs, train=False)
    correlation, pvalue = stats.spearmanr(self.gold_scores, scores, axis=None, nan_policy="omit")
    return correlation, pvalue
    # TODO SpearmanRConstantInputWarning: An input array is constant; the correlation coefficent is not defined.

if __name__ == "__main__":
    print("This file isn't meant to be launched on it's own...")
