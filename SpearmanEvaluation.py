""" SpearmanEvaluation.py
Not meant to be used on its own.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
"""

import torch
from scipy import stats
import torch.nn as nn
import torch.nn.functional as F

class SpearmanEvaluation:
  """ Evaluates embeddings thanks to a corpus of human-scored word pair similarities.
  
  word2vec            Word2Vec, the model to evaluate
  pairs               Tensor, pair ids
  gold_scores         Tensor, human scores
  """

  def __init__(self, similarity_file_path, word2vec):
    """ Initializes based on a corpus of human-scored similarity pairs.
    Only keeps the pair of words that are known by the model (no UNK). If nan or error, check that.

    -> similarity_file_path: string, path to the file with human judgment similarity scores
    -> word2vec: Word2Vec, the model to evaluate
    """
    self.word2vec = word2vec
    self.pairs, self.gold_scores = self.__extract_corpus(similarity_file_path)


  def __extract_corpus(self, path):
    """ Extracts the word ids and gold scores.

    -> path: str, path to the corpus file
    <- pairs: tensor of word id pairs
    <- scores: tensor of similarity scores
    """
    pairs = []
    scores = []
    if self.word2vec.debug: print("\nExtracting human similarity scores...")
    with open(path, 'r', encoding="utf-8") as f:
      for line in f.readlines():
        word1, word2, score = line.split(" ")
        if word1 in self.word2vec.w2i and word2 in self.word2vec.w2i: # To avoid overrepresenting UNK, we
          # do not include unknown words.
          pairs.append([self.word2vec.w2i[word1], self.word2vec.w2i[word2], 0]) # The 0 is only there to
          # have something in gold tag position/the right shape of input for scoring. It will not be used.
          scores.append(float(score))
          if self.word2vec.debug: print(word1+" "+word2+" "+score)

    if len(pairs) < 1:
      print("Careful, Spearman corelation score won't work. Not enough of the word pairs from the similarity corpus are known by the model.")
      pairs.append([self.word2vec.w2i["UNK"], self.word2vec.w2i["UNK"], 0])
      scores.append(0.0)
        
    return torch.tensor(pairs), torch.tensor(scores)


  def evaluate(self):
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
    first_embeds = self.word2vec.context_embeddings(self.pairs[:,0])
    second_embeds = self.word2vec.context_embeddings(self.pairs[:,1])
    cos = nn.CosineSimilarity()
    scores = cos(first_embeds, second_embeds)
    correlation, pvalue = stats.spearmanr(self.gold_scores, scores, axis=None, nan_policy="omit")
    return correlation, pvalue

if __name__ == "__main__":
    print("This file isn't meant to be launched on it's own...")
