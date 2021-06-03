""" SpearmanEvaluation.py
Not meant to be used on its own.
"""

import torch
from scipy import stats

class Spearman_Evaluation: # TODO euh on met ça dans une classe, dans un programme, dans une méthode de W2V, ...?
  """ Evaluates embeddings thanks to a human-scored corpus of word pair similarities.
  
  learned_embeddings  Tensor, word embeddings learnt with Word2Vec, to test
  w2i                 dict, word to index translator, learned with the word embeddings
  pairs               Tensor, pair ids
  gold_scores         Tensor, human_scores
  """

  def init(self, similarity_file_path, w2i, learned_embeddings):
    """ Initializes based on a corpus of human-scored similarity pairs.

    -> similarity_file_path: string, path to the file with human judgment similarity scores
    -> w2i: dict, word to index translator, reference to the same index as the model being evaluated
    -> learned_embeddings: Embeddings, reference to the same embeddings as the model being evaluated
    """
    self.w2i = w2i
    self.learned_embeddings = learned_embeddings
    self.pairs, self.gold_scores = self.__extract_corpus(similarity_file_path)


  def __extract_corpus(self, path):
    """ Extracts the word ids and gold scores.
    """
    pairs = []
    scores = []

    with open(path, 'r', encoding="utf-8") as f:
      for line in f.readlines():
        word1, word2, score = line.split()
        if word1 in self.w2i: # TODO c'est pas très beau comme code ça ^^"
          word1 = self.w2i[word1]
        else:
          word1 = self.w2i['UNK']
        if word2 in self.w2i:
          word2 = self.w2i[word2]
        else:
          word1 = self.w2i['UNK']

        pairs.append([word1, word2])
        scores.append(score)

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
    word1_embeds = self.learned_embeddings(self.pairs[:0])
    word2_embeds = self.learned_embeddings(self.pairs[:1])
    scores = torch.mul(word1_embeds, word2_embeds)
    scores = torch.sum(scores, dim=1)
    scores = F.logsigmoid(scores)
    correlation, pvalue = stats.spearmanr(self.gold_scores, self.learned_scores, axis=None)
    return correlation, pvalue

if __name__ == "__main__":
    print("This file isn't meant to be launched on it's own...")