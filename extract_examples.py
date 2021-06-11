""" extract_examples.py
Creates a corpus of examples and its indexes as an ExampleCorpus object, serializes these elements as a
dict in a json file.

TODO sous-ecahntillonage des mots fréquents
"""

from serialisation import deserialize
from serialisation import serialize
from collections import Counter
import numpy as np
import argparse
import random as rd
import time

class ExampleCorpus:
    """ Corpus of examples, created based on a serialized tokenized doc and the given parameters.
    Positive examples from the doc, negative examples created using negative sampling.

    verbose             bool, verbose mode
    debug               bool, debug mode

    max_vocab_size      int, the maximum number of real-word embeddings to learn
    context_size        int, the size of the context window on each side of the target word
                        if = 2, then for each word, four positive examples are created:
                        (x-2,x,+), (x-1,x,+), (x+1,x,+), (x+2,x,+)
    number_neg_examples int, the number of negative examples per positive example
                        if = 2, then for each word, two negative examples are randomly created:
                        (r1, x, -), (r2, x, -)
    min_occurences      int, the minimum number of occurences for a word to be learned
    sampling            float, sampling rate to calculate the negative example distribution probability

    examples            list of tuples, examples[(int)] = (context_word_id, target_word_id, pos|neg)
    i2w                 list of ints, index to word translator, i2w[index(int)] = "word"
    w2i                 dict, word to index translator, w2i["word"] = index(int)
    """
    def __init__(self,
            tokenized_doc,
            context_size,
            max_vocab_size,
            min_occurences,
            sampling,
            number_neg_examples,
            debug,
            verbose):
        assert type(debug) is bool, "Problem with debug."
        self.debug = debug

        assert type(verbose) is bool, "Problem with verbose."
        self.verbose = verbose

        assert type(tokenized_doc) is list and type(tokenized_doc[0]) is list and \
        type(tokenized_doc[0][0] is str), "Problem with tokenized_doc."
        self._tokenized_doc = tokenized_doc

        assert type(context_size) is int and context_size > 0, "Problem with context_size."
        self.context_size = context_size

        assert type(max_vocab_size) is int, "Problem with max_vocab_size."
        self.max_vocab_size = max_vocab_size

        assert type(min_occurences) is int, "Problem with max_vocab_size."
        self.min_occurences = min_occurences

        assert type(sampling) is float and sampling > 0 and sampling < 1, "Problem with sampling."
        self.sampling = sampling

        assert type(number_neg_examples) is int and number_neg_examples > 0, "Problem with number_neg_examples."
        self.number_neg_examples = number_neg_examples

        if self.verbose:
            print("\Started example extraction.")
            print("\nParameters:")
            print("context size = " + str(self.context_size))
            print("max vocabulary size = " + str(self.max_vocab_size))
            print("min occurences to learn an embedding = "+str(self.min_occurences))
            print("sampling rate = " + str(self.sampling))
            print("negative examples per positive example = " + str(self.number_neg_examples))
        
        self._occurence_counter = self.__get_occurence_counter()
        self.i2w = list(set([token for token in self._occurence_counter]+["UNK"])) # Just in case no unk words...
        self.w2i = {w: i for i, w in enumerate(self.i2w)}
        self._indexed_doc = self.__get_indexed_doc()
        self._prob_dist =  self.__get_prob_dist()
        self.examples = self.__create_examples()


    def __get_occurence_counter(self):
        """Generates the occurence count with only the max_vocab_size most common words and the special words.
        Special words: UNK, *D1*, ...

        cf https://docs.python.org/3/library/collections.html#collections.Counter

        NOTE: We did consider using CountVectorizer but couldn't figure out how to deal with unknown words, which we do want to count too, because we need to create negative examples with them to create the other embeddings, and we need their distribution for that.

        NOTE: a Counter will give a count of 0 for an unknown word and a dict will not.

        NOTE: The occurence_counter need to be set before we replace rare words with UNK and add *D1* and all.
        That's because otherwise, a special word might not appear often enough to make the cut.
        We presumed that adding a few embeddings to the size wouldn't change much in terms of computation.
        However, it's absolutely possible to change it so that we keep max_vocab_size as the total number
        of embeddings learned, an only learn max_vocab_size - 2*self.context_size - 1 real word
        embeddings.
        """
        occurence_counter = Counter() # We want to count the number of occurences of each token, to only keep the max_vocab_size most common ones.

        for sentence in self._tokenized_doc:
            occurence_counter.update(sentence)

        UNK_counter = Counter()

        if self.max_vocab_size > 0 and self.max_vocab_size < len(occurence_counter.keys()): # If a maximum
            # vocab size has been set, and we're over it...
            UNK_counter.update({token : count for (token, count)
                    in occurence_counter.most_common()[self.max_vocab_size:]})

        if self.min_occurences > 0: # if a minimum occurence count has been set...
            UNK_counter.update({token : count for (token, count) in occurence_counter.items()
                    if count >= self.min_occurences})  
        
        occurence_counter.subtract(UNK_counter) # All those other tokens are deleted from the occurence count...
        occurence_counter.update({"UNK": sum([UNK_counter[token] for token in UNK_counter])}) # and counted as occurences of UNK.

        occurence_counter.update({out_of_bounds : len(self._tokenized_doc) for out_of_bounds
            in ["*D"+str(i)+"*" for i in range(1,self.context_size+1)]
            + ["*F"+str(i)+"*" for i in range(1,self.context_size+1)]}) # We add one count of each out-of-bound special word per sentence.

        if self.verbose: print("\nOccurence counter created.")
        if self.debug: print("Occurence count: "+str(+occurence_counter))
        return +occurence_counter # "+" removes 0 or negative count elements.


    def __get_indexed_doc(self):
        """Generates an indexized version of the __tokenized doc, adding out of bound and unknown special words.

        NOTE: If we wanted to adapt this model for other uses (for example, evaluating the 'likelihood'
        of a document), we'd probably need to adapt this method somehow, either for preprocessing input
        in main or for use in pred/forward. Since we don't care about that, it's set to private.
        """
        known_vocab_doc = []
        for sentence in self._tokenized_doc:
            sentence = ["*D"+str(i)+"*" for i in range(1,self.context_size+1)] + sentence + \
                ["*F"+str(i)+"*" for i in range(1,self.context_size+1)] # We add out-of-bound special words.
            for i, token in enumerate(sentence):
                    if token not in self.w2i: # If we don't know a word...
                        sentence[i] = "UNK" # we replace it by UNK.
            known_vocab_doc.append(sentence) # when I tried to change the __tokenized doc directly, the changes got lost, sooo TODO Cécile : look into how referencing works in python again...

        # We switch to indexes instead of string tokens.
        indexed_doc = [[self.w2i[token] for token in sentence] for sentence in known_vocab_doc]

        if self.verbose: print("\nIndexed doc created.")
        if self.debug: print("Indexed doc: "+str(indexed_doc[0:3]))
        return indexed_doc


    def __get_prob_dist(self):
        """Generates the probability distribution of words in the vocabulary to get sampled as negative
        examples, in the form of a dictionary of prob_dist["word"] = probability
        """
        prob_dist = {}

        total_word_count = sum([self._occurence_counter[word]**self.sampling for word in self._occurence_counter])
        for word in self._occurence_counter:
            prob_dist[word] = (self._occurence_counter[word]**self.sampling)/total_word_count

        if self.verbose: print("\nProbability distribution created.")
        if self.debug: print("Probability distribution: "+str(prob_dist))
        return prob_dist


    def __create_examples(self):
        """Creates positive and negative examples using negative sampling. The window size is also sampled
        between context_size and 1.

        An example is a (target word, context word, gold tag) tuple.
        It is tagged 1 for positive (extracted from the corpus) and 0 for negative (randomly created).
        """
        examples = []
        start_time = time.time()

        rng = np.random.default_rng(12345)
        sampling_values = []
        sampling_words = [word for word in self._prob_dist]
        current_total_proba = 0
        overall_total_proba = 100000
        for word in sampling_words:
            current_total_proba += self._prob_dist[word]*overall_total_proba
            sampling_values.append(current_total_proba) # We create a list with the cummulative
            # probability of each word, so that each one takes more of less 'space' in the amount
            # based on its probability. We can then sample between 0 and the total, and each one has
            # (approximately) that probability of being picked.
            # This is less accurate, but quicker than the previous method.
        if self.verbose: print("Negative example sampler created.")

        for i, sentence in enumerate(self._indexed_doc): # For each sentence...
            if self.verbose and i % 100 == 0:
                curr_time = time.time()
                print("sentence ",i,"\t",len(examples),"\t", curr_time-start_time)
            for target_i in range(self.context_size, len(sentence) - self.context_size): # For each word of the actual sentence...
                    sampled_context_size = rd.randint(1, self.context_size) # We sample a context size.
                    for context_i in range(target_i - sampled_context_size, target_i + sampled_context_size + 1): # For each word in the (sampled) context window...
                        if target_i is not context_i:
                                examples.append((sentence[target_i], sentence[context_i], 1))
                                if self.debug: print(self.i2w[sentence[target_i]]+","+self.i2w[sentence[context_i]]+",1")
                        
                    for neg_example in range(self.number_neg_examples): # Now, negative sampling! Using that probability distribution/sampler.
                        random_probability = rng.integers(low=0, high=overall_total_proba)
                        neg_rank = np.searchsorted(sampling_values, random_probability)
                        neg_id = self.w2i[sampling_words[neg_rank]]
                        if self.debug: print(self.i2w[sentence[target_i]]+","+self.i2w[neg_id]+",0")
                        examples.append((sentence[target_i], neg_id, 0))
        
        if self.verbose: print("\n"+str(len(examples))+" examples created.")
        return examples


    def to_dict(self):
        """ Objects are not json serializable, but dictionaries are. This function returns a dict based on
        the ExampleCorpus object.
        """
        dict = {}
        dict["examples"] = self.examples
        dict["i2w"] = self.i2w
        dict["w2i"] = self.w2i
        return dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_path', type=str, default="tokenized_doc.json", help='Path to the serialized tokenized corpus.')
    parser.add_argument('save_as', type=str, default="example_corpus.json", help='Path to the serialized tokenized corpus')
    parser.add_argument('--max_vocab_size', type=int, default=0, help='The maximum number of real-word embeddings to learn, to set to 0 if not applicable')
    parser.add_argument('--min_occurences', type=int, default=3, help='The minimum number of occurences for a word to be learned')
    parser.add_argument('--context_size', type=int, default=2, help='The size of the context window on each side of the target word')
    parser.add_argument('--number_neg_examples', type=int, default=3, help='The number of negative examples per positive example')
    parser.add_argument('--sampling', type=float, default=0.75, help='The sampling rate to calculate the negative example distribution probability')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose mode')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')
    args = parser.parse_args()

    tokenized_doc = deserialize(args.corpus_path)

    corpus = ExampleCorpus(tokenized_doc = tokenized_doc,
        context_size = args.context_size,
        sampling = args.sampling,
        number_neg_examples = args.number_neg_examples,
        max_vocab_size = args.max_vocab_size,
        min_occurences = args.min_occurences,
        verbose = args.verbose,
        debug = args.debug)

    serialize(corpus.to_dict(), args.save_as)