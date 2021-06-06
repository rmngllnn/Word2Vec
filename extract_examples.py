""" extract_examples.py
Creates a corpus of examples and its indexes as an ExampleCorpus object, serializes these elements in.

#TODO dynamic sampling of window size
#TODO min number of occurences for a word to be learned
"""

from serialisation import deserialize
from serialisation import serialize
from collections import Counter
import numpy as np
import argparse

class ExampleCorpus:
    """ Corpus of examples, created based on a serialized tokenized doc and the given parameters.
    Positive examples from the doc, negative examples created using negative sampling.

    verbose             bool, verbose mode
    debug               bool, debug mode

    vocab_size          int, the number of real-word embeddings to learn
    context_size        int, the size of the context window on each side of the target word
                        if = 2, then for each word, four positive examples are created:
                        (x-2,x,+), (x-1,x,+), (x+1,x,+), (x+2,x,+)
    number_neg_examples int, the number of negative examples per positive example
                        if = 2, then for each word, two negative examples are randomly created:
                        (r1, x, -), (r2, x, -)
    sampling            float, sampling rate to calculate the negative example distribution probability

    examples            list of tuples, examples[(int)] = (context_word_id, target_word_id, pos|neg)
    i2w                 list of ints, index to word translator, i2w[index(int)] = "word"
    w2i                 dict, word to index translator, w2i["word"] = index(int)
    """
    def __init__(self,
            tokenized_doc,
            context_size,
            vocab_size,
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

        assert type(vocab_size) is int and vocab_size > 0, "Problem with vocab_size."
        self.vocab_size = vocab_size

        assert type(sampling) is float and sampling > 0 and sampling < 1, "Problem with sampling."
        self.sampling = sampling

        assert type(number_neg_examples) is int and number_neg_examples > 0, "Problem with number_neg_examples."
        self.number_neg_examples = number_neg_examples

        if self.verbose:
            print("\Started example extraction.")
            print("\nParameters:")
            print("context size = " + str(self.context_size))
            print("max vocabulary size = " + str(self.vocab_size))
            print("sampling rate = " + str(self.sampling))
            print("negative examples per positive example = " + str(self.number_neg_examples))
        
        self._occurence_counter = self.__get_occurence_counter()
        self.i2w = list(set([token for token in self._occurence_counter]+["UNK"])) # Just in case no unk words...
        self.w2i = {w: i for i, w in enumerate(self.i2w)}
        self._indexed_doc = self.__get_indexed_doc()
        self._prob_dist =  self.__get_prob_dist()
        self.examples = self.__create_examples()
        
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

        for sentence in self._tokenized_doc:
            occurence_counter.update(sentence) # cf https://docs.python.org/3/library/collections.html#collections.Counter

        if len(occurence_counter.keys()) - self.vocab_size > 0: # If there are tokens left over...
            #print("total:"+str(occurence_counter))
            UNK_counter = {token : count for (token, count)
                    in occurence_counter.most_common()[self.vocab_size:]} # (it's actually a dict not a counter but shrug, doesn't matter for what we're doing with it)
            #print("unk: "+str(UNK_counter))
            occurence_counter.subtract(UNK_counter) # all those other tokens are deleted from the occurence count...
            #print("after subtract:"+str(occurence_counter))
            occurence_counter.update({"UNK": sum([UNK_counter[token] for token in UNK_counter])}) # and counted as occurences of UNK.

        occurence_counter.update({out_of_bounds : len(self._tokenized_doc) for out_of_bounds
            in ["*D"+str(i)+"*" for i in range(1,self.context_size+1)]
            + ["*F"+str(i)+"*" for i in range(1,self.context_size+1)]}) # We add one count of each out-of-bound special word per sentence.

        if self.verbose: print("\nOccurence counter created.")
        if self.debug: print("Occurence count: "+str(+occurence_counter))
        return +occurence_counter # "+" removes 0 or negative count elements.


    def __get_indexed_doc(self):
        """Generates an indexized version of the __tokenized doc, adding out of bound and unknown special words.

        NOTE: If we wanted to adapt this model for other uses (for example, evaluating the 'likelihood' of a
        document), we'd probably need to adapt this method somehow, either for preprocessing input in main or
        for use in pred/forward. Since we don't care about that, it's set to private.
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
        """Generates the probability distribution of known words to get sampled as negative examples.
        """
        prob_dist = {}

        total_word_count = sum([self._occurence_counter[word]**self.sampling for word in self._occurence_counter])
        for word in self._occurence_counter:
            prob_dist[word] = (self._occurence_counter[word]**self.sampling)/total_word_count

        if self.verbose: print("\nProbability distribution created.")
        if self.debug: print("Probability distribution: "+str(prob_dist))
        return prob_dist


    def __create_examples(self):
        """Creates positive and negative examples using negative sampling.

        An example is a (target word, context word, gold tag) tuple.
        It is tagged 1 for positive (extracted from the corpus) and 0 for negative (randomly created).
        """
        examples = []
        if self.debug: print("\nCreating examples...")

        for i, sentence in enumerate(self._indexed_doc): # For each sentence...
            if self.verbose and i % 100 == 0:
                print(str(i)+"\t"+str(len(self._indexed_doc)))
            for target_i in range(self.context_size, len(sentence) - self.context_size): # For each word of the actual sentence...
                    for context_i in range(target_i - self.context_size, target_i + self.context_size + 1): # For each word in the context window...
                        if target_i is not context_i:
                                examples.append((sentence[target_i], sentence[context_i], 1))
                                if self.debug: print(self.i2w[sentence[target_i]]+","+self.i2w[sentence[context_i]]+",1")
                        
                    for neg_example in range(self.number_neg_examples): # Now, negative sampling! Using that probability distribution.
                        random_token = np.random.choice(list(self._prob_dist.keys()), p=list(self._prob_dist.values()))
                        if self.debug: print(self.i2w[sentence[target_i]]+","+random_token+",0")
                        examples.append((sentence[target_i], self.w2i[random_token], 0))
        
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
    parser.add_argument('save_as', type=str, default="example_corpus.json", help='Path to the serialized tokenized corpus.')
    parser.add_argument('--vocab_size', type=int, default=1000, help='The maximum number of real-word embeddings to learn')
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
        vocab_size = args.vocab_size,
        verbose = args.verbose,
        debug = args.debug)

    serialize(corpus.to_dict(), args.save_as)
    print(corpus.i2w)
