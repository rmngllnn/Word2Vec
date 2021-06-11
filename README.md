Dans ce projet, on se propose d'implémenter le modèle word2vec en version SkipGram avec sampling négatif. Un programme d'extraction de corpus est incluse.

Cette implémentation est inspirée de :
- l'implémentation word2vec CBOW du TP8 du cours d'apprentissage automatique 2 de Marie Candito
- l'implémentation word2vec SkipGram de Xiaofei Sun (https://adoni.github.io/2017/11/08/word2vec-pytorch/)

Corpus :
http://www.linguist.univ-paris-diderot.fr/~mcandito/divers/EP.tcs.melt.utf8.a.tgz
http://www.linguist.univ-paris-diderot.fr/~mcandito/divers/EP.tcs.melt.utf8.b.tgz
http://www.linguist.univ-paris-diderot.fr/~mcandito/divers/EP.tcs.melt.utf8.c.tgz 

1) run the extract_corpus program or otherwise creates a json file of a tokenized doc. Command line : `python3 extract_corpus.py corpus_path save_path `
2) run extract_examples on that file, with whatever hyperparameters you want. Command line : `python3 extract_examples.py corpus_path FileName.json`
3) run Word2Vec on the resulting file. Command line : `python3 Word2Vec.py example_corpus_path eval_corpus_path `

`SpearmanEvaluation.py` et `serialisation.py` are not supposed to run "on their own". They contain auxiliary functions used in the runnable files.

