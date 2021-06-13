Projet TAL M1 Linguistique Informatique Université de Paris 2021
Sujet 7 : Modèle "word2vec" pour la construction de vecteurs de mots
Romane Gallienne, Cécile Guitel, Romy Rabemihanta



Dans ce projet, on se propose d'implémenter le modèle word2vec en version SkipGram avec sampling négatif. Un programme d'extraction de corpus est inclus, calibré pour le corpus l'Est Républicain, disponible aux liens suivants :
http://www.linguist.univ-paris-diderot.fr/~mcandito/divers/EP.tcs.melt.utf8.a.tgz
http://www.linguist.univ-paris-diderot.fr/~mcandito/divers/EP.tcs.melt.utf8.b.tgz
http://www.linguist.univ-paris-diderot.fr/~mcandito/divers/EP.tcs.melt.utf8.c.tgz

Un corpus de scores de similarité est aussi nécéssaire. Son extraction n'est pas incluse. Le fichier similarity.txt correspond au jeu de paires de mots RG65 disponible au format HTM sur le lien suivant : https://www.site.uottawa.ca/~mjoub063/wordsims.htm.

Cette implémentation est très inspirée de :
- l'implémentation word2vec CBOW du TP8 du cours d'apprentissage automatique 2 de Marie Candito
- l'implémentation word2vec SkipGram de Xiaofei Sun (https://adoni.github.io/2017/11/08/word2vec-pytorch/)



1) run the `extract_corpus.py` program or otherwise create a json file of a tokenized doc.
Help: python3 extract_corpus.py -h
Example: python3 extract_corpus.py raw_files_folder tokenized_doc.json --number_files=0 --number_sentences=0

2) run `extract_examples.py` on that file, with whatever hyperparameters you want.
Help: python3 extract_examples.py -h
Example: python3 extract_examples.py tokenized_doc.json examples.json --max_vocab_size=0 --min_occurences=5 --sub_sampling=0.001

3) run `Word2Vec.py` on the resulting file.
Help: python3 Word2Vec.py -h
Example: python3 Word2Vec.py examples.json similarity.txt embeddings.pt --embedding_dim=200 --early_stop_delta=0.00001

`SpearmanEvaluation.py` et `serialisation.py` are not supposed to run "on their own". They contain auxiliary functions used in the runnable files.