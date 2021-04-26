#Dans ce fichier, on se propose d'initialiser l'implémentation du modèle word2vec, en exploitant le corpus (extraction, indices, nettoyage, embeddings)
#(Très) inspiré TP8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

torch.manual_seed(1)

CONTEXT_SIZE = 2 #pour commencer
EMBEDDING_DIM = 10 #pour commencer
tokens = []
vocab = set(['D1','D2','F1','F2'])
#TODO : remplir la liste du vocabulaire vocab avec le corpus, séparer en tokens et remplir tokens
#TODO : avec une fonction, ouvrir le fichier donné par le parser, parcourir chaque ligne, supprimer les tags, gérer ou non les majuscules, attention à l'encodage (UTF-8)
#Aussi, compter les occurrences de mots et trier selon un seuil de fréquence
def fill_vocab(infile) :
  pass

#Note de Romy : je ne comprends pas pourquoi dans le TP la prof utilise une liste de tokens et un set des mots de vocabulaire, les deux sont censés faire la même chose, non ?
#D1, D2, F1, F2 sont des mots spéciaux pour les débuts et fins de phrases 
#Leur nombre diffère si on change la taille de la fenêtre
i2w = list(vocab)
w2i = {w: i for i, w in enumerate(i2w)}

#Création des exemples d'apprentissage avec negative sampling
#Aussi, création des classes gold 1/0 (ou 1/-1) dans une liste de même taille que vocab (note Romy : pas sûre...)
def create_examples (tokens, w2i) :
  """Un exemple est une paire (liste de mots contexte, mot cible), la sortie est une liste d'exemple de taille len(vocab) et une liste des classes gold"""
  pass #TO DO

examples = create_examples(tokens, w2i)


#PARTIE MAIN
parser = argparse.ArgumentParser()
#Dans le terminal, on écrira "python3 init.py NOM_DU_FICHIER_CORPUS"
parser.add_argument('examples_file', default=None, help='Corpus utilisé pour la création d\'exemples d\'apprentissage pour les embeddings.')
