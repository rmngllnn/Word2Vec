import json
import os

def extract_corpus(infile):
  """Extracts a file, gets rid of the POS tags, tokenizes it.
  Sentences are split into words based on " ". Nothing is done to uppercase letters or punctuation.
  Calibrated for the "L'Est républicain" corpus, cf README for the original download links.

  -> infile: string, path to the file
  <- tokenized_doc: list of lists of strings, a tokenized doc made of sentences made of words
  """
  tokenized_doc = []
  with open(infile, 'r', encoding = "utf-8-sig") as f:
    for line in f.readlines():
      sentence = []
      
      for word in line.split():
        sentence.append(word.split("/")[0])
      tokenized_doc.append(sentence)
  return tokenized_doc


"""For details about json files see https://www.codeflow.site/fr/article/python-json"""

def serialisation_data(data, title):
    """Serialize data in a json file (in desktop)
    -> data is the variable you want to serialize
    -> title must be a string : "title.json"
    <- Save a json file in desktop
    """

    with open(title, "w+") as file:
        json.dump(data,file)


def open_file(json_file):
    """ open_file opens a json file and puts content in variable data
    -> a json file. Must a string "jsonfile.json"
    <- list of words
    """
    
    with open(json_file) as json_data:
        data = json.load(json_data)
        
    return data

"""
data = extract_corpus("mini_corpus.txt")
serialisation_data(data, "test.json")
print("DONE !")
"""

def extract_create_corpus_file(infile, n_sentences) :
  """Opens a file, gets rid of the POS tags, tokenizes its first n_sentences lines and save the tokenization into a txt file.
  This function is necessary to test our model on a wanted number of sentences.
  We may adjust the size of our corpus later on.

  NOTE : a file contains 100 000 lines.

   -> infile: string, path to the file
  """
  with open(infile, 'r', encoding = "utf-8-sig") as f:
    with open(str(n_sentences)+"_lines_corpus.txt", "w", encoding = "utf-8-sig") as f2 :
      for line in f.readlines()[:n_sentences]:
        sentence = []
        for word in line.split():
          sentence.append(word.split("/")[0])
        f2.write(str(sentence)+'\n')
  f.close()


#extract_create_corpus_file("EP.tcs.melt.utf8.b/EP.tcs.melt.utf8.split-ba", 100)
#Creates 100_lines_corpus.txt

def extract_create_corpus_path(path_name, n_file, n_sentences) :
  """Opens the first n_file files of a folder, gets rid of the POS tags, tokenizes its first n_sentences lines and save the tokenization into a serialised(?) txt file.

  NOTE : a file contains 100 000 lines, one folder contains 25 files, we have 3 folders.

   -> infile: string, path to the folder
  """
  path = os.getcwd()
  directory = path + "/" + path_name
  list_file = os.listdir(directory)[:n_file]
  n_lines = n_file*n_sentences
  f = open(str(n_lines)+"_lines_corpus_txt","w", encoding = "utf-8-sig")
  for fileName in list_file :
    print(fileName)
    if fileName[0] != '.' :
      open_file = open(directory + '/' + fileName, 'r', encoding = "utf-8-sig")
      for line in open_file.readlines()[:n_sentences] :
        sentence = []
        for word in line.split() :
          sentence.append(word.split("/")[0])
        f.write(str(sentence)+'\n')
  f.close()

#extract_create_corpus_path("EP.tcs.melt.utf8.c", 2, 100)
#Creates 200_lines_corpus.txt
