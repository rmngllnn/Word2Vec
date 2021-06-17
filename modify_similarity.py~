import argparse

"""
Modify the file similarity.txt to get pairs with plurals or/and non plurals.
"""

def create_file(similarity_file) :
    """Opens the similarity file and create an ameliorated version of it"""
    s = open(similarity_file, 'r', encoding = "utf-8")
    f = open('similarity_new.txt', 'w', encoding = 'utf-8')
    for line in s.readlines() :
        w1, w2, sc = line.split(" ")
        plural_w1 = ""
        plural_w2 = ""
        if w1[-2:] == "au" or w1[-2:] == "ou" :
            plural_w1 = w1+'x'
        elif w1[-1:] == 's' :
            plural_w1 = w1
        else :
            plural_w1 = w1+'s'
        if w2[-2:] == "au" or w2[-2:] == "ou" :
            plural_w2 = w2+'x'
        elif w2[-1] == 's' :
            plural_w2 = w2
        else :
            plural_w2 = w2+'s'
        f.write(w1 + " " + w2 + " " + str(sc))
        f.write(w1 + " " + plural_w2 + " " + str(sc))
        f.write(plural_w1 + " " + w2 + " " + str(sc))
        f.write(plural_w1 + " " + plural_w2 + " " + str(sc))
    s.close()
    f.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('similarity', type=str)
  args = parser.parse_args()

  create_file(args.similarity)

  
