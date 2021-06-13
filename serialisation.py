""" serialisation.py
Just the serialize and deserialize functions.

For details about json files, see https://www.codeflow.site/fr/article/python-json
"""

import json

def serialize(data, save_as):
    """ Serializes data in a json file saved on desktop.

    -> data: any, the object you want to serialize
    -> save_as: string, the path of the file you want to create, don't forget the
    .json extension
    """
    with open(save_as, "w+") as file:
        json.dump(data, file)


def deserialize(infile):
    """ Reads a json file and returns its contents.

    -> infile: string, path to the json file. Don't forget the ".json" extension!
    <- data: whatever
    """
    data = None
    with open(infile) as json_data:
        data = json.load(json_data)
    return data


if __name__ == "__main__":
    print("This file isn't meant to be launched on it's own...")