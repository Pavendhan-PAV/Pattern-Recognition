import math
import re
import numpy as np

WORD = re.compile(r"\w+")


def get_cosine(vector1, vector2):
    numerator = np.dot(vector1, vector2)
    denominator = np.sqrt(np.dot(vector1, vector1) * np.dot(vector2, vector2))
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def gather_texts(text):
    words = WORD.findall(text)
    return words


def bagofwords(texts1, texts2):
    union = list(set(texts1) - set(texts2))
    for x in texts2:
        union.append(x)
    union = sorted(union)
    return union


def vectorize(texts1, texts2, union):
    vector1 = []
    vector2 = []
    for x in union:
        if x in texts1:
            vector1.append(1)
        else:
            vector1.append(0)
    for x in union:
        if x in texts2:
            vector2.append(1)
        else:
            vector2.append(0)
    return vector1, vector2


def main():
    text1 = "MATLAB is a program for solving engineering and mathematical problems. The basic MATLAB objects are vectors and matrices, so you must be familiar with these before making extensive use of this program."
    text2 = "MATLAB works with essentially one kind of object, a rectangular numerical matrix. Here is some basic information on using MATLAB matrix commands."

    texts1 = gather_texts(text1)
    texts2 = gather_texts(text2)

    # union of texts between texts1 and texts 2
    union = bagofwords(texts1, texts2)
    vector1, vector2 = vectorize(texts1, texts2, union)

    cosine = get_cosine(vector1, vector2)
    print("Cosine Similaryty:", cosine)
    # # print("Cosine Distance:",1-cosine)


if __name__ == "__main__":
    main()