from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch

import unicodedata
import string
def findFiles(path): return glob.glob(path)

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)+1#plus EOS marker
# print(all_letters)
# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor  
import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def onehot(n_sample,index):
    x=torch.zeros((1,n_sample))
    x[0][index]=1
    return x
# One-hot vector for category

def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    return category, line, onehot(n_categories,category_tensor), line_ToTensor(line)

def target(line):
    res=[]
    for x in line:
        res.append(letterToIndex(x))
    res.append(n_letters-1)
    return torch.tensor(res)

def categoryFromOutput(output):
    # output must be at shape (1,n_categories)
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def line_ToTensor(line):
    #convert target line to tensor,included EOS at the end,a line with 5 words have a shape(1,6,n_letters)
    tensor = torch.zeros(len(line)+1, 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    tensor[len(line)][0][n_letters-1]= 1
    return tensor

def wordFromOutput(output):
    # output must be at shape (1,n_letters)
    top_n, top_i = output.topk(1)
    word_i = top_i[0].item()
    try:
        return all_letters[word_i], word_i
    except:
        return "EOS",word_i


if __name__=="__main__":
    print(unicodeToAscii('Ślusàrski'))
    print(findFiles('D:/visual studio/my_python/RNN/data/names/*.txt'))
    print(category_lines['Italian'][:5])
    print(letterToTensor('J'))
    print(lineToTensor('Jones').size())
    for i in range(10):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        print(category,category_tensor)
        print(line,line_tensor)
        print(target("asdf"))
    