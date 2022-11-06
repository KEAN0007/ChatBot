#imports
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

#gets entire file 
with open("ChatBot/intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    #making lists
    words = []
    labels = []
    docs_1 = []
    docs_2 = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_1.append(wrds)
            docs_2.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])
    #make nince and clean            
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_1):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            #if word exists put 1 
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_2[x])] = 1

        training.append(bag)
        output.append(output_row)

        with open("data.pickle","wb") as f:
            pickle.dump((words, labels, training, output),f) 

    training = numpy.array(training)
    output = numpy.array(output)


tensorflow.compat.v1.reset_default_graph()
#Nural network
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
#use except if you add somthing new to intents (coould just add a x to try :) 
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=800, batch_size=8,show_metric=True)
    model.save("model.tflearn")


def bag_of_Words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for b in s_words:
        for i, w in enumerate(words):
            if w == b:
                bag[i] = (1)
    return numpy.array(bag)

def chating():
    print("Hey this is tavernbot what's up(type \"quit\" to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        result = model.predict([bag_of_Words(inp, words)])
        result_index = numpy.argmax(result)
        tag = labels[result_index]

        for sen in data["intents"]:
            if sen['tag']==tag:
                responses = sen['responses']
        
        print(random.choice(responses))

        
chating()