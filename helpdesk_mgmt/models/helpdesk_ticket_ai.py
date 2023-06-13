import random

# Usde to for Contextualisation and Other NLP Tasks.
import nltk

# Used in Tensorflow Model
import numpy as np
import tensorflow as tf
import tflearn

from odoo import models

nltk.download("punkt")
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

# Other
import json
import pickle
import warnings

warnings.filterwarnings("ignore")


class HelpdeskTicketAi(models.TransientModel):
    _name = "helpdesk.ticket.ai"
    _description = "Helpdesk Ticket AI"

    def train_ai(self):
        words = []
        classes = []
        documents = []

        print(
            "Looping through the Intents to Convert them to words, classes, documents and ignore_words......."
        )
        for ticket in self.env["helpdesk.ticket"].search([]):
            for name in ticket.mapped(ticket.name):
                # tokenize each word in the sentence
                w = nltk.word_tokenize(pattern)
                # add to our words list
                words.extend(w)
                # add to documents in our corpus
                documents.append((w, ticket.category_id.name))
                # add to our classes list
                if ticket.category_id.name not in classes:
                    classes.append(ticket.category_id.name)

        print("Creating the Data for our Model.....")
        print("Creating an List (Empty) for Output.....")
        output_empty = [0] * len(classes)

        train_x = []
        train_y = []
        print("Creating Training Set, Bag of Words for our Model....")
        for doc in documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            train_x.append(bag)
            train_y.append(output_row)

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        print("input shape")
        print(train_x.shape)
        print("Building Neural Network for Out Chatbot to be Contextual....")
        print("Resetting graph data....")
        tf.compat.v1.reset_default_graph()

        # net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.input_data(shape=[None, len(words)])
        net = tflearn.fully_connected(net, 3)
        net = tflearn.fully_connected(net, 3)
        net = tflearn.fully_connected(net, 3, activation="softmax")
        net = tflearn.regression(net)
        print("Training....")

        model = tflearn.DNN(net, tensorboard_dir="tflearn_logs")

        print("Training the Model.......")
        model.fit(train_x, train_y, n_epoch=1000, batch_size=9, show_metric=True)
        print("Saving the Model.......")
        model.save("model.tflearn")
        print("Pickle is also Saved..........")
        pickle.dump(
            {
                "words": words,
                "classes": classes,
                "train_x": train_x,
                "train_y": train_y,
            },
            open("training_data", "wb"),
        )
        print("Loading Pickle.....")
        data = pickle.load(open("training_data", "rb"))
        words = data["words"]
        classes = data["classes"]
        train_x = data["train_x"]
        train_y = data["train_y"]

        with open("intents.json") as json_data:
            json.load(json_data)

        print("Loading the Model......")
        # load our saved model
        model.load("./model.tflearn")

    def clean_up_sentence(sentence):
        # It Tokenize or Break it into the constituents parts of Sentense.
        sentence_words = nltk.word_tokenize(sentence)
        # Stemming means to find the root of the word.
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        # print(sentence_words)
        return sentence_words

    # Return the Array of Bag of Words: True or False and 0 or 1 for each word of bag that exists in the Sentence
    def bow(sentence, words, show_details=True):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w.lower() == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        np_bag = np.array(bag)
        reshaped_input = np_bag.reshape((1, -1))
        return reshaped_input

    ERROR_THRESHOLD = 0.001

    def classify(sentence):
        # Prediction or To Get the Posibility or Probability from the Model
        x_bow = bow(sentence, words, show_details=True)
        print(x_bow)
        results = model.predict(x_bow)[0]
        # Exclude those results which are Below Threshold
        results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
        # Sorting is Done because higher Confidence Answer comes first.
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            # print(r)
            return_list.append((classes[r[0]], r[1]))  # Tuppl -> Intent and Probability
        return return_list

    def response(self, ticket):
        results = classify(ticket.name)
        # That Means if Classification is Done then Find the Matching Tag.
        # print(results)
        if results:
            # Long Loop to get the Result.
            while results:
                for i in intents["intents"]:
                    # Tag Finding
                    if i["tag"] == results[0][0]:
                        # Random Response from High Order Probabilities
                        return print(random.choice(i["responses"]))
                results.pop(0)
        return "Sorry, I did not understand"
