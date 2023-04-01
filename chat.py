import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
import openai
from nltk.stem import WordNetLemmatizer 

from tensorflow.keras.models import load_model
bot_name = 'Serenity'
openai.api_key = "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" 
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    bag = [0]* len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] 

    results.sort(key= lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result



def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()


def sad_count(string,response):
    count = string.count('sad')
    if count == 3:
        return"i think you are sad."
    elif count == 5:
        return "Don't be so sad."

    count = string.count('depressed')
    if count == 3:
        return "I think you are depressed \nhere are some wys you can curb it \n1.Take soe deep breaths. \n2. Talk to your family and friends\n3. Visit a theraphist"
    elif count == 5:
        return "Here are the numbers of some psychatrist: \n1.Dr. Vijay Chinchole\n9876542364\n2.Dr. Gourav Trivedi\n9784627562\n3.Dr. Deepak Kelkar\n9761254324"
    else:
        return response

def gpt(msg):
    res = generate_response(msg)
    return res

print("Go! Bot is runnning")

def sere_res(msg):
    ints = predict_class(msg)
    if "search"in msg.lower():
        res = gpt(msg)
    elif "what"in msg.lower():
        res = gpt(msg)
    elif "where"in msg.lower():
        res = gpt(msg)
    elif "how"in msg.lower():
        res = gpt(msg)
    elif "when"in msg.lower():
        res = gpt(msg)
    elif "depressed"in msg.lower():
        res = gpt('depression management')
    else:
        res = get_response(ints, intents)
    res = sad_count(msg,res)
    return res
