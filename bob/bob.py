import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import math
import re
from textblob import TextBlob
import numpy as np 
import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pickle
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import colorama 
colorama.init()
from colorama import Fore, Style, Back

training_sentences = []
training_labels = []
labels = []
responses = []

bob='/content/drive/MyDrive/Colab Notebooks/bob/'

filepath=bob+'all.yml'
lines = open(filepath, 'rt', encoding='utf8').read().split('\n')
for line in lines:
    if(line.startswith('---')):
      temp=line.replace('--- ','')

    if line.startswith('- - '):
      line = line.replace('- - ', '')
      training_sentences.append(line)
      training_labels.append(temp)
        
      
    if line.startswith('  - '):
      line = line.replace('  - ', '')
      responses.append(line)
num_classes=(len(Counter(training_labels).keys()))
#print(len(training_sentences),len(training_labels))

with open(bob+'mod.pkl', 'rb') as file:
    pickle_model1 = pickle.load(file)
with open(bob+'covec.pkl','rb') as file:
  count1=pickle.load(file)
with open(bob+'label.pkl','rb') as file:
  lbl1=pickle.load(file)


cou=0
joy=0
sad=0
sh=0
dc=0
ang=0
fe=0
gu=0
#function to calculate emotion percenatge
def emotion_percentage(emo):
     global cou,joy,sad,sh,dc,ang,fe,gu
     cou+=1
     if emo=='joy':
          joy+=1
     elif emo=='sadness':
          sad+=1
     elif emo=='disgust':
          dc+=1
     elif emo=='guilt':
          gu+=1
     elif emo=='fear':
          fe+=1
     elif emo=='anger':
          ang+=1
     elif emo=='shame':
          sh+=1

def text_to_vector(text):
    WORD = re.compile(r"\w+")
    words = WORD.findall(text)
    return Counter(words)

#function to display emotion percenatges
def display_emopercentage():
      print("============= Result =============\n")
      print("Emotion        Percentage(%)")
      print("JOY            ",((joy-1)/(cou-1))*100,"\nSAD            ",(sad/(cou-1))*100,"\nSHAME          ",
            (sh/(cou-1))*100,"\nFEAR           ",(fe/(cou-1))*100,"\nGUILT          ",(gu/(cou-1))*100,"\nANGER          ",
            (ang/(cou-1))*100,"\nDISGUST        ",(dc/(cou-1))*100)
      #print("you need treatment asap!!!") #hii this is test
      #hi this test 2 
      

      #print(joy,sad,sh,dc,fe,ang,gu,cou-1)
#function to find similarity
def findsim(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator



def chat():
    
  
    # load trained model
    model = keras.models.load_model(bob+'chat_model.h5')

    # load tokenizer object
    with open(bob+'chat_tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open(bob+'chat_label.pkl', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User:    " + Style.RESET_ALL, end="")
        inp = input()
        sentence=[]
        sentence.append(inp)
        tweets=pd.DataFrame(sentence)
        # Doing some preprocessing on these tweets as done before
        tweets[0] = tweets[0].str.replace('[^\w\s]',' ')
        from nltk.corpus import stopwords
        stop = stopwords.words('english')
        tweets[0] = tweets[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        from textblob import Word
        tweets[0] = tweets[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        # Extracting Count Vectors feature from our tweets
        tweet_count = count1.transform(tweets[0])

        tweet_pred = pickle_model1.predict(tweet_count)
        #print("sentiment code : ",tweet_pred)

        emo=lbl1.inverse_transform(tweet_pred)
        emotion_percentage(emo) #calculating_emotion_percentages
        
        
        #print("emotion of sentence:",emo)
        if inp.lower() == "quit":
            display_emopercentage()
            
            break
        
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len,))
        

        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        #tag = np.argmax(result)
        #print(tag)
        #find all questions under this tag, find similarity btw user inp and these
        #questions, randomise the answers of most similar question
        sim=[] 
        strain=[]
        for i in range(len(training_labels)):
          if tag == training_labels[i]:
              #print(training_sentences[i])
              sim.append(findsim(training_sentences[i].lower(),inp.lower()))
              strain.append(training_sentences[i])
        #print(sim)
        pos=sim.index(max(sim))      
        ttext=strain[pos]
        #print(ttext)
        resp=[]
        for line in range(len(lines)): 
          if lines[line].startswith('- - '):
            if (lines[line].replace('- - ','')==ttext):
              line+=1
              while (lines[line].startswith('  - ')):
                resp.append(lines[line].replace('  - ',''))
                line+=1
              break;

        print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,np.random.choice(resp))
        
        
        
       
       
          
        
      

chat()


