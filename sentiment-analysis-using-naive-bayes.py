
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import colorama
from colorama import Fore
colorama.init(autoreset=True)


train_orig=pd.read_csv("train.csv")[:10000]
test_nolabel=pd.read_csv("test.csv")[:10]





import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import re
stop_words = set(stopwords.words('english'))

train = train_orig

def remove_stopwords(line):
    word_tokens = word_tokenize(line)
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    return " ".join(filtered_sentence)

def preprocess(line):
    line = line.lower()  #convert to lowercase
    line = re.sub(r'\d+', '', line)  #remove numbers
    line = line.translate(line.maketrans("","", string.punctuation))  #remove punctuation
    line = remove_stopwords(line)
    return line
for i,line in enumerate(train.tweet):
    train.tweet[i] = preprocess(line)





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train['tweet'], train['label'], test_size=0.5, stratify=train['label'])


# Let us balance the dataset
train_imbalanced = train
from sklearn.utils import resample
df_majority = train[train.label==0]
df_minority = train[train.label==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 

X_train, X_test, y_train, y_test = train_test_split(df_upsampled['tweet'], df_upsampled['label'], test_size=0.5, stratify=df_upsampled['label'])




from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()



# **Convert text data to numerical data**



from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
tf_train=vect.fit_transform(X_train)  #train the vectorizer, build the vocablury
tf_test=vect.transform(X_test)  #get same encodings on test data as of vocabulary built




tf_test_nolabel=vect.transform(test_nolabel.tweet)




model.fit(X=tf_train,y=y_train)




expected = y_test
predicted=model.predict(tf_test)




custom = ["Honesty is the best Policy", "Black Lives Matter", "I'm not feeling Well","i am not satisfied"]
test_custom=pd.DataFrame(custom)
tf_custom = vect.transform(test_custom[0])
ab = model.predict(tf_custom)
print('\n')
for i in range(len(custom)):
    if(ab[i]==1):
        print(f'{custom[i]}      ',end='')
        print(Fore.GREEN+'positive'.upper()+'\n\n')
    else:
        print(f'{custom[i]}      ',end='')
        print(Fore.RED+'negative'.upper()+'\n\n')






