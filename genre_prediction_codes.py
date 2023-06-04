# loading and installing required libraries
!pip install langdetect
from collections import defaultdict
import cufflinks as cf
import numpy as np
import pandas as pd
from langdetect import detect
import matplotlib.pyplot as plt
import string
import re
valid_chars = string.ascii_letters+string.digits+' '
    %matplotlib inline
plt.close('all')
import plotly.offline as pyoff
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
pyoff.init_notebook_mode()
cf.go_offline()

#mounting google drive onto colab (not necessary)
from google.colab import drive
drive.mount('/content/drive')

#bringing in data #please re-specify your file directory
bookdata_path = '/content/drive/MyDrive/Colab Notebooks/book_data.csv'
testdata_path = '/content/drive/MyDrive/Colab Notebooks/test_data.csv'
book = pd.read_csv(bookdata_path , engine = 'python' , encoding='utf-8' , error_bad_lines=False)
test = pd.read_csv(testdata_path , engine = 'python' , encoding='utf-8' , error_bad_lines=False)
book.head()

#genre counts per book
def genre_count(x):
    try:
        return len(x.split('|'))
    except:
        return 0
book['genre_count'] = book['genres'].map(lambda x: genre_count(x))
test['genre_count'] = test['genres'].map(lambda x: genre_count(x))
book.head()

#genre distribution
%matplotlib inline
plt.close('all')
plot_data = [
    go.Histogram(
        x=book['genre_count']
    )
]
plot_layout = go.Layout(
        title='Genre distribution',
        yaxis= {'title': "Frequency"},
        xaxis= {'title': "Number of Genres"}
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show(renderer='colab') 
pyoff.iplot(fig)

#new columns containing genre infos
def genre_listing(x):
    try:
        lst = [genre for genre in x.split("|")]
        return lst
    except: 
        return []

book['genre_list'] = book['genres'].map(lambda x: genre_listing(x))
test['genre_list'] = test['genres'].map(lambda x: genre_listing(x))

genre_dict = defaultdict(int)
for idx in book.index:
    g = book.at[idx, 'genre_list']
    if type(g) == list:
        for genre in g:
            genre_dict[genre] += 1
genre_dict

#number of kinds of genres
len(genre_dict)

#new dataframe containing the genres and their counts
genre_pd = pd.DataFrame.from_records(sorted(genre_dict.items(), key=lambda x:x[1], reverse=True), 
                                     columns=['genre', 'count'])
genre_pd[:100].head()

#view all genre distribution
plot_data = [
    go.Bar(
        x=genre_pd['genre'],
        y=genre_pd['count']
    )
]
plot_layout = go.Layout(
        title='Distribution for all Genres',
        yaxis= {'title': "Count"},
        xaxis= {'title': "Genre"}
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show(renderer='colab') 
pyoff.iplot(fig)

#view top 30 genre distribution
plot_data = [
    go.Bar(
        x=genre_pd[:30]['genre'],
        y=genre_pd[:30]['count']
    )
]
plot_layout = go.Layout(
        title='Top 30 genres',
        yaxis= {'title': "Count"},
        xaxis= {'title': "Genre"}
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show(renderer='colab') 
pyoff.iplot(fig)

#fiction or nonfiction
def determine_fiction(x):
    lower_list = [genre.lower() for genre in x]
    if 'fiction' in lower_list:
        return 'fiction'
    elif 'nonfiction' in lower_list:
        return 'nonfiction'
    else:
        return 'others'
book['label'] = book['genre_list'].apply(determine_fiction)
test['label'] = test['genre_list'].apply(determine_fiction)

#remove all languages except English
def remove_invalid_lang(df):
    invalid_desc_idxs=[]
    for i in df.index:
        try:
            a=detect(df.at[i,'book_desc'])
        except:
            invalid_desc_idxs.append(i)
    
    df=df.drop(index=invalid_desc_idxs)
    return df
book = remove_invalid_lang(book)
test = remove_invalid_lang(test)
test['lang']=test['book_desc'].map(lambda desc: detect(desc))
book['lang']=book['book_desc'].map(lambda desc: detect(desc))
book.head()

#bring in ISO language codes
lang_lookup = pd.read_html('https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes')[1]
langpd = lang_lookup[['ISO language name','639-1']]
langpd.columns = ['language','iso']
langpd

#filter out non-English books in the dataframe
def desc_lang(x):
    if x in list(langpd['iso']):
        return langpd[langpd['iso'] == x]['language'].values[0]
    else:
        return 'nil'
book['language'] = book['lang'].apply(desc_lang)
book.head()
test['language'] = test['lang'].apply(desc_lang)

#show distribution for languages
plot_data = [
    go.Histogram(
        x=book['language']
    )
]
plot_layout = go.Layout(
        title='Distribution for languages',
        yaxis= {'title': "Count"},
        xaxis= {'title': "Language"}
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show(renderer='colab')
pyoff.iplot(fig)

#show non-English books and their distribution
nonen_books = book[book['language']!='English']
plot_data = [
    go.Histogram(
        x=nonen_books['language']
    )
]
plot_layout = go.Layout(
        title='Distribution for non English books',
        yaxis= {'title': "Count"},
        xaxis= {'title': "Language"}
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show(renderer='colab') 
pyoff.iplot(fig)

#cleaning texts
test = test[test['language']=='English']
en_books = book[book['language']=='English']

#English books only
en_books.to_csv('/checkpoint.csv')
test.to_csv('/testcheckpoint.csv')
en_book_path = '/checkpoint.csv'
test_check_path = '/testcheckpoint.csv'

en_books = pd.read_csv(en_book_path)
en_books.head()

#tidying the text
def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = text.replace('(ap)', '')
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)
    text = re.sub('[^a-zA-Z ?!]+', '', text)
    text = _removeNonAscii(text)
    text = text.strip()
    return text

def cleaner(df):
   
    df = df[df['label'] != 'others']

    df = df[df['language'] != 'nil']

    df['clean_desc'] = df['book_desc'].apply(clean_text)

    return df

clean_book = cleaner(en_books)
clean_test = cleaner(test)

#preview the newly cleaned text data
clean_book['desc_len'] = [len(i.split()) for i in clean_book.clean_desc]
clean_book.head()

#see the length of book descriptions
plot_data = [
    go.Histogram(
        x=clean_book['desc_len']
    )
]
plot_layout = go.Layout(
        title='Distribution of description length',
        yaxis= {'title': "Length"},
        xaxis= {'title': "Descriptions"}
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show(renderer='colab') 
pyoff.iplot(fig)

#deciding the optimised lengths for book description
len_df_bins=clean_book.desc_len.value_counts(bins=100, normalize=True).reset_index().sort_values(by=['index'])
len_df_bins['cumulative']=len_df_bins.desc_len.cumsum()
len_df_bins['index']=len_df_bins['index'].astype('str')
len_df_bins.iplot(kind='bar', x='index', y='cumulative')
fig.show(renderer='colab') 
pyoff.iplot(fig)

min_desc_length=5
max_desc_length=300

clean_book=clean_book[(clean_book.clean_desc.str.split().apply(len)>min_desc_length)].reset_index(drop=True)
clean_test=clean_test[(clean_test.clean_desc.str.split().apply(len)>min_desc_length)].reset_index(drop=True)
clean_book.head()

#unique words from all descriptions
vocabulary=set()
def add_to_vocab(df, vocabulary):
    for i in df.clean_desc:
        for word in i.split():
            vocabulary.add(word)
    return vocabulary

vocabulary=add_to_vocab(clean_book, vocabulary)

vocab_dict={word: token+1 for token, word in enumerate(list(vocabulary))}

#tokenizer function
token_dict={token+1: word for token, word in enumerate(list(vocabulary))}

assert token_dict[1]==token_dict[vocab_dict[token_dict[1]]]

def tokenizer(desc, vocab_dict, max_desc_length):
    a=[vocab_dict[i] if i in vocab_dict else 0 for i in desc.split()]
    b=[0] * max_desc_length
    if len(a)<max_desc_length:
        return np.asarray(b[:max_desc_length-len(a)]+a).squeeze()
    else:
        return np.asarray(a[:max_desc_length]).squeeze()

len(vocabulary)

#tokenizing the descriptions
clean_test['desc_tokens']=clean_test['clean_desc'].apply(tokenizer, args=(vocab_dict, max_desc_length))
clean_book['desc_tokens']=clean_book['clean_desc'].apply(tokenizer, args=(vocab_dict, max_desc_length))
clean_book.head()

#training the book model
clean_book.label.value_counts()

#stratified random sampling
def stratified_split(df, target, val_percent=0.2):
    classes=list(df[target].unique())
    train_idxs, val_idxs = [], []
    for c in classes:
        idx=list(df[df[target]==c].index)
        np.random.shuffle(idx)
        val_size=int(len(idx)*val_percent)
        val_idxs+=idx[:val_size]
        train_idxs+=idx[val_size:]
    return train_idxs, val_idxs

_, sample_idxs = stratified_split(clean_book, 'label', 0.1)

train_idxs, val_idxs = stratified_split(clean_book, 'label', val_percent=0.2)
sample_train_idxs, sample_val_idxs = stratified_split(clean_book[clean_book.index.isin(sample_idxs)], 'label', val_percent=0.2)

def test_stratified(df, col):
    classes=list(df[col].unique())
    
    for c in classes:
        print(f'Proportion of records with {c}: {len(df[df[col]==c])*1./len(df):0.2} ({len(df[df[col]==c])} / {len(df)})')
    print("and")

test_stratified(clean_book, 'label')
test_stratified(clean_book[clean_book.index.isin(train_idxs)], 'label')
test_stratified(clean_book[clean_book.index.isin(val_idxs)], 'label')
test_stratified(clean_book[clean_book.index.isin(sample_train_idxs)], 'label')
test_stratified(clean_book[clean_book.index.isin(sample_val_idxs)], 'label')

classes=list(clean_book.label.unique())
classes
sampling=False
x_train=np.stack(clean_book[clean_book.index.isin(sample_train_idxs if sampling else train_idxs)]['desc_tokens'])
y_train=clean_book[clean_book.index.isin(sample_train_idxs if sampling else train_idxs)]['label'].apply(lambda x:classes.index(x))

x_val=np.stack(clean_book[clean_book.index.isin(sample_val_idxs if sampling else val_idxs)]['desc_tokens'])
y_val=clean_book[clean_book.index.isin(sample_val_idxs if sampling else val_idxs)]['label'].apply(lambda x:classes.index(x))

x_test=np.stack(clean_test['desc_tokens'])
y_test=clean_test['label'].apply(lambda x:classes.index(x))

#model building on Keras API
model = Sequential()
model.add(Embedding(len(vocabulary)+1, output_dim=250, input_length=max_desc_length))

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#Model 1 ; play around with epochs, eval_batch_size, and dropout
parameters = {'vocab': vocabulary,
              'eval_batch_size': 30,
              'c': 256,
              'epochs': 6,
              'dropout': 0.2,
              'optimizer': 'Adam',
              'loss': 'binary_crossentropy',
              'activation':'sigmoid'}

def bookLSTM(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(Embedding(len(params['vocab'])+1, output_dim=x_train.shape[1], input_length=x_train.shape[1]))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(200))
    model.add(Dense(1, activation=params['activation']))
    model.compile(loss=params['loss'],
              optimizer=params['optimizer'],
              metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, 
          y_train,
          validation_data=(x_val, y_val),
          batch_size=params['batch_size'], 
          epochs=params['epochs'])
    results = model.evaluate(x_test, y_test, batch_size=params['eval_batch_size'])
    return model

BookMode1 = bookLSTM(x_train, y_train, x_val, y_val, parameters)

#genre prediction function
def reviewBook(model,text):
    labels = ['fiction', 'nonfiction']
    a = clean_text(description)
    a = tokenizer(a, vocab_dict, max_desc_length)
    a = np.reshape(a, (1,max_desc_length))
    output = model.predict(a, batch_size=1)
    score = (output>0.5)*1
    pred = score.item()
    return labels[pred]

description = 'put your book description here'

reviewBook(BookMode1,description)
#it should give you a prediction: either fiction or nonfiction
