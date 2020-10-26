import os, pickle

import flask
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename, redirect

from forms import FileInputForm, PredictionDataForm, TrainModelForm
from util_functions import *

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import time

model = load_model('static/Models/third -  loss 0.3476 accuracy0.8954 val_loss1.2306 val_accuracy0.61.h5')
maxlen = 40
# loading the tokenizer
with open('static/Tokenizer/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = Flask(__name__)
app.config['SECRET_KEY'] = "ERTYUIO"
app.config['UPLOAD_FOLDER'] = "static/File_Upload_Folder/"

GET = 'GET'
POST = "POST"


@app.route("/",  methods=[GET])
@app.route("/home",  methods=[GET])
def home():
    return render_template('home.html')


@app.route("/train",methods=[GET, POST])
def train():
    input_form  = FileInputForm()

    if input_form.validate_on_submit():
        singlefile()
        file = input_form.file.data
        if file.filename[-3:] != 'csv':
            print("ONLY UPLOAD A 'csv' FILE!")
            return redirect('train')
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],str(file.filename)))
        print("File Successfully Uploaded")
        file.close()

    
    trainModel_form = TrainModelForm()
    #if trainModel_form.validate_on_submit():
        


    return render_template("train.html", input_form=input_form, trainModel_form=trainModel_form)

@app.route('/train_model', methods=[POST])
def train_model():
    global model, tokenizer, maxlen

    file_name = os.listdir("static/File_Upload_Folder/")[0]
    path      = "static/File_Upload_Folder/" + file_name

    #Loading the data from csv file
    df       = load_data(path)
    labels   = df['labels'].to_numpy()
    features = df['sentences'].to_numpy()
    total_samples = len(df)

    #Cleaning stop words and converting to lists
    features = filter_func(features)

    #shuffling the data
    rand_features, rand_labels = shuffle(features, labels)

    #Imp numbers to create Embeddings and for padding
    maxlen, count = count_words(features)
    num_words     = len(count)
    maxlen        = maxlen - 20

    #Train Test Split
    ratio = 0.9
    mark  = int(total_samples*ratio)

    train = (rand_features[:mark], rand_labels[:mark])
    test  = (rand_features[mark:], rand_labels[mark:])

    #One hot encoding Labels
    train_labels = convert_labels(train[1])
    test_labels  = convert_labels(test[1])

    #Tokenizing the data
    #tokenizer = Tokenizer(3126)
    tokenizer.fit_on_texts(rand_features)

    # saving
    with open('static/Tokenizer/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    train_sequences = tokenizer.texts_to_sequences(train[0])
    test_sequences  = tokenizer.texts_to_sequences(test[0])

    train_padded = pad_sequences(train_sequences, maxlen=maxlen, padding='post', truncating='post')
    test_padded  = pad_sequences(test_sequences, maxlen=maxlen, padding='post', truncating='post')

    #Training the Model
    model.fit(train_padded, train_labels, epochs=30, validation_data=(test_padded, test_labels))

    #overwriting the model
    model.save('static/Models/new.h5')
    print("Model Trained and Saved!")

    return redirect('train')

def singlefile():
    list = os.listdir("static/File_Upload_Folder/")
    print(list)
    for i in list:
        os.remove("static/File_Upload_Folder/"+i)


@app.route("/test",  methods=[GET,POST])
def test():
    global model, tokenizer, maxlen

    form = PredictionDataForm()
    if form.validate_on_submit():
        text = form.text_area.data
        text = text.split('.')

        text_seq        = tokenizer.texts_to_sequences(text)
        text_seq_padded = pad_sequences(text_seq, maxlen=maxlen, padding='post', truncating='post')

        #print(text_seq_padded)
        prediction = model.predict(text_seq_padded)
        #print(prediction)

        return render_template("test.html", prediction_form=form, prediction=prediction)
    else:
        return render_template("test.html", prediction_form=form)

if __name__ == '__main__':
    app.run()
