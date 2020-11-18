from flask import Flask, render_template, request, url_for, flash,  send_from_directory, send_file
from flask import current_app as app
from werkzeug.utils import secure_filename, redirect
import json
import inspect, nltk
import numpy as np

from .forms import (
    FileInputForm, 
    PredictionDataForm, 
    TrainModelForm, 
    ChangeClassColorsForm, 
    #SubmitAllForm,
    special_form
)

from .util_functions import *

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.math as tfmath
import tensorflow.keras.backend as tfbackend


GET  = 'GET'
POST = 'POST'
TEST_STRING = ''

tokenizer 	 = load_tokenizer()
model     	 = load_model('application/static/Models/model_under_use.h5')
maxlen    	 = 40
class_colors = load_classColors()

@app.route("/",  methods=[GET])
@app.route("/home",  methods=[GET])
def home():
    return render_template('home.html')

@app.route("/changeClassColors",  methods=[GET, POST])
def change_class_colors():
	global class_colors

	form = ChangeClassColorsForm()
	if form.validate_on_submit():
		new_purpose		 = form.new_purpose.data
		new_craftsmaship = form.new_craftsmaship.data
		new_aesthetic 	 = form.new_aesthetic.data
		new_none		 = form.new_none.data

		save_classColors(new_purpose, new_craftsmaship, new_aesthetic, new_none)
		class_colors = load_classColors()

		flash("Class Colors Changed Successfully!", "success")
		return redirect(url_for("home"))

	return render_template("changeClassColors.html",  form=form, class_colors=class_colors)

@app.route("/train", methods=[GET, POST])
def train():
    inputform  = FileInputForm()

    if inputform.validate_on_submit():

        file = inputform.file.data
        if file.filename.split(".")[-1] != 'tsv':

            flash("ONLY UPLOAD A 'tsv' FILE!", "danger")
            return redirect(url_for('train'))

        singlefile(file)
        flash("File Successfully Uploaded", "success")
        file.close()

    
    trainModelform   = TrainModelForm()        

    return render_template("train.html", inputform=inputform, trainModelform=trainModelform)

@app.route('/train_model/<retrain>', methods=[GET, POST])
def train_model(retrain):
    global model, tokenizer, maxlen

    if retrain == 'True':
        file_path       = "application/bin/output2.tsv"
    else:
        file_path = "application/static/File_Upload_Folder/uploaded.tsv"

    try:
        data     = np.genfromtxt(file_path, delimiter='\t', dtype= str)
        labels   = data[:, 0]
        features = data[:, 1]



        if len(features) < 70:
            flash("There must be atleast 70 rows of Data before training", "danger")
            return "less than 70"


        total_samples = data.shape[0]

        #Cleaning stop words and converting to lists
        features = filter_func(features)

        #shuffling the data    
        features, labels = shuffle(features, labels)

        #Imp numbers to create Embeddings and for padding
        maxlen, count = count_words(features)
        num_words     = len(count)
        maxlen        = maxlen - 20

        #One hot encoding Labels
        labels = onehot_encode_labels(labels)

        #Tokenizing the data
        tokenizer.fit_on_texts(features)

        # saving the tokenizer
        save_tokenizer(tokenizer)

        feature_sequences = tokenizer.texts_to_sequences(features)
        feature_padded = pad_sequences(feature_sequences, maxlen=maxlen, padding='post', truncating='post')

        #Training the Model
        model.fit(feature_padded, labels, epochs=20)

        #overwriting the model
        model.save('application/static/Models/model_under_use.h5')

        flash("Model Trained and Saved!", "success")
        return "training done"

    except ValueError as ve:
        flash("ERROR, Plz check if all your sentences end with a period i.e ' . '",  "danger")
        print(ve)
        return "ValueError"

    except KeyError as ke:
        flash("ERROR, Encountered an Unknown Label during Training, Please Check Training Data",  "danger")
        print(ke)
        return "keyError"

    except OSError as oe:
        flash("ERROR! No File uploded to Train on", "danger")
        print(oe)
        return "osError"

    except IndexError as ie:
        flash("There must be atleast 70 rows of Data before training", "danger")
        return "indexError"

@app.route("/restrat_model", methods=[POST])
def restart_model():
	global model

	model = load_model("application/static/Models/scratch_model.h5")
	model.save("application/static/Models/model_under_use.h5")

	flash("Model Started Form Scratch Successful", "success")

	return redirect(url_for('train'))


@app.route("/test",  methods=[GET, POST])
def test():
    global TEST_STRING

    predictionForm = PredictionDataForm()

    if predictionForm.validate_on_submit():
            
           TEST_STRING = predictionForm.text_area.data
           return redirect(url_for("results"))
    
    return render_template("test.html", predictionForm=predictionForm)


@app.route("/results", methods=[GET, POST])
def results():

    global model, tokenizer, maxlen, TEST_STRING  

    sentences   = nltk.sent_tokenize(TEST_STRING)

    text_seq        = tokenizer.texts_to_sequences(sentences)
    text_seq_padded = pad_sequences(text_seq, maxlen=maxlen, padding='post', truncating='post')

    predictions = model.predict(text_seq_padded)
        
    class_num = tfmath.argmax(predictions, axis= 1)
    class_num = tfbackend.eval(class_num)
    labels    = decode_onehot_labels(class_num)

    specialForm = special_form(labels)
    selects     = [
        getattr(specialForm, f"special_{i}")
        for i in range(specialForm.n_attrs)
    ]
    
    data = list(zip(sentences, roundoff(predictions), labels, selects))
    bin_data = loadTSVfromBin()
    print("\n\nBIN LEN:", len(bin_data), '\n\n')

    if specialForm.validate_on_submit():
        corrected_labels = [
            sel.data
            for sel in selects
        ]

        appendTSVtoBin(corrected_labels, sentences)

        flash(f"Added { len(corrected_labels) } rows to the bin, Now total rows in bin are { len(bin_data)+len(corrected_labels) }", "success")
        return redirect(url_for("proceed"))
        

    return render_template(
        "results.html",
        data            = data,
        len_data        = len(data),
        bin_data        = bin_data,
        len_bin_data    = len(bin_data),
        class_colors    = class_colors,
        specialForm     = specialForm
    )


@app.route("/download_file", methods=[GET])
def download_file():
    path = "bin/output2.tsv"
    return send_file(path, as_attachment=True)

@app.route("/clear_bin", methods=[GET])
def clear_bin():

    clearBin()
    flash("Bin Emptied", "success")
    return redirect(url_for("home"))


@app.route("/proceed", methods=[GET])
def proceed():
    return render_template("proceed.html")


@app.route("/predict", methods=["GET","POST"])
def classified():
    text=request.json
    text=text['mytext']
    global model, tokenizer , maxlen
    sentences   = nltk.sent_tokenize(text) 
    text_seq        = tokenizer.texts_to_sequences(sentences)
    text_seq_padded = pad_sequences(text_seq, maxlen=maxlen,padding='post', truncating='post')
    predictions = model.predict(text_seq_padded)
    class_num = tfmath.argmax(predictions, axis= 1) #Returns the index with the largest value across axes of a tensor.
    class_num = tfbackend.eval(class_num) 
    labels    = decode_onehot_labels(class_num)

    dict={

    }
    dat = zip(sentences, labels)
    for i in dat:
        dict[i[0]]=i[1]
    return json.dumps(dict)
