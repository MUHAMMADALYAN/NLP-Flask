from flask import Flask, render_template, request, url_for, flash
from flask import current_app as app
from werkzeug.utils import secure_filename, redirect

from .forms import FileInputForm, PredictionDataForm, TrainModelForm, ModelFromScratchForm,  SelectCorrectClassForm, ChangeClassColorsForm
from .util_functions import *

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.math as tfmath
import tensorflow.keras.backend as tfbackend


GET  = 'GET'
POST = 'POST'

'''classes = [
    'purpose', 
    'craftsmaship', 
    'aesthetic', 
    'narative', 
    'influence', 
    'none',
]'''

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

@app.route("/train",methods=[GET, POST])
def train():
    inputform  = FileInputForm()

    if inputform.validate_on_submit():
        singlefile()
        file = inputform.file.data
        if file.filename.split(".")[-1] != 'tsv':
            flash("ONLY UPLOAD A 'tsv' FILE!", "danger")
            return redirect(url_for('train'))

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], str(file.filename)))
        flash("File Successfully Uploaded", "success")
        file.close()

    
    trainModelform   = TrainModelForm()        
    restratModelform = ModelFromScratchForm()

    return render_template("train.html", inputform=inputform, trainModelform=trainModelform, restratModelform=restratModelform)

@app.route('/train_model', methods=[POST])
def train_model():
    global model, tokenizer, maxlen

    file_name = os.listdir("application/static/File_Upload_Folder/")[0]
    path      = "application/static/File_Upload_Folder/" + file_name

    #Loading the data from csv file
    #df       = load_data(path)
    data     = np.genfromtxt(path, delimiter='\t', dtype= str)
    labels   = data[:, 0] #df['labels'].to_numpy()
    features = data[:, 1] #df['sentences'].to_numpy()

    total_samples = data.shape[0]

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

    train_x, train_y = (rand_features[:mark], rand_labels[:mark])
    test_x, test_y   = (rand_features[mark:], rand_labels[mark:])

    #One hot encoding Labels
    train_labels = onehot_encode_labels(train_y)
    test_labels  = onehot_encode_labels(test_y)

    #Tokenizing the data
    #tokenizer = Tokenizer(3126)
    tokenizer.fit_on_texts(rand_features)

    # saving the tokenizer
    save_tokenizer(tokenizer)
    #with open('application/static/Pickles/tokenizer.pickle', 'wb') as handle:
    #    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    train_sequences = tokenizer.texts_to_sequences(train_x)
    test_sequences  = tokenizer.texts_to_sequences(test_x)

    train_padded = pad_sequences(train_sequences, maxlen=maxlen, padding='post', truncating='post')
    test_padded  = pad_sequences(test_sequences,  maxlen=maxlen, padding='post', truncating='post')

    #Training the Model
    model.fit(train_padded, train_labels, epochs=30, validation_data=(test_padded, test_labels))

    #overwriting the model
    model.save('application/static/Models/model_under_use.h5')
    #print("Model Trained and Saved!")

    flash("Model Trained and Saved!", "success")
    return redirect('train')


@app.route("/restrat_model", methods=[POST])
def restart_model():
	global model

	model = load_model("application/static/Models/scratch_model.h5")
	model.save("application/static/Models/model_under_use.h5")

	flash("Model Started Form Scratch Successful", "success")

	return redirect(url_for('train'))



'''def test():
    global model, tokenizer, maxlen

    predictionForm = PredictionDataForm()
    if predictionForm.validate_on_submit():
        text      = predictionForm.text_area.data
        sentences = text.split('.')

        text_seq        = tokenizer.texts_to_sequences(sentences)
        text_seq_padded = pad_sequences(text_seq, maxlen=maxlen, padding='post', truncating='post')

        predictions = model.predict(text_seq_padded)
        
        class_num = tfmath.argmax(predictions, axis= 1) #[ tfmath.argmax(prediction) for prediction in predictions]
        class_num = tfbackend.eval(class_num)          #[ tfbackend.eval(i) for i in class_num ]
        class_num = decode_onehot_labels(class_num)
        
        data = list(zip(sentences, predictions, class_num))

        correctClassForm = SelectCorrectClassForm()

        return render_template("test.html", predictionForm=predictionForm, data=data, correctClassForm=correctClassForm, class_colors=class_colors)
    
    return render_template("test.html", predictionForm=predictionForm)'''

@app.route("/test",  methods=[GET,POST])
def test():

    global model, tokenizer, maxlen

    predictionForm = PredictionDataForm()
    if predictionForm.validate_on_submit():
        
        sentences           = predictionForm.text_area.data.split('. ')
        correctClassForms   = [ SelectCorrectClassForm() for _ in sentences ]

        text_seq        = tokenizer.texts_to_sequences(sentences)
        text_seq_padded = pad_sequences(text_seq, maxlen=maxlen, padding='post', truncating='post')

        predictions = model.predict(text_seq_padded)
        
        class_num = tfmath.argmax(predictions, axis= 1) #[ tfmath.argmax(prediction) for prediction in predictions]
        class_num = tfbackend.eval(class_num)          #[ tfbackend.eval(i) for i in class_num ]
        labels    = decode_onehot_labels(class_num)

        for form, clss in zip(correctClassForms, class_arr[class_num]): 
            form.dropdown.default = clss
            form.dropdown.process(clss)
        
        data = list(zip(sentences, predictions, labels, correctClassForms))

        return render_template(
            "test.html", 
            predictionForm = predictionForm, 
            data           = data, 
            class_colors   = class_colors,
            #correctClassForms = correctClassForms
        )
    
    return render_template("test.html", predictionForm=predictionForm)