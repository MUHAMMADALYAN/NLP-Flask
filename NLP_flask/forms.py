import os

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField, TextAreaField

class FileInputForm(FlaskForm):
	file   = FileField("Upload CSV file", validators=[FileRequired('File was Empty!')])
	submit = SubmitField("Upload")


class PredictionDataForm(FlaskForm):
	text_area = TextAreaField()
	submit 	  =  SubmitField("Classify")

class TrainModelForm(FlaskForm):
	train = SubmitField("Train Model")
