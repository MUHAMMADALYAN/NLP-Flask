import os

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField, TextAreaField, SelectField, StringField
from wtforms.validators import DataRequired, Length

class FileInputForm(FlaskForm):
	file   = FileField("Upload CSV file", validators=[FileRequired('File was Empty!')])
	submit = SubmitField("Upload")


class PredictionDataForm(FlaskForm):
	text_area = TextAreaField()
	submit 	  =  SubmitField("Classify")

class TrainModelForm(FlaskForm):
	train = SubmitField("Train Model")

class ModelFromScratchForm(FlaskForm):
	restart = SubmitField("Start Model from Scratch")

class SelectCorrectClassForm(FlaskForm):
	dropdown = SelectField(
		u"Select Correct Labels", 
		choices=[
			(1, 'purpose'), 
			(2, 'craftsmaship'), 
			(3, 'aesthetic'), 
			(4, 'narative'), 
			(5, 'influence'), 
			(6, 'none')
	])

class ChangeClassColorsForm(FlaskForm):
	new_purpose 	 = StringField("Purpose:", validators=[DataRequired(), Length(min=7, max=7)])
	new_craftsmaship = StringField("Craftsmaship:", validators=[DataRequired(), Length(min=7, max=7)])
	new_aesthetic 	 = StringField("Aesthetic:", validators=[DataRequired(), Length(min=7, max=7)])
	new_none 	  	 = StringField("None:", validators=[DataRequired(), Length(min=7, max=7)])
	submit 		 	 = SubmitField("Set Colors")
