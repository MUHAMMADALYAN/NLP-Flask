from flask import Flask

def create_app():
    """ Creating the Flask app and setting its config """

    #Creating the flask app
    app = Flask(__name__)

    #setting config variables from DevelopmentConfig class in config file
    app.config.from_object('config.Config')

    with app.app_context():

        #Incuding Routes
        from . import routes

        return app