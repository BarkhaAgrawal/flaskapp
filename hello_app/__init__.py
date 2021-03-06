from flask import Flask
from hello_app.config import TestingConfig, DevelopmentConfig, ProductionConfig
import os
# blueprints
from hello_app.errors.handlers import errors
from hello_app.home.routes import home


app = Flask(__name__)
app.config.from_object(DevelopmentConfig)

app.register_blueprint(errors)
app.register_blueprint(home)


def create_app():
    app = Flask(__name__)
    app.config.from_object(DevelopmentConfig if os.environ.get(
        "PRODUCTION").lower() == 'true' else DevelopmentConfig)

    from hello_app.errors.handlers import errors
    from hello_app.home.routes import home

    app.register_blueprint(errors)
    app.register_blueprint(home)

    return app
