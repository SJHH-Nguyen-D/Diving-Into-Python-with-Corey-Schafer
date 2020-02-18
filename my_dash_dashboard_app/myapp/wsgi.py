from werkzeug.wsgi import DispatcherMiddleware

from dash_app1 import app as app1
from dash_app2 import app as app2
from flask_app import flask_app

application = DispatcherMiddleware(flask_app, {
    '/app1': app1.server,
    '/app2': app2.server,
}) 
