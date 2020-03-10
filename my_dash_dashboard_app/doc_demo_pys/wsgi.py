<<<<<<< HEAD
from werkzeug.wsgi import DispatcherMiddleware

from dashplot import app as app1
from dashtable import app as app2
from dashscatter import app as app3

application = DispatcherMiddleware(flask_app, {
    '/app1': app1.server,
    '/app2': app2.server,
    '/app3', app3.server,
=======
from werkzeug.wsgi import DispatcherMiddleware

from dashplot import app as app1
from dashtable import app as app2
from dashscatter import app as app3

application = DispatcherMiddleware(flask_app, {
    '/app1': app1.server,
    '/app2': app2.server,
    '/app3', app3.server,
>>>>>>> origin/master
})