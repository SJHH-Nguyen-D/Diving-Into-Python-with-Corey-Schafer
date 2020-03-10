from flask import Flask, url_for, render_template, request
from markupsafe import escape, Markup
import requests
from werkzeug.utils import secure_filename

flask_app = Flask(__name__)


@flask_app.route("/")
def index():
    return render_template("index.html")


# @flask_app.route('/login', methods=['POST', 'GET'])
# def login():
#     error = None
#     if request.method == 'POST':
#         if valid_login(request.form['username'],
#                        request.form['password']):
#             return log_the_user_in(request.form['username'])
#         else:
#             error = 'Invalid username/password'
#     # the code below is executed if the request method
#     # was GET or the credentials were invalid
#     return render_template('login.html', error=error)


# def valid_login(username, password):
#     if (request.form.username == username) & (request.form.password == password):
#         return True

# @flask_app.route('/loggedin', methods=['POST', 'GET'])
# def log_the_user_in(username):
#     print("Login successful!")
#     return render_template('successful_login.html', username=username)


# @flask_app.route('/user/<username>')
# def profile(username):
#     return '{}\'s profile'.format(escape(username))

# @server.route('/static', methods=['GET', 'POST'])
# def static():
#     return None

# @flask_app.route('/hello')
# @flask_app.route('/hello/<name>')
# def hello(name=None):
#     return render_template('hello.html', name=name)

# @flask_app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == "POST":
#         f = request.files['the_file']
#         f.save('/var/www/uploads/' + secure_filename(f.filename))


@flask_app.route("/registration", methods=["GET", "POST"])
def register():
    return render_template("registration.html")


if __name__ == "__main__":
    app.run_server(debug=True)
