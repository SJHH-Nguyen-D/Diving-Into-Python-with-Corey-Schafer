from app import app
from flask import render_template, flash, redirect, url_for
from app.forms import LoginForm
from flask_login import current_user, login_user, logout_user, login_required
from app.models import User


@app.route("/")
@app.route("/index")
# flask-login protects a view function against anonymous users
# with a decorator called @login_required
# when you add this decorator to a viewfunction blow the @app.route decorators from Flask,
# the function becomes protected and will not allow access to useres that are not authenticated.
@login_required
def index():
	''' Home page view function '''
    user = {'username': 'Miguel'}
    posts = [
        {
            'author': {'username': 'John'},
            'body': 'Beautiful day in Portland!'
        },
        {
            'author': {'username': 'Susan'},
            'body': 'The Avengers movie was so cool!'
        }
    ]
    return render_template('index.html', title='Home', user=user, posts=posts)


@app.route("/login", methods=["GET", "POST"])
def login():
	''' user login function '''

	# checks to see if the user is logged in and
	# sends the user to the index page
	if current_user.is_authenticated:
		return redirect(url_for("index"))

	# instantiate LoginForm object
    form = LoginForm()

    if form.validate_on_submit():

    	# load the user from the database
    	# returns the user object if it exists
    	# if it doesn't, returns None
    	user = User.query.filter_by(username=form.username.data).first()
    	
    	# login logic
    	# if the password is wrong
    	if user is None or not user.check_password(form.password.data):
    		flash("Invalid username or password")
    		return redirect(url_for("login"))

    	# if the both the login information is correct
    	# we then register the user with the flask_login.login_user() function
    	# it registers the user as logged in
    	# so that future pages the user naviagtes to will have the current_user variable
    	# set to that logged_in user
    	login_user(user, remember=form.remember_me.data)
    	return redirect(url_for("index"))

    return render_template("login.html", title="Sign In", form=form)


@app.route("/logout")
def logout():
	''' logout function '''
	logout_user()
	return redirect(url_for("index"))