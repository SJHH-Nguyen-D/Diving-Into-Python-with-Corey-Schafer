from app import app
from flask import render_template, flash, redirect, url_for, request
from app.forms import LoginForm, RegistrationForm
from flask_login import current_user, login_user, logout_user, login_required
from app.models import User
from werkzeug.urls import url_parse


@app.route("/")
@app.route("/index")

@login_required
def index():
    # ...
    return render_template("index.html", title='Home Page', posts=posts)


@app.route("/login", methods=["GET", "POST"]) # GET and POST type requests are allowed
def login():
	''' user login function '''

    # Scenario 1: If logged in send to home page
	if current_user.is_authenticated:
        return redirect(url_for('index'))

	# Scenario 2: Check to see if the log-in process was successful
    form = LoginForm()

    if form.validate_on_submit():
    	# load the user from the database returns the user object if it exists if it doesn't, returns None
    	user = User.query.filter_by(username=form.username.data).first()
    	
    	# login logic if the password is wrong
    	if user is None or not user.check_password(form.password.data):
    		flash("Invalid username or password")
    		return redirect(url_for("login"))

    	# if the both the login information is correct we then register the user with the flask_login.login_user() function
    	# it registers the user as logged in so that future pages the user naviagtes to will have the current_user variable
    	# set to that logged_in user
    	login_user(user, remember=form.remember_me.data)
    	
    	# if trying to access a page that is protected by the login view, will be redirected to index. A query string "?next=/next_page" 
    	# will be added to the url so that it's "/login?next=index" The next query string argument is set to the original URL so the application
    	# can use that to redirect back after login After the login, the value of the next query string is obtained
    	next_page = requests.args.get("next")

    	# checks whether or not the next argument is a relative path within the application website
    	if not next_page or url_parse(next_page).netloc != '':
    		next_page = url_for("index")

    	return redirect(next_page)

    # display the login.html template
    return render_template("login.html", title="Sign In", form=form)


@app.route("/logout")
def logout():
	''' logout function '''
	logout_user()
	return redirect(url_for("index"))


@app.route("/register", methods=["GET", "POST"])
def register():
    ''' registration form view '''

    # check to see if the user is already logged in
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    # create and register a new user
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("Congratulations, you are now a registered user!")
        return redirect(url_for("login"))

    # display the registration.html template
    return render_template("register.html", title="Register", form=form)


@app.route("/user/<username>")
@login_required
def user(username):
    """ user profile view function """

    user = User.query.filter_by(username=username).first_or_404()
    
    # mock posts
    posts = [ \
    {"author": user, "body": "Test post #1"},
    {"author": user, "body": "Test post #2"}
    ]
    return render_template("user.html", user=user, posts=posts)