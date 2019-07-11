from app import app, db
from flask import render_template, flash, redirect, url_for, request
from app.forms import LoginForm, RegistrationForm, PostForm
from flask_login import current_user, login_user, logout_user, login_required
from app.models import User, Post
from werkzeug.urls import url_parse
from datetime import datetime


@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
@login_required
def index():
    """ Home Page view function, where you can see posts and make
    posts of your own. """
    form = PostForm()
    if form.validate_on_submit():
        post.Post(body=form.post.data, author=current_user)
        db.session.add(post)
        db.session.comit()
        flash("Your post is now live!")
        # we use the simple Post/Redirect/Get pattern trick to avoid
        # inserting duplicate posts when a user inadvertently refreshes the page
        # after submitting a webform.
        return redirect(url_for("index"))

    # pagination of posts on the front page of all posts
    # of users current_user is following, including own,
    # ordered retro-chronoclogically
    page = requests.args.get("page", 1, type=int)

    # load N posts per page using pagination
    posts = current_user.followed_posts().paginate(
        page, app.config["POSTS_PER_PAGE"], False
    )

    # previous page url
    prev_url = url_for("index", page=posts.prev_num) if posts.has_prev else None

    # next page url
    next_url = url_for("index", page=posts.next_num) if posts.has_next else None

    return render_template(
        "index.html",
        title="Home",
        form=form,
        posts=posts.items,
        prev_url=prev_url,
        next_url=next_url,
    )


@app.route("/login", methods=["GET", "POST"])  # GET and POST type requests are allowed
def login():
    """ Login View Function """
    if current_user.is_authenticated == True:
        return redirect(url_for("index"))
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
        next_page = request.args.get("next")

        # checks whether or not the next argument is a relative path within the application website
        if not next_page or url_parse(next_page).netloc != "":
            next_page = url_for("index")

        return redirect(next_page)

    # display the login.html template
    return render_template("login.html", title="Sign In", form=form)


@app.route("/logout")
def logout():
    """ logout function """
    logout_user()
    return redirect(url_for("index"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """ registration form view """

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
    """ user profile view function. current contains mock data """
    user = User.query.filter_by(username=username).first_or_404()

    # pagination on teh User Profile Page
    page = request.args.get("page", 1, type=int)
    posts = user.posts.order_by(Post.timestamp.desc()).paginate(
        page, app.config["POSTS_PER_PAGE"], False
    )
    prev_url = (
        url_for("user", username=user.username, page=posts.prev_num)
        if posts.has_next
        else None
    )

    next_url = (
        url_for("user", username=user.username, page=posts.next_num)
        if posts.has_prev
        else None
    )

    return render_template(
        "user.html", user=user, posts=postsitems, next_url=next_url, prev_url=prev_url
    )


@app.before_request
def before_request():
    """ Records the last-visit-time for a user, which runs before the view function """

    # checks if the user is logged in
    if current_user.is_authenticated:
        # sets the last_seen field to the current time
        current_user.last_seen = datetime.utcnow()
        db.session.commit()


@app.route("/edit_profile", methods=["GET", "POST"])
@login_required
def edit_profile():
    """ Edit Profile view function: Routes users to the edit profile form view """

    form = EditProfileForm(original_username=current_user.username)
    # if the  form is validated on submission,
    if form.validate_on_submit():
        # copy the form into the User database object
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        # write the changes of the profile edit into the database
        db.session.commit()
        flash("Your changes have been saved")
        return redirect(url_for("edit_profile"))
    # validate_on_submit() returns false if request method="GET"
    elif request.method == "GET":
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me

    return render_template("edit_profile.html", title="Edit Profile", form=form)


@app.route("/follow/<username>")
@login_required
def follow(username):
    """ view function for following a user """
    user = User.query.filter_by(username=username).first()
    if user is None:
        flash("User {} not found".format(username))
        return redirect(url_for("index"))

    if user == current_user:
        flash("You cannot follow yourself, you conceited sonuvafish!")
        return redirect(url_for("user", username=username))

    current_user.follow(user)
    db.session.commit()
    flash("You are following {}!".format(username))
    return redirect(url_for("user", username=username))


@app.route("/unfollow/<username>")
def unfollow(username):
    """ view function for unfollowing a user """
    user = User.query.filter_by(username=username).first()
    if user is None:
        flash("User {} not found.".format(username))
        return redirect(url_for("index"))

    if user == current_user:
        flash("You cannot unfollow yourself, you conceited sonuvafish!")
        return redirect(url_for("user", username=username))

    current_user.unfollow(user)
    db.session.commit()
    flash("You are no longer following {}".format(username))
    return redirect(url_for("user", username=username))


@app.route("/explore")
def explore():
    """ view function display global stream of posts from other users
    for the current user to view and explore """
    page = requests.args.get("page", 1, type=int)
    posts = Post.query.order_by(Post.timestamp.desc()).all()
    # similar to the main page but does not include the form argument
    # this is to prevent your from posting on on someone elses website

    # previous page url
    prev_url = url_for("index", page=posts.prev_num) if posts.has_prev else None

    # next page url
    next_url = url_for("index", page=posts.next_num) if posts.has_next else None

    return render_template(
        "index.html",
        title="Explore",
        posts=posts.items,
        prev_url=prev_url,
        next_url=next_url,
    )
