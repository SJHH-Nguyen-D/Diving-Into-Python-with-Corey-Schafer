from app import db
from app import login
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from hashlib import md5

"""
Every time there is a change to the database models, it is important to run the database
migration commands at the command line with:
* flask db migrate -m "<changes> to <table name> table"
* flask db upgrade

This only works when all the scripts are in running order.
"""


@login.user_loader
def load_user(id):
    return User.query.get(int(id))


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    posts = db.relationship("Post", backref="author", lazy="dynamic")
    about_me = db.Column(db.String(140))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    followers = db.Table(
        "followers",
        db.Column("follower_id", db.Integer, db.ForeignKey("user.id")),
        db.Column("followed_id", db.Integer, db.ForeignKey("user.id")),
    )
    followed = db.relationship(
        "User",
        secondary=followers,
        primaryjoin=(followers.c.follower_id == id),
        secondaryjoin=(followers.c.followed_id == id),
        backref=db.backref("followers", lazy="dynamic"),
        lazy="dynamic",
    )

    def __repr__(self):
        return "<User {}>".format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def avatar(self, size):
        """ generate avatar for user profile """

        # the md5 email hashing functions reads bytes instead of string,
        # and expects lower so we apply lower() and encode("utf-8") first
        digest = md5(self.email.lower().encode("utf-8")).hexdigest()
        # the ?s= string value argument dictates the size of the image
        return "https://www.gravatar.com/avatar/{}?d=identicon&s={}".format(
            digest, size
        )

    def follow(self, user):
        """ follow another user if not already following """
        if not self.is_following(user):
            self.followed.append(user)

    def unfollow(self, user):
        """ unfollow a user if already following """
        if self.is_following(user):
            self.followed.remove(user)

    def is_following(self, user):
        """ a method that helps with following-type functions and queries.
        It issues a query on the followed relationship to check if a link 
        between two users already exists. Here we have the quert terminator 
        .count(). Before we have also seen .all() and .first()
        """
        return self.followed.filter(followers.c.followed_id == user.id).count() > 0

    def followed_posts(self):
        """ query to get all posts from all followed users and order them
        retro-chronocologically """
        followed = Post.query.join(
            followers, (followers.c.followed_id == Post.user_id)).filter(
            followers.c.follower_id == self.id)
        # add own posts to the list of posts populating the user's front page
        own = Post.query.filter_by(user_id=self.id)
        # return sorted list of posts
        return followed.union(own).order_by(Post.timestamp.desc())


class Post(db.Model):
    """ Posts table """
    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.String(140))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))

    def __repr__(self):
        return "<Post {}>".format(self.body)
