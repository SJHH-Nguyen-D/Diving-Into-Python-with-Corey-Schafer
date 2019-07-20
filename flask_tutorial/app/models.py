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
    followed = de.relationship(
        "User", secondary=followers,
        primaryjoin=(followers.c.folloewr_id == id),
        secondaryjoin=(followers.c.followed_id == id),
        backref=db.backref("followers", lazy="dynamic"),
        lazy="dynamic"
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


class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.String(140))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))

    def __repr__(self):
        return "<Post {}>".format(self.body)


# manifesting the followers self-referential many-to-many relationship is 
# a association relationship table. The FKs in this table are both pointing at entries in the user table,
# since it is linking users to users
followers = db.Table(
    "followers",
    db.Column("following_id", db.Integer, db.ForeignKey("user.id")),
    db.Column("followed_id", db.Integer, db.ForeignKey("user.id"))
    )

