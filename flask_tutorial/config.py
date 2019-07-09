import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
	# encryption token for security reasons
    SECRET_KEY = os.environ.get("SECRET_KEY") or "you-will-never-guess"
    # set database engine name
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL"
    ) or "sqlite:///" + os.path.join(basedir, "app.db")
    # turn off an irrelevant database option
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # setup for email to send immediately after an error occurs
    MAIL_SERVER = os.environ.get("MAIL_SERVER")
    # by default the email server port is 25
    MAIL_PORT = int(os.environ.get("MAIL_PORT") or 25)
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS") is not None
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")
    # these list of admins will receive the error report
    ADMINS = ["dennisnguyendo@gmail.com"]