from flask_wtf import FlaskForm
from wtforms import TextAreaField, StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, EqualTo, Email, Length, ValidationError
from app.models import User


class LoginForm(FlaskForm):
    """ Form class users use to login to their account """

    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    remember_me = BooleanField("Remember Me")
    submit = SubmitField("Sign In")


class RegistrationForm(FlaskForm):
    """ Form class for registration of user account """

    username = StringField("Username", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    password2 = PasswordField(
        "Repeat Password", validators=[DataRequired(), EqualTo("password")]
    )
    submit = SubmitField("Register")

    def validate_username(self, username):
        """ if username already in database, raise error """
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError(
                "Username already exists. Please try a different username."
            )

    def validate_email(self, email):
        """ if email already in database, raise error 	"""
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError(
                "This email has already been registered. Please try another email address."
            )


class EditProfileForm(FlaskForm):
    """ Form view for users to edit their profiles """

    username = StringField("Username", validators=[DataRequired()])
    about_me = TextAreaField("About me", validators=[Length(min=0, max=140)])
    submit = SubmitField("Submit")

    def __init__(self, original_username, *args, **kwargs):
        super(EditProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username

    def validate_username(self, username):
        """ Perform validation for if the user leaves the original name untouched,
        then the validation should allow it, since that username is already assigned to that
        user."""
        if username.data != self.original_username:
            user = User.query.filter_by(username=self.username.data).first()
            if user is not None:
                raise ValidationError("Please use a different name.")


class PostForm(FlaskForm):
    """ Form view for users to type in new posts """
    post = TextAreaField("Say anything", validators=[
        DataRequired(), Length(min=1, max=140)])
    submit = SubmitField("Submit")

