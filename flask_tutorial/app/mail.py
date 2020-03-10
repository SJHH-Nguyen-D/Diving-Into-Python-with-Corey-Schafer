from flask_mail import Message
from app import mail, app
from threading import Thread

def send_async_email(app, msg):
	""" we want our email to be sent asynchronously 
	to free up our send_email() process, meaning we don't
	have to wait for our email to finish sending before our 
	app works again because this waiting process slows it down
	We use the threading module for this purpose.
	We send the email as a background process
	we need for the mail.send() method to know the configuration 
	values for the email server, and you need to know about 
	the application before hand for this.s
	the application context that is created with the with app.app_context()
	call makes the application instance accessible via the current_app variable
	from Flask.
	"""
	with app.app_context():
		mail.send(msg)


def send_email(subject, sender, recipients, text_body, html_body):
		""" 
		Flask-mail also supports Cc and BCc arguments.
		Make sure to check out the documentation for these features 
		"""
		msg = Message(subject, sender=sender, recipients=recipients)
		msg.body = text_body
		msg.html = html_body
		Thead(target=send_async_email, args=(app, msg)).start()

def send_password_reset_email(user):
	""" sends token for a password reset request """
	token = user.get_reset_password_token()
	send_email(
		subject="[Microblog] Reset Your Password",
		sender=app.config["ADMINS"][0],
		recipients=[user.email],
		text_body=render_template("email/reset_password.txt", user=user, token=token),
		html_body=render_template("email/reset_password.html", user=user, token=token)
		)
		