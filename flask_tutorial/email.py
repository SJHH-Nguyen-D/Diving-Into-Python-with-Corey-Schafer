from flask_mail import Message
from app import mail 

def send_email(subject, sender, recipients, text_body, html_body):
		""" 
		Flask-mail also supports Cc and BCc arguments.
		Make sure to check out the documentation for these features 
		"""
		msg = Message(subject, sender=sender, recipients=recipients)
		msg.body = text_body
		msg.html = html_body
		mail.send(msg)