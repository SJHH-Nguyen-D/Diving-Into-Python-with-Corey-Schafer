from app import app
from flask import render_template

@app.route('/')
@app.route('/index')

def index():
	user = {"username": "Dennis"}
	posts = \
	[
		{
			"author": {"username": "John"},
			"body": "Beautiful day in Portland!"
		},
		{
			"author": {"username": "Susan"},
			"body": "The Avengers are hella cool!"
		},
		{
			"author": {"username": "Jerry"},
			"body": "WHO ARE ALL THESE PEOPLE???"
		},
		{
			"author": {"username": "Fahim"},
			"body": "OJ didn't do it."
		},
		{
			"author": {"username": "Ali"},
			"body": "Kansas City is on my shitlist."
		}
	]
	return render_template("index.html", title="Home", user=user, posts=posts)