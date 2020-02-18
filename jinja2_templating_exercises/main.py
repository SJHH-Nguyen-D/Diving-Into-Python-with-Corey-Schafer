from flask import Flask, render_template
import numpy as np 

app = Flask(__name__)

@app.route('/', methods=['POST', "GET"])
def index(placeholder="random_thing"):
	return render_template("index.html", placeholder=placeholder)


@app.route('/predictions', methods=['POST', "GET"])
def predictions_view(score=0):
	m = 5
	x = 2
	b = 3
	output = m*x+b
	score = output+10
	return render_template("predictions.html", score=score)


if __name__ == "__main__":
	app.run()