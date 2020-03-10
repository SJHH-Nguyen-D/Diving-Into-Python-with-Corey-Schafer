<<<<<<< HEAD
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("/index.html")

if __name__ == "__main__":
=======
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("/index.html")

if __name__ == "__main__":
>>>>>>> origin/master
    app.run(debug=True)