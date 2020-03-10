import dash
import dash_bootstrap_components as dbc
from flask import Flask, render_template, redirect

server = Flask(__name__)

@server.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html', title='Home')

@server.route('/about', methods=['POST', 'GET'])
def about():
    return render_template('about.html', title='About Me')

@server.route('/contact', methods=['POST', 'GET'])
def contact():
    return render_template('contact.html', title='Contact')

external_stylesheets = ["https://codepen.io/chriddyp/pen/dZVMbK.css"]
bootstrap_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, 
                server=server,
                external_stylesheets=external_stylesheets,
                routes_pathname_prefix='/')


app.index_string = '''
    <!DOCTYPE html>
    <html lang="en">

    <head>
        <title>{% block title %}{{title}}{% endblock %}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
    </head>
    <body>
        <nav class="navbar navbar-inverse">
            <div class="container-fluid">
                <div class="navbar-header">
                    <button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#myNavbar">
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                    </button>
                    <a class="navbar-brand" href="#">Logo</a>
                </div>
                <div class="collapse navbar-collapse" id="myNavbar">
                    <ul class="nav navbar-nav">
                        <li class="active"><a href="/">Home</a></li>
                        <li><a href="/apps/app1">Employee Form Submission</a></li>
                        <li><a href="/apps/app2">Dash Visualizations</a></li>
                        <li><a href="/about">About</a></li>
                        <li><a href="/contact">Contact</a></li>
                    </ul>
                    <ul class="nav navbar-nav navbar-right">
                        <li><a href="#"><span class="glyphicon glyphicon-log-in"></span> Login</a></li>
                    </ul>
                </div>
            </div>
        </nav>

        <div class="container-fluid text-center">
            <div class="row content">
                <div class="col-sm-2 sidenav">
                    <p><a href="#">Link</a></p>
                    <p><a href="#">Link</a></p>
                    <p><a href="#">Link</a></p>
                </div>
                <div class="col-sm-8 text-left">
                    {%app_entry%}
                </div>
                <div class="col-sm-2 sidenav">
                    <div class="well">
                        <p>ADS</p>
                    </div>
                    <div class="well">
                        <p>ADS</p>
                    </div>
                </div>
            </div>
        </div>

        <footer class="container-fluid text-center">
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>

    </body>

    </html>
'''

app.config.suppress_callback_exceptions = True

if __name__ == '__main__':
    app.run_server(debug=True)