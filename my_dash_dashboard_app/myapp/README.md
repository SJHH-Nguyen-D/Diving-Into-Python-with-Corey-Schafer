# ABOUT

Flask project is framed as a module with the structure:

    /application.py
    /dash_app1.py
    /dash_app2.py
    /templates
        /hello.html
    /assets
        /static

instead of being a package with an __init__.py file.

## Running

You can run with this command:

    gunicorn wsgi:application

## Heroku Deployment

Heroku dash application name is: 

    https://sjhh-nguyen-d-dashapp.herokuapp.com/


## Why global variables will break your app
Dash is designed to work in multi-user environments where multiple people may view the application at the same time and will have independent sessions.

If your app uses modified global variables, then one user's session could set the variable to one value which would affect the next user's session.

Dash is also designed to be able to run with multiple python workers so that callbacks can be executed in parallel. This is commonly done with `gunicorn` using syntax like

    \$ gunicorn --workers 4 app:server

(app refers to a file named `app.py` and server refers to a variable in that file named server: `server = app.server`).

When Dash apps run across multiple workers, their memory is not shared. This means that if you modify a global variable in one callback, that modification will not be applied to the rest of the workers.
