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
