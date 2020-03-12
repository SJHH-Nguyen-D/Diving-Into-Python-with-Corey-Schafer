# Running the Testing

You can either run the tests in the command line with:

    `python3 -m unittest discover -s tests -v`

Or you can run the shells script that does it for you with:

    `bash runnit.sh`

You can also adjust the code to use *nose2* or pytests if you'd like.

# The Tools of the Trade

*Travis-CI* API allows for automatic testing, CI/CD and integration of other styling/formatting looks like *flake8*, *black*, and *bandit*. So long as you've hooked up your project repo to your Travis-CI account, everytime you push your project to your origin/master, Travis performs all of this for your automatically. You can configure some of the automated operations (linting, formatting, testing) that Travis-CI does for you once you set up your `.travis.yml` file.

*flake8* does style hinting or linting, *black* is a brutally ruthless formatter, and *bandit* is your security-mistakes and vulnerabilities checking app. 

You can test how your code runs in different environments (different version of python, etc) with *tox*.

There's also *pytest-benchmark*(which was not included in this, as I've chosen to use unittest for my test-runner) to run performance benchmarking for function calls for your code.


*Flake8* does linting for your code. You can configure some of the default settings for *flake8* in a `setup.cfg` file for specifications on the type of lint alerting that gets raised. It might look a little something like this:

    [flake8]
    ignore = E305
    exclude = .git,**pycache**
    max-line-length = 120

To run flake8 after setting the configurations:

    `flake8 test.py`

To run black:

    `black test.py`

To start working with tox, you can do:

    tox-quickstart

Tox generates a standard boilerplate config file in `tox.ini`. You can edit it to change your dependenies.

To run tox can be run simply with `tox` in the command line, but you test also test your code in specific environments with the `-e py36` (python3.6 for example), `-r` (to recreate your environment when there are environmental changes to your dependencies), `-q` (less verbose), and `-v` (more verbose) flags.

But by default you can just run tox as such:

    tox

To run bandit:

    `bandit -r my_sum`

To run pytest-benchmark (not included in this project):

    `pytestpytest_benching.py`
