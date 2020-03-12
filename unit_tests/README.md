# About

This portion of the repo is devoted to learning, exploring and understanding some of the fundamentals of unittesting. I've used a few resources from YouTube, PyPI, and Real Python, so some of the code is my own implementation and experimentation.

## What is Testing and Some Ideas About Why to do Testing

We do testing so that our code doesn't break down when unexpected things happen (e.g., taking an unexpected input, some function breaking and affecting another function).

Tests, as I've learned, can be divided into either:

    1) Unit tests (testing your code piecemeal at a time for test cases to make sure that each function works as intended and doesn't break from unexpected inputs)

    2) Integration tests (testing permutations of the components of your code and how they interact with one another to mitigate or prevent side effects and technical debt. Done through the lense of a consumer or user)

But in essence each test REPL (read-evaluate-print-loop) can be broken down into:

    1) Creating inputs for the function you're testing 

    2) Execute the actual methods/function/code that you want to test and capture it's output

    3) The output of the function that you want to test is then tested against your expected mocked input.

If it walks like an error and quacks like an error, then it might just be an error. You have to define what an error even walks and quacks like!

![quackquack](https://i.ytimg.com/vi/28vRib_5Zo0/maxresdefault.jpg
)

## Some Terminology


*Side effects*

These are the unintended effects and consequences of your code that might leak and effect the inputs and functions outside of the code that you are testing.

*Fixtures*

Remember how we needed to create expected inputs for our test to test our functions against? Well, when you run a ton of common tests and you have common test inputs, you want to have a template that you can use over and over again (DRY), so you can create and use cookie-cutter inputs called *fixtures* to test against. Fixtures can be anything from databases, classes, computations, images, etc.

*Parameterization*

When you are testing the same piece of code or function against multiple different inputs, you are said to be *parameterizing* or your testing was *parameterized*.

*DRY*

Not quite terminology specific to testing, but you want to keep your testing, and your code DRY in general. That is, *D*on't *R*epeat *Y*ourself. Refactor. Don't do too much copy and pasting.

## Testing Process

There are some useful testing frameworks in Python and these are some of the more well-known ones: *unittest*, *pytest*, and *nose2*. Each comes with their own *test-runner*, which is an application that detects which parts of your script are tests, and runs them upon detection.

We'll go with the builtin unittest library for simplicity sake (and because of its ubiquitous use in many open source projects).

To test with `unittest`, you'd have to make a subclass of the `unittest.TestCase` class.

    class TestMyApp(unittest.TestCase):
         def test_output_int_instance(self):
            data = (1, 2, 3)
            result = sum(data)
            self.assertIsIntance(result, int)

This class tests your functions using specific test-base methods that you have created for it for each of your different test-cases. This test code should be in a file called `test.py` or if you have multiple tests, they should be in a tests folder. Each test method should be named with the `test_{whatevernameyouwant}` prefix, so that the test-runner recognizes it to run. For example, if you wanted to see if your function can accept an input of a list of integers, you might want to create your method as such:

    def test_list_integers(self):
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6, "This should have evaluated to 6")

## Running the Testing

You can either run the tests in the command line with:

    `python3 -m unittest discover -s tests -v`

Or you can run the shells script that does it for you with:

    `bash runnit.sh`

You can also adjust the code to use *nose2* or pytests if you'd like.

## The Tools of the Trade

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
