# About

This repo goes through the flask mega tutorial, which can be followed along at: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world

As a general note of how pages are constructed, these are the main ingredients that go into making a site:
1. Base.html template
2. Sub-template HTML files for each of the pages
3. A routing script to direct users to where they want to navigate to
4. A models.py file that acts as the database script to store information about the user, wherein there is a user loader function, and each class represents a SQL or NoSQL database model that will store information about various things including information about the user, and another table about the posts made by the user.
5. Configuration script
6. A forms.py script, wherin each class is represents the different types of view functions that will collect the data that are filled out about the user, by the user, to be stored in the backend database.
