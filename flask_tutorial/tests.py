"""
Unit tests that will run each time changes are made
to ensure that code is running smoothly in the future.
"""

from datetime import datetime, timedelta
import unittest
from app import app, db
from app.models import User, Post

class UserModelCase(unittest.TestCase):
	""" Unit Test Casen class that will test the functionality
	of our model classes and functions """
	@classmethod
	def setUpClass(cls):
		print("setUpClass")
	
	@classmethod
	def tearDownClass(cls):
		print("tearDownClass")

	def setUp(self):
		print("setUp")
		app.config["SQLALCHMEY_DATABASE_URI"] = "sqlite://"
		db.create_all()

		# create four user instances
		self.user_1 = User(
			username="john", 
			email="john@example.com"
			)
		self.user_2 = User(
			username="susan",
			email="susan@example.com"
			)
		self.user_3 = User(
			username="mary",
			email="mary@example.com"
			)
		self.user_4 = User(
			username="david",
			email="david@example.com"
			)
		self.followers = User.followers()	

	def tearDown(self):
		print("tearDown\n")
		db.session.remove()
		db.drop_all()

	def test_password_hashing(self):
		print("test_password_hashing")
		self.user_1.set_password("cat")
		self.assertFalse(self.user_1.check_password("dog"))
		self.assertTrue(self.user_1.check_password("cat"))

	def test_avatar(self):
		print("test_avatar")
		self.assertEqual(
			self.user_1.avatar(128),
			("https://www.gravatar.com/avatar/" \
				"d4c74594d841139328695756648b6bd6" \
				"?d=identicon&s=128")
			)

	def test_follow(self):
		print("test_follow")
		db.session.add(self.user_1)
		db.session.add(self.user_2)
		db.session.commit()
		self.assertEqual(self.user_1.followed.all(), [])
		self.assertEqual(self.user_1.followers.all(), [])

		# test if the following a user increases their follower count
		self.user_1.follow(self.user_2)
		db.session.commit()
		self.assertTrue(self.user_1.is_following(self.user_2))
		self.assertEqual(self.user_1.followed.count(), 1)
		self.assertEqual(self.user_1.followed.first().username, "susan")
		self.assertEqual(self.user_2.followers.count(), 1)
		self.assertEqual(self.user_2.followers.first().username, "john")

		# test if unfollowing a user decreases their follower count
		self.user_1.unfollow(self.user_2)
		db.session.commit()
		self.assertFalse(self.user_1.is_following(self.user_2))
		self.assertEqual(self.user_1.followed.count(), 0)
		self.assertEqual(self.user_2.followers.count(), 0)

	def test_follow_posts(self):
		print("test_follow_posts")
		db.session.add_all([self.user_1, self.user_2, self.user_3, self.user_4])

		# create four posts
		now = datetime.utcnow()
		p1 = Post(body="post from John", author=self.user_1,
			timestamp=now + timedelta(seconds=1))
		p2 = Post(body="post from Susan", author=self.user_2,
			timestamp=now + timedelta(seconds=4))
		p3 = Post(body="post from Mary", author=self.user_3,
			timestamp=now + timedelta(seconds=3))
		p4 = Post(body="post from David", author=self.user_4,
			timestamp=now + timedelta(seconds=2))
		db.session.add_all([p1, p2, p3, p4])
		db.session.commit()

		# setup the followers
		self.user_1.follow(self.user_2) # john follows susan
		self.user_1.follow(self.user_4) # john follows david
		self.user_2.follow(self.user_3) # susan follows mary
		self.user_3.follow(self.user_4) # mary follows david
		db.session.commit()

		# check the followed posts of each user
		f1 = self.user_1.followed_posts().all()
		f2 = self.user_2.followed_posts().all()
		f3 = self.user_3.followed_posts().all()
		f4 = self.user_4.followed_posts().all()
		self.assertEqual(f1, [p2, p4, p1])
		self.assertEqual(f2, [p2, p3])
		self.assertEqual(f3, [p3, p4])
		self.assertEqual(f4, [p4])


if __name__ == "__main__":
	unittest.main(verbosity=2)