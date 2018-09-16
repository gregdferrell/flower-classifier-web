#
# WSGI Entry Point
#
from app import app

if __name__ == "__main__":
	app.secret_key = 'TODO:CHANGE-SECRET-KEY!'
	app.run()
