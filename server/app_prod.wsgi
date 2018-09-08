import sys

sys.path.insert(0, '/var/www/flower-classifier-web')

from .app import app as application

# TODO Configure
application.secret_key = 'CHANGEME!'
