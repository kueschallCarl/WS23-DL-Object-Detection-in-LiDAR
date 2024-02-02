import sys
from datetime import datetime

class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Make sure the output is written immediately

    def flush(self):
        for f in self.files:
            f.flush()



