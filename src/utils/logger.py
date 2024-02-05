import sys
from datetime import datetime

class Tee(object):
    """
    A class for redirecting the standard output to multiple files.

    This class allows you to simultaneously write the same output to multiple files.
    It is particularly useful for logging or capturing console output to multiple log files.

    Attributes:
    - files (list of file objects): A list of file objects where the output will be written.

    Methods:
    - write(obj): Writes the provided object to all specified files.
    - flush(): Flushes the output to all specified files to ensure immediate writing.

    """

    def __init__(self, *files):
        """
        Initialize a Tee instance with one or more file objects.

        Args:
            *files: One or more file objects where the output will be written.
        """
        self.files = files

    def write(self, obj):
        """
        Write the provided object to all specified files.

        Args:
            obj: The object (usually a string) to be written to the files.
        """
        for f in self.files:
            f.write(obj)
            f.flush()  # Make sure the output is written immediately

    def flush(self):
        """
        Flush the output to all specified files to ensure immediate writing.
        """
        for f in self.files:
            f.flush()
