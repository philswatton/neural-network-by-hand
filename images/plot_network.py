import os

from graphviz import Source

PATH = os.path.dirname(__file__)
FILE = os.path.join(PATH, "network-diagram")
s = Source.from_file(FILE, format="png")
s.view()
