from collections import Counter
import re
import pprint

nonWordChars = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~\s'

scriptPath=r'C:\Users\Brian Kardon\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Video VI\PyVAQ\Source\PyVAQ.py'

with open(scriptPath, 'r') as f:
    text = f.read()

words = Counter(re.split('[{chars}]'.format(chars=nonWordChars), text))

pprint.pprint(words)
