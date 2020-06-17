from data_preprocess.xml2txt import xml2txt
from data_preprocess.remove_non_Chinese import remove_non_Chinese
from data_preprocess.simplify_Chinese import simplify_Chinese
from data_preprocess.divide_sentences import divide_sentences

if __name__ == '__main__':
  xml2txt()
  remove_non_Chinese()
  simplify_Chinese()
  divide_sentences()
