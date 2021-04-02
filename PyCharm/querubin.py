#!/usr/bin/env python
#-*- coding: utf-8 -*-
###############################################################################
# Exaer
# Author: Spacial
# License: GPL v3 or superior
# Input:
#   [required] -i file with input text (default: not applicable)
#   [optional] -l logfile (default: <empty> - no logfile)
#   [optional] -m maskfile (default: <empty> - no wordcloud mask)
#   [optional] -b bgcolor (default: white)
#   [optional] -d debuglevel (default: 30 - warning)
# Output:
##
###############################################################################
###############################################################################
# Changelog:
# v0.1 (16/03/2021) - Initial version
###############################################################################
"""TP is a Text Processor"""

__AUTHOR__ = "Spacial"
__version__ = "0.1"
__copyright__ = "Copyright 2021 - " + __AUTHOR__
__credits__ = [__AUTHOR__]
__maintainer__ = __AUTHOR__
__email__ = "spacial@gmail.com"
__status__ = "build 1"
__BUILD__ = "2021-03-16"
__LICENSE__ = "GPLv3 or superior"
__program__ = "tptp"
__deion__ = "TP is a Text Processor"


# pipenv install matplotlib nltk spacy pandas numpy wordcloud
# pipenv install pillow gensim==3.8.1
# python -m spacy download en_core_web_sm
# nltk.download('wordnet')
import logging
import argparse
import sys
import os
#import re
import string
import spacy
#import nltk
import pickle
import faulthandler
import gc
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer  # , sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
# from spacy.tokenizer import Tokenizer
from collections import Counter
from PIL import Image
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# Funcoes #####
def initParser():
    parser = argparse.ArgumentParser(deion=__program__,
                                     epilog='All output files will be ' +
                                     '(filename) item.[extension]')
    # 0: no log, 50: critical, 40: error, 30: warning*, 20: info, 10: debug
    parser.add_argument('-d', '--debug', nargs='?', dest='debug',
                        default='30', help='Enable more logging')
    parser.add_argument('-i', '--infile', nargs='?', dest='fileTxt',
                        default='',
                        help='base text file for Text Analysis.')
    parser.add_argument('-l', '--log', nargs='?', dest='fileLog',
                        default='', help='log file, duh.')
    parser.add_argument('-m', '--mask', nargs='?', dest='fileMask',
                        default='mask.png',
                        help='if setted, will use file mask for wordcloud.')
    parser.add_argument('-w', '--wordcloud', nargs='?', dest='fileWC',
                        default='',
                        help='if setted, will use for wordcloud output.')
    parser.add_argument('-t', '--top', nargs='?', dest='top',
                        default='14', help='How much of top on graphics ' +
                        '(words, trigrams, etc..) (default: 14).')
    parser.add_argument('--version', action='version',
                        version='%(prog)s ' + str(__version__))
    args = parser.parse_args()
    if '/' in args.fileTxt:
        args.radical = args.fileTxt.split('.')[0].split('/')[1]
    else:
        args.radical = args.fileTxt.split('.')[0]
    if args.fileTxt == '':
        print("Falta arquivo de texto a analisar.")
        parser.print_help()
        sys.exit(1)
    if args.fileLog == '':
        args.fileLog = args.radical + '.log'
    if args.fileWC == '':
        args.fileWC = args.radical + '_wc.png'
    return args


def setLog(level, fileLog):
    # ##Loglevels:
    # CRITICAL - 50
    # ERROR - 40
    # WARNING - 30
    # INFO - 20
    # DEBUG - 10
    # NOTSET - 0
    fmt = '%(asctime)-20s:%(name)s:%(levelname)-8s= %(message)s'
    df = '%d/%m/%Y %H:%M:%S'
    logging.basicConfig(filename=fileLog, format=fmt, datefmt=df,
                        level=int(level))
    # if (int(level) == 10):
    #     # print('Debug level: 10 (debug)')
    #     logging.basicConfig(filename=fileLog, format=fmt, datefmt=df,
    #                         level=logging.DEBUG)
    # elif (int(level) == 20):
    #     # print('Debug level: 20 (info)')
    #     logging.basicConfig(filename=fileLog, format=fmt, datefmt=df,
    #                         level=logging.INFO)
    # elif (int(level) == 30):
    #     # print('Debug level: 30 (warning)')
    #     logging.basicConfig(filename=fileLog, format=fmt, datefmt=df,
    #                         level=logging.WARNING)
    # elif (int(level) == 40):
    #     # print('Debug level: 40 (error)')
    #     logging.basicConfig(filename=fileLog, format=fmt, datefmt=df,
    #                                                     level=logging.ERROR)
    # elif (int(level) == 50):
    #     # print('Debug level: 50 (critical)')
    #     logging.basicConfig(filename=fileLog, format=fmt, datefmt=df,
    #                                                     level=logging.CRITICAL)
    # else:
    #     # print('Debug level: 0 (not set)')
    #     logging.basicConfig(filename=fileLog, format=fmt, datefmt=df,
    #                                                     level=logging.NOTSET)
    logging.warning('------  >' + __deion__ + '<  ---------')
    return logging.getLogger(__name__)


def loadFile(filename, logger):
    ret = list()
    try:
        fi = open(filename, "r")
        logger.debug('>>>>> Lendo arquivo com texto: %s', filename)
        ret = fi.readlines()
        fi.close()
        # ret = LineSentence(filename)
    except IOError:
        logger.error(' Não consegui abrir o arquivo: %s', filename)
        logger.error(' Erro! ', exc_info=True)
        sys.exit(116)  # define ESTALE  116     /* Stale file handle */
    return ret


def saveTXT(text, logger, txtfname='postprocessed.txt'):
    logger.debug("Salvando Texto [%s] no arquivo: %s", len(text), txtfname)
    with open(txtfname, 'wb') as tf:
        tf.write(text)
        tf.close()
    return


def loadTXT(logger, txtfname='postprocessed.txt'):
    logger.debug("Abrindo Texto do arquivo: %s", txtfname)
    text = ''
    with open(txtfname, 'rb') as tf:
        text = tf.readlines()
        tf.close()
    return text


def saveNLPState(nlp, doc, logger, nlpfname='nlp.pk', docfname='doc.pk'):
    logger.debug("Salvando NLP: ")
    pickle.dump(nlp.to_bytes(), open(nlpfname, 'wb'))
    logger.debug("Salvando doc: ")
    pickle.dump(doc.to_bytes(), open(docfname, 'wb'))
    return


def loadNLPState(logger, nlpfname='nlp.pk', docfname='doc.pk'):
    nlp = spacy.blank('en')
    logger.debug("Abrindo NLP: %s", nlpfname)
    nlp.from_bytes(pickle.load(open(nlpfname, 'rb')))
    logger.debug("Abrindo doc: %s", docfname)
    doc = spacy.tokens.Doc(nlp.vocab).from_bytes(pickle.load(open(docfname,
                                                                  'rb')))
    logger.debug("tipos carregados (doc, nlp): %s, %s", type(doc), type(nlp))
    return nlp, doc


def cleanEmpty(aListofLines, logger, blank=True):
    return [line for line in aListofLines if line != '\n']


def sliceText(text, nlp_ml, logger):
    init = 0
    slicedText = []
    slices = []
    for t in range(nlp_ml-1, len(text), nlp_ml-1):
        logger.debug("Processando: [%s, %s]", init, t)
        slicedText.append(text[init:t])
        slices.append((init, t))
        init = t
    slicedText.append(text[init:len(text)])
    slices.append((init, len(text)))
    logger.debug("Fatias: %s", slices)
    return slicedText, slices


def preprocess(sentence, lang='portuguese', filter=True, tokened=False):
    if isinstance(sentence, str):
        sentence_raw = sentence.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        sentence = "".join([i for i in sentence_raw if i not in
                            string.punctuation])
        tokens = tokenizer.tokenize(sentence)
    elif isinstance(sentence, list):
        print("preprocess got list!!!")
        tokens = sentence
    else:
        tokens = list(sentence)
    if filter:
        filtered_words = [w for w in tokens if w not in stopwords.words(lang)]
        tokens = filtered_words
    if tokened:
        return tokens
    else:
        return " ".join(tokens)


def text2Doc(text, nlp, logger):
    logger.debug("Carregando texto no nlp pra entidade doc.")
    if len(text) >= nlp.max_length:
        logger.debug("nlp.max_length maior que %s, fatiando..: %s,",
                     nlp.max_length, len(text))
        slicedText, _ = sliceText(text, nlp.max_length, logger)
        docs = []
        for d in slicedText:
            logger.debug("Nlp no d: %s", len(d))
            docs.append(nlp(d))
        logger.debug("Docs processados: %s", len(docs))
        logger.debug("Consolidando docs to tipo: %s", type(docs[0]))
        doc = spacy.tokens.Doc.from_docs(docs)
        logger.debug("Consolidado doc: %s", type(doc))
    else:
        logger.debug("Texto menor : %s", len(text))
        doc = nlp(text)
    return doc


# small = sm, medium = md, large = lg
def setup(fileTxt, logger, model='en_core_web_lg', clean=False):
    # for accuracy: en_core_web_trf
    logger.debug("carregando modelo: %s", model)
    nlp = spacy.load(model)
    # nao consigo setar o max lenght
    # nlp.max_lenght=1500000
    logger.debug("carregando texto de : %s", fileTxt)
    textAsList = loadFile(fileTxt, logger)
    logger.debug("Transformando lista em str.. " +
                 "(tamanho em caracteres, limite do spacy): ")
    text_raw = str(" ".join(textAsList))
    # logger.debug(" Aumentanto a capacidade do spacy
    # de processar textos grandes..[E088]")
    # nlp.max_lentgh= len(text)
    logger.debug("(%s, %s)", len(text_raw), nlp.max_length)
    if clean:
        logger.debug("limpando o texto (preprocessing):")
        text = preprocess(text_raw, lang='english')
    else:
        text = text_raw
        logger.debug("Sem limpar o texto (não invoquei preprocessing):")
    logger.debug("Tamanho pos processar: %s", len(text))
    doc = text2Doc(text, nlp, logger)
    return doc, nlp


def getCounts(doc, logger):
    logger.debug("Contando palavras e frases.")
    sentIdx = max([sent_i for sent_i, sent in enumerate(doc.sents)])
    wordIdx = max([token.i for sent_i,
                  sent in enumerate(doc.sents) for token in sent])
    logger.debug("Total de frases: %s", sentIdx)
    logger.debug("Total de palavras: %s", wordIdx)
    return sentIdx, wordIdx


def countWords(doc, logger, filter=True):
    logger.debug("Contando palavras.")
    words = [token.text.strip() for sent in doc.sents
             for token in sent if not token.is_stop and
             not token.is_punct and
             token.text.strip() != '']
    # and token.text.lower().strip() not in stopwords.words('english') ])
    freq = Counter(words)
    logger.debug("Total de palavras contadas: %s", len(freq))
    return freq, words


def autolabel(rects, ax, logger):
    """Attach a text label above each bar in *rects*,
    displaying its height."""
    logger.debug("gerando os autolabels.")
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    return


def plotpng(filename, data, graphlabel, logger):
    # fig = plt.figure(figsize = (10,4))
    fig, ax = plt.subplots(figsize=(10, 10))
    labels = []
    freqs = []
    for d in data:
        labels.append(d[0])
        freqs.append(d[1])
    x = np.arange(len(labels))
    width = 0.35
    rects = ax.bar(x, freqs, width)

    ax.set_ylabel('Frequencia')
    ax.set_title(graphlabel)
    ax.set_xticks(x)
    plt.xticks(rotation=45)
    ax.set_xticklabels(labels)
    #
    autolabel(rects, ax, logger)
    #
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    return


def grams(n, lista):
    if type(lista) == str:
        lista = lista.lower().split()
    ret = []
    for ng in ngrams(lista, n):
        ret.append(ng)
    return ret


def plot_wordcloud(text, logger, max_words=2000, max_font_size=64,
                   title_size=40,
                   figure_size=(24.0, 16.0),
                   title=None, bgc='white',
                   maskfn="mask4.png", filename='cloud.png'):
    stopw = set(STOPWORDS)
    more_stopwords = set(stopwords.words('english'))
    stopw = stopw.union(more_stopwords)
    plt.figure(figsize=figure_size)
    faulthandler.enable()
    gc.collect()
    logger.debug(text, max_words, max_font_size, title_size, figure_size,
                 title, bgc, maskfn, filename)
    if maskfn is None:
        wordcloud = WordCloud(background_color=bgc,
                              stopwords=stopw,
                              max_words=max_words,
                              max_font_size=max_font_size,
                              random_state=42)
        wordcloud.generate(text)
        plt.imshow(wordcloud)
        plt.title(title, fontdict={'size': title_size, 'color': 'black',
                  'verticalalignment': 'bottom'})
    else:
        d = os.getcwd()
        # convert -background white bard.svg mask4.png
        mask = np.array(Image.open(d + '/' + maskfn))
        image_colors = ImageColorGenerator(mask)
        wordcloud = WordCloud(background_color=bgc,
                              stopwords=stopw,
                              max_words=max_words,
                              max_font_size=max_font_size,
                              random_state=42,
                              mask=mask)
        try:
            wordcloud.generate(text)
        except TypeError:
            wordcloud.generate(" ".join(text))
        image_colors = ImageColorGenerator(mask)
        plt.imshow(wordcloud.recolor(color_func=image_colors),
                   interpolation="bilinear")
        plt.title(title, fontdict={'size': title_size,
                  'verticalalignment': 'bottom'})
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    print("file saved:", filename)
    return


def getPlaces(doc, logger):
    places = []
    classes = []
    for token in doc:
        logger.debug("> %s,%s, %s, %s", token.text.strip(), token.pos_,
                     token.tag_, token.ent_type_)
        if token.pos_ == 'NOUN':
            logger.debug(">> NOUN %s", token.text.strip())
            classes.append(('NOUN', token.text))
        elif token.pos_ == 'VERB':
            logger.debug(">> VERB %s", token.text.strip())
            classes.append(('VERB', token.text))
        if token.ent_type_ == 'GPE':
            places.append(token.text)
            logger.debug(">> %s, %s, %s, %s", token.text.strip(),
                         token.pos_, token.dep_, token.ent_type_)
    return Counter(places), set(places), Counter(classes), [x[0] for x in
                                                            classes]


def checkpoint(doc, nlp, logger):
    saveNLPState(nlp, doc, logger)
    doc2, nlp2 = loadNLPState(logger)
    print("doc:", doc == doc2)
    print("nlp:", nlp == nlp2)
    return


def getRatio(nounverb, logger):
    summa = Counter(nounverb)
    noun = summa['NOUN']
    verb = summa['VERB']
    ratio = round(((noun * 100) / verb), 2)
    return ratio


def getSummKW(doc, logge, words=None):
    # stem_text = [w.lemma_ for w in doc]
    # key_words = keywords(" ".join(stem_text), ratio=0.05,
    #                      pos_filter=('NN'), words=10)
    if words is None:
        stem_text = [w.lemma_ for w in doc]
        words = [w for w in doc]
    else:
        wnl = WordNetLemmatizer()
        stem_text = [wnl.lemmatize(p) for p in words]
    try:
        key_wordslemma = keywords(" ".join(stem_text), ratio=0.05,
                                  pos_filter=('NN'), words=10)
    except IndexError as e:
        print(e)
        print(len(stem_text))
        sys.exit(1)
    if (len(doc.text) * 0.05) > 2500:
        try:
            sum_words = summarize(doc.text, ratio=0.05)
            cWds = len(sum_words)
        except IndexError as e:
            print(e)
            print(len(stem_text))
            sys.exit(1)
    else:
        sum_words = summarize(doc.text, word_count=2500)
        cWds = 2500
    try:
        key_words = keywords(" ".join(stem_text), ratio=0.05,
                             pos_filter=('NN'), words=cWds)
    except IndexError as e:
        # print(key_words)
        print(stem_text)
        print(cWds)
        print(e)
        sys.exit(1)
    return sum_words, key_words, key_wordslemma


def genWC(text, filename, maxwords, doc, maxl, logger, maskname=None):
    try:
        if maskname is None:
            plot_wordcloud(text, logger, max_words=maxwords, filename=filename)
        else:
            plot_wordcloud(text, logger, max_words=maxwords,
                           maskfn=maskname, filename=filename)
    except:
        print("erro ao gerar o wordcloud")
        return False
    return


#####
def main():
    args = initParser()

    logger = setLog(args.debug, args.fileLog)
    top = int(args.top)
    wcfilename = args.fileWC
    maskname = args.fileMask
    doc, nlp = setup(args.fileTxt, logger)

    sentIdx, wordIdx = getCounts(doc, logger)
    print("1. Contagem de sentenças: ", sentIdx)
    print("1b. Contagem de palavras: ", wordIdx)
    print("2. Tamanho do vocabulário: ", len(doc.vocab))

    freq, words = countWords(doc, logger)
    print("3. Frequencia de palavras relevantes " +
          "(com gráfico de colunas ou barras): ", freq.most_common(top))
    # arrumar nome de arquivo, etc.. o o n=30 (pode ser maior)
    plotpng(args.radical + '_top' + str(top) + 'palavras.png',
            freq.most_common(top), str(top) + " palavras", logger)
    # foi definido pegar os trigramas sem stop-words ()
    trigrams = Counter(grams(3, words))
    print("4. Trigramas relevantes (com gráfico de colunas ou barras): ",
          trigrams.most_common(top))
    plotpng(args.radical + '_top' + str(top) + 'trigramas',
            trigrams.most_common(top), str(top) +
            " Trigramas que mais aparecem", logger)
    print("5. Quais locais (entidades da classe LOCAL) sao citados no" +
          "texto processado: ")
    countedPlaces, Places, _, nounverb = getPlaces(doc, logger)
    print(Places)
    print("6. Quantas vezes cada local é citado")
    print(countedPlaces)
    print("7. Qual ẽ a proporção de pronomes frente aos verbos do texto")
    ratio = getRatio(nounverb, logger)
    print("A proporção de pronomes em relação aos verbos é de:", ratio, "%")
    print("8. Nuvem de Palavras")
    sum_words, keywords, kw_lemma = getSummKW(doc, logger, words=words)
    genWC(sum_words, wcfilename, len(sum_words), doc,
          nlp.max_length, logger, maskname=maskname)
    print("9. Obtenha um resumo dos textos utilizados, " +
          "acompanhados das palavras-chave")
    print("9a. Resumo:", len(sum_words))
    print("9b. palavras-chave:", keywords[:10])
    return


if __name__ == '__main__':
    main()
