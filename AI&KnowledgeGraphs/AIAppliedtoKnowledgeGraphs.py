import pickle
import time
import string
import os
import re
import numpy as np
import logging
import traceback
import html
import matplotlib.pyplot as plt
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from py2neo import Graph, Node, Relationship
from collections import defaultdict
from six import iteritems
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings #import only what is needed everyt time?
from flair.data import Sentence
from wordcloud import WordCloud
# Parameters
TRAINING = False
UPDATE_EMBEDDINGS = False

#Naming the directories for the models
dir_name = './models/'
corpus_name = os.path.join(dir_name, 'corpus_feedly.mm') #Stemmed Version
dict_name = os.path.join(dir_name, 'dictionary_feedly.dict')
index_tfidf_name = os.path.join(dir_name, 'fd_index_tfidf.index')
index_tfidf_model_name = os.path.join(dir_name, 'fd_model_tfidf.model')
index_lsi_name = os.path.join(dir_name, 'fd_index_lsi.index')
index_lsi_model_name = os.path.join(dir_name, 'fd_model_lsi.model')
index_lda_name = os.path.join(dir_name, 'fd_index_lda.index')
index_lda_model_name = os.path.join(dir_name, 'fd_model_lda.model')
corpus_name_nst = os.path.join(dir_name, 'corpus_feedly_nst.mm') #Not-Stemmed Version
dict_name_nst = os.path.join(dir_name, 'dictionary_feedly_nst.dict')
index_tfidf_name_nst = os.path.join(dir_name, 'fd_index_tfidf_nst.index')
index_tfidf_model_name_nst = os.path.join(dir_name, 'fd_model_tfidf_nst.model')
index_lsi_name_nst = os.path.join(dir_name, 'fd_index_lsi_nst.index')
index_lsi_model_name_nst = os.path.join(dir_name, 'fd_model_lsi_nst.model')
index_lda_name_nst = os.path.join(dir_name, 'fd_index_lda_nst.index')
index_lda_model_name_nst = os.path.join(dir_name, 'fd_model_lda_nst.model')
index_WE_model_cs_name_glove = os.path.join(dir_name, 'fd_model_we_cosine_similarity_glove.dat')
index_WE_model_eu_name_glove = os.path.join(dir_name, 'fd_model_we_euclidean_distance_glove.dat')
index_WE_model_eu_name_paper = os.path.join(dir_name, 'fd_model_we_euclidean_distance_paper_method.dat')
index_WE_model_cs_name_flair = os.path.join(dir_name, 'fd_model_we_cosine_similarity_flair.dat')
index_WE_model_eu_name_flair = os.path.join(dir_name, 'fd_model_we_euclidean_distance_flair.dat')
index_WE_model_cs_name_bert = os.path.join(dir_name, 'fd_model_we_cosine_similarity_bert.dat')
index_WE_model_eu_name_bert = os.path.join(dir_name, 'fd_model_we_euclidean_distance_bert.dat')
index_WE_model_cs_name_bert_glove = os.path.join(dir_name, 'fd_model_we_cosine_similarity_bert_glove.dat')
index_WE_model_eu_name_bert_glove = os.path.join(dir_name, 'fd_model_we_euclidean_distance_bert_glove.dat')
index_WE_model_cs_name_flair_glove_news = os.path.join(dir_name, 'fd_model_we_cosine_similarity_flair_glove_news.dat')
index_WE_model_eu_name_flair_glove_news = os.path.join(dir_name, 'fd_model_we_euclidean_distance_flair_glove_news.dat')
index_WE_model_cs_name_flair_glove_multi = os.path.join(dir_name, 'fd_model_we_cosine_similarity_flair_glove_multi.dat')
index_WE_model_eu_name_flair_glove_multi = os.path.join(dir_name, 'fd_model_we_euclidean_distance_flair_glove_multi.dat')
index_WE_model_cs_title = os.path.join(dir_name, 'fd_model_we_cs_title.dat')
index_WE_model_eu_title = os.path.join(dir_name, 'fd_model_we_eu_title.dat')
glovefilename = os.path.join(dir_name, 'glove.42B.300dda.txt')

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

extended_punctuations = '‘' + '’' + '‚' + '„' + '…' + '``' + '“' + '”' + '£' + '€' + '¥' + '¢' + '₹' + '₱' + '₩' + '฿' + '₫' + '₪' + '‰' + '†' + '‡' + '•' + '¤' + '§' + '©' + '®' + '™' + '℠' + '«' + '»' + '¸' + '·' + '¯' + '¦' + '—'
punctuation_marks_extended = string.punctuation.replace('-','') + extended_punctuations
def extract_punctuation(text):
    """
    Purpose: Clean the text from any punctuation mark, currency and special symbol.
    Input: <List>. List of tokens from a text.
    Output: <List>. Cleaned list of tokens from a text.
    """
    processed_punctuation = []
    acronymregex = re.compile(r'([A-z]{1}\.)([A-z]{1}\.)+') #check for acronyms with punctuation: r'([A-z]{1}\.)([A-z]{1}\.)+\Z'
    for word in text:
        processed = False
        if (len(word) > 1):
            if acronymregex.match(word):
                word = word.replace('.', '')
                processed_punctuation.append(word)
            else:
                for punctmark in punctuation_marks_extended:
                    if word.startswith(punctmark):
                        word = word.replace(punctmark,'')
                    if word.endswith(punctmark):
                        word = word.replace(punctmark,'')
                    if '/' not in word:
                        if len(word) > 1:
                            subwords = word.split(punctmark)
                        if(len(subwords) > 1):
                            processed = True
                            for subword in subwords:
                                if (len(subword)> 1): processed_punctuation.append(subword)
                    else:
                        processed = True
                        break
                if(processed == False): processed_punctuation.append(word)
        #Note: I'm not letting pass one-letter words: unlikely to have a meaning and likely to be a stopword or punctuation mark
    return processed_punctuation

def convert_numbers_to_specialkey(content):
    """
    Purpose: Transform any sum of money to a common token, so they convey the same meaning.
    Input: <String>. Plain text, prefiltered of markups in this case.
    Output: <String>. Text where any mention to amounts of money is substituted for the token 'amountofmoney'.
    """
    amount = re.compile(r'([\$€¥£¢₹₱₩฿₫₪]{1}[0-9]+(,[0-9]+)?)') #modify any amount of money in the document
    content = re.sub(amount,'amountofmoney',content)
    #numbers = re.compile(r'([0-9]+(,[0-9]+)?([a-z]{1,2})?)') #modify any number quantity in the document
    #content = re.sub(numbers,'number',content)
    return content

def extract_markups(text):
    """
    Purpose: Clean a text from markup <tag> elements.
    Input: <String>. Plain text.
    Output: <String>. Text cleaned from markups.
    Note: It is rather a simple one, as it doesn't distinguish <tag> from <tag>(Stuff)</tag>.
    """
    markups = re.compile(r'(<.*?>)') #remove markups
    cltext = re.sub(markups,'',text)
    return cltext

def extract_stopwords(text):
    """
    Purpose: Remove those words that are general in meaning and do not convey any specific context or topic.
    Input: <List>. List of tokens which include stop words.
    Output: <List>. List of tokens which do not include stop words.
    """
    return [word for word in text if word not in stopwords.words('english')]

def stem_words(text):
    """
    Purpose: Transform words to their lexemes.
             This is used for unifying words like 'fast', 'faster' and 'fastest', for example, into one word.
             So words with different forms that mean the same semmantic meaning are unified.
    Input: <List>. List of tokens.
    Output: <List>. List of stemmed tokens.
    """
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in text:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def cleanClipText(cliptext):
    """
    Purpose: Filtering the elements to UTF-16. Tkinter cannot represent items outside this range, and causes an error.
    Input: <String>. A word that can contain UTF-32 characters.
    Output: <String>. Word without the characters out of representable range in Tkinter.
    """
    #Removing all characters > 65535 (that's the range for tcl)
    cliptext = "".join([c for c in cliptext if ord(c) <= 65535])
    return cliptext

def final_clean(text_list):
    """
    Purpose: Do a final sweep over the tokens in search for elements that could have passed the filters.
    Input: <List>. List of preprocessed tokens from the texts.
    Output: <List>. List of cleaned preprocessed tokens from the texts.
    """
    clean_text = []
    final_text = []
    for word in text_list:
        processed = False
        if '/' not in word:
            for punctmark in punctuation_marks_extended:
                if word.startswith(punctmark):
                    word = word.replace(punctmark,'')
                if word.endswith(punctmark):
                    word = word.replace(punctmark,'')
                subwords = word.split(punctmark)
                if(len(subwords) > 1):
                    processed = True
                    for subword in subwords:
                        if (len(subword)> 1): clean_text.append(subword)
            if(processed == False):
                if(len(word) >= 2): clean_text.append(word) #clean remaining single letters and white-spaces
    for word in clean_text:
        clean_word = cleanClipText(word)
        final_text.append(clean_word)
    return final_text

def preprocess_text(article):
    """
    Purpose: Main function for preprocessing a text.
    Input: <String>. Raw plain text coming from a source. In this case, HTML source code.
    Output: <List>. List of clean representable tokens that convey meaning from the raw text.
    """
    
    content = html.unescape(article) #clean unwanted html hexadecimal entities
    content = extract_markups(content) #
    content = convert_numbers_to_specialkey(content)
    content = word_tokenize(content.lower()) #tokenize words
    content = extract_punctuation(content) #remove punctuation marks
    content = final_clean(content)
    content_clean = extract_stopwords(content) #remove stopwords (english)
    return content_clean

def load_articlesNeo4j():
    """
    Purpose: This will load a list of articles in JSON format into Neo4j from the import folder.
    Input: JSON file coming from the Feedly.com API.
    Output: The articles will be imported in the active database in Neo4j as nodes.
    Note: The JSON file must follow the same structure that Feedly.com provides.
    """
    
    queryLoadFeedlyArticles = """
    CALL apoc.load.json('file:///all_complete_articles.json') YIELD value
    UNWIND value.items AS item
    MERGE (a:Article:_AI {id:item.id})
    SET a.created = item.crawled,
        a.image = item.visual.url,
        a.title = trim(item.title),
        a.author = trim(item.author),
        a.content = coalesce(item.content.content,item.fullContent),    
        a.summary = item.summary.content,
        a.url = [],
        a.url = a.url + coalesce(item.canonicalUrl,[]),
        a.highlightedText = []
    FOREACH (annotation IN item.annotations |
        SET a.highlightedText = a.highlightedText + annotation.highlight.text
    )
    FOREACH (alt IN item.alternate |
        SET a.url = a.url + alt.href
    )

    FOREACH (tag IN item.tags |
        FOREACH(ignoreMe IN CASE WHEN left(tag.label,3) = "FA." THEN [1] ELSE [] END |
            MERGE (lfa:LtsFocusArea:_AI {name:trim(substring(tag.label,3))})
            MERGE (a)-[r:RELATES_TO]->(lfa)
        )
    )
    FOREACH (tag IN item.tags |
        FOREACH(ignoreMe IN CASE WHEN left(tag.label,3) = "HS." THEN [1] ELSE [] END |
            MERGE (hs:HorizonScanningArea:_AI {name:trim(substring(tag.label,3))})
            MERGE (a)-[r:IS_AN_INSTANCE_OF]->(hs)
        )
    )
    FOREACH (tag IN item.tags |
        FOREACH(ignoreMe IN CASE WHEN left(tag.label,3) = "MT." THEN [1] ELSE [] END |
            MERGE (mt:Megatrend:_AI {name:trim(substring(tag.label,3))})
            MERGE (a)-[r:RELATES_TO]->(mt)
        )
    )

    WITH count(*) AS ignored

    MATCH (a:Article:_AI)
    WHERE a.highlightedText = []
    SET a.highlightedText = NULL

    WITH count(*) AS ignored

    MATCH (a:Article:_AI)
    WHERE a.url = []
    SET a.url = NULL
    """
    graph.run(queryLoadFeedlyArticles)

def preprocess_articlesNeo4j():
    """
    Purpose: Clean the database from bad examples. Relocate the content in articles where the content is in summary, etc.
    Input: None.
    Output: The database is updated.
    Note: Some properties may be modified, and some articles deleted (empty articles, mostly).
    """
    
    graph.run("MATCH (m:Article:_AI) WHERE NOT EXISTS(m.summary) AND NOT EXISTS(m.content) DETACH DELETE m")
    graph.run("MATCH (m:Article:_AI) WHERE length(m.summary)>length(m.content) SET m.content = m.summary")
    graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.summary) AND NOT EXISTS(m.content) SET m.content = m.summary")

def process_documentsNeo4j():
    """
    Purpose: Preprocess the content of the articles existing in the database.
    Input: None.
    Output: The database is updated. The processed text will be stored in the properties: 'preprocessed' and 'preprocessed_stemmed' of the nodes.
    """
    preprocess_query = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.content) AND NOT EXISTS(m.preprocessed) RETURN m.content AS content, id(m) AS node_id")
    for item in preprocess_query:
        processed_doc = preprocess_text(item['content'])
        query = "MATCH (m:Article:_AI) WHERE id(m) = $node_id SET m.preprocessed = $preproc"
        parameters = {'node_id': item['node_id'], 'preproc': processed_doc}
        graph.run(query, parameters=parameters)
        
    preprocessstem_query = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) AND NOT EXISTS(m.preprocessed_stemmed) RETURN m.preprocessed AS preproc, id(m) AS node_id")
    for art in preprocessstem_query:
        processed_stem_doc = stem_words(art['preproc'])
        query = "MATCH (m:Article:_AI) WHERE id(m) = $node_id SET m.preprocessed_stemmed = $preproc_stem"
        parameters = {'node_id': art['node_id'], 'preproc_stem': processed_stem_doc}
        graph.run(query, parameters=parameters)
    
    #Clean the graph from empty articles
    graph.run("MATCH (n:Article:_AI) WHERE EXISTS(n.content) AND n.preprocessed=[] DETACH DELETE n")
    graph.run("MATCH (n:Article:_AI) WHERE EXISTS(n.preprocessed) AND n.preprocessed_stemmed=[] DETACH DELETE n")

def clean_empty_processed_docs():
    """
    Purpose: Clean the database from bad examples. In this case, empty content articles.
    Input: None.
    Output: The database in Neo4j is updated.
    """
    graph.run("MATCH (n:Article:_AI) WHERE NOT EXISTS(n.content) DETACH DELETE n")

def evaluate_keywordsNeo4j():
    """
    Purpose: Evaluate the key words defining the articles in the database.
    Input: None.
    Output: The database is updated in Neo4j. The results are stored in the node property: 'keywords'.
    Note: This uses the TF-IDF model. Make sure it is up and running.
    """
    
    preprocess_query = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed_stemmed) AND NOT EXISTS(m.keywords) RETURN m.preprocessed_stemmed AS preproc_stem, id(m) AS node_id")
    for item in preprocess_query:
        ptext_bow = corpus_memory_friendly.dictionary.doc2bow(item['preproc_stem'])
        ptext_tfidf = tfidf[ptext_bow] #this is in local memory currently
        #tfidf_doc = [item for item in ptext_tfidf]
        keyw = sorted(ptext_tfidf, key=lambda item: -item[1])
        idwords = []
        idwordsview = []
        for word in keyw:
            (idw,tf) = word
            if(tf >= 0.099):
                idwords.append(idw)
                if(tf >= 0.125):
                    idwordsview.append(idw)
        if not idwords:
            (idw,tf) = keyw[0] #at least put one keyword
            idwords.append(idw)
            idwordsview.append(idw)

        keywords = [corpus_memory_friendly.dictionary[idword] for idword in idwords]
        keywords_viewer = [corpus_memory_friendly.dictionary[idwordv] for idwordv in idwordsview]
        query = "MATCH (m:Article:_AI) WHERE id(m) = $node_id SET m.keywords = $keywords, m.keywords_viewer = $keywords_view"
        parameters = {'node_id': item['node_id'], 'keywords': keywords, 'keywords_view': keywords_viewer}
        graph.run(query, parameters=parameters)
    
def evaluate_keywordsNeo4jNST(): #NST stands for --> Non-STemmed
    """
    Purpose: Evaluate the key words (Non-stemmed) defining the articles in the database.
    Input: None.
    Output: The database is updated in Neo4j. The results are stored in the node property: 'keywords_nst'.
    Note: This uses the NST TF-IDF model. Make sure it is up and running.
    """
    preprocess_query = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) AND NOT EXISTS(m.keywords_nst) RETURN m.preprocessed AS preproc, id(m) AS node_id")
    for item in preprocess_query:
        ptext_bow = corpus_memory_friendly_NST.dictionary.doc2bow(item['preproc'])
        ptext_tfidf = tfidf_nst[ptext_bow] #this is in local memory currently
        keyw = sorted(ptext_tfidf, key=lambda item: -item[1])
        idwords = []
        idwordsview = []
        for word in keyw:
            (idw,tf) = word
            if(tf >= 0.099):
                idwords.append(idw)
                if(tf >= 0.125):
                    idwordsview.append(idw)
        if not idwords:
            (idw,tf) = keyw[0] #at least put one keyword
            idwords.append(idw)
            idwordsview.append(idw)

        keywords = [corpus_memory_friendly_NST.dictionary[idword] for idword in idwords]
        keywords_viewer = [corpus_memory_friendly_NST.dictionary[idwordv] for idwordv in idwordsview]
        query = "MATCH (m:Article:_AI) WHERE id(m) = $node_id SET m.keywords_nst = $keywords, m.keywords_viewer_nst = $keywords_view"
        parameters = {'node_id': item['node_id'], 'keywords': keywords, 'keywords_view': keywords_viewer}
        graph.run(query, parameters=parameters)
        
def create_LiteId_documents():
    """
    Purpose: Create a lite version of id's, only for the articles and that is sequential.
    Input: None.
    Output: The database in Neo4j is updated. The results are stored in the node property: 'liteId'.
    Note: This serves to identify articles when using the models. It is very important that are sequential,
          and that those id's coincide with the rows and columns of the similarity matrix for each article.
          For more information, visit the gensim.similarities.docsim documentation:
          https://radimrehurek.com/gensim/similarities/docsim.html at July 1st, 2019.
    """
    
    queryliteId = """
    MATCH (n:Article:_AI)
    WITH range(coalesce(max(n.liteId)+1,0),count(n)-1,1) AS enum

    MATCH (n:Article:_AI)
    WHERE NOT EXISTS(n.liteId)
    WITH enum, range(0,count(n)-1,1) AS index, collect(id(n)) AS id
    UNWIND index AS indexes
    WITH id[indexes] AS IDs, enum[indexes] AS ENUMs

    MATCH (n:Article:_AI)
    WHERE id(n)=IDs
    SET n.liteId=ENUMs
    """
    graph.run(queryliteId)
    
def check_documents():
    """
    Purpose: Check the coherence of the Lite ID's.
    Input: None.
    Output:
    Note: If the LiteID's are incoherent, it will raise a warning and the application will not let you continue.
          This is for protecting the well-functioning. An incoherence here will provide bad recommendations, or even runtime errors, in some cases.
          That is a very important link (as explained when creating the LiteID's).
          Note that this is simple right now. It only checks that the amount of nodes with LiteID are the equal to the number of Article nodes
          and that the largest LiteID in the database is the same as it should be for the amount of Article nodes in the database.
          It does not check if the LiteID's are completely sequential or if there are duplicates. So in some missused cases, the coherence check
          may come through, when the LiteID's are not sequential. This may cause Errors. Please, check this oftenly when modifying this property or creating a new graph.
    """
    
    check = False
    num_docs_liteid = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.liteId) RETURN count(m) AS count_liteid").data()
    last_num_liteid = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.liteId) RETURN m.liteId AS lite_id ORDER BY m.liteId DESC LIMIT 1").data()
    num_docs_total = graph.run("MATCH (m:Article:_AI) RETURN count(m) AS count_total").data()
    if(num_docs_liteid[0]['count_liteid'] == num_docs_total[0]['count_total']) and ((last_num_liteid[0]['lite_id']+1) == num_docs_total[0]['count_total']): check = True
    return check

def process_wordembeddingsNeo4j():
    """
    Purpose: Store the pre-trained word embeddings from GloVe in the graph.
    Input: CSV file downloaded from GloVe (https://nlp.stanford.edu/projects/glove/, at July 1st, 2019).
    Output: The graph database is updated with a new type of node :Word. The word vectors are stored in the node property 'embedding'.
    Note: The CSV file must contain a header called 'header'. This is for the GloVe vectors with 300 dimensions.
          Please, modify the code if you are going to use a different pre-trained model.
    """
    
    query = """
    USING PERIODIC COMMIT 20000
    LOAD CSV WITH HEADERS FROM 'file:///glove.42B.300d.csv' AS csvLine FIELDTERMINATOR "↨"
    MERGE (w:Word:_AI {name:split(csvLine.header," ")[0]})
    ON CREATE SET w.embedding = [x IN split(csvLine.header,' ')[1..301] | toFloat(x)]
    """
    graph.run("CREATE CONSTRAINT ON (word:Word) ASSERT word.name IS UNIQUE")
    # Take into account that the character " starting at the beginning of a line breaks the query
    # replace every double quot at the beginning of a line for something else, like a single quot
    graph.run(query)

class MyCorpusDashNeo(object):
    """
    Purpose: This is a class that represents both the pre-processed corpus (articles) and the dictionary of words.
             Loads those from disk, if they exist. If not, the class creates and saves them in the computer.
    Input:   None.
    Output:  Object which is iterable and yields the pre-processed corpus. It can also be done tu use dictionary functions.
    """
    
    def __init__(self):
        if (os.path.isfile(dict_name) and not TRAINING):
            self.process = False  
        else:
            self.process = True
        
        if self.process:
            try:
                print("Creating dictionary...")
                query_corpus = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed_stemmed) RETURN m.preprocessed_stemmed AS preprocessed ORDER BY m.liteId ASC")
                self.dictionary = corpora.Dictionary(article['preprocessed'] for article in query_corpus)
                once_ids = (tokenid for tokenid, docfreq in iteritems(self.dictionary.dfs) if docfreq == 1)
                self.dictionary.filter_tokens(once_ids)
                self.dictionary.compactify()
                self.dictionary.save(dict_name)
                print("Dictionary created.")
                print(self.dictionary)
            except Exception as e:
                print("Failed at creating the dictionary. Please check the dictionary generator.")
                print("Type of error: " + str(e))
                print(traceback.format_exc())
            else:
                try:
                    corpora.MmCorpus.serialize(corpus_name, self)
                except Exception as e:
                    print("Error at serializing the corpus in memory. Please check the code snippet at the corpus serializer.")
                
                try:
                    self.__load_corpus()
                except Exception as e:
                    print("There was an error loading the corpus. Please, check the code.")
                else:
                    self.process = False
        
        else:
            try:
                print("Loading dictionary...")
                self.dictionary = corpora.Dictionary.load(dict_name)
                print("Dictionary loaded.")
                print(self.dictionary)
            except Exception as e:
                print("Failed at loading the dictionary. Please check the dictionary file.")
                print("Type of error: " + str(e))
                print(traceback.format_exc())
            else:
                self.__load_corpus()
    
    def __iter__(self):
        if self.process:
            try:
                query_corpus = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed_stemmed) RETURN m.preprocessed_stemmed AS preprocessed ORDER BY m.liteId ASC")
                for idc,art in enumerate(query_corpus):
                    print("Building model: " + str(idc+1) + "/" + str(self.dictionary.num_docs), end='\r') # + "\r"
                    yield self.dictionary.doc2bow(art['preprocessed'])
                print('\n')
            except Exception as e:
                print("Failed at processing the corpus. Please check the transformation dict-->corpus and/or the Neo4j query.")
                print("Check also that Neo4j is open and running.")
                print("Type of error: " + str(e))
                print(traceback.format_exc())
                print("Should I run an old file?")
                # Run an old file if it exists and fails?
        else:
            try:
                for artitem in self.corpus:
                    yield artitem
            except Exception as e:
                print("The generator has failed at yielding the corpus documents. Check the iterator of the corpus.")
                print("Type of error: " + str(e))
                print(traceback.format_exc())
                
    def __load_corpus(self):
        try:
            print("Loading corpus...")
            self.corpus = corpora.MmCorpus(corpus_name)
            print("Corpus loaded.")
            print(self.corpus)
        except Exception as e:
            print("Failed at loading the corpus. Please check that the corpus file is correct.")
            print("Type of error: " + str(e))
            print(traceback.format_exc())
  
class MyCorpusNeoNST(object):
    """
    Purpose: (Non-Stemmed Version) This is a class that represents both the pre-processed corpus (articles) and the dictionary of words.
             Loads those from disk, if they exist. If not, the class creates and saves them in the computer.
    Input:   None.
    Output:  Object which is iterable and yields the pre-processed corpus. It can also be done tu use dictionary functions.
    """
    
    def __init__(self):
        if (os.path.isfile(dict_name_nst) and not TRAINING):
            self.process = False  
        else:
            self.process = True
        
        if self.process:
            try:
                print("Creating dictionary not stemmed...")
                query_corpus = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) RETURN m.preprocessed AS preprocessed ORDER BY m.liteId ASC")
                self.dictionary = corpora.Dictionary(article['preprocessed'] for article in query_corpus)
                once_ids = (tokenid for tokenid, docfreq in iteritems(self.dictionary.dfs) if docfreq == 1)
                self.dictionary.filter_tokens(once_ids)
                self.dictionary.compactify()
                self.dictionary.save(dict_name_nst)
                print("Dictionary created.")
                print(self.dictionary)
            except Exception as e:
                print("Failed at creating the dictionary. Please check the dictionary generator.")
                print("Type of error: " + str(e))
                print(traceback.format_exc())
            else:
                try:
                    corpora.MmCorpus.serialize(corpus_name_nst, self)
                except Exception as e:
                    print("Error at serializing the corpus in memory. Please check the code snippet at the corpus serializer.")
                
                try:
                    self.__load_corpus()
                except Exception as e:
                    print("There was an error loading the corpus. Please, check the code.")
                else:
                    self.process = False
        
        else:
            try:
                print("Loading dictionary...")
                self.dictionary = corpora.Dictionary.load(dict_name_nst)
                print("Dictionary loaded.")
                print(self.dictionary)
            except Exception as e:
                print("Failed at loading the dictionary. Please check the dictionary file.")
                print("Type of error: " + str(e))
                print(traceback.format_exc())
            else:
                self.__load_corpus()
    
    def __iter__(self):
        if self.process:
            try:
                query_corpus = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) RETURN m.preprocessed AS preprocessed ORDER BY m.liteId ASC")
                for idc,art in enumerate(query_corpus):
                    print("Building model: " + str(idc+1) + "/" + str(self.dictionary.num_docs), end='\r') # + "\r"
                    yield self.dictionary.doc2bow(art['preprocessed'])
                print('\n')
            except Exception as e:
                print("Failed at processing the corpus. Please check the transformation dict-->corpus and/or the Neo4j query.")
                print("Check also that Neo4j is open and running.")
                print("Type of error: " + str(e))
                print(traceback.format_exc())
                print("Should I run an old file?")
                # Run an old file if it exists and fails?
        else:
            try:
                for artitem in self.corpus:
                    yield artitem
            except Exception as e:
                print("The generator has failed at yielding the corpus documents. Check the iterator of the corpus.")
                print("Type of error: " + str(e))
                print(traceback.format_exc())
                
    def __load_corpus(self):
        try:
            print("Loading corpus...")
            self.corpus = corpora.MmCorpus(corpus_name_nst)
            print("Corpus loaded.")
            print(self.corpus)
        except Exception as e:
            print("Failed at loading the corpus. Please check that the corpus file is correct.")
            print("Type of error: " + str(e))
            print(traceback.format_exc())
        
from scipy.optimize import minimize
class WMD(object):
    """
    Purpose: The class computes an optimization process called Word Moving Distance (WMD)
    Input:   None.
    Output:  It computes the minimum euclidean distance between two articles using the WMD method.
    Note:    For more information, read the paper: "From Word Embeddings To Document Distances", by Matt J. Kusner et al. (2015)
             The words should not be stemmed for this method, as GloVe do not contain embeddings for stemmed words.
             Make sure that you are using the NST-version of the models to compute this one.
             
    Note2:   This is currently computing the Relaxed Word Moving Distance (RWMD) version, for a reduced computing time.
             If you would like to use the standard version of the WMD, use the commented second constraint in the code.
             Beware that the computation time will increase and the convergence might not happen during optimization.
    """
    def __init__(self):
        self.loaded = False
    def load_docs(self, text1, text2):
        #Parameters
        MAX_WORDS = 20 # tunable: could be 20, 30, 50, 70... also using TF-IDF or not
        USING_TFIDF = True
        
        if(USING_TFIDF):
            bow_1 = corpus_memory_friendly_NST.dictionary.doc2bow(text1)
            bow_1 = tfidf_nst[bow_1] # OBS! careful you have to make sure this is the non-stemmed one all the time!
            bow_2 = corpus_memory_friendly_NST.dictionary.doc2bow(text2)
            bow_2 = tfidf_nst[bow_2] # OBS! careful you have to make sure this is the non-stemmed one all the time!
        else:
            # dict_1 --> nbow_1 --> self.d_1
            dict_1 = corpora.Dictionary([text1])
            bow_1 = dict_1.doc2bow(text1)
            # dict_2 --> nbow_2 --> self.d_2
            dict_2 = corpora.Dictionary([text2])
            bow_2 = dict_2.doc2bow(text2)
            
        bow_1 = sorted(bow_1, key=lambda x: -x[1])[:MAX_WORDS]
        bow_2 = sorted(bow_2, key=lambda x: -x[1])[:MAX_WORDS]
        
        idw_1 = np.array([it for it,val in bow_1])
        nbow_1 = np.array([val for it,val in bow_1])
        self.d_1 = nbow_1/np.sum(nbow_1)
        idw_2 = np.array([it for it,val in bow_2])
        nbow_2 = np.array([val for it,val in bow_2])
        self.d_2 = nbow_2/np.sum(nbow_2)
        n = len(self.d_1) #length of Text 1
        m = len(self.d_2) #length of Text 2
        keep_list_1 = list(range(n))
        keep_list_2 = list(range(m))
        
        c = np.random.rand(n,m)*10000
        for pos1, idword1 in enumerate(idw_1):
            if(USING_TFIDF): word1 = corpus_memory_friendly_NST.dictionary[idword1]
            else: word1 = dict_1[idword1]
            emb1 = graph.run("MATCH (m:Word:_AI {name: $word_1}) RETURN m.embedding AS embedding LIMIT 1", parameters={'word_1': word1}).data()
            if(emb1):
                emb1 = emb1[0]['embedding']
                for pos2, idword2 in enumerate(idw_2):
                    if(USING_TFIDF): word2 = corpus_memory_friendly_NST.dictionary[idword2]
                    else: word2 = dict_2[idword2]
                    emb2 = graph.run("MATCH (m:Word:_AI {name: $word_2}) RETURN m.embedding AS embedding LIMIT 1", parameters={'word_2': word2}).data()
                    if(emb2):
                        emb2 = emb2[0]['embedding']
                        dist = euclidean_distances([emb1,emb2])[0,1]
                        c[pos1,pos2] = dist
                    else:
                        if(pos2 in keep_list_2): keep_list_2.remove(pos2)
            else: 
                if(pos1 in keep_list_1): keep_list_1.remove(pos1)
        
        c = c[keep_list_1,:]
        c = c[:,keep_list_2]
        
        # custom dictionary for the two documents?
        self.n,self.m = c.shape #length of Text1, Text2
        
        self.d_1 = np.ones(self.n) #this is only for single word texts: this needs to be bow
        self.d_2 = np.ones(self.n) #this is only for single word texts: this needs to be bow

        self.c = np.transpose(c.flatten())
        self.loaded = True
        
    #####################
    ## Paper Doc Dist. ##
    #####################
    #Objective
    def __objective(self,T):
        cost_function = np.dot(T,self.c)
        return cost_function

    #Constraints
    def __constr1(self,x):
        jc = x.reshape(self.n,self.m)
        jc = -np.sum(jc, axis=1)
        l = np.add(jc,self.d_1)
        return l

    def __constr2(self,x):
        jc2 = x.reshape(self.n,self.m)
        jc2 = -np.sum(jc2, axis=0)
        l2 = np.add(jc2,self.d_2)
        return l2
    
    #Calculate the document distances
    def calculate_distance(self):
        if(self.loaded):
            cons = [{'type': 'eq', 'fun': self.__constr1}] # , {'type': 'eq', 'fun': constr2}
            T0 = np.random.rand(self.n,self.m)
            T0 = T0.flatten()
            b = (0.0,None) # Bounds for the transformations
            bnds = (b,)*self.n*self.m
            sol = minimize(self.__objective, T0, method='SLSQP', bounds=bnds, constraints=cons, tol=1e-4)
            return sol
        else:
            print("You need to load the documents you want to calculate the distance of first.\nUse the function 'load_docs(doc1,doc2)' for that.")
    
def store_recommendations_single_article_Neo4j(art, recom, algorithm):
    """
    Purpose: The function stores the recommendations based on an algorithm for one certain article defined. 
    Input:   art        - Refers to the current document liteID to store the recommendations for.
             recom      - Refers to the recommendations performed by the algorithm for the 'art' document.
             algorithm  - Refers to the algorithm used to create the recommendation.
    Output:  None. The recommendations will be stored in the graph database.
    """
    recommendations = []
    for rank,recomid in enumerate(recom):
        recommendations.append({'recom': recomid, 'rank': rank+1})
        
    query="""
    WITH $list_recommendations AS list
    UNWIND list AS recommendations
    MATCH (sou:Article:_AI {liteId: $query_liteid})
    MATCH (rec:Article:_AI {liteId: recommendations.recom})
    MERGE (sou)-[REL:%s]->(rec)
    SET REL.rank = recommendations.rank
    """ % (algorithm)
    
    graph.run(query, parameters={'query_liteid': art, 'list_recommendations': recommendations})
    

def store_recommendations_Neo4j():
    """
    Purpose: The function will estimate the recommendations for each one of the articles in the database using the models.
    Input:   None. Uses the database.
    Output:  None. The recommendations will be stored in the graph database.
    """
    all_artic = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) RETURN m.liteId AS lite_id ORDER BY m.liteId ASC").data()
    for article in all_artic:
        #TF-IDF
        rec = find_similardocs_tfidf(doc=article['lite_id'], return_results=True)
        store_recommendations_single_article_Neo4j(article['lite_id'],rec,algorithm='TF_IDF')
        
        #LSA
        rec = find_similardocs_lsi(doc=article['lite_id'], return_results=True)
        store_recommendations_single_article_Neo4j(article['lite_id'],rec,algorithm='LSA')
        
        #LDA
        #rec = find_similardocs_lda(doc=article['lite_id'], return_results=True)
        #store_recommendations_single_article_Neo4j(article['lite_id'],rec,algorithm='LDA')
        
        #Word Embeddings
        rec = find_similardocs_WE(doc=article['lite_id'], return_results=True)
        store_recommendations_single_article_Neo4j(article['lite_id'],rec,algorithm='WORD_EMBEDDINGS')
        
def merge_recommendations_Neo4j():
    """
    Purpose: The function will merge the recommendations among the articles in the database using the algorithms' recommendations.
    Input:   None. Uses the database.
    Output:  None. The recommendations will be stored in the graph database as a :RELATES_TO.
    """
    query="""
    MATCH (a1:Article:_AI)-->(a2:Article_AI)
    WHERE a1 <> a2
    OPTIONAL MATCH (a1)-[r1:WORD_EMBEDDINGS]->(a2)
    OPTIONAL MATCH (a1)-[r2:LSA]->(a2)
    OPTIONAL MATCH (a1)-[r3:TF_IDF]->(a2)
    MERGE (a1)-[com:RELATES_TO]->(a2)
    SET com.weight = (toFloat((10-coalesce(r1.rank,10)) + (10-coalesce(r2.rank,10)) + (10-coalesce(r3.rank,10)))) / 27
    """
    graph.run(query)

#########################
##   GUI APPLICATION   ##
##  -----------------  ##
#########################

import tkinter as tk
import webbrowser
from tkinter import font
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Horizon Scanning AI GUI")
root.resizable(False,False)
HEIGHT = 922
WIDTH = 786

TEXT_FONT = "Volvo Serif Pro"
FONT_ARTICLES = 'Volvo Novum'
FONT_NOT_FOUND = "Volvo Novum Medium"
no_article = -1
no_article_sim = -1
VIEW_DOCUMENT = False
COMPUTE_COHERENCE = False
COMPUTE_RECALL = True
ocult_train = True
algorithm_training = False

#Messages through GUI List of typical functions
def not_implemented_message():
    output_screen['text'] = "This method has not been implemented yet.\nPlease, give time to the engineer responsible."

def showing_recommendations(algorithm):
    output_screen['text'] = "Showing recommendations based on:\n" + str(algorithm)
    
def not_valid_message():
    output_screen['text'] = "The number entered is not valid.\nPlease, enter a valid amount."
    
def algorithm_not_found_message():
    output_screen['text'] = "It seems the engineer responsible has a non-existing algorithm.\nPlease, review this problem."
    
def article_not_selected_message():
    output_screen['text'] = "Please, select the article you want to find similarities to."
    
def cannot_be_implemented():
    output_screen['text'] = "This method cannot be implemented\nfrom the backend currently.\nPlease, use Neo4j for such analysis."
    
def clean_spacelines(text):
    rgx_clsp = re.compile(r'(\n)+')
    tgx_cltb = re.compile(r'(\t)+')
    sgx_clsp = re.compile(r'(\s)+')
    cltext = re.sub(rgx_clsp,'\n',text)
    cltext = re.sub(tgx_cltb,'\t',cltext)
    cltext = re.sub(sgx_clsp,' ',cltext)
    return cltext


#Functions for calculating relations, distances and training models based on metrics
def train_algorithm():
    """
    Purpose: The main function to handle training of algorithms.
    Input:   None.
    Output:  None. The models will be saved in disk, if applicable.
    Note:    Some algorithms are using pre-trained models. However this function will still calculate the distance and relations
             existing among the articles (euclidean distance or cosine similarity).
    """
    global algorithm_training, stemmed_tfidf
    if(algorithmvariable.get()=="Word Embeddings"):
        output_screen['text'] = "Training for Word Embeddings...\nPlease wait, it may take long."
        train_wordembeddings()
        output_screen['text'] = "Training completed."
    elif(algorithmvariable.get()=="TF-IDF"):
        pass
    elif(algorithmvariable.get()=="Doc2vec"):
        not_implemented_message()
    elif(algorithmvariable.get()=="LSA"):
        algorithm_training = True
        if(COMPUTE_RECALL):
            stemmed_tfidf = True
            recall_scores, RBP_scores, RBPacc_scores, start, step, stop = maximum_recall_score('lsa')
            stemmed_tfidf = False
            recall_scores_nst, RBP_scores_nst, RBPacc_scores_nst, start, step, stop = maximum_recall_score('lsa')
            x = range(start, stop+1, step)
            plt.plot(x, recall_scores, 'g', x, recall_scores_nst, 'y')
            plt.xlabel("Number of Topics")
            plt.ylabel("Recall score")
            plt.legend(("Stemmed-words", "NST-words"), loc='best')
            plt.show()

            plt.plot(x, RBP_scores, 'b', x, RBP_scores_nst, 'c')
            plt.xlabel("Number of Topics")
            plt.ylabel("Rank-biased precision (RBP)")
            plt.legend(("Stemmed-words", "NST-words"), loc='best')
            plt.show()

            plt.plot(x, RBPacc_scores, 'r', x, RBPacc_scores_nst, 'm')
            plt.xlabel("Number of Topics")
            plt.ylabel("Rank-biased precision x recall (RBPacc)")
            plt.legend(("Stemmed-words", "NST-words"), loc='best')
            plt.show()
        if(COMPUTE_COHERENCE): coherence_model('lsa')
        algorithm_training = False
    elif(algorithmvariable.get()=="LDA"):
        algorithm_training = True
        if(COMPUTE_RECALL):
            stemmed_tfidf = True
            recall_scores, RBP_scores, RBPacc_scores, start, step, stop = maximum_recall_score('lda')
            stemmed_tfidf = False
            recall_scores_nst, RBP_scores_nst, RBPacc_scores_nst, start, step, stop = maximum_recall_score('lda')
            x = range(start, stop+1, step)
            plt.plot(x, recall_scores, 'g', x, recall_scores_nst, 'y')
            plt.xlabel("Number of Topics")
            plt.ylabel("Recall score")
            plt.legend(("Stemmed-words", "NST-words"), loc='best')
            plt.show()

            plt.plot(x, RBP_scores, 'b', x, RBP_scores_nst, 'c')
            plt.xlabel("Number of Topics")
            plt.ylabel("Rank-biased precision (RBP)")
            plt.legend(("Stemmed-words", "NST-words"), loc='best')
            plt.show()

            plt.plot(x, RBPacc_scores, 'r', x, RBPacc_scores_nst, 'm')
            plt.xlabel("Number of Topics")
            plt.ylabel("Rank-biased precision x recall (RBPacc)")
            plt.legend(("Stemmed-words", "NST-words"), loc='best')
            plt.show()
        if(COMPUTE_COHERENCE): coherence_model('lda')
        algorithm_training = False
    elif(algorithmvariable.get()=="Ensemble Method"):
        cannot_be_implemented()
    elif(algorithmvariable.get()=="Community Finding"):
        cannot_be_implemented()
    else:
        algorithm_not_found_message()

    
def train_wordembeddings():
    """
    Purpose: Compute the similarity matrix to relate documents based on Word Embeddings
    Input:   Word Embeddings, pre-processed documents
    Output:  Serialized similarity matrix for documents
    
    Modes:   paper - Document distance by word embeddings using the Word Moving Distance method.
                (Note!: This method takes a lot of time! Beware of this.
                 Also, I am saving the results as we go in a txt file, so we can retrieve
                 the computed results and not start from the beginning if it breaks).
             title - Document distance and similarity of documents by word embeddings
                     of words appearing in the Title.
             glove - Document distance and similarity of documents by word embeddings using GloVe embeddings.
             berglove - Document distance and similarity of documents by word embeddings using GloVe and BERT
                        pre-trained embeddings.
             
    """
    mode = 'paper' #choose from
    num_articles = graph.run("MATCH (n:Article:_AI) WHERE EXISTS(n.preprocessed) RETURN count(n) AS total").data()[0]['total']
    
    def serialization(documents_embeddings, file_name_cosine_similarity, file_name_euclidean_distance):
        ################################
        ###      Serialization       ###
        ################################
        if(isinstance(documents_embeddings,list) and isinstance(file_name_cosine_similarity,str) and isinstance(file_name_euclidean_distance,str)):
            try:
                print("Initializing serialization...")
                #Cosine similarity:
                cosim = cosine_similarity(X=documents_embeddings)
                cosim.dump(file_name_cosine_similarity)

                #Euclidean distance: to tune in many ways
                euclidean = euclidean_distances(X=documents_embeddings)
                euclidean.dump(file_name_euclidean_distance)
                print("Finished serialization.")
            except:
                print("Something went wrong during serialization.")
        else:
            print("Bad format for the serialization. Please, check the data structure inputted for this.")
        
        
    if(mode=='glove'):
        embed_dim = graph.run("MATCH (m:Word:_AI) RETURN size(m.embedding) AS size LIMIT 1").data()[0]['size']
        documents = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) RETURN m.liteId AS lite_id, m.preprocessed AS preprocessed ORDER BY m.liteId ASC").data()
        total_percentage_glove = 0.0
        documents_embeddings = [] #this is the WE-based Doc2Vec for the documents in the database
        for doc in documents:
            sum_embedding = np.zeros(shape=(embed_dim), dtype=float)
            count_words = 0
            total_words = 0
            for word in doc['preprocessed']:
                we = graph.run("MATCH (n:Word:_AI {name: $word}) RETURN n.embedding AS embedding", parameters={'word':word}).data()
                total_words += 1
                if we:
                    sum_embedding += we[0]['embedding']
                    count_words += 1
            if(count_words > 0):
                total_percentage_glove += count_words/total_words
                doc_embed = sum_embedding/count_words
            else: doc_embed = sum_embedding*count_words
            documents_embeddings.append(doc_embed)
        serialization(documents_embeddings=documents_embeddings,file_name_cosine_similarity=index_WE_model_cs_name_glove, file_name_euclidean_distance=index_WE_model_eu_name_glove)
        print("Percentage of words in GloVe: " + str(total_percentage_glove/num_articles))           
    
    
    if(mode=='title'): 
        documents = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) RETURN m.liteId AS lite_id, m.preprocessed AS preprocessed, m.keywords AS keywords, m.title AS title ORDER BY m.liteId ASC").data()
        total_percentage_glove = 0.0
        sim_test_title = []
        for doc in documents:
            sum_embedding = np.zeros(shape=(embed_dim), dtype=float)
            count_words = 0
            total_words = 0
            title = preprocess_text(doc['title'])
            for word in doc['preprocessed']:
                if(word in title):
                    we = graph.run("MATCH (n:Word:_AI {name: $word}) RETURN n.embedding AS embedding", parameters={'word':word}).data()
                    total_words += 1
                    if we:
                        sum_embedding += we[0]['embedding']
                        count_words += 1
            if(count_words == 0): doc_embed = sum_embedding*count_words
            else:
                doc_embed = sum_embedding/count_words
                total_percentage_glove += count_words/total_words
            sim_test_title.append(doc_embed)
        serialization(documents_embeddings=sim_test_title, file_name_cosine_similarity=index_WE_model_cs_title, file_name_euclidean_distance=index_WE_model_eu_title)
        print("Percentage of words in GloVe: " + str(total_percentage_glove/num_articles))
        
    
    if(mode=='berglove'):
        glove_embedding = WordEmbeddings('en-glove')
        bert_embedding = BertEmbeddings('bert-large-uncased')
        #multi_forward = FlairEmbeddings('multi-forward')
        #multi_backward = FlairEmbeddings('multi-backward')
        stacked_embeddings = StackedEmbeddings([
                                                glove_embedding,
                                                bert_embedding,
                                                #FlairEmbeddings('news-forward'), 
                                                #FlairEmbeddings('news-backward'),
                                                #multi_forward, 
                                                #multi_backward,
                                               ])

        # Embedding dimension
        se = Sentence('grass')
        stacked_embeddings.embed(se)
        embed_dim = len(se[0].embedding)
        documents = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) RETURN m.liteId AS lite_id, m.preprocessed AS preprocessed ORDER BY m.liteId ASC").data()
        num_doc = 1
        documents_embeddings = [] #this is the WE-based Doc2Vec for the documents in the database
        for doc in documents:
            sentences = []
            sum_embedding = np.zeros(shape=(embed_dim), dtype=float)
            print("Computing Doc WE embedding: " + str(num_doc) + "/" + str(num_articles), end="\r")
            count_words = 0
            sen = ""
            for word in doc['preprocessed']:
                temp = sen + word + " "
                if(len(temp)>512):
                    sentences.append(sen)
                    sen = word + " "
                else: sen = temp
            sentences.append(sen)
            for sen in sentences:
                sentence = Sentence(sen)
                stacked_embeddings.embed(sentence)
                for token in sentence:
                    sum_embedding += np.array(token.embedding)
                    count_words += 1
            if(count_words > 0):
                doc_embed = sum_embedding/count_words
            else: doc_embed = sum_embedding*0
            documents_embeddings.append(doc_embed)
            num_doc += 1
        print("Computing Doc WE embedding: " + str(num_doc) + "/" + str(num_articles), end="\n")
        serialization(documents_embeddings=documents_embeddings,file_name_cosine_similarity=index_WE_model_cs_name_bert_glove, file_name_euclidean_distance=index_WE_model_eu_name_bert_glove)
    
    if(mode=='paper'):
        #Paper Document Distances
        time_ini = time.time()
        word_mover_distance = WMD()
        document_distances_paper = np.random.rand(num_articles,num_articles)*10000
        counter = 0
        total_to_count = int(num_articles**2)
        doclist1 = graph.run("MATCH (m:Article:_AI) RETURN m.liteId AS lite_id, m.preprocessed AS preprocessed ORDER BY m.liteId ASC").data()
        for idd1,doc1 in enumerate(doclist1):
            doclist2 = graph.run("MATCH (m:Article:_AI) RETURN m.liteId AS lite_id, m.preprocessed AS preprocessed ORDER BY m.liteId ASC").data()
            for idd2,doc2 in enumerate(doclist2):
                percentage_proc = counter/total_to_count*100
                print("Calculating document distance " + str(idd1+1) + " --> " + str(idd2+1) + "\t Total num. of articles: " + str(num_articles) + " ({0:.1f}%)".format(percentage_proc), end='\r')
                word_mover_distance.load_docs(doc1['preprocessed'],doc2['preprocessed'])
                sol = word_mover_distance.calculate_distance()
                document_distances_paper[idd1,idd2] = sol.fun
                f = open("paper_distances.txt", "a")
                f.write(str(idd1) + " " + str(idd2) + " " + str(sol.fun) + "\n")
                f.close()
                counter += 1
        print("Calculating document distance " + str(idd1+1) + " --> " + str(idd2+1) + "\t Total num. of articles: " + str(num_articles) + " ({0:3d}%)\n".format(100))
        print("Serializing...")
        document_distances_paper.dump(index_WE_model_eu_name_paper)
        time_elapsed = time.time() - time_ini
        elapsed_hours = int(time_elapsed/3600)
        elapsed_minutes = int(int(time_elapsed%3600)/60)
        elapsed_seconds = int(int(time_elapsed%3600)%60)
        print("WMD Document distances serialized.")
        print("Elapsed time Document Distances Paper: " + str(elapsed_hours) + " h " + str(elapsed_minutes) + " min " + str(elapsed_seconds) + " sec\n")

def find_similardocs_lsi_training(lsi_model, lsi_index, num=11, doc=-1, return_results=True):
    """
    Purpose: Find
    Input:   
    Output:  
    """
    global doc_sim_idx, LIST_SIM_DOCS

    for radiobutton in LIST_SIM_DOCS:
        radiobutton.destroy()
    if (doc == -1): doc = doc_var_idx.get()
    LIST_SIM_DOCS = []
    if(stemmed_tfidf): sim_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed_stemmed) AND m.liteId = $query_liteid RETURN m.preprocessed_stemmed AS preprocessed"
    else: sim_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) AND m.liteId = $query_liteid RETURN m.preprocessed AS preprocessed"
    lsiquery = graph.run(sim_query, parameters={'query_liteid': doc}).data()
    if(stemmed_tfidf): doc_bow = [corpus_memory_friendly.dictionary.doc2bow(doc['preprocessed']) for doc in lsiquery]
    else: doc_bow = [corpus_memory_friendly_NST.dictionary.doc2bow(doc['preprocessed']) for doc in lsiquery]
    doc_lsi = lsi_model[doc_bow]
    docs_similar = lsi_index[doc_lsi]
    sort_docs_similar = [sorted(enumerate(val), key=lambda item: -item[1])[:num] for it,val in enumerate(docs_similar)][0]
    recommend_docs_idd = []
    for idd,simil_score in sort_docs_similar[1:]:
        recommend_docs_idd.append(idd)
        #show no similar articles if there are not articles above a certain threshold (OBS: Right now we only show the best 3)
    if return_results: return recommend_docs_idd
    
def find_similardocs_lda_training(lda_model, lda_index, num=11, doc=-1, return_results=True):
    """
    Purpose:
    Input:
    Output:
    """
    global doc_sim_idx, LIST_SIM_DOCS

    for radiobutton in LIST_SIM_DOCS:
        radiobutton.destroy()
    if (doc == -1): doc = doc_var_idx.get()
    LIST_SIM_DOCS = []
    if(stemmed_tfidf): sim_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed_stemmed) AND m.liteId = $query_liteid RETURN m.preprocessed_stemmed AS preprocessed"
    else: sim_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) AND m.liteId = $query_liteid RETURN m.preprocessed AS preprocessed"
    ldaquery = graph.run(sim_query, parameters={'query_liteid': doc}).data()
    if(stemmed_tfidf): doc_bow = [corpus_memory_friendly.dictionary.doc2bow(doc['preprocessed']) for doc in ldaquery]
    else: doc_bow = [corpus_memory_friendly_NST.dictionary.doc2bow(doc['preprocessed']) for doc in ldaquery]
    doc_lda = lda_model[doc_bow]
    docs_similar = lda_index[doc_lda]
    sort_docs_similar = [sorted(enumerate(val), key=lambda item: -item[1])[:num] for it,val in enumerate(docs_similar)][0]
    recommend_docs_idd = []
    for idd,simil_score in sort_docs_similar[1:]:
        recommend_docs_idd.append(idd)
        #show no similar articles if there are not articles above a certain threshold (OBS: Right now we only show the best 3)
    if return_results: return recommend_docs_idd

def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3, algorithm='lsa'):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Feedly Teams corpus
              texts : List of articles
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    print("Computing coherence analysis for " + algorithm + " ...\n")
    coherence_values = []
    model_list = []
    maximum_coherence = -1
    optimum_ntopics = 0
    for num_topics in range(start, stop, step):
        if((algorithm == 'lsa') or (algorithm == 'lsi')):
            # generate LSA model
            model = models.LsiModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
        elif(algorithm == 'lda'):
            # generate LSA model
            model = models.LdaModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
        else:
            model = models.LsiModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
        coherencemodel = models.CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        model_list.append(model)
        coherence_values.append(coherencemodel.get_coherence())
        if(coherencemodel.get_coherence() > maximum_coherence):
            maximum_coherence = coherencemodel.get_coherence()
            optimum_ntopics = num_topics
        print("Coherence analysis: " + str(num_topics) + "/" + str(stop), end="\r")
    return model_list, coherence_values, maximum_coherence, optimum_ntopics

def coherence_model(algorithm):
    start = 2
    stop = 15
    step = 1
    if(stemmed_tfidf): data = graph.run("MATCH (n:Article:_AI) WHERE EXISTS(n.preprocessed_stemmed) RETURN n.preprocessed_stemmed AS preproc ORDER BY n.liteId ASC").data()
    else: data = graph.run("MATCH (n:Article:_AI) WHERE EXISTS(n.preprocessed) RETURN n.preprocessed AS preproc ORDER BY n.liteId ASC").data()
    new_data = [item['preproc'] for item in data] #Careful: you are loading all the articles here!
    if(stemmed_tfidf): model_list, coherence_values, maximum_coherence, optimum_ntopics = compute_coherence_values(dictionary=corpus_memory_friendly.dictionary, doc_term_matrix=corpus_memory_friendly, doc_clean=new_data, start=start, stop=stop, step=step, algorithm=algorithm)
    else: model_list, coherence_values, maximum_coherence, optimum_ntopics = compute_coherence_values(dictionary=corpus_memory_friendly_NST.dictionary, doc_term_matrix=corpus_memory_friendly_NST, doc_clean=new_data, start=start, stop=stop, step=step, algorithm=algorithm)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend("Coherence", loc='best')
    plt.show()
    print("Optimum number of topics: " + str(optimum_ntopics))
    print("Max. coherence: " + str(maximum_coherence))
    
def maximum_recall_score(algorithm):
    """
    Purpose:
    Input:
    Output:
    """
    start = 2
    stop = 30
    step = 1
    if(stemmed_tfidf):
        print("Computing for STEMMED WORDS")
        compare_docs = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed_stemmed) RETURN m.preprocessed_stemmed AS preprocessed ORDER BY m.liteId ASC").data()
        compare_docs_bow = [corpus_memory_friendly.dictionary.doc2bow(doc['preprocessed']) for doc in compare_docs]
    else:
        print("Computing for NON-STEMMED WORDS")
        compare_docs = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) RETURN m.preprocessed AS preprocessed ORDER BY m.liteId ASC").data()
        compare_docs_bow = [corpus_memory_friendly_NST.dictionary.doc2bow(doc['preprocessed']) for doc in compare_docs]
    print("Computing maximum recall score analysis for " + algorithm + " ...\n")
    recall_scores = []
    RBP_scores = []
    RBPacc_scores = []
    model_list = []
    index_list = []
    maximum_recall = -1
    maximum_RBP = -1
    maximum_RBPacc = -1
    optimum_ntopics_recall = 0
    optimum_ntopics_RBP = 0
    optimum_ntopics_RBPacc = 0
    for num_topics in range(start, stop+1, step):
        if((algorithm == 'lsa') or (algorithm == 'lsi')):
            # generate LSA model
            if(stemmed_tfidf):
                model = models.LsiModel(corpus_memory_friendly, id2word=corpus_memory_friendly.dictionary, num_topics=num_topics)
                comp_lsi = model[compare_docs_bow]
                ind_lsi = similarities.Similarity(output_prefix="sim_lsi_idx", corpus=comp_lsi, num_features=len(corpus_memory_friendly.dictionary))
            else:
                model = models.LsiModel(corpus_memory_friendly_NST, id2word=corpus_memory_friendly_NST.dictionary, num_topics=num_topics)
                comp_lsi = model[compare_docs_bow]
                ind_lsi = similarities.Similarity(output_prefix="sim_lsi_nst_idx", corpus=comp_lsi, num_features=len(corpus_memory_friendly_NST.dictionary))
            mod = [model,ind_lsi]
            recall = evaluate_method(mod)
        elif(algorithm == 'lda'):
            # generate LDA model
            if(stemmed_tfidf):
                model = models.LdaModel(corpus_memory_friendly, id2word=corpus_memory_friendly.dictionary, num_topics=num_topics, passes=15, alpha='auto', eval_every=5)
                comp_lda = model[compare_docs_bow]
                ind_lda = similarities.Similarity(output_prefix="sim_lda_idx", corpus=comp_lda, num_features=len(corpus_memory_friendly.dictionary))
            else:
                model = models.LdaModel(corpus_memory_friendly_NST, id2word=corpus_memory_friendly_NST.dictionary, num_topics=num_topics, passes=15, alpha='auto', eval_every=5)
                comp_lda = model[compare_docs_bow]
                ind_lda = similarities.Similarity(output_prefix="sim_lda_nst_idx", corpus=comp_lda, num_features=len(corpus_memory_friendly_NST.dictionary))
            mod = [model,ind_lda]
            recall = evaluate_method(mod)
        model_list.append(model)
        index_list.append(ind_lda)
        recall_scores.append(recall[0])
        RBP_scores.append(recall[1])
        RBPacc_scores.append(recall[2])
        if(recall[0] > maximum_recall):
            maximum_recall = recall[0]
            optimum_ntopics_recall = num_topics
        if(recall[1] > maximum_RBP):
            maximum_RBP = recall[1]
            optimum_ntopics_RBP = num_topics
        if(recall[2] > maximum_RBPacc):
            maximum_RBPacc = recall[2]
            optimum_ntopics_RBPacc = num_topics
        print("Maximum recall analysis: " + str(num_topics) + "/" + str(stop), end="\r")
    model_list[optimum_ntopics_RBPacc-1].save(os.path.join(dir_name, "LDA_MAX.model"))
    index_list[optimum_ntopics_RBPacc-1].save(os.path.join(dir_name, "LDA_MAX_INDEX.index"))
    
    return [recall_scores, RBP_scores, RBPacc_scores, start, step, stop]

#Some callbacks for GUI events
def keyentertitlecallback(event):
    search_documents()

def keyenterdocsamountcallback(event):
    find_similardocuments()

# Feature: Give me some insight! (click in icon at the center of the GUI)
def create_input_wordcloud(input_item):
    string_wordcloud = ""
    if isinstance(input_item, str): #community
        query = "MATCH (n:Article:_AI) WHERE n.community_louvain_filtered_1 = $community_query RETURN n.keywords_nst AS result" #we can do it with content, keywords...
        data = graph.run(query, parameters={'community_query': input_item}).data()
        for article in data: #by content also?
            for word in article['result']:
                string_wordcloud += word + " "
    elif isinstance(input_item, int): #lite_id document
        query = "MATCH (n:Article:_AI {liteId: $lite_id}) RETURN n.preprocessed AS result" #we can do it with content, keywords...
        data = graph.run(query, parameters={'lite_id': input_item}).data()
        for word in data[0]['result']:
            string_wordcloud += word + " "
    return string_wordcloud

def build_wordcloud_ideas(tipology):
    if(doc_insight.get() == 0): #query document
        if(doc_var_idx.get() != no_article):
            if(tipology=="community"):
                community = graph.run("MATCH (n:Article:_AI {liteId: $lite_id}) RETURN n.community_louvain_filtered_1 AS community LIMIT 1", parameters={'lite_id': doc_var_idx.get()}).data()[0]['community']
                item_ext = community
            elif(tipology=="document"):
                item_ext = doc_var_idx.get()
        else:
            output_screen['text'] = "You must select a document first."
            
    else: #recommended document
        if(doc_sim_idx.get() != no_article):
            if(tipology=="community"):
                community = graph.run("MATCH (n:Article:_AI {liteId: $lite_id}) RETURN n.community_louvain_filtered_1 AS community LIMIT 1", parameters={'lite_id': doc_sim_idx.get()}).data()[0]['community']
                item_ext = community
            elif(tipology=="document"):
                item_ext = doc_sim_idx.get()
        else:
            output_screen['text'] = "You must select a document first."
            
    directory_wordcloud = os.path.join(dir_name, 'wordclouds/')
    file_wordcloud = os.path.join(directory_wordcloud, tipology + "_" + str(item_ext) + ".png") 
    
    if not os.path.isfile(file_wordcloud):
        mask = np.array(Image.open(os.path.join(directory_wordcloud, 'cloud.png')))
        wc = WordCloud(background_color="white", mask=mask, max_words=200, stopwords=stopwords.words('english'))
        text = create_input_wordcloud(item_ext)
        wc.generate(text)
        wc.to_file(file_wordcloud)
    window_wordcloud = tk.Toplevel(root, height=HEIGHT, width=WIDTH*2)
    wordcloud_image = ImageTk.PhotoImage(Image.open(file_wordcloud), master=window_wordcloud)
    wordcloud_label = tk.Label(window_wordcloud, image=wordcloud_image)
    wordcloud_label.place(anchor='n', relx=0.5, rely=0, relwidth=1, relheight=1)
    window_wordcloud.mainloop()
    
def find_relations():
    if(doc_var_idx.get() != no_article):
        if(doc_sim_idx.get() != no_article):
            first_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.keywords_nst) AND m.liteId = $query_liteid RETURN m.keywords_nst AS keywords"
            query = graph.run(first_query, parameters={'query_liteid': doc_var_idx.get()}).data()
            if query:
                sec_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.keywords) AND m.liteId = $query_liteid RETURN m.keywords AS keywords"
                second_query = graph.run(sec_query, parameters={'query_liteid': doc_sim_idx.get()}).data()
                if second_query:
                    stemmer = PorterStemmer()
                    list_common_themes = []
                    list_common_stems = []
                    for keyword in query[0]['keywords']:
                        stword = stemmer.stem(keyword)
                        if stword in second_query[0]['keywords'] and stword not in list_common_stems: #
                            list_common_themes.append(keyword)
                            list_common_stems.append(stword)
                            
                    if not list_common_themes:
                        tk.messagebox.showinfo("Info", "There were no words in common or the documents are not actually related.")
                    else:
                        text_insight = "The documents share the next concepts:\n"
                        for word in list_common_themes[:-1]:
                            text_insight += word + ", "
                        text_insight += list_common_themes[-1]
                        output_screen['text'] = "Showing insight of two documents."
                        tk.messagebox.showinfo("Info", text_insight)
                else:
                    tk.messagebox.showwarning("Info", "The application could not find the second document.")
            else:
                tk.messagebox.showwarning("Info", "The application could not find the queried document.")
        else:
            tk.messagebox.showinfo("Info", "You must select a second document.")
    else:
        tk.messagebox.showinfo("Info", "You must select a first document.")

def insightinfocallback(event):
    global doc_insight
    window_insights = tk.Toplevel(root, height=HEIGHT/2-10, width=WIDTH/2-10)
    title_insights = tk.Label(window_insights, text="Select an insight")
    title_insights.config(font=("Volvo Broad Pro", 13))
    title_insights.place(anchor='n', relx=0.5, rely=0.05, relwidth=0.6, relheight=0.08)
    frame_select_document = tk.Frame(window_insights)
    frame_select_document.place(relx=0.05, rely=0.15, anchor='nw', relwidth=0.7, relheight=0.15)
    r1 = tk.Radiobutton(frame_select_document, text="For the query document", selectcolor='#e6f2ff', wraplength=255, variable=doc_insight, value=0)
    r1.config(font=(FONT_ARTICLES, 8))
    r1.pack(anchor = 'w')
    r2 = tk.Radiobutton(frame_select_document, text="For the recommended document", selectcolor='#e6f2ff', wraplength=255, variable=doc_insight, value=1)
    r2.config(font=(FONT_ARTICLES, 8))
    r2.pack(anchor = 'w')
    button_source_material = tk.Button(window_insights, text="Read source material")
    button_source_material.config(font=(TEXT_FONT, 10))
    button_source_material.config(command=lambda: read_source_material())
    button_source_material.place(anchor='n', relx=0.5, rely=0.32, relwidth=0.5, relheight=0.1)
    button_wordcloud_document = tk.Button(window_insights, text="Idea of the document")
    button_wordcloud_document.config(font=(TEXT_FONT, 10))
    button_wordcloud_document.config(command=lambda: build_wordcloud_ideas('document'))
    button_wordcloud_document.place(anchor='n', relx=0.5, rely=0.43, relwidth=0.5, relheight=0.1)
    button_wordcloud_community = tk.Button(window_insights, text="What else is in the topic?")
    button_wordcloud_community.config(font=(TEXT_FONT, 10))
    button_wordcloud_community.config(command=lambda: build_wordcloud_ideas('community'))
    button_wordcloud_community.place(anchor='n', relx=0.5, rely=0.54, relwidth=0.62, relheight=0.1)
    button_why_related = tk.Button(window_insights, text="Why are they related?")
    button_why_related.config(font=(TEXT_FONT, 10))
    button_why_related.config(command=lambda: find_relations())
    button_why_related.place(anchor='n', relx=0.5, rely=0.65, relwidth=0.62, relheight=0.1)
    """
    if(doc_var_idx.get() != no_article):
        if(doc_sim_idx.get() != no_article):
            first_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.keywords) AND m.liteId = $query_liteid RETURN m.keywords_nst AS keywords"
            query = graph.run(first_query, parameters={'query_liteid': doc_var_idx.get()}).data()
            if query:
                sec_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.keywords) AND m.liteId = $query_liteid RETURN m.keywords AS keywords"
                second_query = graph.run(sec_query, parameters={'query_liteid': doc_sim_idx.get()}).data()
                if second_query:
                    stemmer = PorterStemmer()
                    list_common_themes = []
                    list_common_stems = []
                    for keyword in query[0]['keywords']:
                        stword = stemmer.stem(keyword)
                        if stword in second_query[0]['keywords'] and stword not in list_common_stems: #
                            list_common_themes.append(keyword)
                            list_common_stems.append(stword)
                            
                    if not list_common_themes:
                        tk.messagebox.showinfo("Info", "There were no words in common or the documents are not actually related.")
                    else:
                        text_insight = "The documents share the next concepts:\n"
                        for word in list_common_themes[:-1]:
                            text_insight += word + ", "
                        text_insight += list_common_themes[-1]
                        tk.messagebox.showinfo("Info", text_insight)
                else:
                    tk.messagebox.showwarning("Info", "We couldn't find the second document.\nCheck with Ivan.")
            else:
                tk.messagebox.showwarning("Info", "We couldn't find the document.\nCheck with Ivan.")
            #tk.messagebox.showinfo("Info", "This is meant to show insight.")
        else:
            tk.messagebox.showinfo("Info", "You must select a second document.")
    else:
        tk.messagebox.showinfo("Info", "You must select a document.")
    """
    window_insights.mainloop()

def insightlabcallback(event):
    """
    Purpose:
    Input:
    Output:
    """
    global LST_features, ocult_train, stemmed_tfidf, tfidf, index_tfidf
    if(ocult_train):
        answerst = tk.messagebox.askyesno("Want to use the Non-stemmed version?","Do you want to change to the NST version of the TF-IDF?\nIf not, the standard stemmed version will be loaded.")
        stemmed_tfidf = not answerst
        train_button = tk.Button(footer_frame, text="Train Algorithm")
        train_button.config(font=(TEXT_FONT, 10))
        train_button.config(command=lambda: train_algorithm())
        train_button.place(anchor='nw', relx=0.505, rely=0, relwidth=0.2, relheight=0.47)
        evaluate_button = tk.Button(footer_frame, text="Evaluate")
        evaluate_button.config(font=(TEXT_FONT, 10))
        evaluate_button.config(command=lambda: evaluate_method())
        evaluate_button.place(anchor='sw', relx=0.505, rely=1, relwidth=0.2, relheight=0.53)
        LST_features.append(train_button)
        LST_features.append(evaluate_button)
        output_screen['text'] = "Wow, you discovered a new feature!\n(Only for developers)" #change icon to developer?
        ocult_train = not ocult_train
    else:
        for i in LST_features:
            i.destroy()
        ocult_train = not ocult_train
        output_screen['text'] = "" #change icon to developer?

#Functions: Scrolling with the mouse
def _bound_to_mousewheel(event):
    left_dlistcanvas.bind_all("<MouseWheel>", scroll_documentscallback)

def _unbound_to_mousewheel(event):
    left_dlistcanvas.unbind_all("<MouseWheel>")

def _bound_to_mousewheelsim(event):
    right_dlistcanvas.bind_all("<MouseWheel>", scroll_documentscallbacksim)

def _unbound_to_mousewheelsim(event):
    right_dlistcanvas.unbind_all("<MouseWheel>")

def scroll_documentscallback(event):
    if(left_dlistframe.winfo_height() > left_dlistcanvas.winfo_height()):
        left_dlistcanvas.yview_scroll(-1*int((event.delta/120)), "units")
        
def scroll_documentscallbacksim(event):
    if(right_dlistframe.winfo_height() > right_dlistcanvas.winfo_height()):
        right_dlistcanvas.yview_scroll(-1*int((event.delta/120)), "units")

##################################################
#  Functions: Give me related/similar documents  #
#------------------------------------------------#
##################################################

def find_similardocuments():
    """
    Purpose: Using the algorithms to find the closest or most similar documents.
    Input:   None. The function will use the models trained in disk and the selected article from the GUI.
    Output:  None. The representation of related documents will be shown in the GUI.
    """
    global doc_sim_idx
    pred_amount_doc = 10
    valid_number = True
    if(doc_var_idx.get() != no_article):
        for label in LIST_NOT_FOUND_LABELS:
            label.destroy()
    
        #Check if the user has defined an amount#
        if entry_docs.get():
            try:
                amount = int(entry_docs.get(),10)
            except:
                valid_number = False
                amount = pred_amount_doc
            else:
                if(amount <= 0): valid_number=False
                else: annex = "\nShowing the " + str(amount) + " most similar article/s."
        else:
            annex = "\nPredifined: showing the " + str(pred_amount_doc) + " most similar articles."
            amount = pred_amount_doc
        if valid_number:
            amount += 1
            if(algorithmvariable.get()=="Word Embeddings"):
                showing_recommendations(algorithmvariable.get()+annex)
                find_similardocs_WE(amount)
            elif(algorithmvariable.get()=="TF-IDF"):
                showing_recommendations(algorithmvariable.get()+annex)
                if(stemmed_tfidf): find_similardocs_tfidf(amount)
                else: find_similardocs_tfidf_nst(amount)
            elif(algorithmvariable.get()=="Doc2vec"):
                not_implemented_message()
            elif(algorithmvariable.get()=="LSA"):
                showing_recommendations("Latent Semantic Analysis"+annex)
                if(stemmed_tfidf): find_similardocs_lsi(amount)
                else: find_similardocs_lsi_nst(amount)
            elif(algorithmvariable.get()=="LDA"):
                showing_recommendations("Latent Dirichlet Allocation"+annex)
                find_similardocs_lda(amount)
            elif(algorithmvariable.get()=="Ensemble Method"):
                amount -= 1
                find_similardocs_ensemble(amount)
            elif(algorithmvariable.get()=="Community Finding"):
                amount -= 1
                find_similardocs_community(amount)
            else:
                algorithm_not_found_message()
        else:
            not_valid_message()
        
    else: article_not_selected_message()
    update_idle()
        

def find_similardocs_tfidf(num=11, doc=-1, return_results=False):
    global doc_sim_idx, LIST_SIM_DOCS
    
    for radiobutton in LIST_SIM_DOCS:
        radiobutton.destroy()
    if (doc == -1): doc = doc_var_idx.get()
    LIST_SIM_DOCS = []
    sim_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed_stemmed) AND m.liteId = $query_liteid RETURN m.preprocessed_stemmed AS preprocessed"
    query = graph.run(sim_query, parameters={'query_liteid': doc}).data()
    doc_bow = [corpus_memory_friendly.dictionary.doc2bow(doc['preprocessed']) for doc in query]
    doc_tfidf = tfidf[doc_bow]
    docs_similar = index_tfidf[doc_tfidf]
    sort_docs_similar = [sorted(enumerate(val), key=lambda item: -item[1])[:num] for it,val in enumerate(docs_similar)][0]
    recommend_docs_idd = show_results(sort_docs_similar)
    if return_results: return recommend_docs_idd

def find_similardocs_tfidf_nst(num=11, doc=-1, return_results=False):
    global doc_sim_idx, LIST_SIM_DOCS
    
    for radiobutton in LIST_SIM_DOCS:
        radiobutton.destroy()
    if (doc == -1): doc = doc_var_idx.get()
    LIST_SIM_DOCS = []
    sim_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) AND m.liteId = $query_liteid RETURN m.preprocessed AS preprocessed"
    query = graph.run(sim_query, parameters={'query_liteid': doc}).data()
    doc_bow = [corpus_memory_friendly_NST.dictionary.doc2bow(doc['preprocessed']) for doc in query]
    doc_tfidf = tfidf_nst[doc_bow]
    docs_similar = index_tfidf_nst[doc_tfidf]
    sort_docs_similar = [sorted(enumerate(val), key=lambda item: -item[1])[:num] for it,val in enumerate(docs_similar)][0]
    recommend_docs_idd = show_results(sort_docs_similar)
    if return_results: return recommend_docs_idd
    
def find_similardocs_lsi(num=11, doc=-1, return_results=False):
    global doc_sim_idx, LIST_SIM_DOCS

    for radiobutton in LIST_SIM_DOCS:
        radiobutton.destroy()
    if (doc == -1): doc = doc_var_idx.get()
    LIST_SIM_DOCS = []
    sim_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed_stemmed) AND m.liteId = $query_liteid RETURN m.preprocessed_stemmed AS preprocessed"
    lsiquery = graph.run(sim_query, parameters={'query_liteid': doc}).data()
    doc_bow = [corpus_memory_friendly.dictionary.doc2bow(doc['preprocessed']) for doc in lsiquery]
    doc_lsi = lsi[doc_bow]
    docs_similar = index_lsi[doc_lsi]
    sort_docs_similar = [sorted(enumerate(val), key=lambda item: -item[1])[:num] for it,val in enumerate(docs_similar)][0]
    recommend_docs_idd = show_results(sort_docs_similar)
    if return_results: return recommend_docs_idd

def find_similardocs_lsi_nst(num=11, doc=-1, return_results=False):
    global doc_sim_idx, LIST_SIM_DOCS

    for radiobutton in LIST_SIM_DOCS:
        radiobutton.destroy()
    if (doc == -1): doc = doc_var_idx.get()
    LIST_SIM_DOCS = []
    sim_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) AND m.liteId = $query_liteid RETURN m.preprocessed AS preprocessed"
    lsiquery = graph.run(sim_query, parameters={'query_liteid': doc}).data()
    doc_bow = [corpus_memory_friendly_NST.dictionary.doc2bow(doc['preprocessed']) for doc in lsiquery]
    doc_lsi = lsi_nst[doc_bow]
    docs_similar = index_lsi_nst[doc_lsi]
    sort_docs_similar = [sorted(enumerate(val), key=lambda item: -item[1])[:num] for it,val in enumerate(docs_similar)][0]
    recommend_docs_idd = show_results(sort_docs_similar)
    if return_results: return recommend_docs_idd

def find_similardocs_lda(num=11, doc=-1, return_results=False):
    global doc_sim_idx, LIST_SIM_DOCS

    for radiobutton in LIST_SIM_DOCS:
        radiobutton.destroy()
    if (doc == -1): doc = doc_var_idx.get()
    LIST_SIM_DOCS = []
    sim_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed_stemmed) AND m.liteId = $query_liteid RETURN m.preprocessed_stemmed AS preprocessed"
    ldaquery = graph.run(sim_query, parameters={'query_liteid': doc}).data()
    doc_bow = [corpus_memory_friendly.dictionary.doc2bow(doc['preprocessed']) for doc in ldaquery]
    doc_lda = lda[doc_bow]
    docs_similar = index_lda[doc_lda]
    sort_docs_similar = [sorted(enumerate(val), key=lambda item: -item[1])[:num] for it,val in enumerate(docs_similar)][0]
    recommend_docs_idd = show_results(sort_docs_similar)
    if return_results: return recommend_docs_idd
    
def find_similardocs_lda_nst(num=11, doc=-1, return_results=False):
    global doc_sim_idx, LIST_SIM_DOCS

    for radiobutton in LIST_SIM_DOCS:
        radiobutton.destroy()
    if (doc == -1): doc = doc_var_idx.get()
    LIST_SIM_DOCS = []
    sim_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) AND m.liteId = $query_liteid RETURN m.preprocessed AS preprocessed"
    ldaquery = graph.run(sim_query, parameters={'query_liteid': doc}).data()
    doc_bow = [corpus_memory_friendly_NST.dictionary.doc2bow(doc['preprocessed']) for doc in ldaquery]
    doc_lda = lda_nst[doc_bow]
    docs_similar = index_lda_nst[doc_lda]
    sort_docs_similar = [sorted(enumerate(val), key=lambda item: -item[1])[:num] for it,val in enumerate(docs_similar)][0]
    recommend_docs_idd = show_results(sort_docs_similar)
    if return_results: return recommend_docs_idd
    
def find_similardocs_WE(num=11, doc=-1, return_results=False):
    global doc_sim_idx, LIST_SIM_DOCS
    tech = 'pp'
    for radiobutton in LIST_SIM_DOCS:
        radiobutton.destroy()
    if (doc == -1): doc = doc_var_idx.get()
    LIST_SIM_DOCS = []
    if (tech == 'cs'):
        sim_matrix = np.load(index_WE_model_cs_name_bert_glove)[doc]
        sort_docs_similar = [val for it,val in enumerate(sim_matrix)]
        similar_docs = sorted(enumerate(sort_docs_similar), key=lambda item: -item[1])[:num]
    elif (tech == 'eu'):
        dist_matrix = np.load(index_WE_model_eu_name_bert_glove)[doc]
        sort_docs_distances = [val for it,val in enumerate(dist_matrix)]
        similar_docs = sorted(enumerate(sort_docs_distances), key=lambda item: item[1])[:num]
    elif (tech == 'pp'):
        dist_matrix = np.load(index_WE_model_eu_name_paper)[doc]
        sort_docs_distances = [val for it,val in enumerate(dist_matrix)]
        similar_docs = sorted(enumerate(sort_docs_distances), key=lambda item: item[1])[:num]
    recommend_docs_idd = show_results(similar_docs)
    if return_results: return recommend_docs_idd
    
def find_similardocs_ensemble(num=10, doc=-1, return_results=False):
    global doc_sim_idx, LIST_SIM_DOCS

    for radiobutton in LIST_SIM_DOCS:
        radiobutton.destroy()
    if (doc == -1): doc = doc_var_idx.get()
    LIST_SIM_DOCS = []
    similarity_query = "MATCH (m:Article:_AI {liteId: $query_liteid})-[r:RELATES_TO]->(a2:Article:_AI) RETURN a2.title AS title, a2.liteId AS lite_id, r.weight AS weight ORDER BY r.weight DESC LIMIT $num_docs"
    community_query = graph.run(similarity_query, parameters={'query_liteid': doc, 'num_docs': num}).data()
    recommend_docs_idd = []
    for article in community_query:
        label = article['title'] + " (" + str(article['weight']) + ")"
        r = tk.Radiobutton(right_dlistframe, text=label, selectcolor='#e6f2ff', wraplength=255, bg='white', relief='ridge', overrelief='ridge', indicatoron=False, variable=doc_sim_idx, value=article['lite_id'])
        r.config(font=(FONT_ARTICLES, 9))
        r.pack(anchor = 'w', fill='x')
        LIST_SIM_DOCS.append(r)
        recommend_docs_idd.append(article['lite_id'])
    if return_results: return recommend_docs_idd

def find_similardocs_community(num=10, doc=-1, return_results=False):
    global doc_sim_idx, LIST_SIM_DOCS

    for radiobutton in LIST_SIM_DOCS:
        radiobutton.destroy()
    if (doc == -1): doc = doc_var_idx.get()
    LIST_SIM_DOCS = []
    similarity_query = "MATCH (a1:Article:_AI {liteId: $query_liteid})-[r:RELATES_TO]->(a2:Article:_AI) WHERE a1.community_louvain_filtered_1 = a2.community_louvain_filtered_1 RETURN a2.title AS title, a2.liteId AS lite_id, r.weight AS weight ORDER BY r.weight DESC LIMIT $num_docs"
    community_query = graph.run(similarity_query, parameters={'query_liteid': doc, 'num_docs': num}).data()
    recommend_docs_idd = []
    for article in community_query:
        label = article['title'] + " (" + str(article['weight']) + ")"
        r = tk.Radiobutton(right_dlistframe, text=label, selectcolor='#e6f2ff', wraplength=255, bg='white', relief='ridge', overrelief='ridge', indicatoron=False, variable=doc_sim_idx, value=article['lite_id'])
        r.config(font=(FONT_ARTICLES, 9))
        r.pack(anchor = 'w', fill='x')
        LIST_SIM_DOCS.append(r)
        recommend_docs_idd.append(article['lite_id'])
    if return_results: return recommend_docs_idd

def show_results(docs):
    global doc_sim_idx, LIST_SIM_DOCS
    recommend_docs_idd = []
    for idd,simil_score in docs[1:]:
        query = "MATCH (m:Article:_AI) WHERE EXISTS(m.content) AND m.liteId = $lite_id RETURN m.title AS title, m.content AS content, m.liteId AS lite_id"
        retrieve_document = graph.run(query, parameters={'lite_id': idd}).data()
        label = retrieve_document[0]['title'] + " (" + str(simil_score) + ")"
        r = tk.Radiobutton(right_dlistframe, text=label, selectcolor='#e6f2ff', wraplength=255, bg='white', relief='ridge', overrelief='ridge', indicatoron=False, variable=doc_sim_idx, value=retrieve_document[0]['lite_id'])
        r.config(font=(FONT_ARTICLES, 9))
        r.pack(anchor = 'w', fill='x')
        LIST_SIM_DOCS.append(r)
        recommend_docs_idd.append(idd)
    return recommend_docs_idd

#Function: update the GUI, so new content can be showed!
def update_idle():
    right_dlistcanvas.update_idletasks()
    doc_sim_idx.set(no_article)
    right_dlistcanvas.configure(scrollregion=right_dlistcanvas.bbox('all'))

#Secret feature: Evaluate accuracy with the test set!
def evaluate_method(models=[]):
    """
    Purpose: Have a metric to objectively compare different models, also different versions of the same algorithm.
    Input:   Normally it is not used. Only in the version when training a model to maximize the evaluation score
             will be necessary (LSI/LDA for max recall).
             models - [lsi_model, lsi_index] list of the model and index for similarity evaluation.
    Output:  None. The result will be shown in the GUI.
             Provides:
                 Recall: Accuracy metric over what documents of the same cluster appear in a direct recommendation.
                 RBP:    Score that values the rank of the recommendations. The smaller the rank, the higher relevance.
                 RBPacc: A multiplication of the two previous metrics. This is in order to value not only accuracy
                         but also the order in which the recommendations appear in the GUI.
                 
                 For more information, read the paper "Feeling Lucky? Multi-armed Bandits for Ordering Judgements in
                    Pooling-based Evaluation" by David E. Losada et al. (2016) OR the project documentation.
    """
    evaluation_matrix = [[37, 104, 145, 144],
                         [175, 113, 44, 11],
                         [160, 153, 23, 25, 135, 222],
                         [329, 212, 152],
                         [179, 190, 29, 122, 277],
                         [57, 2, 42, 81, 39],
                         [58, 104, 116],
                         [147, 18, 224, 347],
                         [10, 225],
                         [304, 303, 340],
                         [3, 98],
                         [98, 337, 196, 304],
                         [3, 346],
                         [61, 196],
                         [1, 209, 328, 267, 287, 281],
                         [8, 94]] #test set evaluation with known clusters manually evaluated
    total_recommendations = 0
    total_expected_recommendations = 0
    decay = 0.85 #decay coefficient
    RBP = 0 #rank-biased precision
    RBPacc = 0 #rank-biased precision x recall accuracy measurement
    if(algorithmvariable.get()=="Word Embeddings"):
        for cluster in evaluation_matrix:
            n_doc_cluster = len(cluster)
            for itarticle in cluster:
                recom = find_similardocs_WE(doc=itarticle, return_results=True)
                for i,rec in enumerate(recom):
                    if (rec in cluster):
                        total_recommendations += 1
                        RBP += decay**i
                total_recommendations += 1 #to include itself in the cluster recommendation
                total_expected_recommendations += n_doc_cluster
    elif(algorithmvariable.get()=="TF-IDF"):
         for cluster in evaluation_matrix:
            n_doc_cluster = len(cluster)
            for itarticle in cluster:
                if(stemmed_tfidf): recom = find_similardocs_tfidf(doc=itarticle, return_results=True)
                else: recom = find_similardocs_tfidf_nst(doc=itarticle, return_results=True)
                for i,rec in enumerate(recom):
                    if (rec in cluster):
                        total_recommendations += 1
                        RBP += decay**i
                total_recommendations += 1 #to include itself in the cluster recommendation
                total_expected_recommendations += n_doc_cluster
    elif(algorithmvariable.get()=="Doc2vec"):
        not_implemented_message()
    elif(algorithmvariable.get()=="LSA"):
        for cluster in evaluation_matrix:
            n_doc_cluster = len(cluster)
            for itarticle in cluster:
                if(algorithm_training):
                    recom = find_similardocs_lsi_training(doc=itarticle, lsi_model=models[0], lsi_index=models[1], return_results=True)
                else:
                    if(stemmed_tfidf): recom = find_similardocs_lsi(doc=itarticle, return_results=True)
                    else: recom = find_similardocs_lsi_nst(doc=itarticle, return_results=True)
                for i,rec in enumerate(recom):
                    if (rec in cluster):
                        total_recommendations += 1
                        RBP += decay**i
                total_recommendations += 1 #to include itself in the cluster recommendation
                total_expected_recommendations += n_doc_cluster
    elif(algorithmvariable.get()=="LDA"):
        for cluster in evaluation_matrix:
            n_doc_cluster = len(cluster)
            for itarticle in cluster:
                if(algorithm_training):
                    recom = find_similardocs_lda_training(doc=itarticle, lda_model=models[0], lda_index=models[1], return_results=True)
                else:
                    if(stemmed_tfidf): recom = find_similardocs_lda(doc=itarticle, return_results=True)
                    else: recom = find_similardocs_lda_nst(doc=itarticle, return_results=True)
                for i,rec in enumerate(recom):
                    if (rec in cluster):
                        total_recommendations += 1
                        RBP += decay**i
                total_recommendations += 1 #to include itself in the cluster recommendation
                total_expected_recommendations += n_doc_cluster
    elif(algorithmvariable.get()=="Ensemble Method"):
        for cluster in evaluation_matrix:
            n_doc_cluster = len(cluster)
            for itarticle in cluster:
                recom = find_similardocs_ensemble(doc=itarticle, return_results=True)
                for i,rec in enumerate(recom):
                    if (rec in cluster):
                        total_recommendations += 1
                        RBP += decay**i
                total_recommendations += 1 #to include itself in the cluster recommendation
                total_expected_recommendations += n_doc_cluster
    elif(algorithmvariable.get()=="Community Finding"):
        for cluster in evaluation_matrix:
            n_doc_cluster = len(cluster)
            dictionary_evaluation = dict()
            for itarticle in cluster:
                community_found = graph.run("MATCH (n:Article:_AI {liteId: $lite_id}) RETURN DISTINCT n.community_louvain_filtered_1 AS community LIMIT 1", parameters={'lite_id': itarticle}).data()
                dictionary_evaluation[community_found[0]['community']] = dictionary_evaluation.get(community_found[0]['community'], 0) + 1
                recom = find_similardocs_community(doc=itarticle, return_results=True)
                for i,rec in enumerate(recom):
                    if (rec in cluster):
                        RBP += decay**i
            dictionary_evaluation = sorted(dictionary_evaluation.items(), key= lambda item: -item[1])
            total_recommendations += dictionary_evaluation[0][1]
            total_expected_recommendations += n_doc_cluster
    else:
        algorithm_not_found_message()
        
    if(total_expected_recommendations > 0):
        recomendation_evaluation = total_recommendations/total_expected_recommendations
        RBPacc = RBP*recomendation_evaluation
        output_screen['text'] = "Recall score: %.2f\nThe RBP score: %.2f\nThe RBP-accuracy is: %.2f" % (recomendation_evaluation,RBP,RBPacc)
        if(algorithm_training): return [recomendation_evaluation,RBP,RBPacc]
    update_idle() #update the GUI to represent the new things!

#Feature: Searching and visualizing an article in the web browser
def read_source_material():
    """
    Purpose: Open a new tab in the web browser with the url to the selected article in the GUI.
    Input: None. Selection from the GUI (global variable).
    Output: Tab in the preferred web browser.
    """
    query = "MATCH (m:Article:_AI) WHERE EXISTS(m.url) AND m.liteId = $lite_id RETURN m.url AS url"
    if doc_insight.get() == 0:
        if(doc_var_idx.get() != no_article):
            url = graph.run(query, parameters={'lite_id': doc_var_idx.get()}).data()
        else:
            output_screen['text'] = "You must select a document first."
    else:
        if(doc_sim_idx.get() != no_article):
            url = graph.run(query, parameters={'lite_id': doc_sim_idx.get()}).data()
        else:
            output_screen['text'] = "You must select a document first."
    if url:
        output_screen['text'] = ""
        webbrowser.open_new_tab(url[0]['url'][0])
    else:
        output_screen['text'] = "You must select a document first."
    
#Feature: Using the search bar for articles
def search_documents():
    """
    Purpose: Queries the articles in the database that match the words in the search bar.
    Input: None. The input search is done in the GUI (accessible variable).
    Output: List of articles returned by the search, in the left column of the GUI.
    Note: The matching words are only applied by the title.
          The query varies depending on the words and filters used in the GUI.
    """
    global LIST_DOCUMENTS, LIST_SIM_DOCS, LIST_NOT_FOUND_LABELS, LIST_OF_VIEWERS, VIEW_DOCUMENT, entry_title, doc_var_idx, doc_sim_idx
    
    for radiobutton in LIST_DOCUMENTS:
        radiobutton.destroy()
        
    for radiobutton in LIST_SIM_DOCS:
        radiobutton.destroy()
    
    for label in LIST_NOT_FOUND_LABELS:
        label.destroy()
    
    LIST_DOCUMENTS = []
    LIST_SIM_DOCS = []
    LIST_NOT_FOUND_LABELS = []
    title_search = str(entry_title.get()).replace('"', '')
    title_search = title_search.replace("'","")
    words = title_search.split()
    body_query = ""
    for word in words:
        body_query += " AND m.title =~ '(?i).*" + word.lower() + ".*'"
    footer_query = " RETURN m.title AS title, m.liteId AS LID ORDER BY m.liteId ASC"
    
    if(typevar.get() == TYPES[0]):
        header_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.content)"
    else:
        missmatch = False
        if(typevar.get() == TYPES[1]):
            header_query = "MATCH (m:Article:_AI)-[]-(n:HorizonScanningArea:_AI) WHERE EXISTS(m.content)"
        elif(typevar.get() == TYPES[2]):
            header_query = "MATCH (m:Article:_AI)-[]-(n:LtsFocusArea:_AI) WHERE EXISTS(m.content)"
        elif(typevar.get() == TYPES[3]):
            header_query = "MATCH (m:Article:_AI)-[]-(n:Megatrend:_AI) WHERE EXISTS(m.content)"
        else:
            tk.messagebox.showerror("Error: Name Missmatch", "The names appearing do not correspond to the type of nodes in the database.\nPlease check the names in code.\nNo filters will be applied in this case.")
            header_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.content)"
            missmatch = True
        if(not missmatch):
            body_query += " AND n.name =~ '(?i)" + catvar.get()+ "'"
    search_query = graph.run(header_query + body_query + footer_query).data()
    entry_title.delete(0,'end')
    if(not search_query):
        blank_article = tk.Label(left_dlistframe, text="(Article not found)", bg='white')
        blank_article.config(font=(FONT_NOT_FOUND, 9), width=310)
        blank_article.pack(anchor='nw', fill='x')
        LIST_NOT_FOUND_LABELS.append(blank_article)
    else:
        for idd,doc in enumerate(search_query):
            r = tk.Radiobutton(left_dlistframe, text=doc['title'], selectcolor='#e6f2ff', wraplength=255, bg='white', relief='ridge', overrelief='ridge', indicatoron=False, variable=doc_var_idx, value=doc['LID'])
            r.config(font=(FONT_ARTICLES, 9))
            r.pack(anchor = 'w', fill='x')
            LIST_DOCUMENTS.append(r)
            
    left_dlistcanvas.update_idletasks()
    doc_var_idx.set(no_article)
    left_dlistcanvas.configure(scrollregion=left_dlistcanvas.bbox('all'))
    
    blank_simarticle = tk.Label(right_dlistframe, text="", bg='white')
    blank_simarticle.config(font=(FONT_NOT_FOUND, 9), width=310)
    blank_simarticle.pack(anchor='nw', fill='x')
    LIST_NOT_FOUND_LABELS.append(blank_simarticle)
    
    right_dlistcanvas.update_idletasks()
    doc_sim_idx.set(no_article)
    right_dlistcanvas.configure(scrollregion=right_dlistcanvas.bbox('all'))
    
    if VIEW_DOCUMENT:
        change_view_button['text'] = 'Preview'
        for element in LIST_OF_VIEWERS:
            element.destroy()
        LIST_OF_VIEWERS = []
        output_screen['text'] = ""
        VIEW_DOCUMENT = False

def change_view():
    """
    Purpose: Provide a pre-visualization for articles in the GUI. Highlights the predicted key-words for the article.
    Input: None. The article is selected in the GUI, and the LiteID accessible from this function.
    Output: None. A visualization window is opened in the GUI. Close it by clicking again in the button in the GUI.
    """
    global LIST_OF_VIEWERS, VIEW_DOCUMENT
    if(doc_var_idx.get() != no_article): #Makes sure that an article has been selected
        if(VIEW_DOCUMENT == False):
            if search_query:
                #if document selected
                change_view_button['text'] = 'Close Preview'
                vquery = "MATCH (m:Article:_AI) WHERE EXISTS(m.content) AND m.liteId = $lite_id RETURN m.title AS title, m.content AS content, m.keywords_viewer_nst AS keywordsview"
                graph_query = graph.run(vquery, parameters={'lite_id': doc_var_idx.get()}).data()
                nliteid_found = len(graph_query)
                if (nliteid_found > 1):
                    print("Wait, something is wrong with your LiteIDs. There should be a unique one per document.")
                    print("A have found %d documents with the LiteId: %d" % (nliteid_found, doc_var_idx.get()))
                title = 'Title: ' + graph_query[0]['title'] + '\n'
                text = clean_spacelines(extract_markups(html.unescape(graph_query[0]['content'])))
                text = cleanClipText(text)
                keywords = graph_query[0]['keywordsview']
                viewer = tk.Text(inner_dwn_frame_b, bg='white', wrap='word', padx=20)
                viewer.insert(tk.INSERT, title)
                viewer.insert(tk.INSERT,text)
                search_list = []
                for keyword in keywords:
                    if(len(keyword) > 2): search_list.append(keyword)
                    search_list.append(keyword.capitalize()) #look for capitalized version of the words
                    search_list.append(keyword.upper()) #look for upper-cased version of the words
                for keyword in search_list:
                    start = 1.0
                    long = len(keyword)
                    while True:
                        pos = viewer.search(keyword, start, stopindex='end') #this is CASE-SENSITIVE
                        if pos == "": break
                        viewer.tag_add("here", pos, pos+"+%dc" % (long))
                        start = pos+"+%dc" % (long)
                viewer.tag_config("here", background="yellow", foreground="blue")
                viewer.place(anchor='nw', relx=0, rely=0, relwidth=1, relheight=1)
                LIST_OF_VIEWERS.append(viewer)
                output_screen['text'] = "You're previewing the text.\n Highlighted you will find its keywords."
        else:
            change_view_button['text'] = 'Preview'
            for element in LIST_OF_VIEWERS:
                element.destroy()
            LIST_OF_VIEWERS = []
            output_screen['text'] = ""
        VIEW_DOCUMENT = not VIEW_DOCUMENT
    else:
        output_screen['text'] = 'Select a document to view it.'
    
def sort_categories():
    #Changes menu options in the GUI depending on the filter chosen (LtsFocusAreas, Megatrends, HScanning, None)
    if(typevar.get() == TYPES[0]):
        CATEGORIES = [
        "None"
        ]
    elif(typevar.get() == TYPES[1]):
        CATEGORIES = [
        "Business And Economy",
        "Environment And Resources",
        "Politics And Law",
        "Society And Individuals",
        "Technologies And Innovation",
        "Miscellaneous"
        ]
    elif(typevar.get() == TYPES[2]):
        CATEGORIES = [
        "Autonomous Drive",
        "Collaboration",
        "Continuous Learning",
        "Cyber Security",
        "Data And Intelligence",
        "Fleet Operators",
        "Handling Complexity",
        "Mobility Infrastructure",
        "Playable Platform",
        "Services",
        "Urban Mobility",
        "UX And Interactions"
        ]
    elif(typevar.get() == TYPES[3]):
        CATEGORIES = [
        "Demographic Changes",
        "Diffusion of Power",
        "Economic Growth",
        "Globalization",
        "Health And Well-being",
        "Immaterialization",
        "Individualization",
        "Knowledge Society",
        "Sustainability",
        "Technology Development"
        ]
    catvar.set(CATEGORIES[0])
    
    #Clean the categories menu
    localmenu = catmenu["menu"]
    localmenu.delete(0,"end")
    
    for cat in CATEGORIES:
        localmenu.add_command(label=cat, command=lambda value=cat: catvar.set(value))

###########################
##  HERE STARTS THE GUI  ##
## --------------------  ##
###########################

canvas = tk.Canvas(root, height= HEIGHT, width= WIDTH)

#Header
header_frame = tk.Frame(root, bd=5, bg='#d9d9d9')
header_frame.place(relx=0.5, rely=0, anchor='n', relwidth=1, relheight=0.1)
title_h = tk.Label(header_frame, text="AI Applied to Knowledge Graphs") #Title
title_h.config(font=("Volvo Broad Pro", 19))
title_h.place(anchor='n', relx=0.5, rely=0, relwidth=0.6, relheight=0.5)
title_s = tk.Label(header_frame, text="Horizon Scanning AI") #Sub-title
title_s.config(font=("Volvo Broad Pro", 13))
title_s.place(anchor='n', relx=0.5, rely=0.5, relwidth=0.6, relheight=0.5)
insightlab_image = ImageTk.PhotoImage(Image.open('./img/insightlab_logo.png').resize((72,69)), master=root)
insightlab_label = tk.Label(header_frame, image=insightlab_image) #Insight Lab logo
insightlab_label.bind("<Triple-Button-3>", insightlabcallback)
insightlab_label.place(anchor='nw', relx=0.8, rely=0, relwidth=0.2, relheight=1)
volvo_image = ImageTk.PhotoImage(Image.open('./img/volvo_logo2.png').resize((75,75)), master=root)
volvo_label = tk.Label(header_frame, image=volvo_image) #Volvo Cars logo
volvo_label.place(anchor='nw', relx=0, rely=0, relwidth=0.2, relheight=1)

#Body
body_frame = tk.Frame(root)
body_frame.place(relx=0.5, rely=0.1, anchor='n', relwidth=1, relheight=0.77)

# Body-header
inner_upp_frame_b = tk.Frame(body_frame, bd=3)
inner_upp_frame_b.place(anchor='nw', relx=0, rely=0, relwidth=1, relheight=0.1)
inner_upp_left_frame = tk.Frame(inner_upp_frame_b)
inner_upp_left_frame.place(anchor='nw', relx=0, rely=0, relwidth=0.2, relheight=1)
inner_upp_center_frame = tk.Frame(inner_upp_frame_b)
inner_upp_center_frame.place(anchor='nw', relx=0.2, rely=0, relwidth=0.50, relheight=1)
inner_upp_right_frame = tk.Frame(inner_upp_frame_b)
inner_upp_right_frame.place(anchor='nw', relx=0.7, rely=0, relwidth=0.3, relheight=1)

lb_search = tk.Label(inner_upp_left_frame, text="Search by:")
lb_search.config(font=(TEXT_FONT, 10))
lb_search.place(anchor='n', relx=0.5, rely=0.04, relwidth=0.92, relheight=0.44)

lb_title = tk.Label(inner_upp_center_frame, text="Title")
lb_title.config(font=(TEXT_FONT, 9))
lb_title.place(anchor='nw', relx=0.03, rely=0.04, relwidth=0.31, relheight=0.44)

entry_title = tk.Entry(inner_upp_center_frame, justify='center') #Search bar
entry_title.config(font=(TEXT_FONT, 9))
entry_title.place(anchor='ne', relx=0.97, rely=0.04, relwidth=0.60, relheight=0.44)

lb_type = tk.Label(inner_upp_right_frame, text="Type")
lb_type.config(font=(TEXT_FONT, 9))
lb_type.place(anchor='nw', relx=0.04, rely=0.04, relwidth=0.3, relheight=0.44)

TYPES = [
"None",
"H. Scanning",
"Lts Focus Area",
"Megatrend"
]

typevar = tk.StringVar(inner_upp_right_frame)
typevar.set(TYPES[0]) # default value

typemenu = tk.OptionMenu(inner_upp_right_frame, typevar, *TYPES, command=lambda e: sort_categories()) #Filter (Focus Areas, Megatrend, etc.)
typemenu.config(font=(TEXT_FONT,9))
typemenu.place(anchor='ne', relx=0.96, rely=0.04, relwidth=0.6, relheight=0.44)

search_button = tk.Button(inner_upp_left_frame, text="Search")
search_button.config(font=(TEXT_FONT, 10))
search_button.place(anchor='n', relx=0.5, rely=0.56, relwidth=0.6, relheight=0.40)

change_view_button = tk.Button(inner_upp_right_frame, text="Preview", command=lambda: change_view()) #Pre-visualization button
change_view_button.config(font=(TEXT_FONT, 10))
change_view_button.place(anchor='ne', relx=0.96, rely=0.56, relwidth=0.6, relheight=0.40)

lb_cat = tk.Label(inner_upp_center_frame, text="Categories")
lb_cat.config(font=(TEXT_FONT, 9))
lb_cat.place(anchor='nw', relx=0.03, rely=0.52, relwidth=0.31, relheight=0.44)

catvar = tk.StringVar(inner_upp_center_frame)
catvar.set("None") # default value

catmenu = tk.OptionMenu(inner_upp_center_frame, catvar, "None") #Categories existing under the filter
catmenu.config(font=(TEXT_FONT,9))
catmenu.place(anchor='ne', relx=0.97, rely=0.52, relwidth=0.6, relheight=0.44)

# Body-main
inner_dwn_frame_b = tk.Frame(body_frame)
inner_dwn_frame_b.place(anchor='nw', relx=0, rely=0.1, relwidth=1, relheight=0.9)

infoinsight_image = ImageTk.PhotoImage(Image.open('./img/info_insight.png').resize((69,40)), master=root)
infoinsight_label = tk.Label(inner_dwn_frame_b, image=infoinsight_image) #Logo button: "Give me some insight!"
infoinsight_label.bind("<Button-1>", insightinfocallback)
infoinsight_label.place(anchor='n', relx=0.5, rely=0.90, relwidth=0.25, relheight=0.1)


#Footer
footer_frame = tk.Frame(root, bd=8)
footer_frame.place(relx=0.5, rely=0.87, anchor='n', relwidth=1, relheight=0.11)
output_screen = tk.Label(footer_frame, justify='left', bg="white", text="Welcome to the application.") # GUI screen for messages and information
output_screen.place(relx=0.025, rely=0.5, anchor='w', relheight=1, relwidth=0.48)
inner_frame_f = tk.Frame(footer_frame)
inner_frame_f.place(relx=0.975, rely=1, anchor='se', relwidth=0.27, relheight=0.8)
fcc_button = tk.Button(inner_frame_f, text="Find Closest") #Button for executing the search for closest documents
fcc_button.config(font=(TEXT_FONT, 10))
fcc_button.place(anchor='nw', relx=0, rely=0, relwidth=0.55, relheight=0.4)
entry_docs = tk.Entry(inner_frame_f, justify='right') #Defines the amount of documents to be returned
entry_docs.place(anchor='nw', relx=0.6, rely=0, relwidth=0.4, relheight=0.4)

LST_features = []

#Dropdown menu
ALGORITHMS = [
"Word Embeddings",
"TF-IDF",
#"Doc2vec",
"LSA",
"LDA",
"Ensemble Method",
"Community Finding"
]

algorithmvariable = tk.StringVar(inner_frame_f)
algorithmvariable.set(ALGORITHMS[0]) # default value

algorithmenu = tk.OptionMenu(inner_frame_f, algorithmvariable, *ALGORITHMS) #Selection of the algorithm to use!
algorithmenu.config(font=("Volvo Broad Pro",11))
algorithmenu.place(anchor="se", relx=1, rely=1, relwidth=1, relheight=0.5)

canvas.pack()

#Initialization of variables, load models, initialize the py2neo v4 Neo4j backend, checks the status of the database
try:
    #Connect to Neo4j Database
    graph = Graph(auth=('user','password'), host="gotsvl1706.got.volvocars.net", port=7687, secure=True)
    exists_database = graph.run("MATCH (n:Article:_AI) RETURN n LIMIT 1").data() #Checks if there is something in the database or is empty
except Exception as e:
    #This happens if there is no connection with the database, normally
    tk.messagebox.showwarning("Warning: No Connection", "Warning: No connection with Neo4j. Please, make sure that Neo4j is running and you have entered the correct password.")
    root.destroy() #This destroys the application (close it)
else:
    if not exists_database: #If the connection has been made, but the database is empty (it will create one)
        tk.messagebox.showinfo("Info","Info: The Knowledge Graph database is empty.\nWe will need to prepare it for you.\nThis may take several minutes.")
        # Load Neo4j JSON
        try:
            load_articlesNeo4j() # Load JSON file from the import folder
            preprocess_articlesNeo4j() # Pre-filter the documents imported
            process_documentsNeo4j() # Process the content and extract significant words
            clean_empty_processed_docs() #Post-filter the documents that have been processed
            create_LiteId_documents() #This has to be done only once
        except Exception as e:
            print("Something went wrong loading the articles into Neo4j.\nCheck that the JSON file is in the import folder and try again.")
            print("If the problem persists, check the code functions.")
        try:
            process_wordembeddingsNeo4j() #OBS!: Only do this if you don't have GloVe in Neo4j and you want to.
        except Exception as e:
            print("Something went wrong loading GloVe embeddings into Neo4j.\nCheck that the file exists in the import folder and try again.")
            print("If the problem persists, check the code function, the format and dimensions of the CSV file.")
    check_coherence = check_documents()
    if check_coherence:
        # Load/Create the corpus
        # Note: All this strings should maybe appear in the GUI
        corpus_memory_friendly = MyCorpusDashNeo()
        corpus_memory_friendly_NST = MyCorpusNeoNST()
        #(Careful, these are right now generators, or they will yield those)
        #Some functions might be deprecated for generators in the future.
        
        #TF-IDF Load
        if not os.path.isfile(index_tfidf_name):
            # Create the TF-IDF model
            tfidf = models.TfidfModel((bow for bow in corpus_memory_friendly), normalize=True)
            compare_docs_query = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed_stemmed) RETURN m.preprocessed_stemmed AS preprocessed ORDER BY m.liteId ASC")
            # Note: Tests can be done in in-memory batches for large datasets
            compare_docs_bow = [corpus_memory_friendly.dictionary.doc2bow(doc['preprocessed']) for doc in compare_docs_query]
            compare_tfidf = tfidf[compare_docs_bow]
            index_tfidf = similarities.Similarity(output_prefix="sim_tfidf_idx", corpus=compare_tfidf, num_features=len(corpus_memory_friendly.dictionary))
            tfidf.save(index_tfidf_model_name)
            index_tfidf.save(index_tfidf_name)
        
        #TF-IDF Load (NST-Version) (Non-stemmed)
        if not os.path.isfile(index_tfidf_name_nst):
            # (This will be Feedly dashboards)
            tfidf_nst = models.TfidfModel((bow for bow in corpus_memory_friendly_NST), normalize=True)
            compare_docs_query_nst = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) RETURN m.preprocessed AS preprocessed ORDER BY m.liteId ASC")
            # Note: Tests can be done in in-memory batches for large datasets !
            compare_docs_bow_nst = [corpus_memory_friendly_NST.dictionary.doc2bow(doc['preprocessed']) for doc in compare_docs_query_nst]
            compare_tfidf_nst = tfidf_nst[compare_docs_bow_nst]
            index_tfidf_nst = similarities.Similarity(output_prefix="sim_tfidf_idx_nst", corpus=compare_tfidf_nst, num_features=len(corpus_memory_friendly_NST.dictionary))
            # will the length of features be same if we include the entire Feedly? (Nope)
            tfidf_nst.save(index_tfidf_model_name_nst)
            index_tfidf_nst.save(index_tfidf_name_nst)
        stemmed_tfidf = True #This variable defines whether to use the stemmed version or the nst-version in the application
        try:
            tfidf = models.TfidfModel.load(index_tfidf_model_name)
            index_tfidf = similarities.Similarity.load(index_tfidf_name)
            tfidf_nst = models.TfidfModel.load(index_tfidf_model_name_nst)
            index_tfidf_nst = similarities.Similarity.load(index_tfidf_name_nst)
        except RuntimeError:
            print("Something went wrong. Please check that the TF-IDF index exists.")
        evaluate_keywordsNeo4j()    #predict and extract the key-words from the articles (stem version)
        evaluate_keywordsNeo4jNST() #predict and extract the key-words from the articles (nst-version)
        
        #Latent Semantic Analysis (LSA) Load
        if not os.path.isfile(index_lsi_name):
            # (This will be Feedly dashboards)
            lsi = models.LsiModel(corpus_memory_friendly, id2word=corpus_memory_friendly.dictionary, num_topics=37) #to tune (49 prev)
            compare_docs_query = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed_stemmed) RETURN m.preprocessed_stemmed AS preprocessed ORDER BY m.liteId ASC")
            # Note: Tests can be done in in-memory batches for large datasets !
            compare_docs_bow = [corpus_memory_friendly.dictionary.doc2bow(doc['preprocessed']) for doc in compare_docs_query]
            compare_lsi = lsi[compare_docs_bow]
            index_lsi = similarities.Similarity(output_prefix="sim_lsi_idx", corpus=compare_lsi, num_features=len(corpus_memory_friendly.dictionary))
            lsi.save(index_lsi_model_name)
            index_lsi.save(index_lsi_name)
        
        #Latent Semantic Analysis (LSA) Load (NST-Version)
        if not os.path.isfile(index_lsi_name_nst):
            lsi_nst = models.LsiModel(corpus_memory_friendly_NST, id2word=corpus_memory_friendly_NST.dictionary, num_topics=40) #to tune (49 prev)
            compare_docs_query_nst = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) RETURN m.preprocessed AS preprocessed ORDER BY m.liteId ASC")
            # Note: Tests can be done in in-memory batches for large datasets !
            compare_docs_bow_nst = [corpus_memory_friendly_NST.dictionary.doc2bow(doc['preprocessed']) for doc in compare_docs_query_nst]
            compare_lsi_nst = lsi_nst[compare_docs_bow_nst]
            index_lsi_nst = similarities.Similarity(output_prefix="sim_lsi_nst_idx", corpus=compare_lsi_nst, num_features=len(corpus_memory_friendly_NST.dictionary))
            lsi_nst.save(index_lsi_model_name_nst)
            index_lsi_nst.save(index_lsi_name_nst)
        try:
            lsi = models.LsiModel.load(index_lsi_model_name)
            index_lsi = similarities.Similarity.load(index_lsi_name)
            lsi_nst = models.LsiModel.load(index_lsi_model_name_nst)
            index_lsi_nst = similarities.Similarity.load(index_lsi_name_nst)
        except RuntimeError:
            print("Something went wrong. Please check that the LSI index exists.")
        
        #Latent Dirichlet Allocation (LDA) Load
        if not os.path.isfile(index_lda_name):
            # (This will be Feedly dashboards)
            lda = models.LdaModel(corpus_memory_friendly, id2word=corpus_memory_friendly.dictionary, num_topics=12, passes=12, alpha='auto') #to tune, eval_every=5
            ldacompare_docs_query = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed_stemmed) RETURN m.preprocessed_stemmed AS preprocessed ORDER BY m.liteId ASC")
            # Note: Training can be done online for large datasets !
            compare_docs_bow = [corpus_memory_friendly.dictionary.doc2bow(doc['preprocessed']) for doc in ldacompare_docs_query]
            compare_lda = lda[compare_docs_bow]
            index_lda = similarities.Similarity(output_prefix="sim_lda_idx", corpus=compare_lda, num_features=len(corpus_memory_friendly.dictionary))
            # will the length of features be same if we include the entire Feedly?
            lda.save(index_lda_model_name)
            index_lda.save(index_lda_name)
        
        #Latent Dirichlet Allocation (LDA) Load (NST-Version)
        if not os.path.isfile(index_lda_name_nst):
            # (This will be Feedly dashboards)
            lda_nst = models.LdaModel(corpus_memory_friendly_NST, id2word=corpus_memory_friendly_NST.dictionary, num_topics=12, passes=10, alpha='auto') #to tune, eval_every=5
            ldacompare_docs_query = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed) RETURN m.preprocessed AS preprocessed ORDER BY m.liteId ASC")
            # Note: Training can be done online for large datasets !
            compare_docs_bow = [corpus_memory_friendly_NST.dictionary.doc2bow(doc['preprocessed']) for doc in ldacompare_docs_query]
            compare_lda = lda_nst[compare_docs_bow]
            index_lda_nst = similarities.Similarity(output_prefix="sim_lda_idx", corpus=compare_lda, num_features=len(corpus_memory_friendly_NST.dictionary))
            # will the length of features be same if we include the entire Feedly?
            lda_nst.save(index_lda_model_name_nst)
            index_lda_nst.save(index_lda_name_nst)
        try:
            #lda = models.LdaModel.load(os.path.join(dir_name, "LDA_MAX.model"))
            #index_lda = similarities.Similarity.load(os.path.join(dir_name, "LDA_MAX_INDEX.index"))
            lda = models.LdaModel.load(index_lda_model_name)
            index_lda = similarities.Similarity.load(index_lda_name)
            lda_nst = models.LdaModel.load(index_lda_model_name_nst)
            index_lda_nst = similarities.Similarity.load(index_lda_name_nst)
        except RuntimeError:
            print("Something went wrong. Please check that the LSI index exists.")
            
        # List-of-documents for the first visualization and Welcome to the application's GUI
        left_dlist = tk.Frame(inner_dwn_frame_b, bg='white')
        left_dlist.place(anchor='nw', relx=0.05, rely=0.05, relwidth=0.43, relheight=0.85)
        left_dlistcanvas = tk.Canvas(left_dlist, bg='white')
        left_scrollbar = tk.Scrollbar(left_dlist, orient='vertical')
        left_dlistframe = tk.Frame(left_dlistcanvas)
        window = left_dlistcanvas.create_window(0, 0, anchor='nw', window=left_dlistframe, width=314)
        left_scrollbar.pack(fill='y', side='right')
        LIST_OF_VIEWERS = []
        doc_insight = tk.IntVar() #for the Insight! functionality
        doc_insight.set(0) #set query document as pre-defined
        doc_var_idx = tk.IntVar() #to track the selected document in the left column
        LIST_DOCUMENTS = []
        LIST_NOT_FOUND_LABELS = []
        header_query = "MATCH (m:Article:_AI) WHERE EXISTS(m.content) RETURN m.title AS title, m.liteId AS LID ORDER BY m.liteId ASC"
        search_query = graph.run(header_query).data()
        for idd,doc in enumerate(search_query):
            r = tk.Radiobutton(left_dlistframe, text=doc['title'], selectcolor='#e6f2ff', wraplength=255, bg='white', relief='ridge', overrelief='ridge', indicatoron=False, variable=doc_var_idx, value=doc['LID'])
            r.config(font=(FONT_ARTICLES, 9))
            r.pack(anchor = 'w', fill='x')
            LIST_DOCUMENTS.append(r)
        left_dlistcanvas.update_idletasks()
        doc_var_idx.set(no_article)
        left_scrollbar.config(command=left_dlistcanvas.yview)
        left_dlistcanvas.configure(scrollregion=left_dlistcanvas.bbox('all'), yscrollcommand=left_scrollbar.set)
        left_dlistcanvas.pack(fill='both', side='left')
        left_dlistcanvas.bind('<Enter>', _bound_to_mousewheel)
        left_dlistcanvas.bind('<Leave>', _unbound_to_mousewheel)

        doc_sim_idx = tk.IntVar()
        LIST_SIM_DOCS = []

        # List of similar documents
        right_dlist = tk.Frame(inner_dwn_frame_b, bg='white')
        right_dlist.place(anchor='ne', relx=0.95, rely=0.05, relwidth=0.43, relheight=0.85)
        right_dlistcanvas = tk.Canvas(right_dlist, bg='white')
        right_scrollbar = tk.Scrollbar(right_dlist, orient='vertical')
        right_dlistframe = tk.Frame(right_dlistcanvas)
        rwindow = right_dlistcanvas.create_window(0, 0, anchor='nw', window=right_dlistframe, width=314)
        right_scrollbar.pack(fill='y', side='right')
        doc_sim_idx.set(no_article)
        right_scrollbar.config(command=right_dlistcanvas.yview)
        right_dlistcanvas.configure(scrollregion=right_dlistcanvas.bbox('all'), yscrollcommand=right_scrollbar.set)
        right_dlistcanvas.pack(fill='both', side='left')
        right_dlistcanvas.bind('<Enter>', _bound_to_mousewheelsim)
        right_dlistcanvas.bind('<Leave>', _unbound_to_mousewheelsim)
        
        search_button.config(command=lambda : search_documents())
        fcc_button.config(command=lambda: find_similardocuments())
        entry_title.bind("<Return>", keyentertitlecallback)
        entry_docs.bind("<Return>", keyenterdocsamountcallback)
    else:
        tk.messagebox.showerror("Error", "Error: The Graph database is not coherent. The LiteID do not match with the articles. Please, check Neo4j.")
        delete_database = tk.messagebox.askyesno("","Do you want to delete the database before exiting?\nNext time you open, it will be created again.")
        if delete_database:
            try:
                graph.run("MATCH (n:_AI) WHERE NOT '_GlobalConfigurationControl' IN labels(n) DETACH DELETE n")
                for the_file in os.listdir(dir_name):
                    file_path = os.path.join(dir_name, the_file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
            except Exception as e:
                tk.messagebox.showinfo("", "Couldn't delete the database. The error is:\n" + e)
            else:
                tk.messagebox.showinfo("", "Database deleted.")
        root.destroy()
root.mainloop()

#FUTURE UPDATE FUNCTION
def update_database_Neo4j():
    #TODO: API Request JSON, do the middle step conversion, load merge new documents, clean the ones without content, preprocess them, update the tf-idf model and the dictionary, assign keywords, assign LiteId to them 
    pass

###############################################################
###   Check in the dictionary for weird or composed words   ###
###############################################################

for idx,word in iteritems(corpus_memory_friendly_NST.dictionary):
    for punt in punctuation_marks_extended:
        if punt in word:
            print(word)
    if '-' in word:
        print(word)
    if(len(word) < 2): print(word)

###########################################
###   Calculator for different scores   ###
###########################################

Example_to_calculate = [4,2,'x'] # Ranks: [TF-IDF, LSA, Word_Embeddings]
#This is just a few ways to combine the results form the algorithms into one single weight. ('x' means that there is no recommendation from that algorithm)
MRR = 0
RBP = 0
p = 0.85
maxrank = 10
semscore = 0
occurrences = 0
length = len(Example_to_calculate)
for index, rank in enumerate(Example_to_calculate):
    if not isinstance(rank,int): Example_to_calculate[index] = maxrank
    else: occurrences += 1

for rank in Example_to_calculate:
    MRR += 1/rank
    semscore += (maxrank - rank)
    RBP += p**rank

#Normal Score used here
score = semscore/((maxrank - 1)*length)
print("The unmodified Score is: " + str(score))

#Modified Score with penalization for unmatching algorithms
final_score = ((occurrences-1)+semscore/((maxrank-1)*occurrences))/length
print("The modified Score: " + str(final_score))

#Mean reciprocal rank (MRR)
MRR /= length
print("The Mean Reciprocal Rank is: " + str(MRR))

#RBP score normalized
RBP /= p*length
print("The rank exponent score is: " + str(RBP))

############################################
###   Storing results in Matlab format   ###
############################################

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.io

export_MATLAB = False #Choose if you want to export some models to MATLAB format

if(export_MATLAB):
    
    #TF-IDF
    tf_idfmatrix = similarities.Similarity.load(index_tfidf_name)
    corpus_test = MyCorpusDashNeo()
    tfidf_test = models.TfidfModel.load(index_tfidf_model_name)
    index_tfidf_test = similarities.Similarity.load(index_tfidf_name)
    corpus_tfidf = tfidf_test[corpus_test]
    sim = index_tfidf_test[corpus_tfidf]
    scipy.io.savemat('matrix_tf_idf.mat', dict(x=sim))

    #LSA
    docs = graph.run("MATCH (m:Article:_AI) WHERE EXISTS(m.preprocessed_stemmed) RETURN m.preprocessed_stemmed AS preprocessed ORDER BY m.liteId ASC").data()
    bow_test = [corpus_memory_friendly.dictionary.doc2bow(doc['preprocessed']) for doc in docs]
    lsi = models.LsiModel.load(index_lsi_model_name)
    index = similarities.Similarity.load(index_lsi_name)
    docs_lsi = lsi[bow_test]
    simlsa = index[docs_lsi]
    scipy.io.savemat('matrix_lsa.mat', dict(x=simlsa))
    
    #Word Embeddings
    x = np.load(index_WE_model_eu_name_paper)
    scipy.io.savemat('matrix_euclidean_dist_paper.mat', dict(x=x))