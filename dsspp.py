
# This Python file uses the following encoding: utf-8

import cgi

#from google.appengine.api import users

import webapp2

#import jinja2

import webob

import os

#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix

from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
import json, glob
import re, math, operator
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


from sklearn.preprocessing import normalize

import pickle

from random import randint

#template_dir = os.path.join(os.path.dirname(__file__), 'templates')
#jinja_env = jinja2.Environment(loader = jinja2.FileSystemLoader(template_dir), autoescape=True)


## The handler function that generates the main page 
def mainPage(request):
    #return webapp2.Response("index.html")
    return webapp2.Response("""<!doctype html>
<html lang="en">
 <head> <meta charset="utf-8">
   <title>D.S.S.P.P. On Python for GAE</title>
 </head>
 <body>
   <h1>Welcome to the DSSPP. Please choose your language:</h1>
   <form action="proc_form" method="POST">
     <table>
       <tr>
         <th>Language:</th>
         <td>English<input type="radio" name="language" value="English" checked="yes" /><br />
           Spanish<input type="radio" name="language" value="Spanish" /><br />
         </td>
       </tr>
     </table>
     <input type="submit" value="Submit" />
   </form>
 </body>
</html>
""")

def proc_form(request):

    language = request.params.get(cgi.escape('language'))
    if language == "English":
        return webapp2.Response("""<!doctype html>
<html lang="en">
<head>  <meta charset="utf-8">

<title> D.S.S.P.P. On Python For GAE </title>

</head>

  <body>
        <h3>Enter the word in English to find similar words for:</h3>
        <form method="GET" action="eng_results">

          <input type="text" name="word_string"><br />

          <input type="submit" name ="Submit">
        </form>

  </body>
</html>
""" )

    if language =="Spanish":
        return webapp2.Response("""<!doctype html>
<html lang="es">
<head> <meta charset="utf-8">

<title> D.S.S.P.P. En Python para GAE </title>

</head>

  <body>
        <h3> Ingrese la palabra en español para encontrar palabras similares para: </h3> 
         <form method="GET" action="esp_results">

          <input type="text" name="word_string"><br />

          <input type="submit" name ="Enviar">
        </form>

  </body>
</html>
""" )



def eng_results(request):

    word = request.params.get(cgi.escape('word_string'))

    #print(word)

    #print(cgi.escape('word_string'))
    #print(request.params)

    #print webob.multidict.getone('wordstring')

    vocabulary = pickle.load( open( 'vocabulary.pkl', 'rb' ) )

    count_vectorizer = pickle.load( open( 'count_vectorizer.pkl', 'rb' ) )

    tf_idf_matrix = pickle.load( open( 'tf_idf_matrix.pkl', 'rb' ))

    if word in vocabulary:
        word_index  = vocabulary[word]

        words = vocabulary.keys()

        distance_from_other_words = {}
        vector_1 = csr_matrix(tf_idf_matrix[:,word_index])
        vector_1 = np.array(vector_1.todense())

        N = len(vocabulary) 

        for i in range(N):
            try:
                if word==words[i]:
		    pass
		else:
		    vector_2 = tf_idf_matrix[:,i]
		    vector_2 = csr_matrix(tf_idf_matrix[:,i])
		    vector_2 = np.array(vector_2.todense())
		    distance_from_other_words[i] = cosine(vector_1, vector_2)
	    except KeyError:
                pass
        sorted_matrix = sorted(distance_from_other_words.items(), key=operator.itemgetter(1), reverse=True)
        results = []
        for item in sorted_matrix:
            w = words[item[0]]
	    score = item[1]
            if score !=1.0 and score !=0.0 and score >=0.9:
                results.append([w.encode('utf-8'), score])
        #results = [["Hello_%d"%i, np.random.random()] for i in  range(100)]
                #print "Word:", w," Score:",score
	#print len(results)
        rendered = ""

        x = range(1, len(results) + 1)
        y = [r[1] for r in results]
	plt.plot(x, y)
	plt.ylabel('Cosine Similarity')
	plt.title('Word Indices in decreasing order of cosine similarity')
        plt.xlabel('Word Index')

	plt.savefig("images/eng_results.png")
        plt.clf()
	
  	for result in results:
      	    rendered += "<tr><td>%s</td><td>%0.8f</td></tr>" % (result[0], result[1])
        return webapp2.Response("""<!doctype html>
          <html lang="en">
          <head>  <meta charset="utf-8">

          <style>
              td, th {
                text-align: center;
                border: 1px solid black;  
	      }
	      table {
                border-spacing: 0px;
              }
          </style>

          <title> D.S.S.P.P. On Python For GAE </title>

          </head>

            <body>
                  
                  <h1>List of similar words to given word:</h1>
		  <img src="/images/eng_results")/>
                  <table>
                  <tr>
                  	<th>Word</th>
                    <th>Score</th>
                  </tr>
                  %s
                  </table>

            </body>
          </html>
          """ % rendered)
    else:

        print "word not found in corpus. Cannot continue. Redirecting to home page."

        return webapp2.redirect('/')


def esp_results(request):

    word = request.params.get(cgi.escape('word_string'))

    #print(word)

    #print(cgi.escape('word_string'))
    #print(request.params)

    #print webob.multidict.getone('word_string')

    vocabulary = pickle.load( open( 'vocabulary_spanish.pkl', 'rb' ) )

    count_vectorizer = pickle.load( open( 'count_vectorizer_spanish.pkl', 'rb' ) )

    tf_idf_matrix = pickle.load( open( 'tf_idf_matrix_spanish.pkl', 'rb' ))

    if word in vocabulary:
        word_index  = vocabulary[word]

        words = vocabulary.keys()

        distance_from_other_words = {}
        vector_1 = csr_matrix(tf_idf_matrix[:,word_index])
        vector_1 = np.array(vector_1.todense())

        N = len(vocabulary) 

        for i in range(N):
            try:
                if word==words[i]:
		    pass
		else:
		    vector_2 = tf_idf_matrix[:,i]
		    vector_2 = csr_matrix(tf_idf_matrix[:,i])
		    vector_2 = np.array(vector_2.todense())
		    distance_from_other_words[i] = cosine(vector_1, vector_2)
	    except KeyError:
                pass
        sorted_matrix = sorted(distance_from_other_words.items(), key=operator.itemgetter(1), reverse=True)
        results = []
        for item in sorted_matrix:
            w = words[item[0]]
	    score = item[1]
            if score !=1.0 and score !=0.0 and score >=0.9:
                results.append([w.encode('utf-8'), score])
        #results = [["Hello_%d"%i, np.random.random()] for i in  range(100)]
                #print "Word:", w," Score:",score
	#print len(results)
        rendered = ""

        x = range(1, len(results) + 1)
        y = [r[1] for r in results]
	plt.plot(x, y)
	plt.ylabel('Similitud Coseno')
	plt.title('Índices de palabras en orden decreciente de similitud de coseno')
        plt.xlabel('Índice de palabras')

	plt.savefig("images/esp_results.png")
        plt.clf()
	
  	for result in results:
      	    rendered += "<tr><td>%s</td><td>%0.8f</td></tr>" % (result[0], result[1])
        return webapp2.Response("""<!doctype html>
         <html lang="en">
          <head>  <meta charset="utf-8">

          <style>
              td, th {
                text-align: center;
                border: 1px solid black;  
	      }
	      table {
                border-spacing: 0px;
              }
          </style>

          <title> D.S.S.P.P. En Python para GAE </title>

          </head>

            <body>
                  
                  <h1>Lista de palabras similares a la palabra dada:</h1>
		  <img src="/images/esp_results")/>
                  <table>
                  <tr>
                  	<th>Palabra</th>
                    <th>Puntuación</th>
                  </tr>
                  %s
                  </table>

            </body>
          </html>
          """ % rendered)
    else:

        print "palabra no encontrada en el corpus. No puede continuar. Redirigir a la página de inicio."

        return webapp2.redirect('/')


def failure_en():

      return webapp2.Response("""<!doctype html>
<html lang="en">
 <head> <meta charset="utf-8">
   <title>D.S.S.P.P. On Python for GAE</title>
 </head>
 <body>
   <h1>We are extremely sorry. The search word you entered was not found in the corpus. Please click on the link below to go back to the home page:</h1>
   <a href="mainPage">Home Page</a>
 </body>
</html>
""")

def failure_es():

    return webapp2.Response("""<!doctype html>
<html lang="en">
 <head> <meta charset="utf-8">
   <title>D.S.S.P.P. En Python para GAE</title>
 </head>
 <body>
   <h1>Lo sentimos mucho. La palabra de búsqueda que ingresó no se encontró en el corpus. Haga clic en el siguiente enlace para volver a la página de inicio:</h1>
   <a href="mainPage">Página de inicio</a>
 </body>
</html>
""")


def render_image(request):
    img = open('images/eng_results.png')
    resp = webapp2.Response()
    resp.headers['Content-Type'] = 'img/png'
    resp.body_file.write(img.read())
    img.close()
    return resp


def render_es_image(request):
    img = open('images/esp_results.png')
    resp = webapp2.Response()
    resp.headers['Content-Type'] = 'img/png'
    resp.body_file.write(img.read())
    img.close()
    return resp

 
application = webapp2.WSGIApplication([
    ('/', mainPage),
    ('/images/eng_results', render_image),
    ('/images/esp_results', render_es_image),
    ('/proc_form',proc_form),
    ('/eng_results',eng_results),
    ('/esp_results',esp_results)
], debug=True)

def main():

    from paste import httpserver
    httpserver.serve(application, host='127.0.0.1', port='8080')

    #application.run()

if __name__ == "__main__":
    main()
