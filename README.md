# Uni Mannheim - Masters Thesis
Public repo for my masters thesis:

### Identification of Polysemous Entities in a Large Scale Database (WebIsALOD)

First of all the [WebIsALOD](http://data.dws.informatik.uni-mannheim.de/webisa/webisalod-instances.nq.gz) dataset should be downloaded, extracted and saved in the `data` folder.

1. Fix the dataset URI's:
     To fix the dataset URI's run the python script called `fix_dataset_uris.py`.

2. Extract concept documents files and save preprocessed clean files:

   To save the clean preprocessed files run the python script called `Read_And_Clean.py`.

3. Download Wikipedia data:

   Use the following script to download the latest Wikipedia English articles dump:

   ```bash
   curl –O https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
   ```

4. Preprocess Wikipedia data using [Gensim](<https://radimrehurek.com/gensim/index.html>):

   To preprocess the Wikipedia data use the [Gensim](<https://radimrehurek.com/gensim/index.html>)'s script: 

   ```bash
   python -m gensim.scripts.make_wiki
   ```

5. Train LDA model with Wikipedia data:

   `wiki_wordids.txt` and `wiki_tfidf.mm` files generated in the previous step are required by the models using Wikipedia data.

   To train the LDA models with Wikipedia data, run the python script called `wiki_lda.py`.

6. Train LDA model with [WebIsALOD](http://data.dws.informatik.uni-mannheim.de/webisa/webisalod-instances.nq.gz) data:

     To train the LDA models with [WebIsALOD](http://data.dws.informatik.uni-mannheim.de/webisa/webisalod-instances.nq.gz) data, run the python script called `webisalod_lda.py`.

7. Train HDP model:

     To train the LDA models with Wikipedia data, run the python script called `wiki_hdp.py`.

8. Classification using only topic modeling:

     To run the classification model with only topic modeling, run the python script called `polysemous_words.py`.

9. Classification using topic modeling and supervised machine learning algorithms:

     To run the classification model with only topic modeling, run the python script called `supervised_classifier.py`.