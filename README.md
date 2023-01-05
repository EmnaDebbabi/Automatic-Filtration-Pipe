# Automatic-Filtration-Pipe

## Automatic Filtration using NLP and scraping

we will proceed with deeper digging into the business and data understanding.
In this part we will proceed with deeper digging into the business and data understanding then data engineering and preparation process by presenting the proposed framework solution and its process with details and mention of
our dataset, the resources to work with and the inputs and outputs of our solution.

## Problematic

The Web has more than 1 billion websites and many more individual Web pages. The Internet is a
busy place. According to Internet Live Stats, a website of the global Real Time Statistics Project,
every second, over 6,000 tweets are sent, more than 40,000 Google searches are made, and more
than 2 million articles and blogs are published. Moreover, 5.3 billion people will have access to the
internet by 2023, which is about two-thirds of the world’s population. Furthermore, by 2023, they
predict that there will be more internet-connected devices than people.
However, if this information is used properly, it can produce a wide range of advantageous results. Due
to the large volume and diversity of data, the Information Extraction methods required to be updated.
Recent advancements in technology have made it possible to identify and summarize Extraction
problems using natural language processing methods which is a field of Artificial Intelligence (AI)
that enables machines to read, comprehend, and infer meaning from human languages. It plays a
significant role in a variety of software programs that we use on a regular basis.
Yet, it is still difficult to gather pertinent information from the internet that meets exactly our needs
while avoiding noise without the large amount of additional meaningless information and finally
storing this data.

## Study of the existing NLP pipeline:

The NLP pipeline that we used to extract insightful NLP indicators is based on an NLP engine that helps capture relevant, non-trivial information from text.
Develop and gain access to unique analytics and build custom indicators, such as sentiment, emo-
tions, or ESG scores on financial assets, public private companies, brands, products, executives,
commodities, currencies, crypto-assets, economic concepts and more. The NLP pipe analyses millions
of sources to extract sentiments. This feed aggregates messages daily and computes several aggre-
gated indicators suitable for signal construction. To do so, we extract NLP indicators for each asset.
The overall pipeline can be split into five main steps. Data is queried and analysed following the below
initial query shown in the figure below:

NLP pipeline workflow:


![image](https://user-images.githubusercontent.com/47029962/210668037-9fb2fa99-d67b-43f0-ad24-d59bb6f08919.png)

This pipeline consists of 3 main steps, the first of which is the data retrieval from datalake. In this step, the Knowledge Graph is particularly useful to make the search efficient and
relevant. Then there is data filtering. We disambiguate the retrieved documents and make sure
they hold relevance to the study. Finally, we perform sentiment analysis on the data, and generate
informative features from the raw sentiment indicators. Like it is presented in the following figure.

General overview:

![image](https://user-images.githubusercontent.com/47029962/210670198-2dd35df2-922a-4c9a-8e5f-a9a954e50866.png)

We will be interested in the first and second steps in the following part.

## Indicators

The NLP pipeline outputs 9 raw indicators that describe :
<ul>
<li> Volume </li>
<li> Sentiment (positive, negative) </li>
<li> Sentiment standard deviation (spread_positive, spread_negative) </li>
<li> Other indicators as: investor_sentiment, return_investor_sentiment, agreement, bullish_index </li>
</ul>

## Critics of the existing solutions


The previous NLP pipeline consists of queries as input yet even the best-designed schema won’t work well if
the queries are poor and bad quality. Moreover, manually examining the structured query language
of many different entities in an attempt to extract vast amounts of data in its entirety is not particularly effective due to the volume and dynamics present in social media and the web in general.
Since the manual queries creation demand a lot of time and resources consumption also the possible risk of making the queries complicated since the number of conditions to eliminate is uncountable.

##  Proposed solution

Where does it come the main operational goal of our proposed solution which is the development,
for a private equity client, a long- term financial indicators creation project for companies listed
or not on the stock exchange market by creating an automation process with NLP techniques to
filter and disambiguate companies automatically before creating sentiment and financial indicators.
Companies can use an automated crawler to monitor and automatically extract content from web.
Now, the data and its relationships with each other can be examined and categorized with NLP-based
text analysis and causality detection model.

## Project objectives and challenges

The project solution is mainly about developing a long-term financial indicators creation project for companies listed or not on the stock exchange market based on textual
data, sentiment and emotional analysis time series.
The principal aim is to create an automation process to filter and disambiguate companies using
NLP techniques before creating sentiment and financial indicators.
And then from using the data extracted within the previous part we aim implementing another
more efficient causality detection model using advanced NLP techniques that will be used for the
implementation of a Bayesian Network for macro-economic factors, based on causal data detection.

## Expected results

The work is expected to fulfil the following requirements:
<ul>
<li> Functional requirements:
– Create an automation process to filter and disambiguate companies
– Implement a causality detection model 
</li>
<li> 
• Non-Functional requirements:
– Performant: it gives accurate responses in a short amount of time.
– Robust: it attempts to handle erroneous inputs, in the worst-case scenario, it will not be
blocked.
– Reliable: it shall be functional when solicited.
– Accessible: it shall be accessible to every allowed employee or client.
</li> 
</ul>

## Business and data understanding
### Business understanding

Sentiment analysis as a term was first used by [Nasukawa and Yi, 2003], which has gained significant traction in the research world. Sentiment analysis is frequently used in a variety of industries,
including movie reviews., buying behavior in financial products, towards improvement in products
and services by analyzing customers’ review. Stock market changes in general and stock price movements in particular have sparked a lot of discussion and reactions from viewers, traders and investors.
Short-term stock market fluctuations are known to be significantly impacted by participant attitude.
Numerous research have emphasized sentiment analysis and model-based prediction in relation to
stock market movement. Investors’ sentiment on the company’s earnings announcements has an
impact on the price movement of the US stocks. To comprehend and foresee the impact of sentiments on the pricing of chosen sample US enterprises, firm-specific sentiments and general market
sentiments have been researched. Analysis of sentiment signals from experts based on financial news
resources feeds has shown better predictive power for stock market price movements.

### Data understanding
#### Source type

Data Lake includes :
<ul>
<li> 14B+ documents </li>
<li> 4M+ data sources </li>
<li> 12+ years history </li>
<li> 70M+ entities </li>
<li> Multilingual </li>
</ul>
Documents are varied, including - but not limited to - News, Blogs, Forums, Social Media and Social
Trading.

#### Coverage

Given private equity universe, our NLP data is interesting only after 2015 (there is a big shift
in the distribution of NLP data before and after this date). This corresponds to more than 1
billion documents (50 TB of data) retrieved since 2015 according to the scope of the private equity
companies. Our data covers the vast majority of private equity universe, and most of them have
consistent NLP data over time. To add more precision, we focus only on data obtained from reliable
financial news in English and we extract only articles that mention the entities which are mentioned
in the input file.

### Data engineering and preparation

In this part we will proceed with deeper digging into data engineering and preparation process
by presenting the proposed framework solution and its process with details and with mention of our
dataset, the resources to work with and the inputs and outputs of our solution.

### Proposed framework
Data feed from the companies list file provided by the client are used as input. To ensure a good
quality of results a whole phase of data collection and preparation must be done. Firstly data
collection is done by scraping. Secondly these file data pre-processing is done where removal of
punctuation and stopwords is done also tokenization and lemmatization is done. Henceforth, pre-processed data is used either in the first method that uses KG or in the second method without using
KG. After Candidates generation step of the NLP pipeline, filtering is done on which four NLP
techniques namely; TF-IDF, N-grams, Cosine similarity and NER are applied. Finally the feature
set is formatted under these techniques and ready to launch the query of the NLP pipeline on
which ML algorithms are applied to classify sentiment under any one of the categories which are;
Positive, Negative and Neutral. The flow chart of the proposed framework of sentiment analysis and
prediction is depicted in Figure as below

Proposed framework of automatic NLP indicators extracting:

![image](https://user-images.githubusercontent.com/47029962/210672356-4cc3939d-8e2d-492a-a648-035161adbe80.png)


Each step from the above proposed solution will be explained in the following paragraphs:

#### Data requirements

Companies list file is an external file that contains 100k companies listed or not in the stock market
to search via csv for example. Required parameters for the file are the following:
<ul>  
<li> id : Any unique identifier </li>
<li> name: The name of the company </li>
<li> description : A short description of your entity </li>
<li> website: The website of the company </li>
<li> crunchbase_id: The crunchbase identifier </li>
</ul>
We will start with a sample of 500 companies.

#### Data collection
Data collection is done threw scraping which is an automatic process of gathering information from
the Internet. The informations to complete here are empty descriptions of the companies list file
since it contains 40% of null descriptions. The process will search the corresponding company by its
name on different websites and will find the company’s name accompanied by a pattern to extract
text that present the company’s description. The used tools:
<ul>
<li> googlesearch-python: A Google search engine scraping package for Python.</li>
<li> BeautifulSoup: a Python library for extracting data from XML and HTML documents. </li>
<li> nltk-wordnet: An english dictionary which is part of the Natural Language Tool Kit (NLTK)
in Python. </li>
</ul>
##### Data preparation
In order to ensure and enhance performance, data pre-processing is done to the name, website and
description fields threw:
<ul>
<li> Remove punctuation: Remove certain characters from strings. </li>
<li> Remove stopwords: Set the search engine to ignore words that are often used but that can
still be safely discarded without changing the meaning of the sentence. </li>
<li> Tokenization: Splitting the body of the text where strings are converted to streams of token
objects. </li>
<li> Lemmatization: An algorithmic process of finding the lemma of a word depending on its
meaning and context which is the base or the dictionary form of a word without inflectional
endings. </li>
<li> Noun phrase extraction: To expand the keywords list -since the larger the keywords list
is, the more precise the results are- we extract all significant nouns from the given companies
descriptions using the python library Textblob because it has the best time and quality per-
formance; it is fast and it allows to detect compound nouns based on the experience done on
a company’s description sample data as it is shown in the table below. 

Degrees of libraries efficiency:

![image](https://user-images.githubusercontent.com/47029962/210737771-33299a04-74a1-4e89-bdd7-19f35c8a6aa4.png)


</li>

</ul>


#### Using Knowledge graph method
The first method is to request as many relevant documents as possible related to the targeted company
using the Knowledge Graph. The KG includes open, custom, and private structured knowledge
bases :
<ul>
<li> Wikidata (+70M entities) </li>
<li> Crunchbase ( 800k entities ) </li>
</ul>
Objective : Expand the documents requested to the data lake. The KG allows to link an entity to
several of its properties, making the search request more complete and relevant.

KG properties:

![image](https://user-images.githubusercontent.com/47029962/210738228-ab83b887-62ed-43f9-8b51-2e067857ef31.png)

As shown in Figure below we can leverage the Knowledge Graph to gather relevant information about
a company by using the properties that are linked to it. Most commonly used properties are CEO,Subsidiary and Products.

For example :

KG properties example:

![image](https://user-images.githubusercontent.com/47029962/210738432-586dedf8-56cf-407d-a75c-d1b25e95c736.png)

#### Using your own entities method
This method makes it possible to build you request automatically with external information via csv
for example. Required parameters for each entity are the following
Input parameters:
<ul>
<li> filePath: An external file that contains the companies list to search via csv for example.
Required parameters for the file are the following
<li>  id : Any unique identifier </li>
<li>  name: The name of the company </li>
<li>  description : A short description of your entity </li>
<li>  website: The website of the company </li>
<li>  crunchbase_id: The crunchbase identifier </li>
</ul>

#### Candidates generation
When choosing the first method that uses KG, this route makes it possible to return a list of
candidates following a search based on a fuzzy key. An example of payload and response is the
following figure:

Example of candidates generation response:

![image](https://user-images.githubusercontent.com/47029962/210739321-d14f596e-973c-452d-9b5d-cc1a5de49e8c.png)

#### Filtering

After Candidates generation step, Filtering is done on which four NLP techniques namely; TF-IDF,
N-grams, Cosine similarity and NER are applied. And in order to disambiguate the companies threw
applying filtering out among all the requested documents the ones that do not actually mention the
targeted company. This is done using the description matching and the Named Entity Recognition
(NER).

##### Description matching
Description matching is done to determine if the sentence in which the detected entity is mentioned
is actually referring to the target entity we are interested in. Its value ranges between 0 and 1, the
higher the value the closer the matching is. This score is used to filter out non relevant documents.
For the human reader it is obvious that both:

"Apple is a multinational corporation that designs, manufactures, and markets consumer electronics,
personal computers, and software."
and
"Apple Inc. designs, manufactures and markets smartphones, personal computers, tablets, wearables
and accessories, and sells a variety of related services."
are the descriptions of the same company. It is challenging to identify these nearly identical sequences
since for a computer, these are fundamentally distinct. Program Using TF-IDF with N-Grams as
terms to locate related strings is one approach to solving this issue. This turns the issue into a
matrix multiplication problem., which is computationally much cheaper than applying the traditional
approaches to string matching such as the [Winkler and Jaro, 1989] or [Levenshtein, 1965] distance
measure. Using this approach made it possible to search for near duplicates in a set of 663,000
company descriptions in 42 minutes using only a dual-core laptop.
Even if NER has a very high precision, the description matching is a very relevant second filtering
layer. In the example below, figure1.6 the word ’apple’ in the document has already been detected
by the NER as a "COMPANY" type entity. We need to ensure that this entity refers to Apple the
company and not the fruit. To do so, we compute the cosine distance between the embedding vector
of the sentence and the embedding of Apple’s description.

Example of description matching process:

![image](https://user-images.githubusercontent.com/47029962/210739620-6987c7f6-e0c6-4acd-8d32-28fc6671d910.png)

<ul>
<li> TF-IDF:

A method for extracting features from text that involves multiplying a term’s frequency in a
document (the Term Frequency, or TF) by its importance over the entire corpus (the Inverse
Document Frequency, or IDF). This last term gives words that are less important (such the,
it, and etc.) a lower weight and terms that are less common a higher weight. IDF is calculated
as:

IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

An example:

Consider a document containing 100 words in which the word "Apple" appears 3 times. The
term frequency (i.e., tf) for "Apple" is then (3 / 100) = 0.03. Now, assume we have 10 million
documents and the word "Apple" appears in one thousand of these. Then, the inverse document
frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the
product of these quantities: 0.03 * 4 = 0.12.

In text categorization and text clustering, TF-IDF is highly helpful. It is used to convert
written information into comparably simple numerical vectors.
</li>
<li> N-Grams
Although words are typically used as the terms in TF-IDF, this is not required. Since most
company names only contain one or two words, employing words as keywords wouldn’t be
very helpful in our situation. For this reason, we will use n-grams, which are collections of N
connected objects—in this case, characters.

N-grams in "Apple":

![image](https://user-images.githubusercontent.com/47029962/210740128-54cf5a00-fabe-42be-9177-ce3d0e20456d.png)

</li>
<li>
Cosine similarity

Cosine similarity is a metric used to determine how similar the descriptions are irrespective of
their size. Mathematically, it measures the cosine of the angle between two vectors of TF-IDF
values projected in a multi-dimensional space. It turns out, the closer the descriptions are by
angle, the higher is the Cosine Similarity (Cos theta). As it is shown in the following figure1.8.
Empirically, we notice that there is a threshold at which we can be sure of the relevance of a
text, this threshold can vary significantly depending on the source type.

Cosine similarity formula:

![image](https://user-images.githubusercontent.com/47029962/210743107-4ada339f-f33b-490e-b8c4-f45d9b34433e.png)

</li>
</ul>
##### Named Entity Recognition (NER)
NER allows to identify proper mentions in a text and classify them into a set of predefined categories
of interest. By using Spacy library which has the ’ner’ pipeline component that identifies token
spans that match a specified collection of named entities.
<ul>
<li> Non-comprehensive list of categories: Organizations, Persons, Concepts </li>
<li> Variation of Named Entities: Emmanuel Macron / Macron </li>
<li> Ambiguity of Named Entities: May (Person vs date vs verb) </li>

NER example:

![image](https://user-images.githubusercontent.com/47029962/210744265-34b00a22-0fb7-4e5e-87d5-fe4e6087cdc0.png)


</ul>

##### Data Formatting
Formating and transforming the enriched data to the desired output. Each document is formated
into jsonl file. Output format:
<ul>
<li> entity_of_interest: Unique id for the entity of interest </li>
<li> keywords: List of keywords to search  </li>
<li> context: Context description of the entity  </li>
</ul>

##### TR pipeline

Finally the feature set is formatted under these techniques and ready to launch the query of Tex-
treveal pipeline on which ML algorithms are applied to classify sentiment under any one of the
categories which are; Positive, Negative and Neutral. The sentiment is computed on all documents
and sentences that were selected after the disambiguation pipeline. This makes the sentiment score
more accurate because it is related to the targeted entity. This gives a sentiment score at the sentence
level. The daily sentiment score aggregates the sentence level sentiment scores from the same day
related to the targeted entity.

##### Job scheduling
Since each instant social media and web data are increasing, we need to schedule daily routine tasks
in order to take a new daily backup of companies data and that by scheduling tasks that can be
run at a given time or repeatedly, and they typically use the cron command, which is an expression
language popular on Unix systems. These are time-based event triggers that let programs plan tasks
to be carried out on specific dates or at specific times based on cron expressions.
##### Results
Let’s take an example for one company ’Tesla’; the pipeline will complete its description:’Tesla Inc.,
called Tesla Motors until 2017, is an automobile manufacturer of electric cars.’:

Candidates generation for Tesla example:

![image](https://user-images.githubusercontent.com/47029962/210750106-1f6821ad-27dd-4d1a-a2a0-1537278699e6.png)

The pipeline will filter and keep only the first item with item_id="Q478214" from the list of the
proposed items as it is shown in the Figure above. And after request creation and getting the
correspondent data from elasticsearch datalake, timeseries aggregation are generated for the given
instance. Also plotting signal using moving average for a given entity can be done as it is shown in
Figure below:

Display max anger signal for the entity Tesla:

![image](https://user-images.githubusercontent.com/47029962/210750267-b000162a-dd65-4f92-907a-e694bd10a875.png)


We notice from the previous graph that for the entity with the item_id: ’Q478214’ : The "max_anger"
reached a noticeable steady peak on 19 May 2020 before falling a little to reach another second peak
on 19 July 2020, the value of "max_anger" is increasing little by little throw months.
To understand the reason of the "max_anger" fluctuations, we have to extract the informations from
"text" that matches the correspondent dates.
From reading the texts that matches the first peak of 19 May 2020, we can understand that:
The biggest change at Tesla is that the Fremont, California, factory was forced to shut down on
March 23, and the Gigafactory in Nevada is cutting about three-quarters of its workforce. Without
vehicles coming off production lines, Tesla doesn’t have any revenue or margin, meaning cash burn
will likely be high in mid-2020. Moreover Elon Musk reveals Tesla’s electric Cybertruck and smashes
its windows.
From reading the texts that matches the second peak on 19 July 2020, we can understand that:
The stock of Tesla (TSLA) has been on a tear over recent weeks and months; Tesla had a negative
net margin of 0.55% and a negative return on equity of 1.86%. And the US looks into over 100
complaints of Tesla cars suddenly accelerating and crashing.

##### Concluding remarks and future scope
As we saw by visual inspection the cosine similarity gives a good indication of the similarity between
the two companies descriptions. The biggest advantage however, is the speed. By utilizing a dis-
tributed computing environment like Apache Spark, the method previously outlined can be scaled to
considerably bigger datasets. This might be accomplished by sending one of the TF-IDF matrices to
all employees, and parallelizing the second (in our case a copy of the TF-IDF matrix) into multiple
sub-matrices. Each worker on the system can then perform multiplication (using Numpy or the
sparse dot topn library) on the entire first matrix and on a part of the second matrix.




