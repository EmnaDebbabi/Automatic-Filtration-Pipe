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
commodities, currencies, crypto-assets, economic concepts and more. TextReveal analyses millions
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

TextReveal pipeline outputs 9 raw indicators that describe :
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




