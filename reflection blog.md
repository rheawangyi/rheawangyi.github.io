## Reflection Blog Post on PIC 16B Housing Project

* Overall, what did you achieve in your project?

My first task was to find data files related to races and inequality conditions in California, including incomes distribution by races and locations and poverty compositions by races. In order to connect races to housing prices, I merged several datasets to compare housing prices for each race's popular regions. Since I cannot find anything else relating to races and inequality, I decided to do some sentiment analysis and topic modeling on the overall housing discussions on Reddit. I webscrapped Reddit and do topic modeling on the overall housing discussions, topic modeling under each sentiment, and top frequent sentiment words on discussions. I also did the part of cleaning final dataset and drafting the user query function in which the user inputs ideal information and the funciton returns the list of housings.

* What are two aspects of your project that you are especially proud of? 

1. I am proud of that datasets of housing prices, locations, and races information can be merged together that enable me to connect races and prices together.
2. I am also proud of the process of doing topic modeling and simple sentiment analysis. There are interesting results that I did not expect. For instance, in the top frequent words of the overall housing discussions, I am surprised to see that there are a lot of music related words such as "mix" and "track".

* What are two things you would suggest doing to further improve your project? (You are not responsible for doing those things.) 

1. I would webscrap the housing text data on a larger base. I only webscraped for 100000 entries due to my computer's slow operation process, and such amount did not return a large data file especially after text cleaning and subsetting. I would try to clean my computer and webscrapte more data for analysis.
2. I would expand my sentiment analysis to calculate sentiment score for each sentiment. I also want to discover what are the closedness and connections between popular words.

* How does what you achieved compare to what you set out to do in your proposal? (if you didn't complete everything in your proposal, that's fine!

I find data sets relating to races, income, and housing prices in California and produced some plots that could partially reflect the social ineaquality in California. We also did the housing prices prediction part. In addition, we completed the algorithm that is able to recommend Californians or people who want to invest in Caflifornia Houses a housing neighborhood based on their preferences.

* What are three things you learned from the experience of completing your project? Data analysis techniques? Python packages? Git + GitHub? Etc? 

1. I learned to merge data sets in different ways based on our need.
2. I learned the skill of webscraping Reddit and draw wordcloud.
3. I learned the skill of basic topic modeling.
4. I learned to launch a local website and edit website information in html related language.
5. I learned that using different packages and methods for predictions would return different results. We need pick up the ideal method suitable for different situations. I also learned to use prophet to do predictions.
6. I learned to push and pull things on GitHub that allowed me to collaborate with others in a convenient way.
7. I learned that data analysis on social related stuff can be difficult in terms of searching for datasets and ways to generating useful plots. 


* How will your experience completing this project will help you in your future studies or career? Please be as specific as possible. 

This project let me to initiatively search for datasets and generate useful plots to do social analysis. This prompts me to learn methodologies of how to use computing skills and data analysis skills to effectively generate meaningful results on social studies in the future. The website skill also allows me to build and help maintain websites in the future using python in studies or work.I aslo learned that sentiment analysis and topic modeling can be applied on many other research or work such as digital humanites and trading. The experience prompts me to learn deeper skills in natural language processing to generate more meaningful results in future work. I also wish to apply this skill on my next year's internship on trading. 

