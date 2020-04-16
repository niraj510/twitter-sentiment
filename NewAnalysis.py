import pandas as pd
import tweepy as twt
import sys,tweepy,csv,re
from textblob import TextBlob
import matplotlib.pyplot as plt


import tweepy
from textblob import TextBlob
from wordcloud import WordCloud 
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')  

#Twitter API credentials
consumer_key = "Lnv6QWPmxipWi7E84iL0IOLZr"
consumer_secret = "No0unRu7Lu3C7igI4J26t3riOcTDY4PniaobYU0GZA0zqRneTi"
access_key = "1232972624696770560-IWfDQ5PTc042dnAa9v1nosSXBBiga7"
access_secret = "KTuIwHxhU97SmPnbNPMZss6U6CzZa85ABrt2QmdiT7Im1"

alltweets = []	


class Test:

    def __init__(self):
        self.tweets = []
        self.tweetText = []

    def get_all_tweets(self,screen_name):
        auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
        auth.set_access_token(access_key, access_secret)
        api = tweepy.API(auth)
        new_tweets = api.user_timeline(screen_name = screen_name,count=1000)
        alltweets.extend(new_tweets)
        
        oldest = alltweets[-1].id - 1
        while len(new_tweets)>0:
            new_tweets = api.user_timeline(screen_name = screen_name,count=10,max_id=50)
            #save most recent tweets
            alltweets.extend(new_tweets)
            #update the id of the oldest tweet less one
            oldest = alltweets[-1].id - 1
            print ("...%s tweets downloaded so far" % (len(alltweets)))                # tweet.get('user', {}).get('location', {})
     
        outtweets = [[tweet.created_at,tweet.entities["hashtags"],tweet.entities["user_mentions"],tweet.favorite_count,
                      tweet.geo,tweet.id_str,tweet.lang,tweet.place,tweet.retweet_count,tweet.retweeted,tweet.source,tweet.text,
                      tweet._json["user"]["location"],tweet._json["user"]["name"],tweet._json["user"]["time_zone"],
                      tweet._json["user"]["utc_offset"]] for tweet in alltweets]
        
        import pandas as pd
        tweets_df = pd.DataFrame(columns = ["time","hashtags","user_mentions","favorite_count",
                                        "geo","id_str","lang","place","retweet_count","retweeted","source",
                                        "text","location","name","time_zone","utc_offset"])
        tweets_df["time"]  = pd.Series([str(i[0]) for i in outtweets])
        tweets_df["hashtags"] = pd.Series([str(i[1]) for i in outtweets])
        tweets_df["user_mentions"] = pd.Series([str(i[2]) for i in outtweets])
        tweets_df["favorite_count"] = pd.Series([str(i[3]) for i in outtweets])
        tweets_df["geo"] = pd.Series([str(i[4]) for i in outtweets])
        tweets_df["id_str"] = pd.Series([str(i[5]) for i in outtweets])
        tweets_df["lang"] = pd.Series([str(i[6]) for i in outtweets])
        tweets_df["place"] = pd.Series([str(i[7]) for i in outtweets])
        tweets_df["retweet_count"] = pd.Series([str(i[8]) for i in outtweets])
        tweets_df["retweeted"] = pd.Series([str(i[9]) for i in outtweets])
        tweets_df["source"] = pd.Series([str(i[10]) for i in outtweets])
        tweets_df["text"] = pd.Series([str(i[11]) for i in outtweets])
        tweets_df["location"] = pd.Series([str(i[12]) for i in outtweets])
        tweets_df["name"] = pd.Series([str(i[13]) for i in outtweets])
        tweets_df["time_zone"] = pd.Series([str(i[14]) for i in outtweets])
        tweets_df["utc_offset"] = pd.Series([str(i[15]) for i in outtweets])
        tweets_df.to_csv(screen_name+"_tweets.csv")
        DownloadData(self,alltweets)
        return tweets_df



def DownloadData(self,tweets):
   #input for term to be searched and how many tweets to search
    NoOfTerms = int(input("Enter how many tweets to search:"))

    # searching for tweets
    # Open/create a file to append data to
    #csvFile = open('result.csv', 'a')

    # Use csv writer
    #csvWriter = csv.writer(csvFile) 
    
    polarity = 0
    positive = 0
    wpositive = 0
    spositive = 0
    negative = 0
    wnegative = 0
    snegative = 0
    neutral = 0


    # iterating through tweets fetched
    for tweet in tweets:
        #Append to temp so that we can store in csv later. I use encode UTF-8
        self.tweetText.append(cleanTweet(self,tweet.text).encode('utf-8'))
        # print (tweet.text.translate(non_bmp_map))    #print tweet's text
        analysis = TextBlob(tweet.text)
        # print(analysis.sentiment)  # print tweet's polarity
        polarity += analysis.sentiment.polarity  # adding up polarities to find the average later

        if (analysis.sentiment.polarity == 0):  # adding reaction of how people are reacting to find average later
            neutral += 1
        elif (analysis.sentiment.polarity > 0 and analysis.sentiment.polarity <= 0.3):
            wpositive += 1
        elif (analysis.sentiment.polarity > 0.3 and analysis.sentiment.polarity <= 0.6):
            positive += 1
        elif (analysis.sentiment.polarity > 0.6 and analysis.sentiment.polarity <= 1):
            spositive += 1
        elif (analysis.sentiment.polarity > -0.3 and analysis.sentiment.polarity <= 0):
            wnegative += 1
        elif (analysis.sentiment.polarity > -0.6 and analysis.sentiment.polarity <= -0.3):
            negative += 1
        elif (analysis.sentiment.polarity > -1 and analysis.sentiment.polarity <= -0.6):
            snegative += 1


    # Write to csv and close csv file
    #csvWriter.writerow(self.tweetText)
    #csvFile.close()

    # finding average of how people are reacting
    positive = percentage(self,positive, NoOfTerms)
    wpositive = percentage(self,wpositive, NoOfTerms)
    spositive = percentage(self,spositive, NoOfTerms)
    negative = percentage(self,negative, NoOfTerms)
    wnegative = percentage(self,wnegative, NoOfTerms)
    snegative = percentage(self,snegative, NoOfTerms)
    neutral = percentage(self,neutral, NoOfTerms)

    # finding average reaction
    polarity = polarity / NoOfTerms

    # printing out data
    #print("How people are reacting on " + searchTerm + " by analyzing " + str(NoOfTerms) + " tweets.")
    print()
    print("General Report: ")

    if (polarity == 0):
        print("Neutral")
    elif (polarity > 0 and polarity <= 0.3):
        print("Weakly Positive")
    elif (polarity > 0.3 and polarity <= 0.6):
        print("Positive")
    elif (polarity > 0.6 and polarity <= 1):
        print("Strongly Positive")
    elif (polarity > -0.3 and polarity <= 0):
        print("Weakly Negative")
    elif (polarity > -0.6 and polarity <= -0.3):
        print("Negative")
    elif (polarity > -1 and polarity <= -0.6):
        print("Strongly Negative")
    
    print()
    print("Detailed Report: ")
    print(str(positive) + "% people thought it was positive")
    print(str(wpositive) + "% people thought it was weakly positive")
    print(str(spositive) + "% people thought it was strongly positive")
    print(str(negative) + "% people thought it was negative")
    print(str(wnegative) + "% people thought it was weakly negative")
    print(str(snegative) + "% people thought it was strongly negative")
    print(str(neutral) + "% people thought it was neutral")
    plotPieChart(self,positive, wpositive, spositive, negative, wnegative, snegative, neutral)


def cleanTweet(self,tweet):
    # Remove Links, Special Characters etc from tweet
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())

# function to calculate percentage
def percentage(self,part, whole):
    temp = 100 * float(part) / float(whole)
    return format(temp, '.2f')

def plotPieChart(self,positive, wpositive, spositive, negative, wnegative, snegative, neutral):
    labels = ['Positive [' + str(positive) + '%]', 'Weakly Positive [' + str(wpositive) + '%]','Strongly Positive [' + str(spositive) + '%]', 'Neutral [' + str(neutral) + '%]',
              'Negative [' + str(negative) + '%]', 'Weakly Negative [' + str(wnegative) + '%]', 'Strongly Negative [' + str(snegative) + '%]']
    sizes = [positive, wpositive, spositive, neutral, negative, wnegative, snegative]
    colors = ['yellowgreen','lightgreen','darkgreen', 'gold', 'red','lightsalmon','darkred']
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.legend(patches, labels, loc="best")
    #plt.title('How people are reacting on ' + searchTerm + ' by analyzing ' + str(noOfSearchTerms) + ' Tweets.')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()



if __name__== "__main__":
    test = Test()
    I_srk = test.get_all_tweets("CORONA")
    
    
  

#print the last 5 tweets from the account
print("show the 5 recent tweets:\n")
i=1
for tweet in alltweets[0:5]:
    print(str(i)+')'+ tweet.text + '\n')
    i=i+1
    
#create a dataframe with a column called tweets
df = pd.DataFrame([tweet.text for tweet in alltweets],columns=['Tweets'])

#show the first 5 rows of data
df.head()

#Clean the text    
#Create a function to clean the tweets
def cleanTxt(text):
    text =re.sub(r'@[A-Za-z0-9]+','',text) #removing @ Mentios
    text =re.sub(r"#",'',text) #removing the # symbol
    text =re.sub(r'RT[\s]+','',text)     #removing RT
    text =re.sub(r'https?:\/\/\S+','',text) #removing the hyper link
    return text

#Cleaning the text               
df['Tweets']=df['Tweets'].apply(cleanTxt)

#show the cleaned text
df     

#Create a function to get the Subjectivity
def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity
        
#Create a function to get the Polarity
def getPolarity(text):
   return TextBlob(text).sentiment.polarity

# Create two new columns
df['Subjectivity']=df['Tweets'].apply(getSubjectivity)
df['Polarity']=df['Tweets'].apply(getPolarity)

#show the new dataframe with the new columns 
df

#Plot the Word Cloud
allWords=' '.join([twts for twts in df["Tweets"]])
wordCloud=WordCloud(width=500,height=300,random_state=21,max_font_size=119).generate(allWords)

plt.imshow(wordCloud,interpolation="bilinear")
plt.axis('off')
plt.show

# Create a function to compute the negative,neutral and Positive analysis
def getAnalysis(score):
   if score<0:
       return "Negative"
   elif score ==0:
       return "Neutral"
   else:
       return "Positive"
   
df["Analysis"]=df['Polarity'].apply(getAnalysis)    

#show the dataframe   
df   

#print all of the positive tweets
j=1
sortedDF=df.sort_values(by=['Polarity'])
for i in range (0,sortedDF.shape[0]):
    if(sortedDF['Analysis'][i]=='Positive'):
        print(str(i)+')'+ sortedDF['Tweets'][i])
        print()
        j=j+1
     
#print all of the negative tweets
j=1
sortedDF=df.sort_values(by=['Polarity'], ascending='False')
for i in range (0,sortedDF.shape[0]):
    if(sortedDF['Analysis'][i]=='Negative'):
        print(str(i)+')'+ sortedDF['Tweets'][i])
        print()
        j=j+1         

#Plot the Polarity and Subjectivity by Scatter Plot
plt.figure(figsize=(8,6))
for i in range(0,df.shape[0]):
    plt.scatter(df['Polarity'][i],df['Subjectivity'][i],color='Blue')
    
plt.title('Sentiment Analysis')
plt.xlabel("Polarity")
plt.ylabel('Subjectivity')
plt.show()    
            
#Get the percentage of Positive tweets
ptweets = df[df.Analysis=='Positive']
ptweets = ptweets["Tweets"]
ptweets
round((ptweets.shape[0]/df.shape[0])*100,1) 

#Get the percentage of Negative tweets
ntweets = df[df.Analysis=='Negative']
ntweets = ntweets["Tweets"]
ntweets
round((ntweets.shape[0]/df.shape[0])*100,1)

#Get the percentage of Neutral tweets
neutweets = df[df.Analysis=='Neutral']
neutweets = neutweets["Tweets"]
neutweets
round((neutweets.shape[0]/df.shape[0])*100,1)
 
#Plot and Visualization the counts
plt.title('Sentiment Analysis') 
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind='bar')
plt.show
