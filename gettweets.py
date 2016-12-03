import tweepy
import csv #Import csv
import sqlite3
conn=sqlite3.connect('example.db')
c=conn.cursor()
#c.execute('''CREATE TABLE hillall(date text,tweet text)''')
c.execute('''CREATE TABLE trumpall(date text,tweet text)''')

auth = tweepy.auth.OAuthHandler('8HU9EeeJF92etEqzELpFweqXs', 'RnZNwBINAOTwTgAuNbSPmHwJhVpeLqlYstBClMZqRg6gKgPBO2')
auth.set_access_token('722406310986186752-pP5dGyK0ZENxGBl6abP6zxNqBUttKrv', 'CKMdC8r7Oa1i022Ep5kBIKpi5IqgYcq3xYjlYVxDs6Cq5') 
api = tweepy.API(auth) # Open/Create a file to append data
#csvFile = open('hillarypostelections.csv', 'a') #Use csv Writer
#csvWriter = csv.writer(csvFile)
hq="Hillary OR clinton #shewon OR #imwithher OR #electionnight OR #electionday OR #election2016 OR #uselection2016 OR #myvote2016 -filter:media -filter:images -filter:native_video -filter:links -filter:twimg -filter:vine -filter:periscope since:2016-11-09 until:2016-11-13"
tq="Donald OR Trump #maga OR #electionnight OR #electionday OR #election2016 OR #uselection2016 OR #myvote2016 -filter:media -filter:images -filter:native_video -filter:links -filter:twimg -filter:vine -filter:periscope since:2016-11-09 until:2016-11-13"
hqall="Hillary OR clinton OR #electionnight OR #electionday OR #election2016 OR #uselection2016 OR #myvote2016 -filter:media -filter:images -filter:native_video -filter:links -filter:twimg -filter:vine -filter:periscope since:2016-11-09 until:2016-11-13"
tqall="Donald OR Trump OR #electionnight OR #electionday OR #election2016 OR #uselection2016 OR #myvote2016 -filter:media -filter:images -filter:native_video -filter:links -filter:twimg -filter:vine -filter:periscope since:2016-11-09 until:2016-11-13"
count=0
for tweet in tweepy.Cursor(api.search, q=tqall, lang="en").items(300): 
#Write a row to the csv file/ I use encode utf-8
    if not tweet.retweeted and 'RT @' not in tweet.text and count<100:
        c.execute("insert into trumpall(date,tweet) values(?,?)",(tweet.created_at,tweet.text))
        #c.execute("insert into hillall(date,tweet) values(?,?)",(tweet.created_at,tweet.text))
        count=count+1
        #csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
#csvFile.close()
conn.commit()
conn.close()
