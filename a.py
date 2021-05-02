import re
import tweepy
from tweepy import OAuthHandler


# class tweet:
#     def init(self):
consumer_key = "FkTWTI6Iu34DekHnwuTXEpxI7"
consumer_secret = "WnWoF9xRpqGPAQvYc08OJwU6uKmHrIGsZmxCI7dnAlyDORAAwP"
access_key = "4670586138-uMKyCBEKSuxWoqb87970hA3XOpNwhCO7EAJrXuo"
access_secret = "y2ZYm6OthCpKe98tg6cWeAFQq69Fv066a24pkU1vFU9hs"
# try:
#     auth = OAuthHandler(consumer_key, consumer_secret)
#     auth.set_access_token(access_token, access_token_secret)
#     api = tweepy.API(self.auth)
#     print("Successfully")

# except:
#     print("Authentication Failed")


def get_tweets(username):

        # Authorization to consumer key and consumer secret
        # try:
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

            # Access to user's access key and access secret
            auth.set_access_token(access_key, access_secret)

            # Calling api
            api = tweepy.API(auth)

            # 200 tweets to be extracted
            number_of_tweets=8
            tweets = api.user_timeline(screen_name=username)

            # Empty Array
            tmp=[] 

            # create array of tweet information: username, 
            # tweet id, date/time, text
            tweets_for_csv = [tweet.text for tweet in tweets] # CSV file created 
            counter=0
            for j in tweets_for_csv:
                counter+=1
                # Appending tweets to the empty array tmp
                tmp.append(j) 
                if counter>7:
                    break
            return tmp        
            # Printing the tweets
    

# text to be sent
            # text = "chutiya"

# sending the direct message
        

        # except:
        #     print("Error!!")


if __name__ == '__main__':
    user_name=input("Please Enter the username: ")
    get_tweets(user_name)
