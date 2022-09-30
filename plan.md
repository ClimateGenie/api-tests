The goal of these tests is to simulate a some number of users using the extention at one and track their experience.

First we generate user objects, where each user has a random browsing pattern taken from reddits r/news.
We want to simulate a 'check phone in the morning' scenario, where everone will go on social media and read the top posts for the day, and this will go on for 20 minutes or so.

By taking the N users and using the bayers filter, we can determine which of the sites will get an api call, and add the api call to a list with the time it should be sent.
We can then run the program for 20 mins and see the length of time it takes for each api call to be handeled.

Using these results we will have a good picture of the number of concurrent users that we can have at once.
