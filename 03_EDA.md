---
nav_include: 1
title: EDA
notebook: 03_EDA.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}

## Exploration of Review Table

We first look at the review table with size (4166778, 9). The histogram below shows the distribution of stars from total reviews in yelp. 5-star reviews takes a great proportion.







![png](03_EDA_files/03_EDA_5_0.png)


We manipulate the date column to show the relationships between review scores and year / month / day.












![png](03_EDA_files/03_EDA_8_0.png)


The amount of reviews is increasing since yelp started (2017 is lower because there is not enough data). Similarly reviews received in the later months are a little bit fewer. The following is the relationships between reviews and date (month/day/year). 







![png](03_EDA_files/03_EDA_10_0.png)








![png](03_EDA_files/03_EDA_11_0.png)


There is no apparent pattern for stars w.r.t. month and day. For year as variable, at the beginning (2004/2005) most of the reviewes received high stars. However, the distribution becomes spread and polarization is severe. That means users may tend to give extreme reviews.







![png](03_EDA_files/03_EDA_13_0.png)


This is the correlation graph for stars of each review and three attributes received by this review. There are some interesting facts:

1. distribution of stars w.r.t. useful: reviews given by low stars are more likely to be agreed as 'useful' by other users.

2. useful, cool and funny has certain positive relationships. Perhaps yelp would rank this review at top by default so they are much easier to be commented by users.


## Exploration of Business Table






We explore the restaurants information by merging business talbe with data_review table. We get a table with size (4166778, 95). Here stars_x represent star given by single user and stars_y represent average stars of restaurant. We first have a glance at the distribution of restaurants' average stars (star_y) and wordcloud for 1-star and 5-star review.







![png](03_EDA_files/03_EDA_18_0.png)








![png](03_EDA_files/03_EDA_19_0.png)








![png](03_EDA_files/03_EDA_20_0.png)


Comparing the categories between 1-star and 5-star reviews, we found that they differ with each other to some extent. For example, american traditional appear more in 1-star review, while American new appear more in 5-star. 

















![png](03_EDA_files/03_EDA_24_0.png)







Both individual stars and average stars seem to have some links with location (longitude and latitude, similar postal code refers to places close with each other as well). There are clustering effect for some locations, but the pattern is not clear. So simply including the numeric values of location variables in the later analysis is unreasonable.







![png](03_EDA_files/03_EDA_27_0.png)


We compare individual stars, restaurants' average stars and review count in the pair plot.

1. Stars given by single user has positive relationships with average stars of the restaurants

2. Higher average star is usually related to higher review count. Reason might be customers would choose high star restaurant to eat and comment. But there is no clear pattern between individual w.r.t. review_count.

In the following, we explore the relationships between average stars and different attributes of the restaurants.







![png](03_EDA_files/03_EDA_30_0.png)








![png](03_EDA_files/03_EDA_31_0.png)


Some day of week such as Friday and Saturday have higher average stars and higher reviews.







![png](03_EDA_files/03_EDA_33_0.png)








![png](03_EDA_files/03_EDA_34_0.png)


High level ambience might have fewer reviews but they also relate to higher stars. Most of the restaurants are casual and trendy, because high-end environment require more effort invested by restaurants.







![png](03_EDA_files/03_EDA_36_0.png)








![png](03_EDA_files/03_EDA_37_0.png)


The distributions for food restriction vary but star seem not to be related to these varialbes.







![png](03_EDA_files/03_EDA_39_0.png)


Average score for restaurants good for dancing is relatively lower compared to other type. The reason might be they are usually indicated as noisy bar/pubs, which corresponds with our result in ambience.







![png](03_EDA_files/03_EDA_41_0.png)








![png](03_EDA_files/03_EDA_42_0.png)


Similar with results in "ambience" and "good for", upscale enviroment with live music have higher stars than those with viedo, dj or karaoke, which refers to noisy and crowded enviroment.

In the following, we select some interesting attributes that might indicate various stars from the corresponding boxplot.







![png](03_EDA_files/03_EDA_45_0.png)


The pattern for stars distribution is different from now (open) and past (closed). Users are more critique (lower gap between five levels) at present.

## Exploration of User Table






We analyze the users' information by merging user table and data_review table. We try to find relationships between star and users' properties of this merged table with shape (4166778, 24).







![png](03_EDA_files/03_EDA_50_0.png)


1. stars w.r.t. fans / review counts:
    Except for some outliners, users with more fans are more likely to give higher stars. The reason might be users with more followers are food bloggers, who is passionate about finding good restaurants. This phenonmen is less apparent for review counts.
    
2. stars w.r.t. users average_star:
    Users giving higher scores on average tend to have a positive review for each review.







![png](03_EDA_files/03_EDA_52_0.png)


Higher score comments have more probability to be complimented as useful/funny/cool. Also, there are positive relationship between useful/funny/cool reviews. This may refers active users who prefer to comment on other's reviews.







![png](03_EDA_files/03_EDA_54_0.png)


Most of users have utilized yelp for 2 - 7 years. Very old and loyalty yelp users tend to give higher and concentratated scores than new users (0-1 year). 






We select some attributes of complement for analysis while others might have less obvious patterns with other variables. We could see there are positive relationships between these variables.












![png](03_EDA_files/03_EDA_59_0.png)


The number of elite users is increasing, but the average scores (one user per count) given by them don't vary a lot.