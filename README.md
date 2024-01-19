# cs210-project 
## Hilal Sümeyra Aydoğdu
This is my Data science project using my own instagram likes, step counts and lecture hours data. 
You can see the website of my project [here](https://hilalsay.github.io/cs210-project/). This website includes the presentation of my project. In the analysis page if you click on the graphs you can see their explanations.

The code behind the graphs is available on the website. Also it is uploaded as .py and .jpynb if anyone wants to download them.

### Motivation
My motivation on this project was to discover how my daily activites were distributed. I was searching for relationships between my Instagram likes and step counts in my both busy days and holidays. The idea on my mind while starting this project was "Does my free time and lecture hours affect my Instagram likes and step counts?". It was interesting to see the results become visualized.
### Data Source
I got my Instagram likes data by downloading it from Instagram as a json file (liked_posts.json). Then I cleaned the data to make a proper data frame to use. I used the Samsung Healths app to retrieve my step counts data (com.samsung.shealth.activity.day_summary.20240115183382.csv). It was in csv format and I also cleaned and turned it into a data frame. I get the information of my daily lecture hours from my schedule and converted it to an xlsx file and then turned it into a data frame (schedule.xlsx). I combined data frames to use while analyzing.
### Data Analysis
In my analysis of datasets, I had a comparative approach, utilizing date information to explore trends on a weekly, monthly, and yearly basis. Additionally, I conducted comparisons between weekends and weekdays, as well as distinctions between summer and winter holidays versus school days.
### Findings
I've noticed different patterns in my data related to Instagram likes and step counts, particularly on weekends and in the days with more lectures. On weekends and holidays, when I had more free time my instagram likes were more. On these days, my step counts tend to be lower, especially when I spend more time indoors. Interestingly, on days with more lectures, my step counts are higher compared to other days. This is noticeable when I have physical lectures that require me to go out. However, during periods of hybrid lectures where I have the option to attend the classes online, I've observed a decrease in my step counts. It appears that I tend to stay indoors more when given the opportunity to attend lectures remotely. In summary, my Instagram likes aligns with periods of increased free time, while my step counts correlate with the days with more physical lectures. And step counts are lower during online or hybrid lecture periods.
### Limitations and Future Work
I could not reach to my Instagram usage time data. It could be more accurate to compare if I could use it. Also in machine learning part the predictions could be better with an advanced machine learning. My aim is to learn more about machine learning in the future and make this true. I would also like to improve my project and see if my step counts and Instagram likes would still be in the same relation next semester!
