import pandas as pd

#This reads the crime1 cvs file into pandas dataframe
crimeDataFrame= pd.read_csv("crime1.csv")

#Specifically gets the ViolentCrimesPerPop column from the crimeDataFrame
ViolentCrimesPerPopColumn = crimeDataFrame["ViolentCrimesPerPop"]

#Using pandas methods, this calculates the statistical measures needed to interpret the numerical data from the column
MeanValueOfViolentCrimesPerPop = ViolentCrimesPerPopColumn.mean()
MedianValueOfViolentCrimesPerPop = ViolentCrimesPerPopColumn.median()
StandardDeviationValueOfViolentCrimesPerPop = ViolentCrimesPerPopColumn.std()
MinimumValueOfViolentCrimesPerPop = ViolentCrimesPerPopColumn.min()
MaximumValueOfViolentCrimesPerPop = ViolentCrimesPerPopColumn.max()

#Displays the calculated statistical measures
print(f"Statistical Measures for ViolentCrimesPerPop Column:\n"
      f"Mean: {MeanValueOfViolentCrimesPerPop}\n"
      f"Median: {MedianValueOfViolentCrimesPerPop}\n"
      f"Standard Deviation: {StandardDeviationValueOfViolentCrimesPerPop}\n"
      f"Minimum Value: {MinimumValueOfViolentCrimesPerPop}\n"
      f"Maximum Value: {MaximumValueOfViolentCrimesPerPop}\n")

"""
Answers:
1. Compare the mean and median. Does the distribution look symmetric or skewed? Explain.

The mean has a value of approximately 0.44, while the median has a value of 0.39. This shows that the mean is greater than the median and
suggests that the distribution is skewed rather than symmetric. Specifically, the distribution appears to be right-skewed
We can conclude from this that it is skewed, specifically right-skewed as higher values pull the mean towards the right.

2. If there are extreme values (very large or very small), which statistic is more affected: mean or median? Explain why.

If there is an outlier, the mean value is more affected by extreme values than the median since the mean is the average of all values in a dataset. A single big or small 
outlier can significantly change the average. As per the median, it only depends on the middle value.

"""