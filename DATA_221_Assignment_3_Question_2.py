import pandas as pd
import matplotlib.pyplot as plt

#This reads the crime1 cvs file into pandas dataframe
crimeDataFrame= pd.read_csv("crime1.csv")

#Specifically gets the ViolentCrimesPerPop column from the crimeDataFrame
ViolentCrimesPerPopColumn = crimeDataFrame["ViolentCrimesPerPop"]

#Creates a histogram to show the frequency distribution of violent crimes per population
plt.hist(ViolentCrimesPerPopColumn, edgecolor = "white")

#Adds the title of the visualization
plt.title("Frequency Distribution of Violent Crimes Per Population")

#This creates labels for the x-axis and y-axis of the histogram
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Frequency")

#Displays the histogram with the title and labels
plt.show()

#Creates a box plot to display another way of showing the distribution of violent crimes per population
plt.boxplot(ViolentCrimesPerPopColumn)

#Adds the name of the box plot
plt.title("Boxplot for Violent Crimes Per Population")

#Created labels for the x-axis and y-axis
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Statistical Distribution")

#Displays the boxplot with the title and labels
plt.show()

"""
Based on the data from the ViolentCrimesPerPopulation column of the crime1.csv file, the histogram shows a 
right-skewed distribution. This indicates that the majority of 
violent crime rates falls within the lower region of the scale. This can also be seen in the box plot, where
the 25th to approximately 50th percentile range is located in the lower part of the distribution. The box plot 
also shows that the median value is at 0.39, which sits significantly close to the 50th percentile of the data. 
The box plot also confirms the absence of any outliers, as all data points falls within the lower and upper bounds of the distribution. 

"""
