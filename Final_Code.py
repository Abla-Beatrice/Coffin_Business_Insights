#!/usr/bin/env python
# coding: utf-8

# ## 1. Data Preparation 

# In[1]:


# Dependencies and Setup 
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import csv
import os
import requests
import time
import sys
import calendar
from scipy.stats import linregress
import statsmodels.api as sm
import seaborn; seaborn.set()
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt


# In[2]:


# Loading our CSV file containing data for 2014-2019
file_1 = "Data_files/Counts_of_Deaths_by_Select_Causes__2014-2019.csv"

# Read the data back into the dataframe from the csv file
df1 = pd.read_csv(file_1)

df1.head()


# In[3]:


# Loading our CSV file containing data for 2020-2023
file_2 = "Data_files/Counts_of_Deaths_by_Select_Causes__2020-2023.csv"

# Read the data back into the dataframe from the csv file
data_2 = pd.read_csv(file_2)
#data_2.head()

# Standardizing the two non-matching attribute names across both source data files
#data_2["Nephritis, Nephrotic Syndrome and Nephrosis"] = data_2["Nephritis, Nephrotic Syndrome, and Nephrosis"]
df2 = data_2.rename(columns = {"Nephritis, Nephrotic Syndrome and Nephrosis":"Nephritis, Nephrotic Syndrome, and Nephrosis", "Symptoms, Signs and Abnormal Clinical and Laboratory Findings, Not Elsewhere Classified": "Symptoms, Signs, and Abnormal Clinical and Laboratory Findings, Not Elsewhere Classified"}) 

df2.head()


# In[4]:


# Concatenating the two source data files
data = pd.concat([df1, df2], ignore_index=True)
data.head()


# In[5]:


#Renaming column for simplification
data = data.rename(columns={"Symptoms, Signs, and Abnormal Clinical and Laboratory Findings, Not Elsewhere Classified":"Unclassified"})

# Removing the columns and rows that we don't want included in the analysis

# Dropping unecessary columns
data = data.drop(columns=["Jurisdiction of Occurrence","Start Date","End Date", "Data As Of", "flag_accid", "flag_mva", "flag_suic", "flag_homic", "flag_drugod"])


# Dropping non-chronic illness columns (but keeping Covid)
data = data.drop(columns=["All Cause", "Natural Cause", "Influenza and Pneumonia", "Other Diseases of Respiratory System", "Unclassified", "Drug Overdose","Assault (Homicide)", "Intentional Self-Harm (Suicide)", "Motor Vehicle Accidents", "Accidents (Unintentional Injuries)"])

# Removing months in 2023. Adjust as needed.
#data = data.drop(["2023-Jan", "2023-Feb", "2023-Mar", "2023-Apr", "2023-May", "2023-Jun", "2023-Jul", "2023-Aug"])
recentmonths = range(108,116)
data = data.drop(recentmonths)

data.head()


# In[6]:


data.tail(10)


# ## 2. Data Quality Check and Cleaning

# In[7]:


# Summary statistics 

data.describe()


# In[8]:


#Replace "NaN" with 0
data = data.fillna(0)

data.head()
#data.tail()


# In[9]:


# Updated summary statistics and data quality review
data.describe()


# In[10]:


# Creating total selected causes of death columns

total = data.sum(axis = 1) 
total_without_covid = data.sum(axis = 1) - data["COVID-19 (Multiple Cause of Death)"] - data["COVID-19 (Underlying Cause of Death)"]
print(total)


# In[11]:


# Adding total selected causes of death column to duplicate/alternate dataset

data["Total"] = total
data["Total w/o Covid"] = total_without_covid

data.head()


# In[12]:


# Importing file to csv

file = "data.csv"
data.to_csv(file, index=False)


# ## 3a. Analysis: With regards to the U.S. population, which top two chronic illnesses should we target based on their impact and trend?

# In[13]:


# Create a reference to relevant list of illnesses 
chronic_illnesses = ["Diseases of Heart", "Malignant Neoplasms","Septicemia","Diabetes Mellitus","Alzheimer Disease","Chronic Lower Respiratory Diseases","Nephritis, Nephrotic Syndrome, and Nephrosis","Cerebrovascular Diseases"]

# Aggregating deaths per year
deaths_per_year = data.groupby('Year')[chronic_illnesses].sum()

# Create total column
total_deaths_per_year = deaths_per_year.sum(axis = 1) 
deaths_per_year["Total"] = total_deaths_per_year 

deaths_per_year.head(9)


# In[14]:


deaths_per_year.describe()


# In[50]:


# # Plot the annual death counts for selected chronic illnesses (absolute)

deaths_per_year.plot( y = chronic_illnesses, figsize = (11,6.5))
plt.title("Annual Death Counts for Each Chronic Illness", fontsize= 14)
plt.ylabel("Number of deaths");

plt.savefig('Figures/Annual_count_for_hronic illnesses.png', dpi=300, bbox_inches='tight')


# In[51]:


# Plot the annual death counts for selected chronic illnesses (ratio of all selected chronic illnesses)

share_in_deaths = deaths_per_year.divide(deaths_per_year["Total"], axis = 0)
share_in_deaths.plot(y = chronic_illnesses, figsize = (11,6.5)); 
plt.title("Share of Total Annual Deaths for Each Chronic Illness", fontsize= 14)
plt.ylabel("% of Total Chronic Illness Deaths")

plt.savefig('Figures/Share_of_Total_Annual_Deaths_Chronic Illness.png', dpi=300, bbox_inches='tight')


# ## Summary:
# 
# For our analysis, we would focus our efforts on these top two chronic illnesses:
# 
# 1. Diseases of Heart: From the selected chronic illnesses, this illness is the largest cause of death, as of 2022, and the share of deaths caused by this illness has been increasing over recent years.
# 
# 2. Cerebrovascular Diseases: As of 2022, this illness is the third largest cause of death, and the share of deaths caused by this illness has been increasing over recent years.
# 

# ## 3b. Analysis: Should we adjust our marketing plan to account for seasonality?

# In[18]:


# Aggregating deaths per month
deaths_per_month = data.groupby(['Year', 'Month'])[chronic_illnesses].sum()

# Create total column
total_deaths_per_month = deaths_per_month.sum(axis = 1) 
deaths_per_month["Total"] = total_deaths_per_month

deaths_per_month.head(9)


# ### Standardizing Average Monthly Death Counts

# In[19]:


# Setting up analysis of annual averages and standardization
newdata = data.copy()
newdata.drop(columns = ["Septicemia", "Diabetes Mellitus", "Alzheimer Disease", "Malignant Neoplasms", "Chronic Lower Respiratory Diseases","Nephritis, Nephrotic Syndrome, and Nephrosis", "COVID-19 (Multiple Cause of Death)", "COVID-19 (Underlying Cause of Death)", "Total", "Total w/o Covid"], inplace = True)
newdata["Year"] = newdata["Year"].astype(int)
newdata["Month"] = newdata["Month"].astype(int)
year_avg = newdata.groupby(["Year"]).mean()
year_avg.drop(columns = "Month", inplace = True)

year_stddev = newdata.groupby(["Year"]).std()
year_stddev.drop(columns = "Month", inplace = True)
year_stddev


# In[20]:


by_year = newdata.reindex()
by_year.set_index(["Year", "Month"], inplace=True)
by_year


# In[54]:


import calendar

fig2 = plt.figure(figsize=(11, 4))
fig2.suptitle("Number of Death by Year", fontsize=14)

fig2_ax1 = fig2.add_subplot(1, 2, 1)
fig2_ax2 = fig2.add_subplot(1, 2, 2)

# Graph 1
years_data = by_year["Cerebrovascular Diseases"].groupby(level=0)
colors = [plt.cm.rainbow(i) for i in np.linspace(0, 1, len(years_data))]
for (year, group), color in zip(years_data, colors):
    group.plot(ax=fig2_ax1, color=color, label=year)
fig2_ax1.set_xticks(group.index.levels[1])
fig2_ax1.set_xticklabels([calendar.month_abbr[month] for month in group.index.levels[1]])
fig2_ax1.set_title("Cerebrovascular Diseases")
fig2_ax1.legend(ncol=3)

# Graph 2
years_data = by_year["Diseases of Heart"].groupby(level=0)
colors = [plt.cm.rainbow(i) for i in np.linspace(0, 1, len(years_data))]
for (year, group), color in zip(years_data, colors):
    group.plot(ax=fig2_ax2, color=color, label=year)
fig2_ax2.set_xticks(group.index.levels[1])
fig2_ax2.set_xticklabels([calendar.month_abbr[month] for month in group.index.levels[1]])
fig2_ax2.set_title("Diseases of Heart")
fig2_ax2.legend(ncol=3)
fig2.savefig('Figures/Line_Chart_Seasonality_by_Year.png', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()


# In[42]:


# Create a copy of the original dataset to store the averaged data
averages = newdata.copy()

# Determine the number of columns and rows in the 'year_avg' DataFrame
column_count = year_avg.shape[1]
year_count = year_avg.shape[0]

# Initialize column index for looping
k = 0

# Loop through each column in the 'year_avg' DataFrame
while k < column_count:
    # Start year for the inner loop
    y = 2014
    # Extract the column name based on current index 'k'
    name = year_avg.columns.values[k]
    
    # Loop through each year from 2014 to 2022
    while y < 2023:
        # Get the average value for the current year and column
        mu = year_avg.loc[y, name]
        
        # The below line, though commented, provides an alternate way to achieve the same result as the subsequent line
        #ratios[name] = np.where(ratios["Year"] == y, mu)
        
        # For each matching year in 'averages' DataFrame, update the respective column with the average value
        averages.loc[averages["Year"] == y, name] = mu.astype(int)
        
        # Move to the next year
        y = y + 1
    
    # Move to the next column
    k = k + 1
averages


# In[43]:


# Create a deep copy of the original dataset to store the standard deviation data.
# Deep copy ensures that changes to 'deviations' do not affect the original 'newdata' DataFrame.
deviations = newdata.copy(deep=True)

# Determine the number of columns and rows in the 'year_stddev' DataFrame.
column_count = year_stddev.shape[1]
year_count = year_stddev.shape[0]

# Initialize column index for looping.
k = 0

# Loop through each column in the 'year_stddev' DataFrame.
while k < column_count:
    # Start year for the inner loop.
    y = 2014
    
    # Extract the column name based on the current index 'k'.
    name = year_stddev.columns.values[k]
    
    # Loop through each year from 2014 to 2022.
    while y < 2023:
        # Get the standard deviation value for the current year and column.
        sdev = year_stddev.loc[y, name]
        
        # For each matching year in the 'deviations' DataFrame, 
        # update the respective column with the standard deviation value.
        deviations.loc[deviations["Year"] == y, name] = sdev.astype(int)
        
        # Move to the next year.
        y = y + 1
    
    # Move to the next column.
    k = k + 1
deviations


# In[44]:


# Calculate the standardized spread by subtracting 'averages' from 'newdata' 
# and then dividing by 'deviations'. This provides a measure of how many standard deviations 
# each data point deviates from the average.
standard_spread = (newdata.sub(averages)).div(deviations)

# Copy the 'Year' and 'Month' columns from 'averages' to 'standard_spread'.
standard_spread["Year"] = averages["Year"]
standard_spread["Month"] = averages["Month"]

# Convert the 'Year' and 'Month' columns to string data type.
standard_spread["Year"] = standard_spread["Year"].astype(str)
standard_spread["Month"] = standard_spread["Month"].astype(str)

# Combine 'Year' and 'Month' to create a 'date' column in the format "Year Month".
standard_spread["date"] = standard_spread["Year"] + " " + standard_spread["Month"]

# Convert the 'date' column strings to datetime objects using the specified format.
standard_spread.date = standard_spread["date"].map(lambda x: datetime.strptime(x,"%Y %m"))

# Sort the 'standard_spread' DataFrame based on the 'date' column.
standard_spread = standard_spread.sort_values(by="date")

# Set the 'date' column as the index for the 'standard_spread' DataFrame.
standard_spread = standard_spread.set_index(["date"])

# Display the 'standard_spread' DataFrame with the new updates.
standard_spread


# In[45]:


# Create a new figure with specified dimensions (12 by 4 inches).
fig1 = plt.figure(figsize=(12, 4))

# Set a title for the entire figure.
fig1.suptitle("Standardized Monthly Deviation for Each Year", fontsize=12, y=1.05)

# Create the first subplot (ax1) 
fig1_ax1 = fig1.add_subplot(1, 2, 1)

# Create the second subplot (ax2)
fig1_ax2 = fig1.add_subplot(1, 2, 2)

# Graph 1
# Plot the standardized data for "Cerebrovascular Diseases" on ax1.
fig1_ax1.plot(standard_spread["Cerebrovascular Diseases"], c="blue")
# Set the title for this subplot.
fig1_ax1.set_title("Cerebrovascular Diseases")

# Graph 2
# Plot the standardized data for "Diseases of Heart" on ax2.
fig1_ax2.plot(standard_spread["Diseases of Heart"], c="red")
# Set the title for this subplot.
fig1_ax2.set_title("Diseases of Heart")

# Save the figure to a file in the "Figures" folder.
fig1.savefig('Figures/Seasonalyty.png', dpi=300, bbox_inches='tight')


# ### Summary:
# There is certainly seasonality to justify specific marketing for winter seasons.

# ## 3c. Analysis: Since the onset of Covid-19, have there been any significant shifts in death trends regarding the top three chronic illness?

# In[53]:


# Initialize a new figure with specified dimensions (12 by 4 inches).
fig1b = plt.figure(figsize=(12, 4))

# Set a title for the figure.
fig1b.suptitle("Standardized Monthly Deviation for Diseases of Heart", fontsize=14)

# Add a subplot to the figure.
ax = fig1b.add_subplot()

# Plot the standardized data for "Diseases of Heart" with a red color.
ax.plot(standard_spread["Diseases of Heart"], c="red")

# Save the figure to a file in the "Figures" folder.
plt.savefig('Figures/Standardized_Monthly_Deviation_Diseases_of_Heart', dpi=300, bbox_inches='tight')


# In[48]:


# Initialize a new figure with specified dimensions (12 by 4 inches).
fig1b = plt.figure(figsize=(12, 4))

# Set a title for the figure.
fig1b.suptitle("Standardized Monthly Deviation for Cerebrovascular Diseases", fontsize=14)

# Add a subplot to the figure.
ax = fig1b.add_subplot()

# Plot the standardized data for "Cerebrovascular Diseases" with a blue color.
ax.plot(standard_spread["Cerebrovascular Diseases"], c="blue")

# Save the figure to a file in the "Figures" folder.
plt.savefig('Figures/Standardized_Monthly_Deviation_Cerebrovascular_Diseases', dpi=300, bbox_inches='tight')

