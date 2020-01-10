# import relevant libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

crime_df = pd.read_csv("datafinal.csv")
crime_df.isnull().sum().any()
crime_df['SHOOTING'].value_counts()
updated_crimedf = crime_df.drop(['SHOOTING'], axis=1)
updated_crimedf.columns
cleaned_crimedf = updated_crimedf.dropna()
cleaned_crimedf["OCCURRED_ON_DATE"] = cleaned_crimedf["OCCURRED_ON_DATE"].apply(lambda x: \
datetime.strptime(x,"%d-%m-%Y %H:%M"))
cleaned_crimedf['DATE'] = [d.date() for d in cleaned_crimedf['OCCURRED_ON_DATE']]
cleaned_crimedf['TIME'] = [d.time() for d in cleaned_crimedf['OCCURRED_ON_DATE']]
cleaned_crimedf['YEAR_MONTH'] = pd.to_datetime(cleaned_crimedf['OCCURRED_ON_DATE']).dt.to_period('M')
crimedf = cleaned_crimedf
crimedf.OFFENSE_DESCRIPTION.unique()

crime_count_by_type = pd.DataFrame(crimedf.groupby('OFFENSE_DESCRIPTION').size().sort_values(ascending=False).rename('COUNT').reset_index())
crime_count_by_type.head(10)

sns.set(style="whitegrid")

# create the matplotlib figure
fig, ax = plt.subplots(figsize=(20, 10))

# plot the graph of number of crimes vs. OFFENSE_CODE_GROUP
# all types of OFFENSE_CODE_GROUP types will be plotted
barplot_alltypes = sns.barplot(x="OFFENSE_DESCRIPTION", y="COUNT", data=crime_count_by_type, color="g")
# set the axis labels
ax.set(ylabel="Number of Crimes", xlabel="Crime Type")
# rotate xticklabels
barplot_alltypes.set_xticklabels(barplot_alltypes.get_xticklabels(),
                        rotation=90,
                        fontweight='light',
                        fontsize='large'
                        )