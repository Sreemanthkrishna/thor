# https://www.kaggle.com/wendykan/sf-crime/don-t-know-what-i-want-to-do-yet
import pandas as pd

import matplotlib.pyplot as plt

train = pd.read_csv('crime.csv', header=0, parse_dates=['OCCURRED_ON_DATE'])


train['Year'] = train['OCCURRED_ON_DATE'].map(lambda x: x.year)
train['Week'] = train['OCCURRED_ON_DATE'].map(lambda x: x.week)
train['Hour'] = train['OCCURRED_ON_DATE'].map(lambda x: x.hour)

train['OFFENSE_DESCRIPTION'] = 1
hourly_district_events = train[['DISTRICT', 'HOUR', 'OFFENSE_DESCRIPTION']].groupby(['DISTRICT', 'HOUR']).count().reset_index()
hourly_district_events_pivot = hourly_district_events.pivot(index='HOUR', columns='DISTRICT',
                                                            values='OFFENSE_DESCRIPTION').fillna(method='ffill')
hourly_district_events_pivot.interpolate().plot(title='number of cases hourly by district', figsize=(10,6))

plt.show()
#plt.savefig('hourly_events_by_district.png')
