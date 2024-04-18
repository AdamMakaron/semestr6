import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

car_crashes = sns.load_dataset('car_crashes')

speeding_threshold = car_crashes['speeding'].median()
alcohol_threshold = car_crashes['alcohol'].median()
distraction_threshold = car_crashes['not_distracted'].median()
previous_threshold = car_crashes['no_previous'].median()

car_crashes['Speeding_Involved'] = car_crashes['speeding'].apply(lambda x: 'Yes' if x > speeding_threshold else 'No')
car_crashes['Alcohol_Involved'] = car_crashes['alcohol'].apply(lambda x: 'Yes' if x > alcohol_threshold else 'No')
car_crashes['Distracted_Involved'] = car_crashes['not_distracted'].apply(lambda x: 'No' if x > distraction_threshold else 'Yes')
car_crashes['Previous_Accidents'] = car_crashes['no_previous'].apply(lambda x: 'No' if x > previous_threshold else 'Yes')

speeding_accidents = car_crashes[car_crashes['Speeding_Involved'] == 'Yes']['total'].sum()
alcohol_accidents = car_crashes[car_crashes['Alcohol_Involved'] == 'Yes']['total'].sum()
distracted_accidents = car_crashes[car_crashes['Distracted_Involved'] == 'Yes']['total'].sum()
previous_accidents = car_crashes[car_crashes['Previous_Accidents'] == 'Yes']['total'].sum()

plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 1)
sns.barplot(x='Speeding_Involved', y='total', data=car_crashes, color='red')
plt.title('Total Accidents Involving Speeding')
plt.xlabel('Speeding Involved')
plt.ylabel('Total Accidents')

plt.subplot(2, 2, 2)
sns.barplot(x='Alcohol_Involved', y='total', data=car_crashes, color='blue')
plt.title('Total Accidents Involving Alcohol')
plt.xlabel('Alcohol Involved')
plt.ylabel('Total Accidents')

plt.subplot(2, 2, 3)
sns.barplot(x='Distracted_Involved', y='total', data=car_crashes, color='green')
plt.title('Total Accidents Involving Distractions')
plt.xlabel('Distraction Involved')
plt.ylabel('Total Accidents')

plt.subplot(2, 2, 4)
sns.barplot(x='Previous_Accidents', y='total', data=car_crashes, color='purple')
plt.title('Total Accidents and Previous Accidents')
plt.xlabel('Previous Accidents Involved')
plt.ylabel('Total Accidents')

plt.tight_layout()
plt.show()

categories = ['Speeding', 'Alcohol', 'Distractions', 'Previous Accidents']
totals = [speeding_accidents, alcohol_accidents, distracted_accidents, previous_accidents]

plt.figure(figsize=(10, 8))
plt.pie(totals, labels=categories, autopct='%1.1f%%', startangle=90, colors=['red', 'blue', 'green', 'purple'])
plt.title('Proportion of Total Accidents by Factor')
plt.show()

numeric_data = car_crashes.select_dtypes(include=[np.number])  # Only numeric columns

correlation = numeric_data.corr()

mask = np.triu(np.ones_like(correlation, dtype=bool))

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', square=True, fmt=".2f")
plt.title('Correlation Matrix for Numeric Variables in the Car Crashes Dataset')
plt.show()

print("Można łatwo wywnioskować, że wypadki głównie są spowodowane alkoholem i szybką jazdą")
print("Często do tych dwóch dołączają inne czynniki, jak rozkojarzenie")
print("Ciekawym jest to, że większość wypadków zdarza się komuś po raz pierwszy, i mattrix pokazuje, że połączone jest to z alkoholem i nieuwagą")