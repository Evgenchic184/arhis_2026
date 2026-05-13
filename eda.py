import pandas as pd

df = pd.read_csv('cyberbullying_tweets.csv')

import matplotlib.pyplot as plt
import seaborn as sns

# Исходное распределение (6 классов)
print("Распределение по исходным классам:")
print(df['cyberbullying_type'].value_counts())
print(f"\nУникальные классы: {df['cyberbullying_type'].unique()}")

# Визуализация
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='cyberbullying_type', order=df['cyberbullying_type'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Распределение по классам кибербуллинга (исходное)')
plt.xlabel('Тип буллинга')
plt.ylabel('Количество')
plt.tight_layout()
plt.savefig('eda/class_distribution_original.png')
plt.show()

# Создание бинарного признака is_toxic
df['is_toxic'] = df['cyberbullying_type'].apply(
    lambda x: 0 if x == 'not_cyberbullying' else 1
)

# Распределение после маппинга
print("\nРаспределение после маппинга (binary):")
print(df['is_toxic'].value_counts())
print(f"\nБаланс классов: {df['is_toxic'].value_counts()[1] / df['is_toxic'].value_counts()[0]:.2f}")

# Визуализация бинарного распределения
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='is_toxic')
plt.xticks([0, 1], ['Not Toxic', 'Toxic'])
plt.title('Распределение по классам (бинарное)')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.tight_layout()
plt.savefig('eda/class_distribution_binary.png')
plt.show()