# EDA Plan: Cyberbullying Classification Dataset

## 📋 Контекст

**Датасет:** Cyberbullying Classification  
**URL:** https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification  
**Объём:** 47,000+ твитов  
**Колонки:** `tweet_text`, `cyberbullying_type`  
**Задача:** Бинарная классификация токсичности (Toxic / Non-Toxic)

---

## 1. Описание источников данных и способов получения

### 1.1. Источник данных

| Параметр | Значение |
|----------|----------|
| **Название** | Cyberbullying Classification |
| **Автор** | andrewmvd |
| **Платформа** | Kaggle |
| **URL** | https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification |
| **Лицензия** | Kaggle (проверить перед commercial use) |

### 1.2. Способ получения

```bash
# Вариант 1: Скачать через Kaggle CLI
kaggle datasets download -d andrewmvd/cyberbullying-classification
unzip cyberbullying-classification.zip

# Вариант 2: Скачать вручную через веб-интерфейс
# Вариант 3: Использовать Kaggle API в Python
```

```python
# Загрузка в Python
import pandas as pd

df = pd.read_csv('cyberbullying.csv')
# или
df = pd.read_csv('cyberbullying_classification.csv')  # точное имя файла
```

### 1.3. Структура данных (ожидаемая)

```python
# Ожидаемые колонки
df.columns
# ['tweet_text', 'cyberbullying_type']

# Ожидаемые типы данных
df.dtypes
# tweet_text          object
# cyberbullying_type  object
```

---

## 2. EDA (Exploratory Data Analysis)

### 2.1. Базовая статистика

```python
import pandas as pd
import numpy as np

# Общая информация
df.shape  # (ожидаемо: ~47000, 2)
df.info()
df.head(10)

# Проверка типов данных
print(f"Тип tweet_text: {df['tweet_text'].dtype}")
print(f"Тип cyberbullying_type: {df['cyberbullying_type'].dtype}")
```

**Что проверяем:**
- [ ] Количество строк и колонок
- [ ] Типы данных (object для текста и класса)
- [ ] Первые 10 строк для понимания формата

---

### 2.2. Распределение классов (Class Distribution)

```python
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
print(f"\nБаланс классов: {df['is_toxic'].value_counts().ratio:.2f}")

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
```

**Что проверяем:**
- [ ] Количество уникальных классов (ожидаемо: 6)
- [ ] Баланс исходных классов (~8000 каждый)
- [ ] Баланс после маппинга (ожидаемо: ~8K not_toxic, ~39K toxic = соотношение 5:1)
- [ ] Визуализация распределения

**Метрики баланса:**
```python
toxic_count = df[df['is_toxic'] == 1].shape[0]
non_toxic_count = df[df['is_toxic'] == 0].shape[0]
imbalance_ratio = toxic_count / non_toxic_count
print(f"Соотношение классов: {imbalance_ratio:.2f}:1 (toxic:not_toxic)")
```

---

### 2.3. Анализ длин текстов

```python
# Статистика длин
df['text_length'] = df['tweet_text'].apply(len)
df['word_count'] = df['tweet_text'].apply(lambda x: len(x.split()))

print("Статистика длин текстов:")
print(df[['text_length', 'word_count']].describe())

# Распределение длин (общее)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['text_length'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Распределение длин текстов (символы)')
plt.xlabel('Длина (символы)')
plt.ylabel('Количество')
plt.axvline(df['text_length'].mean(), color='r', linestyle='--', label=f'Среднее: {df["text_length"].mean():.0f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(df['word_count'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Распределение длин текстов (слова)')
plt.xlabel('Количество слов')
plt.ylabel('Количество')
plt.axvline(df['word_count'].mean(), color='r', linestyle='--', label=f'Среднее: {df["word_count"].mean():.0f}')
plt.legend()

plt.tight_layout()
plt.savefig('eda/text_length_distribution.png')
plt.show()

# Распределение длин по классам
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='is_toxic', y='text_length')
plt.xticks([0, 1], ['Not Toxic', 'Toxic'])
plt.title('Длина текстов по классам (символы)')
plt.xlabel('Класс')
plt.ylabel('Длина (символы)')

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='is_toxic', y='word_count')
plt.xticks([0, 1], ['Not Toxic', 'Toxic'])
plt.title('Длина текстов по классам (слова)')
plt.xlabel('Класс')
plt.ylabel('Количество слов')

plt.tight_layout()
plt.savefig('eda/text_length_by_class.png')
plt.show()

# Статистика по классам
print("\nСтатистика длин по классам:")
print(df.groupby('is_toxic')['text_length'].describe())
print("\nСтатистика слов по классам:")
print(df.groupby('is_toxic')['word_count'].describe())
```

**Что проверяем:**
- [ ] Мин/макс/средняя длина текста
- [ ] Выбросы (очень длинные/короткие тексты)
- [ ] Различия длин между toxic и not_toxic
- [ ] Выбор максимального размера для модели (например, 280 символов для Twitter)

---

### 2.4. Анализ пропусков (Missing Values)

```python
# Проверка пропусков
missing_stats = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Пропуски': missing_stats,
    'Процент': missing_pct
})
print("Пропуски в данных:")
print(missing_df)

# Визуализация
plt.figure(figsize=(8, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Пропуски в данных')
plt.savefig('eda/missing_values_heatmap.png')
plt.show()
```

**Что проверяем:**
- [ ] Количество пропусков в `tweet_text` (ожидаемо: 0)
- [ ] Количество пропусков в `cyberbullying_type` (ожидаемо: 0)
- [ ] Допустимая доля пропусков: 0% (оба поля required)

**Действия при пропусках:**
```python
# Если есть пропуски в тексте
df = df.dropna(subset=['tweet_text'])  # Удалить строки с пропусками
# или
df['tweet_text'] = df['tweet_text'].fillna('')  # Заполнить пустой строкой

# Если есть пропуски в классе
df = df.dropna(subset=['cyberbullying_type'])  # Удалить (класс критичен)
```

---

### 2.5. Анализ дубликатов

```python
# Дубликаты текстов
duplicate_texts = df['tweet_text'].duplicated().sum()
print(f"Дубликаты текстов: {duplicate_texts} ({duplicate_texts/len(df)*100:.2f}%)")

# Дубликаты полных строк
duplicate_rows = df.duplicated().sum()
print(f"Полные дубликаты строк: {duplicate_rows} ({duplicate_rows/len(df)*100:.2f}%)")

# Топ дублирующихся текстов
if duplicate_texts > 0:
    duplicate_df = df[df['tweet_text'].duplicated(keep=False)]
    print("\nТоп-10 дублирующихся текстов:")
    print(duplicate_df['tweet_text'].value_counts().head(10))
```

**Что проверяем:**
- [ ] Количество дубликатов текстов
- [ ] Количество полных дубликатов строк
- [ ] Допустимая доля дубликатов: < 5%

**Действия при дубликатах:**
```python
# Удаление дубликатов
df = df.drop_duplicates(subset=['tweet_text'])
# или
df = df.drop_duplicates()  # Полные дубликаты
```

---

### 2.6. Топ слов по классам (Word Frequency Analysis)

```python
from collections import Counter
import re

def preprocess_text(text):
    """Базовая предобработка для анализа слов"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Удалить URL
    text = re.sub(r'@\w+|#\w+', '', text)  # Удалить @mentions и #hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Оставить только буквы
    words = text.split()
    words = [w for w in words if len(w) > 2]  # Удалить короткие слова
    return words

# Разделение по классам
toxic_texts = df[df['is_toxic'] == 1]['tweet_text']
non_toxic_texts = df[df['is_toxic'] == 0]['tweet_text']

# Токенизация
toxic_words = []
for text in toxic_texts:
    toxic_words.extend(preprocess_text(text))

non_toxic_words = []
for text in non_toxic_texts:
    non_toxic_words.extend(preprocess_text(text))

# Частотность
toxic_word_counts = Counter(toxic_words)
non_toxic_word_counts = Counter(non_toxic_words)

# Топ-20 слов
print("Топ-20 слов в TOXIC комментариях:")
for word, count in toxic_word_counts.most_common(20):
    print(f"  {word}: {count}")

print("\nТоп-20 слов в NOT_TOXIC комментариях:")
for word, count in non_toxic_word_counts.most_common(20):
    print(f"  {word}: {count}")

# Визуализация (Word Cloud)
from wordcloud import WordCloud

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
wc_toxic = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(' '.join(toxic_words))
plt.imshow(wc_toxic, interpolation='bilinear')
plt.title('Toxic Comments')
plt.axis('off')

plt.subplot(1, 2, 2)
wc_non_toxic = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(' '.join(non_toxic_words))
plt.imshow(wc_non_toxic, interpolation='bilinear')
plt.title('Not Toxic Comments')
plt.axis('off')

plt.tight_layout()
plt.savefig('eda/wordcloud_comparison.png')
plt.show()
```

**Что проверяем:**
- [ ] Топ частых слов для каждого класса
- [ ] Слова-маркеры токсичности (оскорбления, угрозы)
- [ ] Слова-маркеры нормальных комментариев
- [ ] Наличие стоп-слов (I, you, the, etc.)

---

### 2.7. Анализ специфичных токенов

```python
# Проверка на наличие специфичных паттернов
df['has_url'] = df['tweet_text'].str.contains(r'http\S+|www\S+', regex=True)
df['has_mention'] = df['tweet_text'].str.contains(r'@\w+', regex=True)
df['has_hashtag'] = df['tweet_text'].str.contains(r'#\w+', regex=True)
df['has_caps'] = df['tweet_text'].apply(lambda x: x.isupper())
df['caps_ratio'] = df['tweet_text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)

# Статистика по классам
print("Наличие URL по классам:")
print(df.groupby('is_toxic')['has_url'].mean())

print("\nНаличие @mention по классам:")
print(df.groupby('is_toxic')['has_mention'].mean())

print("\nНаличие #hashtag по классам:")
print(df.groupby('is_toxic')['has_hashtag'].mean())

print("\nДоля CAPS по классам:")
print(df.groupby('is_toxic')['caps_ratio'].mean())
```

**Что проверяем:**
- [ ] Наличие URL (возможный спам)
- [ ] Наличие @mentions (обращения к пользователям)
- [ ] Наличие #hashtags (тематики)
- [ ] CAPS LOCK (агрессия)

---

### 2.8. Примеры комментариев (Manual Review)

```python
# Примеры токсичных комментариев
print("=" * 80)
print("ПРИМЕРЫ TOXIC комментариев (5 случайных):")
print("=" * 80)
toxic_sample = df[df['is_toxic'] == 1]['tweet_text'].sample(5, random_state=42)
for i, text in enumerate(toxic_sample, 1):
    print(f"\n{i}. {text}")

# Примеры не токсичных комментариев
print("\n" + "=" * 80)
print("ПРИМЕРЫ NOT_TOXIC комментариев (5 случайных):")
print("=" * 80)
non_toxic_sample = df[df['is_toxic'] == 0]['tweet_text'].sample(5, random_state=42)
for i, text in enumerate(non_toxic_sample, 1):
    print(f"\n{i}. {text}")

# Примеры пограничных случаев (можно добавить колонку с уверенностью, если есть)
```

**Что проверяем:**
- [ ] Качество разметки (ручная проверка 5-10 примеров)
- [ ] Пограничные случаи (сложно классифицировать)
- [ ] Наличие шума в разметке

---

### 2.9. Проверка на утечки данных (Data Leakage)

```python
# 1. Проверка дубликатов (уже сделано выше)
print(f"Дубликаты текстов: {duplicate_texts}")

# 2. Проверка на ложные корреляции
# Анализ слов, которые встречаются только в одном классе
toxic_vocab = set(toxic_words)
non_toxic_vocab = set(non_toxic_words)

unique_toxic = toxic_vocab - non_toxic_vocab
unique_non_toxic = non_toxic_vocab - toxic_vocab

print(f"\nУникальные слова только в toxic: {len(unique_toxic)}")
print(f"Уникальные слова только в non_toxic: {len(unique_non_toxic)}")

# Топ уникальных слов
print("\nТоп-10 уникальных слов для toxic:")
for word in list(unique_toxic)[:10]:
    count = toxic_word_counts[word]
    print(f"  {word}: {count}")

# 3. Проверка на временную утечку (если есть timestamp)
# В этом датасете timestamp нет, но если бы был:
# df.sort_values('timestamp') и сплит по времени
```

**Что проверяем:**
- [ ] Дубликаты между train/test (удалить перед сплитом)
- [ ] Ложные корреляции (слова-маркеры, которые модель запомнит)
- [ ] Временная утечка (если есть timestamp)

---

### 2.10. Языковое распределение (Language Detection)

```python
# Проверка языка (выборочно, т.к. langdetect медленный)
from langdetect import detect

# Семплирование для скорости
sample_df = df.sample(1000, random_state=42)
sample_df['language'] = sample_df['tweet_text'].apply(lambda x: detect(x) if len(x) > 10 else 'unknown')

print("Распределение языков (выборка 1000):")
print(sample_df['language'].value_counts())

# Если есть не-English комментарии
non_english = sample_df[sample_df['language'] != 'en']
print(f"\nНе-English комментарии: {len(non_english)} ({len(non_english)/len(sample_df)*100:.1f}%)")
```

**Что проверяем:**
- [ ] Основной язык (ожидаемо: English)
- [ ] Доля не-English комментариев
- [ ] Решение: фильтровать или обрабатывать отдельно

---

### 2.11. Сводная статистика EDA

```python
# Финальный отчёт
print("=" * 80)
print("СВОДНЫЙ ОТЧЁТ EDA")
print("=" * 80)

print(f"\n1. Объём данных: {len(df)} строк, {len(df.columns)} колонок")
print(f"2. Классы: {df['cyberbullying_type'].nunique()} уникальных")
print(f"3. Баланс классов (binary): {df['is_toxic'].value_counts().to_dict()}")
print(f"4. Соотношение: {imbalance_ratio:.2f}:1")
print(f"5. Средняя длина текста: {df['text_length'].mean():.0f} символов")
print(f"6. Пропуски: {df.isnull().sum().sum()}")
print(f"7. Дубликаты: {duplicate_texts} ({duplicate_texts/len(df)*100:.2f}%)")
print(f"8. Доля URL: {df['has_url'].mean()*100:.1f}%")
print(f"9. Доля CAPS (>50%): {(df['caps_ratio'] > 0.5).mean()*100:.1f}%")
```

---

## 3. Data Contract

### 3.1. Схема данных

| feature_name | dtype | required | описание | диапазон/ограничения |
|:------------ |:------ | -------- | -------- | ------------------- |
| tweet_text | object | True | Текст твита | 1-500 символов, не пустой |
| cyberbullying_type | object | True | Тип буллинга | {not_cyberbullying, age, ethnicity, gender, religion, sexual_orientation, other_cyberbullying} |
| is_toxic | int64 | True | Бинарный признак (производный) | {0, 1} |

### 3.2. Требования к данным

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| **Допустимая доля пропусков** | 0% | Оба поля required |
| **Допустимая доля дубликатов** | < 5% | Иначе переобучение |
| **Мин. длина текста** | 1 токен | Пустые тексты не информативны |
| **Макс. длина текста** | 512 токенов | Ограничение моделей |
| **Требования к свежести** | Не критично | Язык стабилен |
| **Периодичность обновления** | Единоразово | Для обучения v1 |
| **Язык** | English | Требуется детекция non-English |

### 3.3. Валидация данных (Data Validation)

```python
def validate_data(df):
    """Проверка соответствия Data Contract"""
    errors = []
    
    # Проверка колонок
    required_cols = ['tweet_text', 'cyberbullying_type']
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    
    # Проверка пропусков
    if df['tweet_text'].isnull().sum() > 0:
        errors.append(f"Null values in tweet_text: {df['tweet_text'].isnull().sum()}")
    
    if df['cyberbullying_type'].isnull().sum() > 0:
        errors.append(f"Null values in cyberbullying_type: {df['cyberbullying_type'].isnull().sum()}")
    
    # Проверка допустимых значений
    valid_types = {'not_cyberbullying', 'age', 'ethnicity', 'gender', 'religion', 'sexual_orientation', 'other_cyberbullying'}
    actual_types = set(df['cyberbullying_type'].unique())
    invalid_types = actual_types - valid_types
    if invalid_types:
        errors.append(f"Invalid cyberbullying_type values: {invalid_types}")
    
    # Проверка длины текстов
    empty_texts = (df['tweet_text'].str.len() == 0).sum()
    if empty_texts > 0:
        errors.append(f"Empty tweet_text: {empty_texts}")
    
    # Проверка дубликатов
    duplicate_pct = df['tweet_text'].duplicated().sum() / len(df) * 100
    if duplicate_pct > 5:
        errors.append(f"High duplicate rate: {duplicate_pct:.2f}%")
    
    if errors:
        print("DATA VALIDATION FAILED:")
        for error in errors:
            print(f"  ❌ {error}")
        return False
    else:
        print("DATA VALIDATION PASSED ✅")
        return True
```

---

## 4. Визуализации для отчёта

### Список обязательных графиков:

1. **class_distribution_original.png** — Распределение 6 исходных классов
2. **class_distribution_binary.png** — Бинарное распределение (toxic/not_toxic)
3. **text_length_distribution.png** — Гистограмма длин текстов (2 подграфика)
4. **text_length_by_class.png** — Box plot длин по классам
5. **missing_values_heatmap.png** — Тепловая карта пропусков
6. **wordcloud_comparison.png** — Word clouds для toxic vs non-toxic
7. **bar_chart_top_words.png** — Топ-20 слов по классам

---

## 5. Чек-лист завершения EDA

- [ ] Загружен датасет (47K строк)
- [ ] Проверена структура (2 колонки)
- [ ] Создан бинарный признак `is_toxic`
- [ ] Распределение классов: ~8K not_toxic, ~39K toxic (5:1)
- [ ] Статистика длин: mean ~100-150 символов
- [ ] Пропуски: 0 (или обработаны)
- [ ] Дубликаты: < 5% (или удалены)
- [ ] Топ слов по классам: выявлены маркеры
- [ ] Word clouds: визуализированы
- [ ] Примеры: 5 toxic + 5 non_toxic просмотрены вручную
- [ ] Утечки: проверены дубликаты, ложные корреляции
- [ ] Язык: подтверждён English (или выявлены исключения)
- [ ] Data Contract: задокументирован
- [ ] Визуализации: 7 графиков сохранены
- [ ] Сводный отчёт: распечатан

---

## 6. Выводы EDA (шаблон)

```
EDA SUMMARY
===========

1. DATASET: 47,000 tweets, 2 columns (tweet_text, cyberbullying_type)

2. CLASS DISTRIBUTION:
   - Original: 6 balanced classes (~8K each)
   - Binary: 8K not_toxic (17%), 39K toxic (83%)
   - Imbalance ratio: 5:1 (requires class_weight='balanced')

3. TEXT LENGTH:
   - Mean: XXX characters, YY words
   - Min: X, Max: XXXX
   - Toxic comments: slightly longer/shorter

4. DATA QUALITY:
   - Missing values: 0 (clean)
   - Duplicates: X.X% (acceptable/removed)
   - Language: XX% English

5. KEY INSIGHTS:
   - Toxic words: [top 5 markers]
   - Non-toxic words: [top 5 markers]
   - Specific patterns: URLs X%, CAPS X%

6. DATA LEAKAGE:
   - Duplicates: None/Low (handled)
   - False correlations: [identified words]
   - Temporal: N/A (no timestamp)

7. RECOMMENDATIONS:
   - Use class_weight='balanced' for model training
   - Max features: 10,000 (TF-IDF)
   - Handle URLs/mentions in preprocessing
   - Consider removing duplicates before train/test split
```

---

## 7. Python Notebook Structure

```python
# EDA for Cyberbullying Classification Dataset
# =============================================

# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
from langdetect import detect

# 2. Load Data
df = pd.read_csv('cyberbullying.csv')

# 3. Create Binary Target
df['is_toxic'] = df['cyberbullying_type'].apply(
    lambda x: 0 if x == 'not_cyberbullying' else 1
)

# 4. Basic Stats
# ... (код из разделов выше)

# 5. Visualizations
# ... (сохранение всех графиков в папку eda/)

# 6. Save EDA Report
# ... (сводный отчёт)
```

---

**Следующий шаг:** Выполнить EDA, сохранить визуализации, задокументировать инсайты в отчёте.
