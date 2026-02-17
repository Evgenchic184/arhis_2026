# Анализ использования двух датасетов + контекст поста

## 📊 Почему можно и нужно использовать оба датасета

### Аргументы ЗА:

| Преимущество | Описание |
|--------------|----------|
| **Больше данных** | 47K (Cyberbullying) + ~100K (Reddit) = ~150K примеров |
| **Разнообразие** | Твиты (короткие) + Комментарии (развёрнутые) = лучшая генерализация |
| **Разные типы негатива** | Cyberbullying: токсичность<br>Reddit: эмоциональный негатив |
| **Контекст поста** | Только в Reddit: Post_Title, Post_ID → можно использовать контекст |
| **Комбинированная модель** | Одна модель на объединённых данных может быть лучше двух отдельных |

---

### Аргументы ПРОТИВ:

| Риск | Описание | Митигация |
|------|----------|-----------|
| **Разная разметка** | Cyberbullying: toxic/not_toxic<br>Reddit: Positive/Negative/Neutral | Привести к общей схеме (см. ниже) |
| **Разный стиль** | Твиты vs Комментарии | Использовать универсальные эмбеддинги (BERT) |
| **Разные источники** | Formspring/Instagram vs Reddit | Domain adaptation или fine-tuning |
| **Сложнее EDA** | Нужно анализировать 2 датасета | Сделать отдельный notebook для каждого |

---

## 🎯 Варианты объединения датасетов

### Вариант 1: Объединение для Toxic Detection ⭐

**Идея:** Создать общий бинарный признак `is_toxic`

**Маппинг:**

```python
# Cyberbullying Dataset
cyberbullying_map = {
    'not_cyberbullying': 0,
    'age': 1,
    'ethnicity': 1,
    'gender': 1,
    'religion': 1,
    'sexual_orientation': 1,
    'other_cyberbullying': 1
}

# Reddit Dataset
reddit_map = {
    'Positive': 0,
    'Neutral': 0,
    'Negative': 1  # Предполагаем, что Negative ≈ токсичный
}
```

**Проблема:** В Reddit `Negative` — это **эмоциональный негатив** (грусть, злость, разочарование), а не обязательно токсичность.

**Пример:**
```
"Negative" из Reddit: "I'm so depressed, nobody cares about me"
→ Это НЕ токсично, это крик о помощи

"Toxic" из Cyberbullying: "You're worthless, kill yourself"
→ Это токсично, это буллинг
```

**Вывод:** Прямое объединение **некорректно** — разные определения "негатива".

---

### Вариант 2: Две отдельные модели (рекомендую) ⭐⭐⭐

**Идея:** Использовать каждый датасет для своей задачи

```
┌─────────────────────────────────────────────────────┐
│              Объединённая система                   │
└─────────────────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Toxic Detector     │      │  Sentiment Detector │
│  (Cyberbullying)    │      │  (Reddit)           │
│                     │      │                     │
│  is_toxic: 0/1      │      │  sentiment:         │
│                     │      │  Positive/Negative/ │
│                     │      │  Neutral            │
└─────────────────────┘      └─────────────────────┘
```

**Бизнес-логика (Decision Matrix):**

| is_toxic | sentiment | Действие | Пример |
|----------|-----------|----------|--------|
| ✅ 1 | Negative | **Блокировка** (токсичный + агрессивный) | "You're stupid, shut up" |
| ✅ 1 | Positive/Neutral | **Флаг** (токсичный, но спокойный — спам, бот) | "All [group] are inferior" |
| ❌ 0 | Negative | **Поддержка** (грустный, но не токсичный) | "I feel so alone today" |
| ❌ 0 | Positive/Neutral | **Пропустить** (нормальный) | "Great post, thanks!" |

**Преимущества:**
- ✅ Каждая модель обучается на "своих" данных с корректной разметкой
- ✅ 4 сценария вместо 2 — более гибкая система
- ✅ Можно использовать контекст поста для Sentiment модели
- ✅ Можно дообучать модели независимо

---

### Вариант 3: Multi-task модель (продвинутый)

**Идея:** Одна модель с двумя output heads

```
                    ┌─────────────────┐
                    │   Комментарий   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Encoder       │
                    │   (BERT/RoBERTa)│
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌─────────────────┐          ┌─────────────────┐
     │ Toxic Head      │          │ Sentiment Head  │
     │ (binary)        │          │ (3-class)       │
     └─────────────────┘          └─────────────────┘
```

**Преимущества:**
- ✅ Общие эмбеддинги для обеих задач
- ✅ Эффективнее inference (один проход модели)

**Недостатки:**
- ⚠️ Сложнее в реализации
- ⚠️ Нужны данные с обоими типами разметки (а у нас раздельно)
- ⚠️ Для учебного проекта — overengineering

---

## 📝 Использование контекста поста (Reddit)

### Что даёт контекст:

| Фича из поста | Как использовать | Пример |
|---------------|------------------|--------|
| **Post_Title** | Конкатенация с комментарием | Заголовок: "Support for depression"<br>Коммент: "Me too" → понятнее контекст |
| **Post_ID** | Группировка комментариев | Все комменты к одному посту → агрегация |
| **Num_Comments** | Популярность поста | Много комментов → возможный холивар |
| **Created** | Время поста | Свежий пост → выше активность |
| **URL** | Тип контента | Ссылка на новость → возможные споры |

---

### Способы использования контекста:

#### Способ 1: Конкатенация текста ⭐ (простой)

```python
# Объединяем заголовок поста с комментарием
df['text_with_context'] = df['Post_Title'] + " [SEP] " + df['Body']

# Используем как один текст для модели
# "Support for depression [SEP] Me too, feeling alone"
```

**Плюсы:**
- ✅ Простая реализация
- ✅ Модель видит контекст

**Минусы:**
- ⚠️ Увеличивает длину текста (может быть > 512 токенов для BERT)
- ⚠️ Заголовок может быть шумом

---

#### Способ 2: Separate embeddings (продвинутый)

```python
# Кодируем отдельно
comment_emb = model.encode(comment_text)
post_title_emb = model.encode(post_title)

# Конкатенируем эмбеддинги
combined_emb = np.concatenate([comment_emb, post_title_emb])

# Используем combined_emb для классификации
```

**Плюсы:**
- ✅ Контроль над представлением
- ✅ Можно добавить больше фич (num_comments, etc.)

**Минусы:**
- ⚠️ Сложнее реализация
- ⚠️ Требует кастомной архитектуры

---

#### Способ 3: Attention over context (сложный)

```python
# Модель с attention механизмом
# Учит, какие части комментария важны
# с учётом контекста поста
```

**Плюсы:**
- ✅ Наилучшее качество (теоретически)

**Минусы:**
- ⚠️ Очень сложно
- ⚠️ Не для v1

---

### Рекомендация для контекста:

**Для v1:** Использовать **Способ 1 (конкатенация)** с `[SEP]` токеном

```python
# Пример для Reddit датасета
df['full_text'] = df['Post_Title'].fillna('') + ' [SEP] ' + df['Body']

# Для модели
# "What's your favorite programming language? [SEP] Python is great!"
```

**Проверка влияния контекста:**
```python
# Обучить 2 модели:
# 1. Только комментарий (Body)
# 2. Комментарий + заголовок (Post_Title + Body)

# Сравнить метрики на test set
# Если контекст помогает → используем
```

---

## 🎯 Итоговая архитектура (рекомендация)

### Для учебного проекта (Checkpoint 1-4):

```
┌─────────────────────────────────────────────────────┐
│              Chekpoint 1-2: v1                      │
│                                                     │
│  Задача: Toxic Detection                            │
│  Датасет: Cyberbullying Classification              │
│  Модель: Logistic Regression + TF-IDF               │
│  Метрики: Precision, Recall, F1                     │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Chekpoint 3-4: v2                      │
│                                                     │
│  Добавить: Sentiment Detection                      │
│  Датасет: Reddit Sentiment Analysis                 │
│  Фичи: Comment + Post_Title (конкатенация)          │
│  Decision Engine: 4 сценария                        │
└─────────────────────────────────────────────────────┘
```

---

## 📋 Конкретный план работы

### Этап 1 (сейчас): Toxic Detection на Cyberbullying

```python
# 1. Загрузка
df_cb = pd.read_csv('cyberbullying.csv')

# 2. Маппинг
df_cb['is_toxic'] = df_cb['cyberbullying_type'].apply(
    lambda x: 0 if x == 'not_cyberbullying' else 1
)

# 3. EDA
# - Распределение is_toxic
# - Длины текстов
# - Топ слов

# 4. Модель
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer(max_features=10000)
X = tfidf.fit_transform(df_cb['tweet_text'])
y = df_cb['is_toxic']

model = LogisticRegression(class_weight='balanced')
model.fit(X, y)
```

---

### Этап 2 (позже): Sentiment Detection на Reddit с контекстом

```python
# 1. Загрузка
df_reddit = pd.read_csv('reddit_sentiment.csv')

# 2. Маппинг
df_reddit['is_negative'] = df_reddit['Sentiment'].apply(
    lambda x: 1 if x == 'Negative' else 0
)

# 3. Контекст
df_reddit['text_with_context'] = (
    df_reddit['Post_Title'].fillna('') + ' [SEP] ' + df_reddit['Body']
)

# 4. EDA с контекстом
# - Влияет ли длина заголовка на качество?
# - Какие посты имеют больше негативных комментариев?

# 5. Модель
X = tfidf.transform(df_reddit['text_with_context'])
y = df_reddit['is_negative']

# Сравнить с моделью без контекста
```

---

### Этап 3: Decision Engine

```python
def classify_comment(comment_text, post_title=None):
    # Toxic detection (всегда)
    toxic_prob = toxic_model.predict_proba([comment_text])[0][1]
    
    # Sentiment detection (если есть контекст)
    if post_title:
        text_with_context = f"{post_title} [SEP] {comment_text}"
        negative_prob = sentiment_model.predict_proba([text_with_context])[0][1]
    else:
        negative_prob = sentiment_model.predict_proba([comment_text])[0][1]
    
    # Decision
    if toxic_prob > 0.7:
        if negative_prob > 0.7:
            return "BLOCK", "Toxic + aggressive"
        else:
            return "FLAG", "Toxic content"
    
    if negative_prob > 0.6:
        return "SUPPORT", "User seems upset"
    
    return "PASS", "No action needed"
```

---

## 📊 Сравнение подходов

| Подход | Сложность | Качество | Гибкость | Рекомендация |
|--------|-----------|----------|----------|--------------|
| **1 датасет (Cyberbullying)** | Низкая | Хорошее | Низкая | ✅ Для v1 |
| **2 датасета, 2 модели** | Средняя | Отличное | Высокая | ✅⭐ Для v2 |
| **2 датасета, 1 модель** | Высокая | Неясно | Средняя | ❌ Не имеет смысла |
| **Multi-task** | Очень высокая | Отличное | Средняя | ❌ Для production |

---

## 💡 Выводы

### Почему **два датасета — это хорошо**:

1. ✅ **Разные задачи:** Toxic (модерация) + Sentiment (поддержка)
2. ✅ **Больше данных:** ~150K примеров
3. ✅ **Контекст из Reddit:** Post_Title улучшает понимание
4. ✅ **4 сценария:** Блокировка / Флаг / Поддержка / Пропустить

### Почему **начать с одного (Cyberbullying)**:

1. ✅ Проще для v1 (одна задача, один датасет)
2. ✅ Чёткая бизнес-логика (модерация)
3. ✅ Можно успеть в сроки
4. ✅ Легче объяснить на защите

### План:
- **Checkpoint 1-2:** Cyberbullying → Toxic Detection
- **Checkpoint 3-4:** + Reddit → Sentiment Detection + Decision Engine
- **Production:** + Spam detection + контекст поста

---

**Вопрос:** Начинаем с Cyberbullying (Toxic Detection) для v1, или хочешь сразу делать комбинированную систему с двумя моделями?
