
1. Выбор темы и постановка задачи
- Классификация настроения пользователя по комментарию к посту.
- Всем грустным комментаторам будут отправлены фразы поддержки; Из ограничений - время ответа не более 5-10 секунд. 
- Всех считаем грустными.
2. Первичный сбор данных и EDA
- Источник - https://www.kaggle.com/datasets/vijayj0shi/reddit-dataset-with-sentiment-analysis
- EDA?
- Утечки маловероятны, но может быть ложная корреляция между конкретными словами и меткой класса.
3. Data Contract
Схема данных 
Комментарии

| feature_name | dtype  | required |
|:------------ |:------ | -------- |
| Sentiment    | object | True     |
| Body         | object | True     |
| Post_Title   | object | True     |
| Post_ID      | object | True     |
| Author       | object | False    |
| Parent_ID    | object | False    |
| Comment_ID   | object | False    |

Посты

| feature_name | dtype   | required |
|:------------ |:------- | -------- |
| Text         | object  | True     |
| Post_ID      | object  | True     |
| Title        | object  | True     |
| Author       | object  | False    |
| Created      | float64 | False    |
| Num_Comments | int64   | False    |
| URL          | object  | False    |



4. Архитектура системы 


5. Риски (v0)
- Сленг -> Невозможность классификации, снижение точности
- Другой язык -> Невозможность классификации, снижение точности
- Большой поток комментариев -> Перегруз сервиса
- Злоупотребление сервисом -> Перегруз сервиса