# Competition on credit history data
 The decision of the competition from ODS.ai and Alfa Bank

В данном репозитории представлено решение, занявшее 7 место в "Соревновании на данных кредитных историй" от ODS.ai и Альфа Банка.

Ссылка на соревнование: https://ods.ai/competitions/dl-fintech-bki

**Задача:**

В этом соревновании участникам предлагалось решить задачу кредитного скоринга клиентов Альфа-Банка, используя только данные кредитных историй.

**Данные:**
Датасет соревнования устроен таким образом, что кредиты для тренировочной выборки взяты за период в М месяцев, а кредиты для тестовой выборки взяты за последующие K месяцев.

Каждая запись кредитной истории содержит самую разнообразную информацию о прошлом кредите клиента, например, сумму, отношение клиента к кредиту, дату открытия и закрытия, информацию о просрочках по платежам и др. Все публикуемые данные тщательно анонимизированы.

Целевая переменная – бинарная величина, принимающая значения 0 и 1, где 1 соответствует дефолту клиента по кредиту.

Подробное описание файлов и полей датасета соревнования участники могут найти по ссылке: https://ods.ai/competitions/dl-fintech-bki/data, а также в файле "description".

**Решение:**
Для решения поставленной задачи был реализован стекинг из 9 моделей(6 нейросетевых модели и 3 модели, основанных на градиентном бустинге).

***Нейросетевые модели:*** 
 - 4 модели, основанных на RNN, с разными вариациями архитектур,
 - 1 модель, основанная на архитектуре Transformers
 - 1 модель, основанная на сверточных нейронных сетях.

Подготовка данных для нейросетевых моделей представлена в файлах "dataset_preprocessing_utils_with_mask.py" и "data_generators_with_mask.py", архитектуры моделей прописаны в файле "models.py", циклы обучения прописаны в "train_models.py".
Вспомогательные функции в "utils.py" и "training_aux.py".
