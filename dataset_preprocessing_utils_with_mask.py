from typing import Dict
import numpy as np
import pandas as pd
import pickle
from tqdm.notebook import tqdm
from sklearn.utils import shuffle

features = ["pre_since_opened", "pre_since_confirmed", "pre_pterm", "pre_fterm", "pre_till_pclose", "pre_till_fclose",
            "pre_loans_credit_limit", "pre_loans_next_pay_summ", "pre_loans_outstanding", "pre_loans_total_overdue",
            "pre_loans_max_overdue_sum", "pre_loans_credit_cost_rate",
            "pre_loans5", "pre_loans530", "pre_loans3060", "pre_loans6090", "pre_loans90",
            "is_zero_loans5", "is_zero_loans530", "is_zero_loans3060", "is_zero_loans6090", "is_zero_loans90",
            "pre_util", "pre_over2limit", "pre_maxover2limit", "is_zero_util", "is_zero_over2limit", "is_zero_maxover2limit",
            "enc_paym_0", "enc_paym_1", "enc_paym_2", "enc_paym_3", "enc_paym_4", "enc_paym_5", "enc_paym_6", "enc_paym_7", "enc_paym_8",
            "enc_paym_9", "enc_paym_10", "enc_paym_11", "enc_paym_12", "enc_paym_13", "enc_paym_14", "enc_paym_15", "enc_paym_16",
            "enc_paym_17", "enc_paym_18", "enc_paym_19", "enc_paym_20", "enc_paym_21", "enc_paym_22", "enc_paym_23", "enc_paym_24",
            "enc_loans_account_holder_type", "enc_loans_credit_status", "enc_loans_credit_type", "enc_loans_account_cur",
            "pclose_flag", "fclose_flag"]


def pad_sequence(array: np.ndarray, max_len: int) -> np.ndarray:
    """
    Принимает на вход массив массивов ``array`` и производит padding каждого вложенного массива до ``max_len``.

    Параметры:
    -----------
    array: numpy.ndarray
        Входной массив массивов.
    max_len: int
        Длина, до которой нужно сделать padding вложенных массивов.

    Возвращаемое значение:
    ----------------------
    output: numpy.ndarray
        Выходной массив.
    """
    if isinstance(max_len, float):
        print(max_len)
    output = np.zeros((len(features), max_len))
    
    output[:, :array.shape[1]] = array
    
    return output


def mask_sequence(array: np.ndarray, max_len: int) -> np.ndarray:
    """
    Принимает на вход массив массивов ``array``(результат работы pad_sequence) и производит masking для padding элементов.

    Параметры:
    -----------
    array: numpy.ndarray
        Входной массив массивов после padding.
    max_len: int
        Длина, до которой нужно сделать padding вложенных массивов.

    Возвращаемое значение:
    ----------------------
    output: numpy.ndarray
        Выходной массив.
    """
    if isinstance(max_len, float):
        print(max_len)

    mask = np.zeros(max_len)
  
    mask[:array.shape[1]] = 1
    
    return mask


def truncate(x, num_last_credits: int = 0):
    return pd.Series({"sequences": x.values.transpose()[:, -num_last_credits:]})


def transform_credits_to_sequences(credits_frame: pd.DataFrame,
                                   num_last_credits: int = 0, shuffle_rn: bool = False) -> pd.DataFrame:
    """
    Принимает pandas.DataFrame с записями кредитных историй клиентов, сортирует кредиты по клиентам
    (внутри клиента сортирует кредиты от старых к новым), берет ``num_last_credits`` кредитов,
    возвращает новый pandas.DataFrame с двумя колонками: id и sequences.
    
    Каждое значение в столбце sequences - это массив массивов.
    Каждый вложенный массив - значение одного признака во всех кредитах клиента.
    Всего признаков len(features), поэтому будет len(features) массивов.
    
    Если shuffle_rn = True, кредиты в рамках одного id перемешиваются случайным образом.

    Параметры:
    -----------
    credits_frame: pandas.DataFrame
        Датафрейм с записями кредитных историй клиентов.
    num_last_credits: int, default=0
        Количество кредитов клиента, которые будут включены в выходные данные. Если 0, то берутся все кредиты.
    shuffle_rn: bool = False
        Если shuffle_rn = True, кредиты в рамках одного id перемешиваются случайным образом.

    Возвращаемое значение:
    ----------------------
    output: pandas.DataFrame
        Выходной датафрейм с двумя столбцами: "id", "sequences".
    """
    if shuffle_rn:
        credits_frame = shuffle(credits_frame)
        return credits_frame \
            .sort_values(["id"]) \
            .groupby(["id"])[features] \
            .apply(lambda x: truncate(x, num_last_credits=num_last_credits)) \
            .reset_index()
    else:
        return credits_frame \
            .sort_values(["id", "rn"]) \
            .groupby(["id"])[features] \
            .apply(lambda x: truncate(x, num_last_credits=num_last_credits)) \
            .reset_index()
        


def create_padded_buckets(frame_of_sequences: pd.DataFrame, bucket_info: Dict[int, int],
                          save_to_file_path: str = None, has_target: bool = True):
    """
    Реализует Sequence Bucketing технику для обучения рекуррентных нейронных сетей.
    Принимает на вход датафрейм ``frame_of_sequences`` с двумя столбцами: "id", "sequences"
    (результат работы функции transform_credits_to_sequences),
    словарь ``bucket_info``, где для последовательности каждой длины указано, до какой максимальной длины нужно делать
    padding, группирует кредиты по бакетам (на основе длины), производит padding нулями и сохраняет результат
    в pickle файл, если требуется.

    Параметры:
    -----------
    frame_of_sequences: pandas.DataFrame
        Входной датафрейм с двумя столбцами "id", "sequences" (результат работы функции transform_credits_to_sequences).
    bucket_info: Dict[int, int]
        Cловарь, где для последовательности каждой длины указано, до какой максимальной длины нужно делать padding.
    save_to_file_path: str, default=None
        Опциональный путь до файла, куда нужно сохранить результат. Если None, то сохранение не требуется.
    has_target: bool, deafult=True
        Флаг, есть ли в frame_of_sequences целевая переменная или нет. Если есть, то она также будет записана в выходной словарь.

    Возвращаемое значение:
    ----------------------
    dict_result: dict
        Выходной словарь со ключами:  "id", 
                                      "padded_sequences", 
                                      "masked_sequences", 
                                      "lenght_sequences" - длина последовательности кредитов до padding, 
                                      "id_segments" - сегмент, к которому относится id (учитывая, что id увеличивается по времени, все id                                                         заявок разделены на группы по 100000 шт.), 
                                      "target".
    """
    frame_of_sequences["sequence_length"] = frame_of_sequences["sequences"].apply(lambda x: len(x[1]))
    frame_of_sequences["bucket_idx"] = frame_of_sequences["sequence_length"].map(bucket_info)
    frame_of_sequences["id_segment"] = frame_of_sequences["id"].apply(lambda x: x // 100000)
    frame_of_sequences = frame_of_sequences.sort_values('sequence_length', ascending=False)
    padded_seq = []
    masks_seq = []
    lenght_seq = []
    id_seg = []
    targets = []
    ids = []

    for size, bucket in tqdm(frame_of_sequences.groupby("bucket_idx", sort=False), desc="Extracting buckets"):
        padded_sequences = bucket["sequences"].apply(lambda x: pad_sequence(x, size)).values
        mask_sequences = bucket["sequences"].apply(lambda x: mask_sequence(x, size)).values
        lenght_sequences = bucket['sequence_length'].values 
        id_segments = bucket['id_segment'].values
        
        padded_seq.append(np.stack(padded_sequences, axis=0))
        masks_seq.append(np.stack(mask_sequences, axis=0))
        lenght_seq.append(np.stack(lenght_sequences, axis=0))
        id_seg.append(np.stack(id_segments, axis=0))

        if has_target:
            targets.append(bucket["flag"].values)

        ids.append(bucket["id"].values)

    frame_of_sequences.drop(columns=["bucket_idx"], inplace=True)

    dict_result = {
        "id": np.array(ids, dtype=np.object),
        "padded_sequences": np.array(padded_seq, dtype=np.object),
        "masked_sequences": np.array(masks_seq),
        "lenght_sequences": np.array(lenght_seq),
        "id_segments":  np.array(id_seg),
        "target": np.array(targets, dtype=np.object) if targets else []
    }

    if save_to_file_path:
        with open(save_to_file_path, "wb") as f:
            pickle.dump(dict_result, f)
    return dict_result
