from typing import List
import numpy as np
import pickle
import torch


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


def batches_generator(list_of_paths: List[str], batch_size: int = 32, shuffle: bool = False,
                      is_infinite: bool = False, verbose: bool = False, device: torch.device = None, is_train: bool = True):
    """
    Создает батчи на вход рекуррентных нейронных сетей, реализованных на фреймворках tensorflow и pytorch.

    Параметры:
    -----------
    list_of_paths: List[str]
        Список путей до файлов с предобработанными последовательностями.
    batch_size: int, default=32
        Размер батча.
    shuffle: bool, default=False
        Перемешивать ли данные перед генерацией батчей.
    is_infinite: bool, default=False
        Должен ли генератор быть бесконечным.
    verbose: bool, default=False
        Печатать ли имя текущего обрабатываемого файла.
    device: torch.device, default=None
        Девайс, на который переместить данные при ``output_format``="torch". Игнорируется, если ``output_format``="tf".
    is_train: bool, default=True
        Используется ли генератор для обучения модели или для инференса.

    Возвращаемое значение:
    ----------------------
    result: dict
        Выходной словарь , с ключами "id_", "features", "mask", "lenght", "id_segment"  и "label", если is_train=True,
        и содержащий идентификаторы заявок, признаки и тагрет соответственно.
        Признаки и таргет помещаются на девайс, указанный в ``device``..
    """

    while True:
        if shuffle:
            np.random.shuffle(list_of_paths)

        for path in list_of_paths:
            if verbose:
                print(f"Reading {path}")

            with open(path, "rb") as f:
                data = pickle.load(f)


            ids, padded_sequences, targets, masked_sequences, lenght_sequences, id_segments = \
            data["id"], data["padded_sequences"], data["target"], data['masked_sequences'], data['lenght_sequences'], data["id_segments"]
            
            indices = np.arange(len(ids))
            if shuffle:
                np.random.shuffle(indices)
                ids = ids[indices]
                padded_sequences = padded_sequences[indices]
                masked_sequences = masked_sequences[indices]
                lenght_sequences = lenght_sequences[indices]
                id_segments = id_segments[indices]
                
                if is_train:
                    targets = targets[indices]

            
            for idx in range(len(ids)):
                bucket_ids = ids[idx]
                bucket = padded_sequences[idx]
                mask = masked_sequences[idx]
                lenght = lenght_sequences[idx]
                id_segment = id_segments[idx]
                
                if is_train:
                    bucket_targets = targets[idx]

                
                for jdx in range(0, len(bucket), batch_size):
                    batch_ids = bucket_ids[jdx: jdx + batch_size]
                    batch_sequences = bucket[jdx: jdx + batch_size]
                    batch_mask = mask[jdx: jdx + batch_size]
                    batch_lenght = lenght[jdx: jdx + batch_size]
                    batch_id_segment = id_segment[jdx: jdx + batch_size]
                    
                    if is_train:
                        batch_targets = bucket_targets[jdx: jdx + batch_size]


                    batch_sequences = [torch.LongTensor(batch_sequences[:, i]).to(device) for i in range(len(features))]
                    batch_mask = torch.LongTensor(batch_mask).unsqueeze(1).unsqueeze(2).to(device)
                                                
                    if is_train:
                        yield dict(id_=batch_ids,
                                   features=batch_sequences,
                                   mask=batch_mask, 
                                   lenght = batch_lenght,
                                   id_segment = batch_id_segment,
                                   label=torch.LongTensor(batch_targets).to(device))
                    else:
                        yield dict(id_=batch_ids,
                                   features=batch_sequences,
                                   mask=batch_mask,
                                   id_segment = batch_id_segment,
                                   lenght = batch_lenght)
        if not is_infinite:
            break
