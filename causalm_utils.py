import logging
from datetime import datetime
from glob import glob
from multiprocessing import cpu_count
from os import listdir, path
from pathlib import Path
from subprocess import Popen, PIPE, run
from typing import Dict

import pandas as pd
import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split

####### Dataset utils
# from spacy.lang.tag_map import TAG_MAP
# import spacy

HOME_DIR = str(Path.home())
INIT_TIME = datetime.now().strftime('%e-%m-%y_%H-%M-%S').lstrip()


def init_logger(name=None, path=None, screen=True):
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('{asctime} - {message}', datefmt="%H:%M:%S", style="{")
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(f"{path}/{name}-{INIT_TIME}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if screen:
        screen_handler = logging.StreamHandler()
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)
    return logger


def get_free_gpu():
    if torch.cuda.is_available():
        gpu_output = Popen(["nvidia-smi", "-q", "-d", "PIDS"], stdout=PIPE, text=True)
        gpu_processes = Popen(["grep", "Processes"], stdin=gpu_output.stdout, stdout=PIPE, text=True)
        gpu_output.stdout.close()
        processes_output = gpu_processes.communicate()[0]
        for i, line in enumerate(processes_output.strip().split("\n")):
            if line.endswith("None"):
                print(f"Found Free GPU ID: {i}")
                cuda_device = f"cuda:{i}"
                torch.cuda.set_device(cuda_device)
                return torch.device(cuda_device)
        print("WARN - No Free GPU found! Running on CPU instead...")
    return torch.device("cpu")


def count_num_cpu_gpu():
    if torch.cuda.is_available():
        num_gpu_cores = torch.cuda.device_count()
        num_cpu_cores = (cpu_count() // num_gpu_cores // 2) - 1
    else:
        num_gpu_cores = 0
        num_cpu_cores = (cpu_count() // 2) - 1
    return num_cpu_cores, num_gpu_cores


class StreamToLogger:
    """
   Fake file-like stream object that redirects writes to a logger instance.
   written by: Ferry Boender
   https://www.electricmonk.nl/log/2011/08/14/redirect-stdout-and-stderr-to-a-logger-in-python/
   """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())


def save_predictions(folder, sample_idx_list, predictions_list, true_list, correct_list, class_probs, name):
    df_dict = {
        "sample_index": sample_idx_list,
        "prediction": predictions_list,
        "true": true_list,
        "correct": correct_list,
    }
    df_dict.update({f"class_{i}_prob": class_i_prob for i, class_i_prob in enumerate(class_probs)})
    df = pd.DataFrame.from_dict(df_dict)
    df = df.set_index("sample_index").sort_index()
    df.to_csv(f"{folder}/{name}-predictions.csv")


class GoogleDriveHandler:
    def __init__(self,
                 local_root: str = f"{HOME_DIR}/GoogleDrive",
                 drive_binary: str = f"{HOME_DIR}/bin/go/packages/bin/drive",
                 default_timeout: int = 600):
        self.local_root = local_root
        self.drive_binary = drive_binary
        self.default_args = ["-no-prompt"]
        self.default_timeout = default_timeout

    def _execute_drive_cmd(self, subcommand: str, path: str, cmd_args: list):
        if subcommand not in ("pull", "push"):
            raise ValueError("Only pull and push commands are currently supported")
        cmd = [self.drive_binary, subcommand] + self.default_args + cmd_args + [path]
        cmd_return = run(cmd, capture_output=True, text=True, timeout=self.default_timeout, cwd=HOME_DIR)
        return cmd_return.returncode, cmd_return.stdout, cmd_return.stderr

    def push_files(self, path: str, cmd_args: list = []):
        try:
            push_return = self._execute_drive_cmd("push", path, ["-files"] + cmd_args)
            if push_return[0] == 0:
                message = f"Successfully pushed results to Google Drive: {path}"
            else:
                message = f"Failed to push results to Google Drive: {path}\nExit Code: {push_return[0]}\nSTDOUT: {push_return[1]}\nSTDERR: {push_return[2]}"
        except Exception as e:
            message = f"ERROR: {e}\nFailed to push results to Google Drive: {path}"
        return message

    def pull_files(self, path: str, cmd_args: list = []):
        return self._execute_drive_cmd("pull", path, ["-files"] + cmd_args)


def get_checkpoint_file(ckpt_dir: str):
    for file in sorted(listdir(ckpt_dir)):
        if file.endswith(".ckpt"):
            return f"{ckpt_dir}/{file}"
    else:
        return None


def find_latest_model_checkpoint(models_dir: str):
    model_ckpt = None
    while not model_ckpt:
        model_versions = sorted(glob(models_dir), key=path.getctime)
        if model_versions:
            latest_model = model_versions.pop()
            model_ckpt_dir = f"{latest_model}/checkpoints"
            model_ckpt = get_checkpoint_file(model_ckpt_dir)
        else:
            raise FileNotFoundError(f"Couldn't find a model checkpoint in {models_dir}")
    return model_ckpt


def print_final_metrics(name: str, metrics: Dict, logger=None):
    if logger:
        logger.info(f"{name} Metrics:")
        for metric, val in metrics.items():
            logger.info(f"{metric}: {val:.4f}")
        logger.info("\n")
    else:
        print(f"{name} Metrics:")
        for metric, val in metrics.items():
            print(f"{metric}: {val:.4f}")
        print()


####### Dataset utils

### BERT constants
WORDPIECE_PREFIX = "##"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

### POS Tags constants
# TOKEN_SEPARATOR = " "
# WORD_POS_SEPARATOR = "_"
# ADJ_POS_TAGS = ("ADJ", "ADV")
# POS_TAGS_TUPLE = tuple(sorted(TAG_MAP.keys()))
# POS_TAG_IDX_MAP = {str(tag): int(idx) for idx, tag in enumerate(POS_TAGS_TUPLE)}
# ADJ_POS_TAGS_IDX = {"ADJ": 0, "ADV": 2}
# NUM_POS_TAGS_LABELS = len(POS_TAGS_TUPLE)

sentiment_output_datasets = {0: 'negative', 1: 'positive'}


# def clean_review(text: str) -> str:
#     review_text = re.sub("\n", "", text)
#     review_text = re.sub(" and quot;", '"', review_text)
#     review_text = re.sub("<br />", "", review_text)
#     review_text = re.sub(WORD_POS_SEPARATOR, "", review_text)
#     review_text = re.sub("\s+", TOKEN_SEPARATOR, review_text)
#     # review_text = re.sub(";", ",", review_text)
#     return review_text.strip()


# class PretrainedPOSTagger:
#
#     """This module requires en_core_web_lg model to be installed"""
#     tagger = spacy.load("en_core_web_lg")
#
#     @staticmethod
#     def tag_review(review: str) -> str:
#         review_text = clean_review(review)
#         tagged_review = [f"{token.text}{WORD_POS_SEPARATOR}{token.pos_}"
#                          for token in PretrainedPOSTagger.tagger(review_text)]
#         return TOKEN_SEPARATOR.join(tagged_review)


def split_data(df: DataFrame, path: str, prefix: str, label_column: str = "label"):
    train, test = train_test_split(df, test_size=0.2, stratify=df[label_column], random_state=RANDOM_SEED)
    train, dev = train_test_split(train, test_size=0.2, stratify=train[label_column], random_state=RANDOM_SEED)
    df.sort_index().to_csv(f"{path}/{prefix}_all.csv")
    train.sort_index().to_csv(f"{path}/{prefix}_train.csv")
    dev.sort_index().to_csv(f"{path}/{prefix}_dev.csv")
    test.sort_index().to_csv(f"{path}/{prefix}_test.csv")
    return train, dev, test


# def print_text_stats(df: DataFrame, text_column: str):
#     sequence_lengths = df[text_column].apply(lambda text: int(len(str(text).split(TOKEN_SEPARATOR))))
#     print(f"Number of sequences in dataset: {len(sequence_lengths)}")
#     print(f"Max sequence length in dataset: {np.max(sequence_lengths)}")
#     print(f"Min sequence length in dataset: {np.min(sequence_lengths)}")
#     print(f"Median sequence length in dataset: {np.median(sequence_lengths)}")
#     print(f"Mean sequence length in dataset: {np.mean(sequence_lengths)}")


RANDOM_SEED = 212


def bias_random_sampling(df: DataFrame, bias_column: str, biasing_factor: float, seed: int = RANDOM_SEED):
    return df.sample(frac=biasing_factor, random_state=seed)


def bias_ranked_sampling(df: DataFrame, bias_column: str, biasing_factor: float):
    return df.sort_values(by=bias_column, ascending=False).head(int(len(df) * biasing_factor))


def bias_aggressive(df_a, df_b, label_column, bias_column,
                    biased_label, biasing_factor, sampling_method=bias_random_sampling):
    """
    Biases selected class by biasing factor, and uses same factor to inversely bias all other classes.
    :param bias_column:
    :param label_column:
    :param sampling_method:
    :param df_a:
    :param df_b:
    :param biased_label:
    :param biasing_factor:
    :return:
    """
    df_biased = pd.DataFrame(columns=df_a.columns)
    for label in sorted(df_a[label_column].unique()):
        df_label_a = df_a[df_a[label_column] == label]
        df_label_b = df_b[df_b[label_column] == label]
        if label == biased_label:
            df_biased = df_biased.append(df_label_a, ignore_index=True)
            df_sampled_b = sampling_method(df_label_b, bias_column, biasing_factor)
            df_biased = df_biased.append(df_sampled_b, ignore_index=True)
        else:
            df_biased = df_biased.append(df_label_b, ignore_index=True)
            df_sampled_a = sampling_method(df_label_a, bias_column, biasing_factor)
            df_biased = df_biased.append(df_sampled_a, ignore_index=True)
    return df_biased


def bias_gentle(df_a, df_b, label_column, bias_column,
                biased_label, biasing_factor, sampling_method=bias_random_sampling):
    """
    Biases selected class by biasing factor, and leaves other classes untouched.
    :param bias_column:
    :param label_column:
    :param sampling_method:
    :param df_a:
    :param df_b:
    :param biased_label:
    :param biasing_factor:
    :return:
    """
    df_biased = pd.DataFrame(columns=df_a.columns)
    for label in sorted(df_a[label_column].unique()):
        df_label_a = df_a[df_a[label_column] == label]
        df_label_b = df_b[df_b[label_column] == label]
        if label == biased_label:
            df_biased = df_biased.append(df_label_a, ignore_index=True)
            df_sampled_b = sampling_method(df_label_b, bias_column, biasing_factor)
            df_biased = df_biased.append(df_sampled_b, ignore_index=True)
        else:
            df_biased = df_biased.append(df_label_a, ignore_index=True)
            df_biased = df_biased.append(df_label_b, ignore_index=True)
    return df_biased


def bias_binary_rank_aggressive(df, label_column, bias_column,
                                biased_label=1, biasing_factor=0.5):
    """
    Biases selected class by biasing factor, and uses same factor to inversely bias all other classes.
    :param df:
    :param label_column:
    :param bias_column:
    :param biased_label:
    :param biasing_factor:
    :return:
    """
    df_biased = pd.DataFrame(columns=df.columns)
    df_label = df[df[label_column] == biased_label]
    df_not_label = df[df[label_column] != biased_label]
    num_samples = int(len(df_label) * biasing_factor)
    df_sampled_not_label = df_not_label.sort_values(by=bias_column, ascending=True).head(num_samples)
    df_sampled_label = df_label.sort_values(by=bias_column, ascending=False).head(num_samples)
    df_biased = df_biased.append(df_sampled_not_label, ignore_index=True)
    df_biased = df_biased.append(df_sampled_label, ignore_index=True)
    return df_biased


def bias_binary_rank_gentle(df, label_column, bias_column, biased_label=1, biasing_factor=0.5):
    """
    Biases selected class by biasing factor, and leaves other classes untouched.
    :param df:
    :param label_column:
    :param bias_column:
    :param biased_label:
    :param biasing_factor:
    :return:
    """
    df_biased = pd.DataFrame(columns=df.columns)
    df_label = df[df[label_column] == biased_label]
    df_not_label = df[df[label_column] != biased_label]
    num_samples = int(len(df_label) * biasing_factor)
    df_sampled_not_label = df_not_label.sort_values(by=bias_column, ascending=True).head(num_samples)
    df_biased = df_biased.append(df_sampled_not_label, ignore_index=True)
    df_biased = df_biased.append(df_label, ignore_index=True)
    return df_biased


def validate_dataset(df, stats_columns, bias_column, label_column, logger=None):
    if not logger:
        logger = init_logger("validate_dataset")
    logger.info(f"Num reviews: {len(df)}")
    logger.info(f"{df.columns}")
    for col in df.columns:
        if col.endswith("_label"):
            logger.info(f"{df[col].value_counts(dropna=False)}\n")
    for col in stats_columns:
        col_vals = df[col]
        logger.info(f"{col} statistics:")
        logger.info(f"Min: {col_vals.min()}")
        logger.info(f"Max: {col_vals.max()}")
        logger.info(f"Std: {col_vals.std()}")
        logger.info(f"Mean: {col_vals.mean()}")
        logger.info(f"Median: {col_vals.median()}")
    logger.info(
        f"Correlation between {bias_column} and {label_column}: {df[bias_column].corr(df[label_column].astype(float))}\n")
