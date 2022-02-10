# This was meant as a script to rename files into matching formats and index what experiments we have already run.


# import os
# import json
#
# import pandas as pd
#
# from filename_formats import get_experiment_title, get_additional_pretrain_title, get_finetune_params_str, \
#     get_domain_title, get_details_file_name, pivot_names_fileformat, full_pivots_csv_format
# from paths import DATA_ROOT, EXPERIMENTS_ROOT

# concept words
# {dataset_path} / {pivot_names_fileformat}

# partitions with concepts already created
# {dataset_path} / {full_pivots_filename}

# directories for pregenerated epochs
# {source_path} / {get_experiment_title(config)}

# pretrained models
# {exps_path} / 'pretraining' / {source_title} / {get_additional_pretrain_title(config)}

# modified finetuned models
# {exps_path} / 'finetuned' / {get_domain_title(config)} / {get_details_file_name(config)}

# original finetuned models
# {exps_path} / 'finetuned' / {get_domain_title(config)} / {get_details_file_name(config, True)}

# columns in 'pairs' files
# {exps_path} / 'pairs' / {get_domain_title(config)}_{get_domain_title(config, is_target=True)}.csv

# per-sample predictions files
# {exps_path} / 'per_sample' / {get_domain_title(config)}_{get_domain_title(config, is_target=True)} / {get_details_file_name(config, is_gold=True)}.csv'
# {exps_path} / 'per_sample' / {get_domain_title(config)}_{get_domain_title(config, is_target=True)} / {get_details_file_name(config, is_gold=False)}.csv

if __name__ == '__main__':
    pass
