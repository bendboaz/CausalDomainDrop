from enum import Enum, auto


# def parse_single_link(link, prefix=None, suffix=None):
#     if prefix is not None:
#         actual_prefix, link = link[:len(prefix)], link[len(prefix):]
#         assert actual_prefix == prefix
#     if suffix is not None:
#         link, actual_suffix = link[:-len(suffix)], link[-len(suffix):]
#         assert actual_suffix == suffix
#     return link


def get_experiment_title(config, old_title=False, **kwargs):
    tc_index = config.get('tc_index', None)
    cc_index = config.get('cc_index', None)
    shap = config.get('shap', False)
    shap_footer = 'S' if shap else ''
    tc_section = '' if tc_index is None else f'_T{tc_index}{shap_footer}'
    cc_section = '' if cc_index is None else f'_C{cc_index}{shap_footer}'
    ngram_title = config['n_grams']
    if old_title:
        ngram_title = ngram_title.upper()
    return f"{config['model_class']}_{config['n_pivots']}pv_{ngram_title}grams{tc_section}{cc_section}"


# def reverse_experiment_title(title):
#     params = {}
#     title = title.split('_')
#     params['model_class'] = title[0]
#     n_pivots = parse_single_link(title[1], suffix='pv')
#     params['n_pivots'] = int(n_pivots)
#     ngram_type = parse_single_link(title[2], suffix='grams')
#     params['n_grams'] = str(ConceptType.str2concept(ngram_type))
#     if len(title) > 3:
#         concept1_prefix, concept1_idx = title[3][:1], title[3][1:]
#         concept1_prefix = concept1_prefix.lower() + 'c_index'
#         params[concept1_prefix] = int(concept1_idx)
#     if len(title) > 4:
#         concept1_prefix, concept1_idx = title[4][:1], title[4][1:]
#         concept1_prefix = concept1_prefix.lower() + 'c_index'
#         params[concept1_prefix] = int(concept1_idx)
#     return params


def get_additional_pretrain_title(config, **kwargs):
    """ - task
        - gradient_accumulation_steps
        - batch_size
        - warmup_steps
        - adam_epsilon
        - learning_rate"""
    task = config['task']
    pretrain_params = config['pretrain']
    accumulation = pretrain_params['gradient_accumulation_steps']
    batch = pretrain_params['batch_size']
    warmup = pretrain_params['warmup_steps']
    adam_e = pretrain_params['adam_epsilon']
    lr = pretrain_params['learning_rate']
    return f"{get_experiment_title(config)}_{task}_{accumulation}acc_{batch}batch_{warmup}warm_{adam_e}epsilon_{lr}lr"


# def reverse_additional_pretrain_title(title):
#     title = title.split('_')
#     params = reverse_experiment_title('_'.join(title[:-6]))
#     params['task'] = title[-6]
#     accumulation = parse_single_link(title[-5], suffix='acc')
#     batch = parse_single_link(title[-4], suffix='batch')
#     warmup = parse_single_link(title[3], suffix='warm')
#     adam_e = parse_single_link(title[-2], suffix='epsilon')
#     lr = parse_single_link(title[-1], suffix='lr')
#     params['pretrain'] = {
#         'gradient_accumulation_steps': accumulation,
#         'batch_size': batch,
#         'warmup_steps': warmup,
#         'adam_epsilon': adam_e,
#         'learning_rate': lr
#     }
#     return params


def get_finetune_params_str(config, **kwargs):
    ft_params = config['ft']
    batch_size = ft_params['batch_size']
    init_lr = ft_params['init_lr']
    lr_gamma = ft_params['lr_gamma']
    max_epochs = ft_params['max_epochs']
    weight_decay = ft_params['weight_decay']
    return f'ftbatch{batch_size}_ftinitlr{init_lr}_ftlrgamma{lr_gamma}_ftepochs{max_epochs}_ftwd{weight_decay}'


# def reverse_finetune_params_str(title):
#     title = title.split('_')
#     batch = parse_single_link(title[0], prefix='ftbatch')
#     init_lr = parse_single_link(title[1], prefix='ftinitlr')
#     lr_gamma = parse_single_link(title[2], prefix='ftlrgamma')
#     epochs = parse_single_link(title[3], prefix='ftepochs')
#     weight_decay = parse_single_link(title[4], prefix='ftwd')
#     return {
#         'batch_size': batch,
#         'init_lr': init_lr,
#         'lr_gamma': lr_gamma,
#         'max_epochs': epochs,
#         'weight_decay': weight_decay
#     }


def get_domain_title(config, is_target=False, **kwargs):
    relevant_entry = 'target' if is_target else 'source'
    domain_info = config[relevant_entry]
    return f"{domain_info['dataset']}_{domain_info['domain']}"


# def reverse_domain_title(title):
#     # This is the only non-trivial part here, IMO.
#     pass


def get_details_file_name(config, is_gold=False, **kwargs):
    if is_gold:
        return f"{get_domain_title(config)}_{get_domain_title(config, True)}_{get_finetune_params_str(config)}_orig"
    else:
        return f"{get_additional_pretrain_title(config)}_{get_finetune_params_str(config)}_CF"


# def reverse_details_file_name(title: str):
#     if title.endswith('CF'):
#         # modified models
#         pass
#     else:
#         # original models (is_gold=True)
#         pass


class ConceptType(Enum):
    UNIGRAM = auto()
    BIGRAM = auto()
    CLUSTER = auto()

    @staticmethod
    def concept_name_dict():
        return {
            'uni': ConceptType.UNIGRAM,
            'bi': ConceptType.BIGRAM,
            'kmeans': ConceptType.CLUSTER
        }

    def __str__(self):
        for name, value in self.concept_name_dict().items():
            if value is self:
                return name

    @classmethod
    def str2concept(cls, description: str):
        normalized = description.lower()
        return ConceptType.concept_name_dict()[normalized]


def pivot_names_fileformat(partition, n_pivots, concept_type=ConceptType.UNIGRAM, shap=False):
    shap_footer = 'S' if shap else ''
    return f'pivot_names_{concept_type}_{n_pivots}{shap_footer}.json'


# def reverse_pivot_names_fileformat(title):
#     pass


def full_pivots_csv_format(partition, n_pivots, concept_type=ConceptType.UNIGRAM, shap=False):
    shap_footer = 'S' if shap else ''
    return f'{partition}_wpivots_{concept_type}_{n_pivots}{shap_footer}.csv'

# def reverse_full_pivots_csv_format(title):
#     pass
