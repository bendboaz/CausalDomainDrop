import json
import logging
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Optional, Dict

import torch
from torch import nn
from torch.autograd import Function
from transformers import BertConfig, BertPreTrainedModel, BertModel, PretrainedConfig
from transformers.modeling_bert import BertPredictionHeadTransform, \
    BertLMPredictionHead

from utils import MaskedAvgPooler, change_module_grad


class FineTuningHead(torch.nn.Module):
    DEFAULT_GRL_ALPHA = 1.0
    WEIGHTS_NAME = 'pytorch_model.bin'
    CONFIG_NAME = 'config.json'
    CLASS_DATA = 'specific.json'

    def __init__(self, config: Optional[BertConfig], num_labels=2,
                 do_transform=True, do_grl=False):
        super(FineTuningHead, self).__init__()
        self.config = config
        self.num_labels = num_labels
        self.do_transform = do_transform
        self.do_grl = do_grl

        self.pooler = MaskedAvgPooler()

        if self.config is not None:
            self.init_config_dependent_attributes()

    def init_config_dependent_attributes(self):
        if self.do_transform:
            self.add_module('transform', BertPredictionHeadTransform(self.config))
        if not hasattr(self, 'classifier'):
            self.__setattr__('classifier', nn.Linear(self.config.hidden_size,
                                                     self.num_labels))

    def set_config(self, config: BertConfig):
        self.config = config
        self.init_config_dependent_attributes()

    def forward(self, sequence_outputs, sequence_masks, **kwargs):
        pooled_outputs = self.pooler(sequence_outputs, sequence_masks)
        if self.do_grl:
            alpha = kwargs.get('alpha', self.DEFAULT_GRL_ALPHA)
            pooled_outputs = ReverseLayerF.apply(pooled_outputs, alpha)

        if self.do_transform:
            pooled_outputs = self.transform(pooled_outputs)

        return self.classifier(pooled_outputs)

    def save_pretrained(self, save_dir):
        """ Save a model and its configuration file to a directory, so that it
                    can be re-loaded using the `:func:`~FineTuningHead.from_pretrained`` class method.
            Adapted from huggingface's transformers package.
        """
        assert os.path.isdir(
            save_dir
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        if self.config is not None:
            # Attach architecture to the config
            model_to_save.config.architectures = [model_to_save.__class__.__name__]

            # Save configuration file
            model_to_save.config.save_pretrained(save_dir)

        class_data = {
            'num_labels': self.num_labels,
            'do_transform': self.do_transform,
            'do_grl': self.do_grl
        }
        with open(os.path.join(save_dir, self.CLASS_DATA), 'w+') as dump_file:
            json.dump(class_data, dump_file)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_dir, self.WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logging.info("Model weights saved in {}".format(output_model_file))

    @classmethod
    def from_pretrained(cls, pretrained_path, *model_args, **kwargs):
        """
        Adapted from huggingface's transformers package.
        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)

        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_path
            config, model_kwargs = BertConfig.from_pretrained(
                config_path,
                return_unused_kwargs=True,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        if os.path.isdir(pretrained_path):
            if os.path.isfile(os.path.join(pretrained_path, cls.WEIGHTS_NAME)):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(pretrained_path, cls.WEIGHTS_NAME)
            else:
                raise EnvironmentError(f'Error no file named {cls.WEIGHTS_NAME} '
                                       f'found in directory {pretrained_path}')
        elif os.path.isfile(pretrained_path):
            archive_file = pretrained_path
        else:
            raise EnvironmentError(f'Error no file or directory named '
                                   f'{pretrained_path} found.')

        with open(os.path.join(pretrained_path, cls.CLASS_DATA), 'r+') as dump_file:
            class_data = json.load(dump_file)

        class_data.update(model_kwargs)
        # Instantiate model:
        model = cls(config, *model_args, **class_data)

        if state_dict is None:
            try:
                state_dict = torch.load(archive_file, map_location="cpu")
            except Exception:
                raise OSError(
                    "Unable to load weights from pytorch checkpoint file. "
                )

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        model.load_state_dict(state_dict)

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()
        return model


class BertForFineTuning(BertPreTrainedModel):
    FINETUNE_HEADS_SAVE_DIR = 'finetuning_heads'

    def __init__(self, config: BertConfig,
                 finetuning_tasks: Dict[str, FineTuningHead] = None,
                 unfreeze_layers=0):
        super(BertForFineTuning, self).__init__(config)
        if finetuning_tasks is None:
            finetuning_tasks = {}

        self.unfreeze_layers = unfreeze_layers
        self.finetuning_tasks = OrderedDict()
        self.config = config

        self.bert = BertModel(config)

        change_module_grad(self.bert, False)  # Freezing BERT
        map(
            partial(change_module_grad, new_grad=True),
            self.bert.encoder.layer[:unfreeze_layers],
        )

        for task_name, task in sorted(finetuning_tasks.items(), key=lambda x: x[0]):
            self.add_finetuning_task(task_name, task)

        self.init_weights()
        self.tie_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None,
                labels: Dict[str, torch.Tensor] = None, **kwargs):
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=device)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        sequence_outputs = outputs[0]
        prediction_scores = {}
        for name in self.finetuning_tasks:
            head = self.__getattr__(f'task_{name}')
            prediction_scores[name] = head(sequence_outputs, attention_mask,
                                           **kwargs)

        # add hidden states and attention if they are here:
        outputs = ((prediction_scores,)
                   + outputs[2:])

        loss = 0
        if labels is not None:
            for (task_name, task_labels), scores, ft_head \
                    in zip(
                labels.items(),
                prediction_scores.values(),
                self.finetuning_tasks.values(),
            ):
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
                loss = loss_fct(
                    scores.view(-1, ft_head.num_labels),
                    task_labels.view(-1)
                )
                loss += loss

            outputs = (loss,) + outputs
        # Outputs:
        #   (
        #   [loss],
        #   prediction_scores: Dict,
        #   [hidden_states],
        #   [attentions]
        #   )
        return outputs

    def add_finetuning_task(self, name, finetuning_head, override=False):
        if name not in self.finetuning_tasks or override:
            self.finetuning_tasks[name] = finetuning_head
            finetuning_head.set_config(self.config)
            self.add_module(f'task_{name}', finetuning_head)
        else:
            logging.warning('A task named {} was already registered in {} '
                            'instance. If you want to override an existing '
                            'finetuning task, please add \'override=True\' '
                            'to your add_finetuning_task() call',
                            name, type(self).__name__)
        self.finetuning_tasks = OrderedDict(sorted(self.finetuning_tasks.items(),
                                                   key=lambda x: x[0]))

    def save_pretrained(self, save_directory):
        heads_dir = Path(save_directory) / self.FINETUNE_HEADS_SAVE_DIR
        heads_dir.mkdir(parents=True, exist_ok=True)
        for head_name, head in self.finetuning_tasks.items():
            head_path = heads_dir / head_name
            head_path.mkdir(parents=True, exist_ok=True)
            head.save_pretrained(head_path)
        super().save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        finetuning_tasks = kwargs.get('finetuning_tasks', None)
        heads = OrderedDict() if finetuning_tasks is None else finetuning_tasks
        if finetuning_tasks is None:
            if not os.path.isdir(pretrained_model_name_or_path):
                logging.info('{} is not a directory path, therefore no finetuning '
                             'heads are loaded with the model. To use it for '
                             'predictions, please add prediction heads with '
                             'add_finetuning_task()')
            else:
                heads_dir = Path(pretrained_model_name_or_path, cls.FINETUNE_HEADS_SAVE_DIR)
                for head_name in sorted(os.listdir(heads_dir)):
                    head = FineTuningHead.from_pretrained(heads_dir / head_name)
                    heads[head_name] = head

        kwargs['finetuning_tasks'] = heads
        if 'config' not in kwargs:
            kwargs['config'] = BertConfig.from_pretrained(pretrained_model_name_or_path)
        model = super().from_pretrained(pretrained_model_name_or_path,
                                        *model_args,
                                        **kwargs)
        return model


class AdditionalPretrainingHeads(nn.Module):
    def __init__(self, config: BertConfig,
                 additional_tasks: Dict[str, FineTuningHead]):
        """
        Module to combine MLM head with additional sequence tasks for
            pretraining BERT.
        :param config: BertConfig.
        :param additional_tasks: A dictionary of name -> FineTuningHead
        """
        super(AdditionalPretrainingHeads, self).__init__()
        self.config = config
        self.mlm_predictions = BertLMPredictionHead(config)
        self.additional_tasks = additional_tasks

        for task_name, task in additional_tasks.items():
            task.set_config(self.config)
            self.add_module(f'task_{task_name}', task.classifier)

    def forward(self, sequence_output, sequence_masks, **kwargs):
        lm_prediction_scores = self.mlm_predictions(sequence_output)
        additional_prediction_scores = {
            name: task(sequence_output, sequence_masks, **kwargs)
            for name, task in self.additional_tasks.items()
        }
        return lm_prediction_scores, additional_prediction_scores


class BertForAdditionalPreTraining(BertPreTrainedModel):
    MLM_NAME = 'masked_lm'

    def __init__(self, config: BertConfig,
                 aux_tasks: Dict[str, FineTuningHead] = None,
                 unfreeze_layers=0):
        super(BertForAdditionalPreTraining, self).__init__(config)
        if aux_tasks is None:
            aux_tasks = {}
        self.unfreeze_layers = unfreeze_layers
        self.config = config

        self.bert = BertModel(self.config)
        change_module_grad(self.bert, False)  # Freezing BERT

        map(partial(change_module_grad, new_grad=True),
            self.bert.encoder.layer[:self.unfreeze_layers])

        self.cls = AdditionalPretrainingHeads(self.config, aux_tasks)

        self.init_weights()
        self.tie_weights()

    def get_output_embeddings(self):
        return self.cls.mlm_predictions.decoder

    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        sequence_output, pooled_output = outputs[:2]

        lm_prediction_scores, additional_prediction_score = self.cls(
            sequence_output,
            attention_mask,
            **kwargs
        )

        # add hidden states and attention if they are here:
        outputs = ((lm_prediction_scores, additional_prediction_score,)
                   + outputs[2:])

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = 0
            masked_lm_labels = labels[self.MLM_NAME]
            del labels[self.MLM_NAME]

            if masked_lm_labels is not None:
                masked_lm_loss = loss_fct(
                    lm_prediction_scores.view(-1, self.config.vocab_size),
                    masked_lm_labels.view(-1)
                )
                loss += masked_lm_loss

            if labels is not None:
                for task_name, task_labels in labels:
                    task_loss = loss_fct(
                        additional_prediction_score[task_name].view(-1, 2),
                        task_labels.view(-1)
                    )
                    loss += task_loss

            outputs = (loss,) + outputs

        # Outputs now:
        #   (
        #   [loss],
        #   mlm_prediction_scores,
        #   additional_prediction_scores,
        #   [hidden_states],
        #   [attentions]
        #   )

        return outputs


class ReverseLayerF(Function):
    """
    Taken from https://github.com/fungtion/DANN_py3
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
