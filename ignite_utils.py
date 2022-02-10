from typing import Dict, Tuple

import torch
import transformers
from ignite.engine import Engine
from ignite.metrics import Metric, Precision, Recall
from ignite.utils import convert_tensor


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.
        Batch is of the shape: (list(x), list(masks), list(y))
    """

    def _convert(tens):
        return convert_tensor(tens, device=device, non_blocking=non_blocking)

    (x, x_mask), y = batch
    return {'input_ids': _convert(x),
            'attention_mask': _convert(x_mask),
            'labels': _convert(y)}


def transformers_prep_batch(batch, device=None, non_blocking=False):
    """
    Prepare a batch for transformers training pass.
    :param batch: Pair of (input, label_dict) where:
        - Input is a tuple of tensors
            (input_ids, attention_mask, token_type_ids)
        - label_dict is a dictionary str -> int
    :param device:
    :param non_blocking:
    :return: Dictionary of:
        {
            input_ids: torch.Tensor (N, L),
            token_type_ids: torch.Tensor (N, L),
            attention_mask: torch.Tensor (N, L),
            labels: dict[str, torch.Tensor]
        }
        Where N is the batch size and L is the maximum length.
    """
    processed_batch = {}

    if len(batch) == 3:
        sample_ids, inputs, labels = batch
        processed_batch['sample_ids'] = sample_ids
    else:
        inputs, labels = batch

    input_ids, attention_masks, token_type_ids = inputs

    def _convert(tens):
        return convert_tensor(tens, device=device, non_blocking=non_blocking)

    processed_batch.update({
        'input_ids': _convert(input_ids),
        'attention_mask': _convert(attention_masks),
        'token_type_ids': _convert(token_type_ids),
        'labels': {task_name: _convert(task_labels)
                   for task_name, task_labels in labels.items()}
    })
    return processed_batch


def _dict_output_transform(x, y, y_pred, loss=None):
    return {
        'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
        'y_pred': y_pred,
        'y': y
    }


def create_transformers_trainer(model: transformers.PreTrainedModel,
                                optimizer, device=None,
                                non_blocking=False,
                                prepare_batch=_prepare_batch,
                                output_transform=_dict_output_transform,
                                average_outputs=False):
    """
    Factory function for creating a trainer for supervised
    models from the transformers library.

    Args:
        model (`transformers.PreTrainedModel`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification.
        Defaults None. Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between
        CPU and GPU, the copy may occur asynchronously
        with respect to the host.
        For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives
            `batch`, `device`, `non_blocking` and outputs
            a dictionary of all the kwargs that need
            to be passed to model.
        output_transform (callable, optional): function that receives
            'x', 'y', 'y_pred', 'loss' and returns
            value to be assigned to engine's state.output after
            each iteration.
            Default is returning a dict for loss, y_pred, y.
        average_outputs (boolean, defaults False): Whether or not
            to average the outputs over the first dimension
            (used for multi-gpu training).

    Note: `engine.state.output` for this engine is defined by
        `output_transform` parameter and is a dict with
        y, y_pred and loss by default.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        batch = prepare_batch(
            batch,
            device=device,
            non_blocking=non_blocking
        )
        loss, y_pred = model(**batch)[:2]
        if average_outputs:
            loss = loss.mean(dim=0)
            if not isinstance(y_pred, dict):
                y_pred = y_pred.mean(dim=0)
        loss.backward()
        optimizer.step()
        return output_transform(batch['input_ids'], batch['labels'],
                                y_pred, loss)

    return Engine(_update)


def create_transformers_evaluator(model: transformers.PreTrainedModel,
                                  metrics: Dict[str, Metric] = None,
                                  device=None, non_blocking=False,
                                  prepare_batch=_prepare_batch,
                                  output_transform=_dict_output_transform,
                                  average_outputs=False):
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`transformers.PreTrainedModel`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`):
            a map of metric names to Metrics.
        device (str, optional): device type specification.
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is
            between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument
            has no effect.
        prepare_batch (callable, optional): function that receives
            `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_masks, batch_y)`.
        output_transform (callable, optional): function that receives
            'x', 'y', 'y_pred' and returns value to be assigned to
            engine's state.output after each iteration.
            Default is returning `{'y_pred': y_pred, 'y': y}`
            which fits output expected by metrics.
            If you change it you should use
            `output_transform` in metrics.
        average_outputs (boolean, defaults False): Whether or not
            to average the outputs over the first dimension
            (used for multi-gpu training).

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = prepare_batch(batch, device=device,
                                  non_blocking=non_blocking)
            y_pred = model(**batch)[1]
            if average_outputs:
                if not isinstance(y_pred, dict):
                    y_pred = y_pred.mean(dim=0)
            return output_transform(
                batch['input_ids'],
                batch['labels'],
                y_pred
            )

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def output_for_metrics(output, task_name='sentiment') -> \
        Tuple[torch.Tensor, torch.Tensor]:
    return output['y_pred'][task_name], output['y'][task_name]


class IgnoreIndexMixin:
    def __init__(self, *args, ignore_index=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index

    def update(self, output):
        if self.ignore_index is not None:
            y_pred, y = output
            mask = y != self.ignore_index
            y_pred = y_pred[mask]
            y = y[mask]
            output = y_pred, y
            if y.shape[0] == 0:
                return

        super().update(output)


class PrecisionIgnore(IgnoreIndexMixin, Precision):
    pass


class RecallIgnore(IgnoreIndexMixin, Recall):
    pass


def FbetaIgnore(beta, average=True, output_transform=None, device=None,
                ignore_index=None):
    """Calculates F-beta score

    Args:
        beta (float): weight of precision in harmonic mean
        average (bool, optional): if True, F-beta score is computed
            as the unweighted average (across all classes in multiclass
             case), otherwise, returns a tensor with F-beta score for
              each class in multiclass case.
        output_transform (callable, optional): a callable that is used
            to transform the :class:`~ignite.engine.Engine`'s
            `process_function`'s output into the form expected by the
            metric.
            It is used only if precision or recall are not provided.
        device (str of torch.device, optional): device specification in
            case of distributed computation usage.
            In most of the cases, it can be defined as
            "cuda:local_rank" or "cuda" if already set
            `torch.cuda.set_device(local_rank)`.
            By default, if a distributed process group is
            initialized and available, device is set to `cuda`.
        ignore_index (int, optional): index to ignore in metric
            computation.
    Returns:
        MetricsLambda, F-beta metric
    """
    if not (beta > 0):
        raise ValueError("Beta should be a positive integer, but given {}".format(beta))

    precision = PrecisionIgnore(
        ignore_index=ignore_index,
        output_transform=(lambda x: x) if output_transform is None else output_transform,
        average=False, device=device
    )

    recall = RecallIgnore(
        ignore_index=ignore_index,
        output_transform=(lambda x: x) if output_transform is None else output_transform,
        average=False, device=device
    )

    fbeta = (1.0 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-15)

    if average:
        fbeta = fbeta.mean().item()

    return fbeta
