# docstyle-ignore-file
import logging
from abc import ABC, abstractmethod
from typing import List, Union

import torch
from torch import nn

from .adapter_composition import AdapterCompositionBlock, Fuse, Parallel, Split, Stack, parse_composition
from .adapter_heads import (
    ClassificationHead,
    ModelWithFlexibleHeadsAdaptersMixin,
    MultiLabelClassificationHead,
    MultipleChoiceHead,
    QuestionAnsweringHead,
    TaggingHead,
)
from .adapter_model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin
from .adapter_modeling import Adapter, BertFusion


logger = logging.getLogger(__name__)


class BertAdaptersBaseMixin(ABC):
    """An abstract base implementation of adapter integration into a Transformer block.
    In BERT, subclasses of this module are placed in the BertSelfOutput module and in the BertOutput module.
    """

    # override this property if layer norm has a different name
    @property
    def layer_norm(self):
        return self.LayerNorm

    @property
    @abstractmethod
    def adapter_modules(self):
        """Gets the module dict holding the adapter modules available in this block."""
        pass

    @property
    @abstractmethod
    def adapter_config_key(self):
        """Gets the name of the key by which this adapter location is identified in the adapter configuration."""
        pass

    @property
    def layer_idx(self):
        return getattr(self, "_layer_idx", -1)

    @layer_idx.setter
    def layer_idx(self, layer_idx):
        idx = getattr(self, "_layer_idx", layer_idx)
        assert idx == layer_idx
        setattr(self, "_layer_idx", idx)

    def _init_adapter_modules(self):
        self.adapter_fusion_layer = nn.ModuleDict(dict())

    def add_adapter(self, adapter_name: str, layer_idx: int):
        self.layer_idx = layer_idx

        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config[self.adapter_config_key]:
            adapter = Adapter(
                input_size=self.config.hidden_size,
                down_sample=self.config.hidden_size // adapter_config["reduction_factor"],
                add_layer_norm_before=adapter_config["ln_before"],
                add_layer_norm_after=adapter_config["ln_after"],
                non_linearity=adapter_config["non_linearity"],
                residual_before_ln=adapter_config["adapter_residual_before_ln"],
            )
            self.adapter_modules[adapter_name] = adapter

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        """See BertModel.add_fusion_layer"""
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(",")
        if self.config.adapters.common_config_value(adapter_names, self.adapter_config_key):
            self.adapter_fusion_layer[",".join(adapter_names)] = BertFusion(self.config)

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        """Unfreezes a given list of adapters, the adapter fusion layer, or both

        :param adapter_names: names of adapters to unfreeze (or names of adapters part of the fusion layer to unfreeze)
        :param unfreeze_adapters: whether the adapters themselves should be unfreezed
        :param unfreeze_fusion: whether the adapter attention layer for the given adapters should be unfreezed
        """
        if unfreeze_adapters:
            for adapter_name in adapter_setup.flatten():
                if adapter_name in self.adapter_modules:
                    for param in self.adapter_modules[adapter_name].parameters():
                        param.requires_grad = True
        if unfreeze_fusion:
            if isinstance(adapter_setup, Fuse):
                if adapter_setup.name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[adapter_setup.name].parameters():
                        param.requires_grad = True
            for sub_setup in adapter_setup:
                if isinstance(sub_setup, Fuse):
                    if sub_setup.name in self.adapter_fusion_layer:
                        for param in self.adapter_fusion_layer[sub_setup.name].parameters():
                            param.requires_grad = True

    def get_adapter_preparams(
        self,
        adapter_config,
        hidden_states,
        input_tensor,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration
        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        query = None

        if adapter_config["residual_before_ln"]:
            residual = hidden_states

        if hasattr(self.config, "adapter_fusion") and self.config.adapter_fusion["query_before_ln"]:
            query = hidden_states

        if adapter_config["original_ln_before"] and self.layer_norm:
            hidden_states = self.layer_norm(hidden_states + input_tensor)

        if not adapter_config["residual_before_ln"]:
            residual = hidden_states

        if hasattr(self.config, "adapter_fusion") and not self.config.adapter_fusion["query_before_ln"]:
            query = hidden_states

        return hidden_states, query, residual

    def adapter_stack(self, adapter_setup: Stack, hidden_states, input_tensor, lvl=0):
        """
        Forwards the given input through the given stack of adapters.
        """
        for adapter_stack_layer in adapter_setup:
            # Break if setup is too deep
            if isinstance(adapter_stack_layer, AdapterCompositionBlock) and lvl >= 1:
                raise ValueError(
                    "Specified adapter setup is too deep. Cannot have {} at level {}".format(
                        adapter_stack_layer.__class__.__name__, lvl
                    )
                )
            # Case 1: We have a nested fusion layer -> call fusion method
            if isinstance(adapter_stack_layer, Fuse):
                hidden_states = self.adapter_fusion(adapter_stack_layer, hidden_states, input_tensor)
                up = hidden_states  # TODO
            # Case 2: We have a nested split layer -> call split method
            elif isinstance(adapter_stack_layer, Split):
                hidden_states = self.adapter_split(adapter_stack_layer, hidden_states, input_tensor)
                up = hidden_states  # TODO
            # Case 3: We have a single adapter which is part of this module -> forward pass
            elif adapter_stack_layer in self.adapter_modules:
                adapter_layer = self.adapter_modules[adapter_stack_layer]
                adapter_config = self.config.adapters.get(adapter_stack_layer)
                hidden_states, _, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)
                hidden_states, _, up = adapter_layer(hidden_states, residual_input=residual)
            # Case X: No adapter which is part of this module -> ignore

        return hidden_states, up

    def adapter_fusion(self, adapter_setup: Fuse, hidden_states, input_tensor, lvl=0):
        """
        Performs adapter fusion with the given adapters for the given input.
        """
        # config of _last_ fused adapter is significant
        adapter_config = self.config.adapters.get(adapter_setup.last())
        hidden_states, query, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)

        up_list = []

        for adapter_block in adapter_setup:
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                _, up = self.adapter_stack(adapter_block, hidden_states, input_tensor, lvl=lvl + 1)
            # Case 2: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.adapter_modules:
                adapter_layer = self.adapter_modules[adapter_block]
                _, _, up = adapter_layer(hidden_states, residual_input=residual)
            # Case 3: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore
            up_list.append(up)

        if len(up_list) > 0:
            up_list = torch.stack(up_list)
            up_list = up_list.permute(1, 2, 0, 3)

            hidden_states = self.adapter_fusion_layer[adapter_setup.name](
                query,
                up_list,
                up_list,
                residual,
            )

        return hidden_states

    def adapter_split(self, adapter_setup: Split, hidden_states, input_tensor, lvl=0):
        """
        Splits the given input between the given adapters.
        """
        # config of _first_ of splitted adapters is significant
        adapter_config = self.config.adapters.get(adapter_setup.first())
        hidden_states, query, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)

        # split hidden representations and residuals at split index
        split_hidden_states = [
            hidden_states[:, : adapter_setup.split_index, :],
            hidden_states[:, adapter_setup.split_index :, :],
        ]
        split_input_tensor = [
            input_tensor[:, : adapter_setup.split_index, :],
            input_tensor[:, adapter_setup.split_index :, :],
        ]
        split_residual = [
            residual[:, : adapter_setup.split_index, :],
            residual[:, adapter_setup.split_index :, :],
        ]

        for i, adapter_block in enumerate(adapter_setup):
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                _, up = self.adapter_stack(adapter_block, split_hidden_states[i], split_input_tensor[i], lvl=lvl + 1)
            # Case 2: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.adapter_modules:
                adapter_layer = self.adapter_modules[adapter_block]
                split_hidden_states[i], _, _ = adapter_layer(split_hidden_states[i], residual_input=split_residual[i])
            # Case 3: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore

        hidden_states = torch.cat(split_hidden_states, dim=1)
        return hidden_states

    def adapter_parallel(self, adapter_setup: Parallel, hidden_states, input_tensor):
        """For parallel execution of the adapters on the same input. This means that the input is repeated N times
        before feeding it to the adapters (where N is the number of adapters).

        TODO: this POC currently only works with a batchsize of 1 (initial input). The AdapterDrop paper also reports
        resuls with larger batchsizes, for which we have a POC in the old AdapterHub v1 (TODO: implement here)

        TODO: classification heads? We need to choose different heads for different instances in the batch.
        """
        # We assume that all adapters have the same config
        adapter_config = self.config.adapters.get(adapter_setup.children[0])
        hidden_states, _, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)

        # if the input batch size is equal to one, this means that here we have the first occurrence where we need to
        # repeat the input such that we can feed it into different adapters (and then process the normal transformer
        # layers in parallel
        if input_tensor.shape[0] == 1:
            # TODO change this such that we support batched input as well (which is then repeated N times)
            # This requires comparing the input batch size to the new one after repeating the input N times
            n_repeat = len(adapter_setup.children)
            input_tensor = input_tensor.repeat(n_repeat, 1, 1)
            hidden_states = hidden_states.repeat(n_repeat, 1, 1)
            # query = query.repeat(len(flat_adapter_names), 1, 1)
            residual = residual.repeat(n_repeat, 1, 1)

        # sequentially feed different parts of the blown-up batch into different adapters
        children_hidden = []
        for i, child in enumerate(adapter_setup.children):
            adapter_layer = self.adapter_modules[child]
            child_hidden_states, _, _ = adapter_layer(hidden_states[i : i + 1], residual_input=residual[i : i + 1])
            children_hidden.append(child_hidden_states)

        # concatenate all outputs and return
        hidden_states = torch.cat(tuple(children_hidden), 0)
        return hidden_states, input_tensor

    def adapters_forward(self, hidden_states, input_tensor):
        """
        Called for each forward pass through adapters.
        """
        skip_layer = (
            self.config.adapters.skip_layers is not None and self.layer_idx in self.config.adapters.skip_layers
        )
        adapter_setup = self.config.adapters.active_setup if hasattr(self.config, "adapters") else None
        if (
            not skip_layer
            and adapter_setup is not None
            and (len(set(self.adapter_modules.keys()) & adapter_setup.flatten()) > 0)
        ):
            if isinstance(adapter_setup, Stack):
                hidden_states, _ = self.adapter_stack(adapter_setup, hidden_states, input_tensor)
            elif isinstance(adapter_setup, Fuse):
                hidden_states = self.adapter_fusion(adapter_setup, hidden_states, input_tensor)
            elif isinstance(adapter_setup, Split):
                hidden_states = self.adapter_split(adapter_setup, hidden_states, input_tensor)
            elif isinstance(adapter_setup, Parallel):
                # notice that we are overriding input tensor here to keep the same dim as hidden_states for the residual
                # in case we were blowing up the batch for parallel processing of multiple adapters for the same input
                hidden_states, input_tensor = self.adapter_parallel(adapter_setup, hidden_states, input_tensor)
            else:
                raise ValueError(f"Invalid adapter setup {adapter_setup}")

            last_config = self.config.adapters.get(adapter_setup.last())
            if last_config["original_ln_after"] and self.layer_norm:
                hidden_states = self.layer_norm(hidden_states + input_tensor)

        elif self.layer_norm:
            hidden_states = self.layer_norm(hidden_states + input_tensor)
        else:
            hidden_states = hidden_states + input_tensor

        return hidden_states


class BertSelfOutputAdaptersMixin(BertAdaptersBaseMixin):
    """Adds adapters to the BertSelfOutput module."""

    @property
    def adapter_modules(self):
        return self.attention_adapters

    @property
    def adapter_config_key(self):
        return "mh_adapter"

    def _init_adapter_modules(self):
        super()._init_adapter_modules()
        self.attention_adapters = nn.ModuleDict(dict())


class BertOutputAdaptersMixin(BertAdaptersBaseMixin):
    """Adds adapters to the BertOutput module."""

    @property
    def adapter_modules(self):
        return self.output_adapters

    @property
    def adapter_config_key(self):
        return "output_adapter"

    def _init_adapter_modules(self):
        super()._init_adapter_modules()
        self.output_adapters = nn.ModuleDict(dict())


class BertLayerAdaptersMixin:
    """Adds adapters to the BertLayer module."""

    def add_fusion_layer(self, adapter_names):
        self.attention.output.add_fusion_layer(adapter_names)
        self.output.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str, layer_idx: int):
        self.attention.output.add_adapter(adapter_name, layer_idx)
        self.output.add_adapter(adapter_name, layer_idx)

    def enable_adapters(
        self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_attention: bool
    ):
        self.attention.output.enable_adapters(adapter_setup, unfreeze_adapters, unfreeze_attention)
        self.output.enable_adapters(adapter_setup, unfreeze_adapters, unfreeze_attention)


class BertEncoderAdaptersMixin:
    """Adds adapters to the BertEncoder module."""

    def add_fusion_layer(self, adapter_names):
        for layer in self.layer:
            layer.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str):
        adapter_config = self.config.adapters.get(adapter_name)
        leave_out = adapter_config.get("leave_out", [])
        for i, layer in enumerate(self.layer):
            if i not in leave_out:
                layer.add_adapter(adapter_name, i)

    def enable_adapters(
        self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_attention: bool
    ):
        for layer in self.layer:
            layer.enable_adapters(adapter_setup, unfreeze_adapters, unfreeze_attention)


class BertModelAdaptersMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the BertModel module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_adapter_modules(self):
        super()._init_adapter_modules()

        # add adapters specified in config; invertible adapter will only be added if required
        for adapter_name in self.config.adapters.adapters:
            self.encoder.add_adapter(adapter_name)
            self.add_invertible_adapter(adapter_name)
        # fusion
        if hasattr(self.config, "fusion_models"):
            for fusion_adapter_names in self.config.fusion_models:
                self.add_fusion_layer(fusion_adapter_names)

    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        """Sets the model into mode for training the given adapters."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.encoder.enable_adapters(adapter_setup, True, False)
        self.enable_invertible_adapters(adapter_setup.flatten())
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)

    def train_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.encoder.enable_adapters(adapter_setup, False, True)
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)
        # TODO implement fusion for invertible adapters

    def add_adapter(self, adapter_name: str, config=None):
        """Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            config (str or dict or AdapterConfig, optional): The adapter configuration, can be either:
                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
        """
        self.config.adapters.add(adapter_name, config=config)
        self.encoder.add_adapter(adapter_name)
        self.add_invertible_adapter(adapter_name)

    def _add_fusion_layer(self, adapter_names):
        self.encoder.add_fusion_layer(adapter_names)

    def get_fusion_regularization_loss(self):
        reg_loss = 0.0
        target = torch.zeros((self.config.hidden_size, self.config.hidden_size)).fill_diagonal_(1.0).to(self.device)
        for _, v in self.encoder.layer._modules.items():

            for _, layer_fusion in v.output.adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

            for _, layer_fusion in v.attention.output.adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

        return reg_loss


class BertModelHeadsMixin(ModelWithFlexibleHeadsAdaptersMixin):
    """Adds flexible heads to a BERT-based model class."""

    def add_prediction_head_from_config(self, head_name, config, overwrite_ok=False):
        id2label = (
            {id_: label for label, id_ in config["label2id"].items()}
            if "label2id" in config.keys() and config["label2id"]
            else None
        )
        if config["head_type"] == "classification":
            self.add_classification_head(
                head_name,
                config["num_labels"],
                config["layers"],
                config["activation_function"],
                id2label=id2label,
                overwrite_ok=overwrite_ok,
            )
        elif config["head_type"] == "multilabel_classification":
            self.add_classification_head(
                head_name,
                config["num_labels"],
                config["layers"],
                config["activation_function"],
                multilabel=True,
                id2label=id2label,
                overwrite_ok=overwrite_ok,
            )
        elif config["head_type"] == "tagging":
            self.add_tagging_head(
                head_name,
                config["num_labels"],
                config["layers"],
                config["activation_function"],
                id2label=id2label,
                overwrite_ok=overwrite_ok,
            )
        elif config["head_type"] == "multiple_choice":
            self.add_multiple_choice_head(
                head_name,
                config["num_choices"],
                config["layers"],
                config["activation_function"],
                id2label=id2label,
                overwrite_ok=overwrite_ok,
            )
        elif config["head_type"] == "question_answering":
            self.add_qa_head(
                head_name,
                config["num_labels"],
                config["layers"],
                config["activation_function"],
                id2label=id2label,
                overwrite_ok=overwrite_ok,
            )
        else:
            if config["head_type"] in self.config.custom_heads:
                self.add_custom_head(head_name, config, overwrite_ok=overwrite_ok)
            else:
                raise AttributeError("Please register the PredictionHead before loading the model")

    def add_classification_head(
        self,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        overwrite_ok=False,
        multilabel=False,
        id2label=None,
    ):
        """Adds a sequence classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
            multilabel (bool, optional): Enable multilabel classification setup. Defaults to False.
        """

        if multilabel:
            head = MultiLabelClassificationHead(head_name, num_labels, layers, activation_function, id2label, self)
        else:
            head = ClassificationHead(head_name, num_labels, layers, activation_function, id2label, self)
        self.add_prediction_head(head, overwrite_ok)

    def add_multiple_choice_head(
        self, head_name, num_choices=2, layers=2, activation_function="tanh", overwrite_ok=False, id2label=None
    ):
        """Adds a multiple choice head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_choices (int, optional): Number of choices. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = MultipleChoiceHead(head_name, num_choices, layers, activation_function, id2label, self)
        self.add_prediction_head(head, overwrite_ok)

    def add_tagging_head(
        self, head_name, num_labels=2, layers=1, activation_function="tanh", overwrite_ok=False, id2label=None
    ):
        """Adds a token classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 1.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = TaggingHead(head_name, num_labels, layers, activation_function, id2label, self)
        self.add_prediction_head(head, overwrite_ok)

    def add_qa_head(
        self, head_name, num_labels=2, layers=1, activation_function="tanh", overwrite_ok=False, id2label=None
    ):
        head = QuestionAnsweringHead(head_name, num_labels, layers, activation_function, id2label, self)
        self.add_prediction_head(head, overwrite_ok)
