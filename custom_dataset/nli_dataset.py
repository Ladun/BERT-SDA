
import logging

import torch
from torch.utils.data import TensorDataset

from custom_dataset.dataset_utils import InputExample, InputFeatures, DataProcessor


logger = logging.getLogger(__name__)


class CustomNLIProcessor(DataProcessor):

    def get_train_examples(self, dataset_name):
        """Gets a collection of :class:`InputExample` for the train set."""
        return self._create_examples(
            self._read_huggingface_datasets(dataset_name, data_type='train'), "train")

    def get_dev_examples(self, dataset_name):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise self._create_examples(
            self._read_huggingface_datasets(dataset_name, data_type='validation'), "validation")

    def get_test_examples(self, dataset_name):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise self._create_examples(
            self._read_huggingface_datasets(dataset_name, data_type='test'), "test")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return [0, 1, 2]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(dataset):
            guid = "%s-%s" % (set_type, i)
            text_a = line['premise']
            text_b = line['hypothesis']
            label = line['label'] if set_type != 'test' else None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def nli_convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):

    features = []
    for idx, example in enumerate(examples):
        if idx % 10000 == 0:
            logger.info("Writing example %d" % (idx))

        text = "[CLS] {} [SEP] {} [SEP]".format(example.text_a, example.text_b)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length
        )

        features.append(
            InputFeatures(input_ids=inputs['input_ids'],
                          attention_mask=inputs['input_ids'],
                          token_type_ids=inputs['input_ids'],
                          label=example.label))

    # Construct dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    if is_training:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_labels
        )
    else:
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids
        )

    return features, dataset




