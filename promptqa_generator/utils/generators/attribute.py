from collections import Counter
from typing import List
import inflect
from .utils import make_and_description, make_one_data, normalize_attributes
from .attribute_classifier import AttributeClassifier
from ..base import BaseGenerator
from ..dataset import AttributeDataset, Attributes


class ExistsAttributeGenerator(BaseGenerator):
    qa_templates = [
        {
            "prompt": "How many {name}?",
            "response": "{count}."
        }
    ]
    des_templates = [
        {
            "description": "There {be} {count_name}."
        }
    ]
    inflect_engine = inflect.engine()

    def __init__(self, dataset: AttributeDataset, numeric=True, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.numeric = numeric

    def _generate(self, annotation: Attributes, templates: List) -> List:
        data_list = []

        labels = []
        for attr, label in zip(annotation.attributes, annotation.labels):
            if len(attr):
                attr = make_and_description(normalize_attributes(attr), self.rng)
                labels.append(f'{attr} {label}')

        for name, cnt in Counter(labels).items():
            be = 'is'
            if self.inflect_engine.singular_noun(name):
                if self.template_mode == 'qa':
                    continue
                be = 'are'
                cnt_word = ''
                count_name = name
            else:
                if cnt > 1:
                    be = 'are'
                    name = self.inflect_engine.plural(name)
                cnt_word = str(cnt) if self.numeric else self.inflect_engine.number_to_words(cnt)
                count_name = cnt_word + ' ' + name

            data_list += make_one_data(
                {
                    "name": name,
                    "count_name": count_name,
                    "be": be,
                    "count": cnt_word
                },
                templates=templates,
                rng=self.rng,
                enumerate_templates=True
            )
        return data_list


class AttributeBBoxGenerator(BaseGenerator):
    qa_templates = [
        {
            "prompt": "What are {attribute_type} of {name}?",
            "response": "{attribute_values}."
        }
    ]
    des_templates = [
        {
            "description": "The {name} is {attribute_values}."
        }
    ]
    attribute_classifier = AttributeClassifier()

    def _generate(self, annotation: Attributes, templates: List) -> List:
        data_list = []

        for name, attributes in zip(annotation.labels, annotation.attributes):
            if not attributes:
                continue

            if self.template_mode == 'qa':
                type_to_attributes = {}
                for attr in attributes:
                    attribute_type = self.attribute_classifier.classify(attr)
                    if not attribute_type:
                        continue  # 如果分类器返回 None 或空值，跳过这个属性

                    if attribute_type not in type_to_attributes:
                        type_to_attributes[attribute_type] = []
                    type_to_attributes[attribute_type].append(attr)

                for attribute_type, attr in type_to_attributes.items():
                    if not attr:
                        continue
                    attr = make_and_description(normalize_attributes(attr), self.rng)
                    data_list += make_one_data(
                        {
                            "name": name or "unknown",
                            "attribute_values": attr or "unknown",
                            "attribute_type": attribute_type or "unknown",
                        },
                        templates=templates,
                        rng=self.rng,
                        enumerate_templates=True
                    )
            else:
                attr = make_and_description(normalize_attributes(attributes), self.rng)
                data_list += make_one_data(
                    {
                        "name": name or "unknown",
                        "attribute_values": attr or "unknown",
                    },
                    templates=templates,
                    rng=self.rng,
                    enumerate_templates=True
                )

        return data_list



AttributeGeneratorList = [
    ExistsAttributeGenerator,
    AttributeBBoxGenerator
]
