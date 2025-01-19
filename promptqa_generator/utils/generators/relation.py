from collections import defaultdict
from typing import List

import inflect

from .utils import make_and_description, make_one_data
from ..base import BaseGenerator
from ..dataset import RelationDataset, Relations


class ExistsRelationGenerator(BaseGenerator):
	qa_templates = [
		{
			"prompt"  : "What is the relation between {object1} and {object2}?",
			"response": "{object1} {be} {relation} {object2}"
		}
	]
	des_templates = [
		{
			"description": "{object1} {be} {relation} {object2}"
		}
	]
	inflect_engine = inflect.engine()

	def __init__(self, dataset: RelationDataset, numeric=True, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.numeric = numeric

	def _generate(self, annotation: Relations, templates: List) -> List:
		data_list = []

		agg_relations = defaultdict(list)
		for o1, rel, o2 in annotation.relations:
			o1, o2 = annotation.labels[o1], annotation.labels[o2]
			agg_relations[(o1, o2)].append(rel)

		for (o1, o2), rel in agg_relations.items():
			if 'of' in rel:
				continue
			if len(rel) > 1:
				rel = self.rng.choice(rel)
			else:
				rel = rel[0]
			be = 'are' if self.inflect_engine.singular_noun(o1) else 'is'

			data_list += make_one_data(
				{
					"object1" : o1,
					"be"      : be,
					"relation": rel,
					"object2" : o2
				},
				templates=templates,
				rng=self.rng,
				enumerate_templates=True
			)

		return data_list


class HeadRelationGenerator(BaseGenerator):
	qa_templates = [
		{
			"prompt"  : "Among {candidates}, what is {relation} {object2}?",
			"response": "{object1}"
		}
	]
	inflect_engine = inflect.engine()

	def __init__(self, dataset: RelationDataset, numeric=True, n=3, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.numeric = numeric
		self.n = n

	def _generate(self, annotation: Relations, templates: List) -> List:
		data_list = []

		target_to_relations = defaultdict(dict)
		for o1, rel, o2 in annotation.relations:
			o1, o2 = annotation.labels[o1], annotation.labels[o2]
			if 'of' in rel:
				continue
			if rel not in target_to_relations[o2]:
				target_to_relations[o2][rel] = set()
			target_to_relations[o2][rel].add(o1)

		all_objects = set(annotation.labels)

		for o2, rels in target_to_relations.items():
			for rel, o1s in rels.items():
				object1 = make_and_description(o1s, self.rng)
				candidates = list(all_objects - o1s - {o2})
				if len(candidates) == 0:
					continue
				else:
					candidates = self.rng.choice(candidates, min(len(candidates), self.n - len(o1s)), replace=False)
				candidates = list(candidates) + list(o1s)
				candidates = make_and_description(candidates, self.rng)
				data_list += make_one_data(
					{
						"object1"   : object1,
						"relation"  : rel,
						"object2"   : o2,
						"candidates": candidates
					},
					templates=templates,
					rng=self.rng,
					enumerate_templates=True
				)

		return data_list


RelationGeneratorList = [
	ExistsRelationGenerator,
	HeadRelationGenerator
]
