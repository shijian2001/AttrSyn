from typing import List

import inflect

from .scene_graph_caption import get_sg_desc
from .scene_graph_qa import generate_attribute_qa, generate_object_qa, generate_relation_qa
from .utils import make_and_description, normalize_attributes
from ..base import BaseGenerator
from ..dataset import SceneGraph


class SceneGraphCaptionGenerator(BaseGenerator):
	des_templates = [
		{}
	]
	inflect_engine = inflect.engine()

	def _generate(self, annotation: SceneGraph, templates: List) -> List:
		descriptions = []
		subgraphs = annotation.decompose()
		singular_node = []
		for subgraph in subgraphs:
			if len(subgraph) == 1:
				singular_node.append(subgraph)
			else:
				description = self._describe_subgraph(subgraph)
				descriptions.append(description)
		if len(singular_node):
			descriptions = [self._describe_nodes(singular_node)] + descriptions
		descriptions = [d.capitalize() for d in descriptions]
		description = '\n\n'.join(descriptions)
		return [
			{
				"description": description
			}
		]

	def _describe_nodes(self, nodes):
		labels = []
		for node in nodes:
			attr = node.attributes[0]
			label = node.labels[0]
			if len(attr):
				attr = make_and_description(normalize_attributes(attr), self.rng)
				labels.append(f'{attr} {label}')
			else:
				labels.append(label)
		be = 'are'
		if len(labels) == 1 and not self.inflect_engine.singular_noun(labels[0]):
			be = 'is'
		d = f'there {be} {make_and_description(labels, self.rng)}.'
		return d

	def _describe_subgraph(self, subgraph: SceneGraph) -> List:
		subgraph = subgraph.single_edge_scene_graph(self.rng)
		G = subgraph.graph.copy()
		return get_sg_desc(G, self.rng)


def graph_to_json(graph: SceneGraph):
	objects = {}
	for i, (name, attrs) in enumerate(zip(graph.labels, graph.attributes)):
		objects[i] = {
			"name"      : name,
			"attributes": attrs,
			"relations" : []
		}
	for (o1, rel, o2) in graph.relations:
		objects[o1]["relations"].append({
			"object": o2,
			"name"  : rel
		})
	return {
		"objects": objects
	}


class SceneGraphQAGenerator(BaseGenerator):
	qa_templates = [
		{}
	]
	inflect_engine = inflect.engine()

	def _generate(self, annotation: SceneGraph, templates: List) -> List:
		qas = []
		subgraphs = annotation.decompose()
		for subgraph in subgraphs:
			if len(subgraph) > 1:
				qas += self._qa_subgraph(subgraph)
		return qas

	def _qa_subgraph(self, subgraph: SceneGraph) -> List:

		subgraph = subgraph.single_edge_scene_graph(self.rng)
		subgraph_json = graph_to_json(subgraph)
		qas = []
		for q, a in generate_relation_qa(subgraph_json):
			qas.append({
				"prompt"  : q,
				"response": make_and_description(a, self.rng),
				"type"    : "relation"
			})
		for q, a in generate_attribute_qa(subgraph_json):
			qas.append({
				"prompt"  : q,
				"response": make_and_description(a, self.rng),
				"type"    : "attribute"
			})
		for q, a in generate_object_qa(subgraph_json):
			qas.append({
				"prompt"  : q,
				"response": make_and_description(a, self.rng),
				"type"    : "object"
			})
		return qas
