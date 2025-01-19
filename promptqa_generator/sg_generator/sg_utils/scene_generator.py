import time
from typing import List

import networkx as nx
import numpy as np

from .metadata import Text2ImageMetaData
from .scene_graph import get_sg_desc, add_seed_graph_to_template_graph
from .utils import make_and_description, mention_scene_attributes, normalized_sentence, capitalize_first_char_if_letter, get_element_num_dict, convert_sg_to_json, convert_json_to_sg
import sys

def get_scene_attribute_desc(scene_attributes):
	scene_attributes = [
		f"{mention_scene_attributes(type, v)}"
		for type, v in scene_attributes.items()
	]
	global_desc = make_and_description(scene_attributes)
	return global_desc

def get_prompt(scene_attribute_desc, sg_desc):
	scene_attribute_desc = normalized_sentence(scene_attribute_desc)
	sg_desc = normalized_sentence(sg_desc)
	if scene_attribute_desc == "":
		return f"{capitalize_first_char_if_letter(sg_desc)}."
	else:
		return f"{capitalize_first_char_if_letter(sg_desc)}. {capitalize_first_char_if_letter(scene_attribute_desc)}."


class Text2SceneGraphGenerator():
	metadata: Text2ImageMetaData
	allowed_scene_attributes = {
		"style": "all",
		"scene setting": "all",
		"camera setting": "all",
	}
	def __init__(self, metadata: Text2ImageMetaData, seed=42):
		self.metadata = metadata
		self.rng = np.random.default_rng(seed=seed)

	def _task_plan_to_str(self, task_plan):
		return get_sg_desc(task_plan["scene_graph"])

	def _complete_sg(self, scene_graph: nx.DiGraph):
		assert isinstance(scene_graph, nx.DiGraph)
		# first adding data for each object nodes
		for node, data in scene_graph.nodes(data=True):
			if data["type"] == "object_node":
				if "value" not in data:
					data["value"] = self.metadata.sample_metadata(
						self.rng, element_type="object"
					)
				for neighbor in scene_graph.neighbors(node):
					if scene_graph.nodes[neighbor]["type"] == "attribute_node":
						k, v = self.metadata.sample_metadata(
							self.rng, element_type="attribute"
						)
						scene_graph.nodes[neighbor]["value_type"] = k
						scene_graph.nodes[neighbor]["value"] = v
		for s, t, data in scene_graph.edges(data=True):
			if "value" not in data:
				if data.get("type") == "relation_edge":
					k, v = self.metadata.sample_metadata(
						self.rng, element_type="relation"
					)
					data["value_type"] = k
					data["value"] = v
		return scene_graph

	def _sample_scene_graph(self, complexity, seed_graph, seed_graph_element_num_dict, element_num_dict, retry=50):
		sg_templates = self.metadata.query_sg_templates(
			complexity, seed_graph_element_num_dict, element_num_dict
		)
		#返回一个sg_template
		if len(sg_templates) == 0:
			raise ValueError("No specific template scene graph found")

		conditioned_template = None
		for i in self.rng.permutation(len(sg_templates)):
			template_graph = sg_templates[i]
			conditioned_templates = add_seed_graph_to_template_graph(
				seed_graph, template_graph
			)
			# randomly pick one of the conditioned templates
			if len(conditioned_templates) != 0:
				index = self.rng.integers(len(conditioned_templates))
				conditioned_template = conditioned_templates[index]
				break

		if conditioned_template is None:
			raise ValueError("No template scene graph matches seed graph")
		
		scene_graph = self._complete_sg(conditioned_template)
		return scene_graph

	def _sample_scene_attributes(self, number_of_scene_attributes, allowed_scene_attributes):
		return self.metadata.sample_scene_attribute(self.rng, number_of_scene_attributes, allowed_scene_attributes)

	def generate(
			self,
			complexity=5,
			number_of_scene_attributes=1,
			sample_numbers=100,
			time_limit=60,
			seed_graph: nx.DiGraph = None,
			allowed_scene_attributes: list = None,
			element_num_dict: dict = None,
	) -> List:

		# check whether user input is legal
		if seed_graph is None:
			seed_graph = nx.DiGraph()
		if allowed_scene_attributes is None:
			allowed_scene_attributes = self.allowed_scene_attributes

		seed_graph_element_num_dict = get_element_num_dict(seed_graph)
		assert sum(seed_graph_element_num_dict.values()) <= complexity

		if element_num_dict is None:
			element_num_dict = {
				"object"   : None,
				"attribute": None,
				"relation" : None,
			}
		n_elements = 0
		for k in ['object', 'relation', 'attribute']:
			if element_num_dict[k] is not None:
				assert seed_graph_element_num_dict[k] <= element_num_dict[k]
				n_elements += element_num_dict[k]
		assert n_elements <= complexity

		# sample task plans
		task_plans = []
		start_time = time.time()
		while len(task_plans) < sample_numbers:
			# make sure the time limit is not exceeded
			if time.time() - start_time > time_limit:
				print("Time limit: 60s exceeded. Exiting the sampling process.")
				break
			scene_graph = self._sample_scene_graph(complexity, seed_graph, seed_graph_element_num_dict, element_num_dict)
			scene_attributes = self._sample_scene_attributes(number_of_scene_attributes, allowed_scene_attributes)
			scene_graph_str = convert_sg_to_json(scene_graph)
			task_plans.append(
				
				{
					"scene_attributes": scene_attributes,
					"annotation"      : scene_graph_str,
				}
			)
		if(len(task_plans) == 1):
			task_plans = {
					"scene_attributes": scene_attributes,
					"annotation"      : scene_graph_str,
				}
		# print(f"sampling {len(task_plans)} task plans.")
		return task_plans









