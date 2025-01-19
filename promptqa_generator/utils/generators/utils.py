import string
from typing import Dict, List



def normalize_attributes(attributes):
	attr = set()
	for a in attributes:
		if a == 'grey':
			a = 'gray'
		attr.add(a)
	return list(attr)


def make_and_description(names, rng=None):
	if not isinstance(names, list):
		names = list(names)

	if len(names) == 0:
		return ""
	if len(names) == 1:
		return names[0]

	if rng is not None:
		names = list(rng.permutation(names))

	if len(names) == 2:
		return ' and '.join(names)
	return ', '.join(names[:-1] + [f'and {names[-1]}'])


def _fill_formatted_string(formatted_string, **kwargs):
	key_words = [tup[1] for tup in string.Formatter().parse(formatted_string) if tup[1] is not None]
	contained_kwargs = {k: v for k, v in kwargs.items() if k in key_words}
	# fill in the formatted string
	return formatted_string.format(**contained_kwargs)


def _make_one_data(template: Dict, **kwargs) -> Dict:
	# fill in the template
	return {k: _fill_formatted_string(item, **kwargs) for k, item in template.items()}


def make_one_data(data_info: Dict, templates: List, rng, enumerate_templates: bool = True) -> List:
	# initialize a random number generator
	if enumerate_templates:
		instruction = [_make_one_data(template, **data_info) for template in templates]
	else:
		# choose a random template
		template = rng.choice(templates)
		instruction = [_make_one_data(template, **data_info)]

	return instruction


def make_data(data_info_list: List[Dict], templates: List, rng, enumerate_templates: bool = True) -> List:
	# initialize a random number generator
	data_list = []
	for data_info in data_info_list:
		data_list += make_one_data(data_info, templates, rng, enumerate_templates)
	return data_list
