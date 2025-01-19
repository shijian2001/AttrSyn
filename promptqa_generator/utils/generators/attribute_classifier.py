from .attribute_category import ATTRIBUTE_CATEGORY


class AttributeClassifier:
	def __init__(self):
		attribute_category = ATTRIBUTE_CATEGORY
		self.categories = list(attribute_category.keys())
		self.attribute_to_category = {}
		for category, attributes in attribute_category.items():
			for attribute in attributes:
				self.attribute_to_category[attribute] = category
		self.category_to_attribute = attribute_category.copy()

	def classify(self, attribute):
		return self.attribute_to_category.get(attribute, None)

	def sample_category(self, rng, n=1):
		return rng.choice(self.categories, n)

	def sample_attribute(self, category, rng, n=1):
		return rng.choice(self.category_to_attribute[category], n)
