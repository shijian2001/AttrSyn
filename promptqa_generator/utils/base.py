import json
from abc import abstractmethod
from typing import Dict, List, Callable, Type
import numpy as np


class BaseDataset:
    """
    Base class for datasets, supporting loading data from a file or a directly passed list of scene graphs.
    """
    scene_graph_ids: List[str]
    annotations: List
    sources: List[str]

    def __init__(self, annotation_path=None, data=None):
        """
        Initialize BaseDataset, supporting loading data from a file or directly passed scene graph list.

        Args:
            annotation_path (str, optional): Path to the annotation file.
            data (list, optional): Directly passed list of scene graphs.
        """
        self.scene_graph_ids = []
        self.annotations = []
        self.sources = []

        if data is not None:
            # Load data from scene graph list
            for ann in data:
                self.scene_graph_ids.append(ann['scene_graph_id'])
                self.annotations.append(ann['annotation'])
                self.sources.append(ann.get('source', None))
        elif annotation_path is not None:
            # Load data from JSON file
            self.annotation_path = annotation_path
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
                for ann in annotation:
                    self.scene_graph_ids.append(ann['scene_graph_id'])
                    self.annotations.append(ann['annotation'])
                    self.sources.append(ann.get('source', None))
        else:
            raise ValueError("Either 'annotation_path' or 'data' must be provided.")

        self._load()

    @abstractmethod
    def _load(self):
        """
        (Abstract method) Custom loading logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __getitem__(self, idx):
        return self.scene_graph_ids[idx], self.annotations[idx], self.sources[idx]

    def __len__(self):
        return len(self.scene_graph_ids)


class BaseGenerator:
    """
    Base generator class, used to generate descriptions or QA from the dataset.
    """
    dataset: BaseDataset
    qa_templates = []
    des_templates = []

    def __init__(self, dataset: BaseDataset, template_mode: str = 'description', seed: int = 42):
        """
        Initialize BaseGenerator.

        Args:
            dataset (BaseDataset): The dataset object.
            template_mode (str): Template mode, either 'qa' or 'description'.
            seed (int): Random seed.
        """
        self.dataset = dataset
        if template_mode == 'qa':
            self.templates = self.qa_templates
            self.template_mode = 'qa'
        elif template_mode == 'description':
            self.templates = self.des_templates
            self.template_mode = 'description'
        else:
            raise ValueError(f"Invalid template mode: {template_mode}")
        self.rng = np.random.default_rng(seed)

    def generate(self) -> List:
        """
        Iterate through the dataset and generate data.

        Returns:
            List: A list containing the generated results.
        """
        if len(self.templates) == 0:
            return []
        data_list = []
        for data_path, annotation, source in (
                zip(self.dataset.scene_graph_ids, self.dataset.annotations, self.dataset.sources)
        ):
            if len(annotation.labels) > 0:  # Accessing object properties
                for data in self._generate(annotation, self.templates):
                    data['scene_graph_id'] = data_path
                    data['generator'] = self.__class__.__name__
                    data_list.append(data)
        return data_list

    @abstractmethod
    def _generate(self, annotation, templates: List) -> List[Dict]:
        """
        Abstract method implemented by subclasses to generate data.

        Args:
            annotation: The scene graph annotation (SceneGraph object).
            templates (list): List of templates.

        Returns:
            List[Dict]: List of generated data.
        """
        pass


class JointGenerator(BaseGenerator):
    """
    Joint generator class, used to combine multiple generators.
    """
    def __init__(self, dataset: BaseDataset, generators: List[Type[BaseGenerator]], template_mode: str = 'description', seed: int = 42):
        """
        Initialize JointGenerator.

        Args:
            dataset (BaseDataset): The dataset object.
            generators (List[Type[BaseGenerator]]): List of generator classes.
            template_mode (str): Template mode, either 'qa' or 'description'.
            seed (int): Random seed.
        """
        super().__init__(dataset=dataset, template_mode=template_mode, seed=seed)

        # Ensure generators is an iterable list, and the elements are callable objects (e.g., classes)
        if not isinstance(generators, list) or not all(callable(generator) for generator in generators):
            raise ValueError("The 'generators' argument must be a list of callable objects (e.g., generator classes).")

        # Initialize generator instances
        self.generators = [
            generator(dataset=dataset, template_mode=template_mode, seed=seed) 
            for generator in generators
        ]

    def generate(self) -> List:
        """
        Iterate through all generators and generate data.

        Returns:
            List: A list containing data generated by all generators.
        """
        data_list = []
        for scene_graph_id, annotation, source in (
                zip(self.dataset.scene_graph_ids, self.dataset.annotations, self.dataset.sources)
        ):
            for generator in self.generators:
                if len(annotation.labels) > 0:  # Accessing object properties
                    for data in generator._generate(annotation, generator.templates):
                        data['scene_graph_id'] = scene_graph_id
                        data['generator'] = generator.__class__.__name__
                        data_list.append(data)
        return data_list

    def _generate(self, annotation, templates: List) -> List[Dict]:
        """
        Placeholder, actually implemented by the sub-generators.

        Args:
            annotation: The scene graph annotation (SceneGraph object).
            templates (List): List of templates.

        Returns:
            List[Dict]: List of generated data.
        """
        pass
