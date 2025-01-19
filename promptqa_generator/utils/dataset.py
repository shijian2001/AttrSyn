from dataclasses import dataclass
from typing import List, Optional, Tuple
import networkx as nx
import numpy as np

from .base import BaseDataset


def boxOverlap(box1, box2):
    # keep the function but won't use it in the new code
    pass

def boxInclude(box1, box2):
    # keep the function but won't use it in the new code
    pass


@dataclass
class AnnotationList:
    def subset(self, indices):
        data = self.__dict__.copy()
        for key, value in data.items():
            if key == 'relations':
                relations = []
                for head, relation, target in value:
                    if head in indices and target in indices:
                        relations.append((indices.index(head), relation, indices.index(target)))
                data[key] = relations
            else:
                if isinstance(value, list) or isinstance(value, np.ndarray):
                    data[key] = [value[i] for i in indices]
                else:
                    data[key] = value
        return self.__class__(**data)


@dataclass
class Labels(AnnotationList):
    labels: List[str]
    attributes: List[List[str]]
    scores: Optional[List[float]]  # Removing bboxes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.labels[item], self.attributes[item], self.scores[item]


class LabelDataset(BaseDataset):
    def _load(self):
        self.annotations = [
            Labels(scores=label.get("scores", None), labels=label['labels'], attributes=label['attributes'])
            for label in self.annotations.copy()
        ]


@dataclass
class Attributes(AnnotationList):
    labels: List[str]
    attributes: List[List[str]]
    scores: Optional[List[float]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.labels[item], self.attributes[item], self.scores[item]


class AttributeDataset(BaseDataset):
    def _load(self):
        self.annotations = [
            Attributes(scores=label.get("scores", None), labels=label['labels'], attributes=label['attributes'])
            for label in self.annotations.copy()
        ]


@dataclass
class Relations(AnnotationList):
    relations: List[Tuple[int, str, int]]  # No bboxes, just relations
    labels: List[str]

    def __len__(self):
        return len(self.labels)


class RelationDataset(BaseDataset):
    def _load(self):
        self.annotations = [
            Relations(relations=label['relations'], labels=label['labels'])
            for label in self.annotations.copy()
        ]


@dataclass
class SceneGraph(AnnotationList):
    labels: List[str]
    attributes: List[List[str]]
    relations: List[Tuple[int, str, int]]
    scores: Optional[List[float]]  # No bboxes

    def __len__(self):
        return len(self.labels)

    @property
    def graph(self):
        if not hasattr(self, 'graph_'):
            self.graph_ = self._create_graph()
        return self.graph_

    def _create_graph(self):
        scene_graph = nx.MultiDiGraph()
        for i, label in enumerate(self.labels):
            scene_graph.add_node(i, value=label, attributes=self.attributes[i])
        for head, relation, target in self.relations:
            scene_graph.add_edge(head, target, value=relation)
        return scene_graph
        
    def single_edge_scene_graph(self, rng):
        uv_to_relations = {}
        for head, relation, target in self.relations:
            if head < target:
                head_to_target = True
            else:
                head, target = target, head
                head_to_target = False
            if (head, target) not in uv_to_relations:
                uv_to_relations[(head, target)] = []
            uv_to_relations[(head, target)].append((relation, head_to_target))
        relations = []
        for (head, target), rels in uv_to_relations.items():
            if len(rels) == 1:
                selected_rel, head_to_target = rels[0]
            else:
                selected_rel, head_to_target = rng.choice(rels)
            if head_to_target:
                relations.append((head, selected_rel, target))
            else:
                relations.append((target, selected_rel, head))
        return SceneGraph(labels=self.labels, attributes=self.attributes, relations=relations, scores=self.scores)
        
	

    def decompose(self) -> List:
        subgraphs = []
        G = self.graph
        connected_nodes = nx.connected_components(G.to_undirected())
        for ids in sorted(connected_nodes, key=len):
            ids = list(ids)
            labels = [self.labels[i] for i in ids]
            attributes = [self.attributes[i] for i in ids]
            relations = [
                (ids.index(head), relation, ids.index(target))
                for head, relation, target in self.relations if head in ids and target in ids
            ]
            scores = [self.scores[i] for i in ids] if self.scores is not None else None
            subgraphs.append(SceneGraph(labels=labels, attributes=attributes, relations=relations, scores=scores))
        return subgraphs


class SceneGraphDataset(BaseDataset):
    def _load(self):
        self.annotations = [
            SceneGraph(labels=label['labels'], attributes=label['attributes'], relations=label['relations'], scores=label.get("scores", None))
            for label in self.annotations.copy()
        ]
