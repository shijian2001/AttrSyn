import torch
import numpy as np
from PIL import Image
from .vqa_models import ImageQAModel
from .vqa_models.prompt import detailed_imageqa_prompt
from .wrench.labelmodel import *

label_models = {
    "MajorityWeightedVoting": MajorityWeightedVoting(),
    "MajorityVoting": MajorityVoting(),
    "DawidSknene": DawidSkene(),
    "MeTaL": MeTaL(),
    "EBCC": EBCC(), # need Weaklabel
    "FlyingSquid": FlyingSquid(),
    "GenerativeModel": GenerativeModel(),
    "GoldCondProb": GoldCondProb(), # need labels
    "NaiveBayesModel": NaiveBayesModel(),
    "Snorkel": Snorkel(),
    "Weapo": Weapo(), # cannot use
    "IBCC": IBCC(), # need Weaklabel
    # "Fable": Fable(), need kernel_function
    "HyperLM": HyperLM() # need kernel_function
}

class BinaryQuesitonGenerator:
    pass

class VQAEnsembler:

    def __init__(self, vqa_models, images, questions):
        """
        vqa_models: list of UnifiedVQA instances (multiple VQA model instances)
        """
        self.vqa_models = vqa_models
        self.images = images
        self.questions = questions # TODO: change this into a class object

    def generate_vqa_score_matrix(self):
        """
        images: input images
        questions: list of binary questions list (e.g., questions = [["Is there a cat?", "Is there a girl?", "Is there a boy?"],["Is it cat?", "Is there a girl?", "Is there a boy?"]])
        
        return: np.array, VQA score matrix, shape (num_images, num_questions, num_models)
        """
        images = self.images
        questions =self.questions
        choices = [["Yes", "No"]] * len(questions)
        combined_questions = [list(q) for q in zip(*questions)]
        score_matrix = []
        
        # For each VQA model, generate scores (0 or 1) for each question
        for model in self.vqa_models:
            vqa_scores = [] 
            for question in combined_questions:
                answers = model.batch_multiple_choice_qa(images, question, choices) 
                binary_scores = [1 if answer['multiple_choice_answer'] == 'Yes' else 0 for answer in answers]
                vqa_scores.append(binary_scores)
            score_matrix.append(vqa_scores)
        
        score_matrix = np.array(score_matrix)
        # Use transpose to change the dimension order, convert to (images, questions, models), original order was (models, questions, images)
        score_matrix = score_matrix.transpose(2, 1, 0)
        
        # Convert to NumPy array for further processing
        print(score_matrix)
        return score_matrix

    # TODO: Implement method for non-matrix question set


class WeakSupervisor:
    '''
    A weak supervision class for combining results from multiple VQA models, primarily performing voting or aggregation strategies.
    '''
    def __init__(self, strategy_name: str):
        """
        Initialize the WeakSupervision class.

        Args:
            strategy_name: The name of the voting or aggregation strategy.
        """
        self.strategy_name = strategy_name
        self.voting_strategy = None 

    def _load_strategy(self):
        """
        Lazily load the voting or aggregation strategy to reduce initial overhead.
        """
        if self.voting_strategy is None:
            if self.strategy_name in label_models:
                self.voting_strategy = label_models[self.strategy_name]
            else:
                raise ValueError(f"Strategy '{self.strategy_name}' not found in the label_models dictionary")

    def compute_soft_score(self, score_matrix): 
        """
        Compute the soft score using the specified voting or aggregation strategy.

        Args:
            score_matrix: A 2D array representing the hard scores.

        Returns:
            np.array: The soft score matrix.
        """
        self._load_strategy()  # Ensure the strategy is loaded
        self.voting_strategy.fit(score_matrix.reshape(1, -1))
        soft_score = self.voting_strategy.predict_proba(score_matrix.reshape(1, -1))
        return soft_score

class Pipeline:
    def __init__(self, vqa_ensembler, weak_supervisor):
        """
        vqa_ensembler: VQAEnsembler instance
        weak_supervisor: WeakSupervisor instance
        """
        self.vqa_ensembler = vqa_ensembler
        self.weak_supervisor = weak_supervisor
    
    def process(self, images, questions):
        hard_score_matrix = self.vqa_ensembler.generate_vqa_score_matrix()
        soft_score = [self.weak_supervisor.compute_soft_score(hard_score_matrix_item) for hard_score_matrix_item in hard_score_matrix]
        final_score = [item[0, 1] for item in soft_score]
        return final_score

    def __call__(self, images, questions):
        return self.process(images, questions)

