import torch
import numpy as np
from PIL import Image
from .vqa_models import ImageQAModel
from .vqa_models.prompt import detailed_imageqa_prompt
from .wrench.labelmodel import *
from .questions_generation import generate_qid_data_from_prompt

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
    def __init__(self, prompt):
        self.prompt = prompt

    def generate_binary_question(self):
        prompt = self.prompt
        self.questions = generate_qid_data_from_prompt(prompt)
    

class VQAEnsembler:

    def __init__(self, vqa_models):
        """
        vqa_models: list of UnifiedVQA instances (multiple VQA model instances)
        """
        self.vqa_models = vqa_models
        

    def generate_vqa_score_matrix(self, image, prompt):
        binaryQuesitonGenerator = BinaryQuesitonGenerator(prompt)
        # questions = binaryQuesitonGenerator.generate_binary_question()
        questions = {'prompt': 'there are two beautiful chinese girls in school.', 'qid2tuple': {1: 'entity - whole', 2: 'other - count', 3: 'entity - whole', 4: 'attribute - type', 5: 'attribute - beauty', 6: 'relation - spatial'}, 'qid2dependency': {1: [0], 2: [1], 3: [0], 4: [1], 5: [1], 6: [1, 3]}, 'qid2question': {1: 'Are there boys?', 2: 'Are there two girls?', 3: 'Is there a school?', 4: 'Are the girls Chinese?', 5: 'Are the girls beautiful?', 6: 'Are the girls in the school?'}}
        # example -> questions : {'prompt': 'there are two beautiful chinese girls in school.', 'qid2tuple': {1: 'entity - whole', 2: 'other - count', 3: 'entity - whole', 4: 'attribute - type', 5: 'attribute - beauty', 6: 'relation - spatial'}, 'qid2dependency': {1: [0], 2: [1], 3: [0], 4: [1], 5: [1], 6: [1, 3]}, 'qid2question': {1: 'Are there boys?', 2: 'Are there two girls?', 3: 'Is there a school?', 4: 'Are the girls Chinese?', 5: 'Are the girls beautiful?', 6: 'Are the girls in the school?'}}
        qid2dependency = questions['qid2dependency']
        qid2questions = questions['qid2question']

        choice = ["Yes", "No"]
        score_matrix = []
        
        # For each VQA model, generate scores (0 or 1) for each question
        for model in self.vqa_models:
            vqa_scores = [] 
            for idx, question in qid2questions.items():
                answer = model.multiple_choice_qa(image, question, choice) 
                print(answer)
                binary_scores = 1 if answer['multiple_choice_answer'] == 'Yes' else 0 
                vqa_scores.append(binary_scores)

            for id, parent_ids in qid2dependency.items():
                # zero-out scores if parent questions are answered 'no'
                any_parent_answered_no = False
                for parent_id in parent_ids:
                    if parent_id == 0:
                        continue
                    if vqa_scores[parent_id - 1] == 0:
                        any_parent_answered_no = True
                        break
                if any_parent_answered_no:
                    vqa_scores[id - 1] = 0
            
            score_matrix.append(vqa_scores)
            
        score_matrix = np.array(score_matrix)
        return score_matrix


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
        self.voting_strategy.fit(score_matrix.reshape(1,-1))
        soft_score = self.voting_strategy.predict_proba(score_matrix.reshape(1,-1))
        return soft_score

class Pipeline:
    def __init__(self, vqa_ensembler, weak_supervisor): 
        self.vqa_ensembler = vqa_ensembler
        self.weak_supervisor = weak_supervisor
    
    def process(self, image, prompt):
        hard_score_matrix = self.vqa_ensembler.generate_vqa_score_matrix(image, prompt)
        soft_score = self.weak_supervisor.compute_soft_score(hard_score_matrix) 
        final_score = soft_score[0][1]
        return final_score

    def __call__(self, image, prompt):
        return self.process(image, prompt)

