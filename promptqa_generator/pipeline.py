import json
from .utils.base import *
from .utils.dataset import *
from .utils.generators.attribute import *
from .utils.generators.object import *
from .utils.generators.relation import *
from .utils.generators.scene_graph import *
from .sg_generator.scene_generation import *

class PromptQAGenerator:
    def __init__(self, metadata_path, output_file):
        """
        Initialize the PromptQAGenerator class

        Args:
            output_file (str): Path to the output JSON file
        """
        self.metadata_path = metadata_path
        self.output_file = output_file

    def parse_scene_param(self, param):
        """Parse the parameter: if it's a number, return it as both min and max; if it's a tuple, return it as min and max"""
        if isinstance(param, tuple):
            return param
        else:
            return param, param

    def generate_prompt_and_qa(self, scene_graphs):
        results = []
        sg_dataset = SceneGraphDataset(data=scene_graphs)

        for scene in scene_graphs:
            scene_graph_id = scene['scene_graph_id']
            result = {"scene_graph": scene, "prompts": [], "qa": []}

            # Prompt generation section
            prompt_generators = {
                "Objects": (ObjectGeneratorList, True),  # (generator, is_list)
                "Attributes": (AttributeGeneratorList, True),
                "Relations": (RelationGeneratorList, True),
                "SceneGraph": (SceneGraphCaptionGenerator, False),
            }

            for name, (generator, is_list) in prompt_generators.items():
                if is_list:
                    gen = JointGenerator(
                        dataset=sg_dataset,
                        generators=generator,  # Directly pass the entire list
                        template_mode='description'
                    )
                else:
                    gen = generator(  # Directly instantiate a single generator
                        dataset=sg_dataset,
                        template_mode='description'
                    )
                
                for data in gen.generate():
                    if data['scene_graph_id'] == scene_graph_id:
                        result["prompts"].append({
                            "type": name,
                            "generator": data['generator'],
                            "description": data['description']
                        })

            # QA generation section
            qa_generators = {
                "Objects": (ObjectGeneratorList, True),
                "Attributes": (AttributeGeneratorList, True),
                "Relations": (RelationGeneratorList, True),
                "SceneGraph": (SceneGraphQAGenerator, False),
            }

            for name, (generator, is_list) in qa_generators.items():
                if is_list:
                    gen = JointGenerator(
                        dataset=sg_dataset,
                        generators=generator,  # Directly pass the entire list
                        template_mode='qa'
                    )
                else:
                    gen = generator(  # Directly instantiate a single generator
                        dataset=sg_dataset,
                        template_mode='qa'
                    )
                
                for data in gen.generate():
                    if data['scene_graph_id'] == scene_graph_id:
                        result["qa"].append({
                            "type": name,
                            "generator": data['generator'],
                            "question": data['prompt'],
                            "response": data['response']
                        })

            results.append(result)
        return results

    def save_to_json(self, data):
        """
        Save the generated Prompts and QA as a JSON file.

        Args:
            data (list): List containing the Prompt and QA information for each scene graph
        """
        try:
            with open(self.output_file, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Results saved to {self.output_file}")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")

    def process(self, total_scenes, scene_complexity, scene_attributes, num_workers=1):
        """
        Main process, generates Prompts and QA from scene graphs and saves them as a JSON file.

        Args:
            scene_graphs (list): List of scene graphs
        """
        # Parse the input parameters
        min_complexity, max_complexity = self.parse_scene_param(scene_complexity)
        min_attributes, max_attributes = self.parse_scene_param(scene_attributes)

        # Generate scene graphs
        scene_graphs = generate_scene(self.metadata_path, total_scenes, min_complexity, max_complexity, min_attributes, max_attributes, num_workers)
        data = self.generate_prompt_and_qa(scene_graphs)
        self.save_to_json(data)
