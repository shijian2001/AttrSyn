import json
import os
import random
import argparse
import multiprocessing as mp
from tqdm import tqdm
from .sg_utils.scene_generator import Text2SceneGraphGenerator
from .sg_utils.metadata import Text2ImageMetaData
import sys


def create_metadata(metadata_path):
    return Text2ImageMetaData(path_to_metadata=metadata_path)


def generate_batch(batch_idx, complexities, scene_attributes, prompts_per_attribute, metadata_path, seed):
    scene_graphs_list = []
    metadata = create_metadata(metadata_path)
    generator = Text2SceneGraphGenerator(metadata=metadata, seed=seed)

    for complexity in tqdm(complexities, desc=f"Batch {batch_idx} - Complexity"):
        for num_attributes in scene_attributes:
            for _ in range(prompts_per_attribute):
                scene_graphs_list.append(generator.generate(number_of_scene_attributes=num_attributes, complexity=complexity, sample_numbers=1))
    for idx, scene_graph in enumerate(scene_graphs_list):
        scene_graph.setdefault("scene_graph_id", idx)

    # Return the entire list
    return scene_graphs_list


def generate_scene(metadata_path, total_scenes, min_complexity, max_complexity, min_attributes, max_attributes, num_workers = 1):
    
    if num_workers > total_scenes:
        raise ValueError("Number of workers cannot exceed total scenes.")
    complexities = range(min_complexity, max_complexity + 1)
    scene_attributes = range(min_attributes, max_attributes + 1)
    scenes_per_file = total_scenes // num_workers
    scenes_per_complexity = scenes_per_file // len(complexities)
    scenes_per_attribute = scenes_per_complexity // len(scene_attributes)

    print("complexities: " + str(complexities))
    print("scene_attributes: " + str(scene_attributes))
    print("scenes_per_file: " + str(scenes_per_file))
    print("len(complexities): " + str(len(complexities)))
    print("scenes_per_complexity: " + str(scenes_per_complexity))
    print("scenes_per_attribute: " + str(scenes_per_attribute))

    seeds = [random.randint(0, 100) for _ in range(num_workers)]

    # Use starmap to collect the results from each process
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(generate_batch, [
            (batch_idx, complexities, scene_attributes, scenes_per_attribute, metadata_path, seeds[batch_idx])
            for batch_idx in range(num_workers)
        ])

    # Merge all the scene_graphs returned by the processes
    all_scene_graphs = [scene for batch_result in results for scene in batch_result]
    return all_scene_graphs


if __name__ == "__main__":
    scene_graphs = generate_scene()
    print(f"Generated {len(scene_graphs)} scene graphs.")
    print(scene_graphs)
