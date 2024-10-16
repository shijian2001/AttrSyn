import torch
from vqa_feedback import VQAEnsembler, WeakSupervisor, detailed_imageqa_prompt, ImageQAModel

# Initialize models
vqa_model_1 = ImageQAModel(
    model_name="llavav1.5-7b", 
    prompt_func=detailed_imageqa_prompt, 
    enable_choice_search=True, 
    precision=torch.float16, 
    use_lora=False
)

vqa_model_2 = ImageQAModel(
    model_name="llavav1.5-7b",
    prompt_func=detailed_imageqa_prompt, 
    enable_choice_search=True, 
    precision=torch.float16, 
    use_lora=False
)

# Example usage
image1 = "test/flux-dev1.png"
image2 = "test/flux-dev2.png"
image3 = "test/flux-dev3.png"
images = [image1, image2, image3]

questions = [["Is there a cat?", "Is there a girl?", "Is there a boy?"],["Is it cat?", "Is there a girl?", "Is there a boy?"],["Is it cat?", "Is there a girl?", "Is there a boy?"]]

vqa_ensembler = VQAEnsembler([vqa_model_1, vqa_model_2], images, questions)

weak_supervisor = WeakSupervisor('MajorityVoting')

pipeline = vqa_feedback.Pipeline(vqa_ensembler, weak_supervisor)


soft_vqa_score = pipeline(images, questions)

print(f"Soft VQA Score: {soft_vqa_score}")
