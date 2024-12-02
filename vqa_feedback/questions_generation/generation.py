from PIL import Image
from .query_utils import generate_dsg
import openai
from .openai_utils import openai_setup, openai_completion
from .parse_utils import parse_tuple_output, parse_dependency_output, parse_question_output

def generate_qid_data_from_prompt(input_text_prompt):
    """
    Generate questions based on the descriptive text and return a dictionary 
    containing prompt, qid2tuple, qid2dependency, and qid2question.

    Parameters:
    - input_text_prompt: A prompt describing the image to be generated.

    Returns:
    - qid_data: A dictionary containing prompt, qid2tuple, qid2dependency, and qid2question.
    """

    openai_setup()

    assert openai.api_key is not None

    id2prompts = {'custom_0': {'input': input_text_prompt}}

    id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(
        id2prompts,
        generate_fn=openai_completion 
    )

    qid2tuple = parse_tuple_output(id2tuple_outputs['custom_0']['output'])
    qid2dependency = parse_dependency_output(id2dependency_outputs['custom_0']['output'])
    qid2question = parse_question_output(id2question_outputs['custom_0']['output'])

    qid_data = {
        'prompt': input_text_prompt,
        'qid2tuple': qid2tuple,
        'qid2dependency': qid2dependency,
        'qid2question': qid2question
    }

    return qid_data
