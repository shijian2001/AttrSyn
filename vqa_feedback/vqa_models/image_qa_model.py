import base64
import json
import tempfile
import time
from typing import Callable, Union

import diskcache
import openai
import torch
from PIL import Image
from torch.nn.parallel import DataParallel

from .base_qa_model import QAModelInstance, QAModel

imageqa_models = {

	"instructblip-flant5xl" 		: ("InstructBlip", "Salesforce/instructblip-flan-t5-xl"),
	"instructblip-flant5xxl"		: ("InstructBlip", "Salesforce/instructblip-flan-t5-xxl"),
	"instructblip-vicuna7b" 		: ("InstructBlip", "Salesforce/instructblip-vicuna-7b"),
	"instructblip-vicuna13b"		: ("InstructBlip", "Salesforce/instructblip-vicuna-13b"),
	"blip2-flant5xxl"       		: ("BLIP2", "Salesforce/blip2-flan-t5-xxl"),
	"llavav1.5-7b"          		: ("LLaVA", "llava-hf/llava-1.5-7b-hf"),
	"llavav1.5-7b-trl"      		: ("LLaVA", "HuggingFaceH4/vsft-llava-1.5-7b-hf-trl"),
	"llavav1.5-7b-10k-tune" 		: ("LLaVA", "/linxindisk/VQAPromptBench/logs/multi-templates-finetuning/mask_instruct_all_param_generator_10k_llava_data_3epochs/best_model"),
	"llavav1.5-7b-259k-tune"		: ("LLaVA", "/linxindisk/VQAPromptBench/logs/multi-templates-finetuning/mask_instruct_all_param_generator_259k_llava_data_3epochs/best_model"),
	"llavav1.5-7b-lora-templated"	: ('LLaVA', "shijianS01/llava-v1.5-7b-lora-templated"),
	"llavav1.5-7b-lora"				: ('LLaVA', "shijianS01/llava-v1.5-7b-lora"),
	"llavav1.5-13b"         		: ("LLaVA", "llava-hf/llava-1.5-13b-hf"),
	"llavav1.6-7b"          		: ("LLaVA", "llava-hf/llava-v1.6-vicuna-7b-hf"),
	"llavav1.6-13b"         		: ("LLaVA", "llava-hf/llava-v1.6-vicuna-13b-hf"),
	"llavav1.6-34b"         		: ("LLaVA", "llava-hf/llava-v1.6-34b-hf"),
	"llava1.6-34b-api"      		: ("LLaVA34B", '<replicate-api>'),
	"qwenvl"                		: ("QwenVL", "Qwen/Qwen-VL"),
	"qwenvl-chat"           		: ("QwenVLChat", "Qwen/Qwen-VL-Chat"),
	"internvl-chat-v1.5"    		: ("InternVLChat", 'failspy/InternVL-Chat-V1-5-quantable'),
	"deepseek-vl-7b-chat"   		: ("DeepSeekVLChat", 'deepseek-ai/deepseek-vl-7b-chat'),
	"idefics2-8b"           		: ("IDEFICS2", "HuggingFaceM4/idefics2-8b"),
	"phi-3-vision"          		: ("Phi3Vision", "microsoft/Phi-3-vision-128k-instruct"),

	"gpt4v"                 		: ("GPT4V", "<openai-api>"),
	"gpt4o"                 		: ("GPT4O", "<openai-api>"),
	"qwen-vl-plus"          		: ("QwenVLPlus", ['<qwen-api>', '<aliyun-access-id>', '<aliyun-access-secret>']),
	"qwen-vl-max"           		: ("QwenVLMax", ['<qwen-api>', '<aliyun-access-id>', '<aliyun-access-secret>']),
	"gemini-vision-pro"     		: ("GeminiVisionPro", "<google-api>"),
}


def set_imageqa_model_key(model_name, key):
	imageqa_models[model_name] = (imageqa_models[model_name][0], key)


def list_imageqa_models():
	return list(imageqa_models.keys())


def image_to_base64(pil_image):
	import io
	import base64
	img_byte_arr = io.BytesIO()
	pil_image.save(img_byte_arr, format='PNG')
	img_byte_arr = img_byte_arr.getvalue()
	base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
	return base64_str


class ImageQAModel(QAModel):
	def __init__(
			self,
			model_name: str,
			prompt_func: Callable,
			model: QAModelInstance = None,
			torch_device: Union[int, str] = -1,
			precision=torch.bfloat16,
			choice_format='letter',
			enable_choice_search: bool = False,
			cache_path: str = None,
			enable_interpretation: bool = False,
			use_lora: bool = False

	):
		super().__init__(model_name, prompt_func, choice_format, enable_choice_search, cache_path, enable_interpretation, use_lora)

		if isinstance(torch_device, str):
			torch_device = torch.device(torch_device)
		else:
			if torch_device == -1:
				torch_device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
			else:
				torch_device = torch.device(f"cuda:{torch_device}")

		if model is None:
			print(f"Loading {model_name}...")
			class_name, ckpt = imageqa_models[model_name]
			self.model_precision = precision
			self.model = eval(class_name)(ckpt, torch_device, self.model_precision, self.use_lora)
			print(f"Finish loading {model_name}")
		else:
			print(f"Using provided model...")
			self.model = model

	def _data_to_str(self, data):
		if isinstance(data, str):
			return data
		else:
			return image_to_base64(data)
		
	# @torch.no_grad()
	# def _get_explanation(self, data, qa_prompt: str, multiple_choice_answer: str):
	# 	prompt = qa_prompt + "\n" + f"You have given your choice:{multiple_choice_answer}, please explain your answer according to the given image."
	# 	explanation = self._qa(data, prompt)
	# 	explanation = explanation.strip()
	# 	return explanation

class BLIP2(QAModelInstance):
	def __init__(self, ckpt="Salesforce/blip2-flan-t5-xxl", torch_device=torch.device("cuda"), model_precision=torch.float32, use_lora=False):
		from transformers import Blip2Processor, Blip2ForConditionalGeneration
		self.processor = Blip2Processor.from_pretrained(ckpt, device_map=torch_device)
		self.model = Blip2ForConditionalGeneration.from_pretrained(
			ckpt,
			device_map=torch_device,
			torch_dtype=model_precision,
			low_cpu_mem_usage=True,
		).eval()

	def qa(self, image, prompt):
		if isinstance(image, str):
			image = Image.open(image).convert('RGB')
		inputs = self.processor(image, prompt, return_tensors="pt").to(self.model.device)
		out = self.model.generate(**inputs, max_new_tokens=200)
		answer = self.processor.decode(out[0], skip_special_tokens=True)
		return answer


class InstructBlip(QAModelInstance):
	def __init__(self, ckpt="Salesforce/instructblip-flan-t5-xxl", torch_device=torch.device("cuda"), model_precision=torch.float32, use_lora=False):
		from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, InstructBlipConfig, AutoModelForVision2Seq
		from accelerate import infer_auto_device_map, init_empty_weights
		if ckpt == "Salesforce/instructblip-vicuna-34b": # temp changed
			# Load the model configuration.
			config = InstructBlipConfig.from_pretrained(ckpt)
			# Initialize the model with the given configuration.
			with init_empty_weights():
				model = AutoModelForVision2Seq.from_config(config)
				model.tie_weights()
			# Infer device map based on the available resources.
			device_map = infer_auto_device_map(model, max_memory={0: "40GiB", 1: "40GiB"},
											   no_split_module_classes=['InstructBlipEncoderLayer', 'InstructBlipQFormerLayer', 'LlamaDecoderLayer'])
			device_map['language_model.lm_head'] = device_map['language_projection'] = device_map[('language_model.model.embed_tokens')]
		else:
			device_map = torch_device
		self.processor = InstructBlipProcessor.from_pretrained(ckpt, device_map="auto")
		self.model = InstructBlipForConditionalGeneration.from_pretrained(
			ckpt,
			device_map=device_map,
			torch_dtype=model_precision,
			low_cpu_mem_usage=True,
		).eval()

	def qa(self, image, prompt):
		if isinstance(image, str):
			image = Image.open(image).convert('RGB')
		inputs = self.processor(image, prompt, return_tensors="pt").to(self.model.device)

		out = self.model.generate(**inputs, max_new_tokens=200)
		answer = self.processor.decode(out[0], skip_special_tokens=True)
		return answer


class LLaVA(QAModelInstance):
	def __init__(self, ckpt="llava-hf/llava-1.5-7b-hf", torch_device=torch.device("cuda"), model_precision=torch.float32, use_lora=False):

		if ckpt == "llava-hf/llava-v1.6-34b-hf":  # run model on multi gpus
			from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
			model = LlavaNextForConditionalGeneration.from_pretrained(ckpt,
																	  torch_dtype=torch.float16,
																	  low_cpu_mem_usage=True,
																	  load_in_4bit=True,
																	  # use_flash_attention_2=True,
																	  )
			self.model = DataParallel(model)
			self.processor = LlavaNextProcessor.from_pretrained(ckpt)
		elif ckpt == "llava-hf/llava-v1.6-vicuna-13b-hf" or ckpt == "llava-hf/llava-v1.6-vicuna-7b-hf":
			from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
			self.model = LlavaNextForConditionalGeneration.from_pretrained(
				ckpt,
				torch_dtype=model_precision,
				low_cpu_mem_usage=True,
			).to(torch_device).eval()
			self.processor = LlavaNextProcessor.from_pretrained(ckpt)
		# Need update after uploading the finetuned model to huggingface
		elif ckpt == "/linxindisk/VQAPromptBench/logs/multi-templates-finetuning/mask_instruct_all_param_generator_10k_llava_data_3epochs/best_model":
			from transformers import AutoProcessor, LlavaForConditionalGeneration
			self.model = LlavaForConditionalGeneration.from_pretrained(
				ckpt,
				torch_dtype=model_precision,
				low_cpu_mem_usage=True,
				device_map="auto"
			).eval()
			self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
		# Need update after uploading the finetuned model to huggingface
		elif ckpt == "/linxindisk/VQAPromptBench/logs/multi-templates-finetuning/mask_instruct_all_param_generator_259k_llava_data_3epochs/best_model":
			from transformers import AutoProcessor, LlavaForConditionalGeneration
			self.model = LlavaForConditionalGeneration.from_pretrained(
				ckpt,
				torch_dtype=model_precision,
				low_cpu_mem_usage=True,
				device_map="auto"
			).eval()
			self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
		elif ckpt == "shijianS01/llava-v1.5-7b-lora-templated" or ckpt == "shijianS01/llava-v1.5-7b-lora":
			from transformers import AutoProcessor, LlavaForConditionalGeneration
			self.model = LlavaForConditionalGeneration.from_pretrained(
				ckpt,
				torch_dtype=model_precision,
				low_cpu_mem_usage=True,
				device_map="auto"
			).eval()
			self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
		else:
			from transformers import AutoProcessor, LlavaForConditionalGeneration
			self.model = LlavaForConditionalGeneration.from_pretrained(
				ckpt,
				torch_dtype=model_precision,
				low_cpu_mem_usage=True,
				device_map="auto"
			).eval()
			self.processor = AutoProcessor.from_pretrained(ckpt, device_map="auto")
		
		if use_lora:
			print("[Lauching Lora] Lora is merging...")
			from peft import PeftModel, LoraConfig
			lora_config = LoraConfig(
				r=4,
				lora_alpha=4,
				lora_dropout=0.1,
				bias="none",
				target_modules=["q_proj", "k_proj", "v_proj"],
				task_type="CAUSAL_LM",
				use_dora=False
			)
			# Use absolute path; need change
			self.model = PeftModel.from_pretrained(
				self.model, 
				model_id="/linxindisk/VQAPromptBench/logs/reasoning-finetuning/259k_llava_30_templates_without_reasoning/best_model", 
				config=lora_config
			)

	def qa(self, image, prompt):
		if isinstance(image, str):
			image = Image.open(image).convert('RGB')

		prompt = "USER: <image>\n" + prompt + "\nASSISTANT:"
		if isinstance(self.model, torch.nn.DataParallel):
			inputs = self.processor(prompt, image, return_tensors='pt').to(next(self.model.parameters()).device)
			out = self.model.module.generate(**inputs, max_new_tokens=200, do_sample=False)
		else:
			inputs = self.processor(prompt, image, return_tensors='pt').to(self.model.device)
			out = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
		answer = self.processor.decode(out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

		return answer
	
	def batch_qa(self, images, prompts):
		if isinstance(images[0], str):
			images = [Image.open(image).convert('RGB') for image in images]

		prompts = ["USER: <image>\n" + prompt + "\nASSISTANT:" for prompt in prompts]

		if isinstance(self.model, torch.nn.DataParallel):
			# padding must be setted to True for batch generation
			inputs = self.processor(prompts, images, return_tensors='pt', padding=True).to(next(self.model.parameters()).device)
			out = self.model.module.generate(**inputs, max_new_tokens=200, do_sample=False)
		else:
			inputs = self.processor(prompts, images, return_tensors='pt', padding=True).to(self.model.device)
			out = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
		answers = self.processor.batch_decode(out, skip_special_tokens=True)
		answers = [answer.split("ASSISTANT:")[-1].strip() for answer in answers]
		
		return answers


class QwenVL(QAModelInstance):
	def __init__(self, ckpt="Qwen/Qwen-VL", torch_device=torch.device("cuda"), model_precision=torch.float32, use_lora=False):
		from transformers import AutoModelForCausalLM, AutoTokenizer
		self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
		if model_precision == torch.float32:
			self.model = AutoModelForCausalLM.from_pretrained(
				ckpt,
				device_map=torch_device,
				trust_remote_code=True,
				fp32=True,
				low_cpu_mem_usage=True,
			).eval()
		else:
			self.model = AutoModelForCausalLM.from_pretrained(
				ckpt,
				device_map=torch_device,
				trust_remote_code=True,
				bf16=True,
				low_cpu_mem_usage=True,
			).eval()

	def qa(self, image, prompt):
		if isinstance(image, Image.Image):
			# Check if the image is a PIL.Image object and save to a temporary file if so
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				image.save(tmp.name)
				image_path = tmp.name

				# Use the temporary image path for the tokenizer
				query = self.tokenizer.from_list_format([
					{'image': image_path},
					{'text': prompt},
				])

				inputs = self.tokenizer(query, return_tensors='pt').to(self.model.device)
				out = self.model.generate(**inputs)

		else:
			# If `image` is not a PIL.Image object, use it directly
			query = self.tokenizer.from_list_format([
				{'image': image},
				{'text': prompt},
			])

			inputs = self.tokenizer(query, return_tensors='pt').to(self.model.device)
			out = self.model.generate(**inputs)

		answer = self.tokenizer.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()

		return answer
		
	def batch_qa(self, images, prompts):
		queries = [
			self.tokenizer.from_list_format([{'image': img, 'text': prompt}])
			for img, prompt in zip(images, prompts)
		]
		
		inputs = [self.tokenizer(query, return_tensors='pt') for query in queries]
		
		inputs = [input.to(self.model.device) for input in inputs]
		
		outputs = [self.model.generate(**input_data) for input_data in inputs]
		
		answers = [self.tokenizer.decode(output[0], skip_special_tokens=True) for output in outputs]
		
		return answers




class QwenVLChat(QAModelInstance):
	def __init__(self, ckpt="Qwen/Qwen-VL-Chat", torch_device=torch.device("cuda"), model_precision=torch.float32, use_lora=False):
		from transformers import AutoModelForCausalLM, AutoTokenizer
		from transformers.generation import GenerationConfig

		self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
		if model_precision == torch.float32:
			self.model = AutoModelForCausalLM.from_pretrained(
				ckpt,
				device_map=torch_device,
				trust_remote_code=True,
				fp32=True,
				low_cpu_mem_usage=True,
			).eval()
		else:
			self.model = AutoModelForCausalLM.from_pretrained(
				ckpt,
				device_map=torch_device,
				trust_remote_code=True,
				bf16=True,
				low_cpu_mem_usage=True,
			).eval()

		# Specify hyperparameters for generation
		self.model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

	def qa(self, image, prompt):
		if isinstance(image, Image.Image):
			# Check if the image is a PIL.Image object and save to a temporary file if so
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				image.save(tmp.name)
				image_path = tmp.name

				# Use the temporary image path for the tokenizer
				query = self.tokenizer.from_list_format([
					{'image': image_path},
					{'text': prompt},
				])

				answer, history = self.model.chat(self.tokenizer, query=query, history=None)
		else:
			# If `image` is not a PIL.Image object, use it directly
			query = self.tokenizer.from_list_format([
				{'image': image},
				{'text': prompt},
			])

			answer, history = self.model.chat(self.tokenizer, query=query, history=None)

		return answer

	def batch_qa(self, images, prompts):
		"""
		Batch QA method for handling multiple images and prompts.
		"""
		# Handle images: save them temporarily if they are PIL Images
		image_paths = []
		for image in images:
			if isinstance(image, Image.Image):
				with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
					image.save(tmp.name)
					image_paths.append(tmp.name)
			else:
				# If the image is already a path, append directly
				image_paths.append(image)

		# Build queries for all prompts and images
		queries = []
		for image_path, prompt in zip(image_paths, prompts):
			queries.append(self.tokenizer.from_list_format([
				{'image': image_path},
				{'text': prompt},
			]))

		answers = []
		for query in queries:
			answer, history = self.model.chat(self.tokenizer, query=query, history=None)
			answers.append(answer)

		return answers




IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T


def build_transform(input_size):
	MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
	transform = T.Compose([
		T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
		T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
		T.ToTensor(),
		T.Normalize(mean=MEAN, std=STD)
	])
	return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
	best_ratio_diff = float('inf')
	best_ratio = (1, 1)
	area = width * height
	for ratio in target_ratios:
		target_aspect_ratio = ratio[0] / ratio[1]
		ratio_diff = abs(aspect_ratio - target_aspect_ratio)
		if ratio_diff < best_ratio_diff:
			best_ratio_diff = ratio_diff
			best_ratio = ratio
		elif ratio_diff == best_ratio_diff:
			if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
				best_ratio = ratio
	return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
	orig_width, orig_height = image.size
	aspect_ratio = orig_width / orig_height

	# calculate the existing image aspect ratio
	target_ratios = set(
		(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
		i * j <= max_num and i * j >= min_num)
	target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

	# find the closest aspect ratio to the target
	target_aspect_ratio = find_closest_aspect_ratio(
		aspect_ratio, target_ratios, orig_width, orig_height, image_size)

	# calculate the target width and height
	target_width = image_size * target_aspect_ratio[0]
	target_height = image_size * target_aspect_ratio[1]
	blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

	# resize the image
	resized_img = image.resize((target_width, target_height))
	processed_images = []
	for i in range(blocks):
		box = (
			(i % (target_width // image_size)) * image_size,
			(i // (target_width // image_size)) * image_size,
			((i % (target_width // image_size)) + 1) * image_size,
			((i // (target_width // image_size)) + 1) * image_size
		)
		# split the image
		split_img = resized_img.crop(box)
		processed_images.append(split_img)
	assert len(processed_images) == blocks
	if use_thumbnail and len(processed_images) != 1:
		thumbnail_img = image.resize((image_size, image_size))
		processed_images.append(thumbnail_img)
	return processed_images


def load_image(image_file, input_size=448, max_num=6):
	image = Image.open(image_file).convert('RGB')
	transform = build_transform(input_size=input_size)
	images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
	pixel_values = [transform(image) for image in images]
	pixel_values = torch.stack(pixel_values)
	return pixel_values


class InternVLChat(QAModelInstance):
	def __init__(self, ckpt="OpenGVLab/InternVL-Chat-V1-5", torch_device=torch.device("cuda"), model_precision=torch.float32, use_lora=False):
		from transformers import AutoTokenizer, AutoModel
		# Required a 80GB A100. current not support multi gpus now, internvl's bug. 
		self.model = AutoModel.from_pretrained(
			ckpt,
			torch_dtype=torch.bfloat16,
			low_cpu_mem_usage=True,
			trust_remote_code=True,
			device_map='auto').eval()
		self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

	def qa(self, image, prompt):
		if isinstance(image, Image.Image):
			# Check if the image is a PIL.Image object and save to a temporary file if so
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				image.save(tmp.name)
				image_path = tmp.name
				pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).cuda()
		else:
			pixel_values = load_image(image, max_num=6).to(torch.bfloat16).cuda()

		generation_config = dict(
			num_beams=1,
			max_new_tokens=512,
			do_sample=False,
		)

		response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
		return response

class DeepSeekVLChat(QAModelInstance):
	def __init__(self, ckpt="deepseek-ai/deepseek-vl-7b-chat", torch_device=torch.device("cuda"), model_precision=torch.bfloat16, use_lora=False):
		from transformers import AutoModelForCausalLM
		from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

		self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(ckpt)
		self.tokenizer = self.vl_chat_processor.tokenizer

		vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
		self.model = vl_gpt.to(model_precision).to(torch_device).eval()

	def qa(self, image, prompt):

		from deepseek_vl.utils.io import load_pil_images
		
		with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
			if isinstance(image, Image.Image):
				# Check if the image is a PIL.Image object and save to a temporary file if so
					image.save(tmp.name)
					# Use the temporary image path for the tokenizer
					image_path = tmp.name
			else:
				# If `image` is not a PIL.Image object, use it directly
				image_path = image
			
			# single image conversation
			conversation = [
				{
					"role": "User",
					"content": f"<image_placeholder>{prompt}",
					"images": [image_path],
				},
				{"role": "Assistant", "content": ""},
			]

			# load images and prepare for inputs
			pil_images = load_pil_images(conversation)
			
		prepare_inputs = self.vl_chat_processor(
			conversations=conversation,
			images=pil_images,
			force_batchify=True
		).to(self.model.device)

		# run image encoder to get the image embeddings
		inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

		# run the model to get the response
		outputs = self.model.language_model.generate(
			inputs_embeds=inputs_embeds,
			attention_mask=prepare_inputs.attention_mask,
			pad_token_id=self.tokenizer.eos_token_id,
			bos_token_id=self.tokenizer.bos_token_id,
			eos_token_id=self.tokenizer.eos_token_id,
			max_new_tokens=512,
			do_sample=False,
			# change from True to False
			use_cache=False
		)

		answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

		return answer

	
	


	

class IDEFICS2(QAModelInstance):
	def __init__(self, ckpt="HuggingFaceM4/idefics2-8b", torch_device=torch.device("cuda"), model_precision=torch.float16, use_lora=False):
		from transformers import AutoProcessor, AutoModelForVision2Seq

		self.processor = AutoProcessor.from_pretrained(ckpt)
		self.model = AutoModelForVision2Seq.from_pretrained(
			ckpt,
			torch_dtype=model_precision,
			_attn_implementation="flash_attention_2",
			device_map="auto"
		)
		# .to(torch_device)

		if use_lora:
			print("[Lauching Lora] Lora is merging...")
			from peft import PeftModel, LoraConfig
			lora_config = LoraConfig(
				r=4,
				lora_alpha=4,
				lora_dropout=0.1,
				bias="none",
				target_modules=["q_proj", "k_proj", "v_proj"],
				task_type="CAUSAL_LM",
				use_dora=False
			)
			# Use absolute path; need change
			self.model = PeftModel.from_pretrained(self.model, model_id="/linxindisk/VQAPromptBench/logs/multi_prompt_finetune/idefics2-6k-templates/best_lora", config=lora_config)
	
	def _extract_assistant_content(self, text: str):
		parts = text.split('\nAssistant:', 1)
		if len(parts) > 1:
			return 'Assistant:' + parts[1]
		return text

	def qa(self, image, prompt):

		from transformers.image_utils import load_image

		messages = [
			{
				"role": "user",
				"content": [
					{"type": "image"},
					{"type": "text", "text": prompt},
				]
			},
			# {
			# 	"role": "assistant",
			# 	"content": [
			# 		{"type": "text", "text": ""},
			# 	]
			# },
		]

		input_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

		if isinstance(image, Image.Image):
			inputs = self.processor(text=input_prompt, images=[image], return_tensors="pt")
		else:
			inputs = self.processor(text=input_prompt, images=[load_image(image)], return_tensors="pt")

		inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

		generated_ids = self.model.generate(**inputs, max_new_tokens=500)
		generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

		# print(generated_texts[0])
		return self._extract_assistant_content(generated_texts[0])

class Phi3Vision(QAModelInstance):
	def __init__(self, ckpt="microsoft/Phi-3-vision-128k-instruct", torch_device=torch.device("cuda"), model_precision=torch.bfloat16, use_lora=False):
		from transformers import AutoModelForCausalLM, AutoProcessor

		self.device = torch_device
		# use _attn_implementation='eager' to disable flash attention
		self.model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=model_precision, _attn_implementation='flash_attention_2').to(self.device)
		self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True) 

	
	def qa(self, image, prompt):
		messages = [ 
			{
				"role": "user", 
				"content": f"<|image_1|>\n{prompt}"
			}, 
			{
				"role": "assistant", 
				"content": ""
			}, 
		] 

		input_prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

		inputs = self.processor(input_prompt, [image], return_tensors="pt").to(self.device) 

		generation_args = { 
			"max_new_tokens": 500, 
			# "temperature": 0.0, 
			# "do_sample": False, 
		} 

		generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 

		# remove input tokens 
		generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
		response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
		
		return response

class GPT4V(QAModelInstance):
	model_stamp = 'gpt-4-turbo'

	def __init__(self, ckpt, *args, **kwargs):
		from openai import OpenAI
		if isinstance(ckpt, str):
			self.client = OpenAI(api_key=ckpt)
		elif isinstance(ckpt, list):
			self.client = [OpenAI(api_key=c) for c in ckpt]
		self.completion_tokens = 0
		self.prompt_tokens = 0

	def _get_response(self, client, image, prompt):
		while True:
			try:
				response = client.chat.completions.create(
					model=self.model_stamp,
					messages=[
						{
							"role"   : "user",
							"content": [
								{"type": "text", "text": f"{prompt}"},
								{
									"type"     : "image_url",
									"image_url": {
										"url": f"data:image/jpeg;base64,{image}",
									},
								},
							],
						}
					],
					max_tokens=300,
					temperature=0.,
					seed=42,
				)
			except openai.BadRequestError as e:
				if e.code == "sanitizer_server_error":
					continue
				elif e.code == "content_policy_violation":
					response = ""
				else:
					raise e
			except openai.InternalServerError as e:
				continue
			break
		return response

	def cost(self):
		return (0.03 * self.completion_tokens + 0.01 * self.prompt_tokens) / 1000

	def qa(self, image, prompt):
		if isinstance(image, str):
			with open(image, "rb") as image_file:
				base64_image = base64.b64encode(image_file.read()).decode('utf-8')
		else:
			base64_image = image_to_base64(image)

		if isinstance(self.client, list):
			pointer = 0
			while True:
				client = self.client[pointer]
				try:
					response = self._get_response(client, base64_image, prompt)
				except openai.RateLimitError as e:
					if pointer < len(self.client) - 1:
						pointer += 1
						continue
					else:
						raise e
				break
		else:
			response = self._get_response(self.client, base64_image, prompt)
		if isinstance(response, str):
			return response
		else:
			self.completion_tokens += response.usage.completion_tokens
			self.prompt_tokens += response.usage.prompt_tokens
			return response.choices[0].message.content
		

class GPT4O(GPT4V):
	model_stamp = 'gpt-4o'


def upload_image_to_oss(image_path, bucket_name='benverse', endpoint='http://oss-cn-hongkong.aliyuncs.com',
						access_key_id='<your access key>', access_key_secret='<you access key secret>'):
	import oss2
	import secrets

	endpoint = endpoint
	auth = oss2.Auth(access_key_id, access_key_secret)
	bucket = oss2.Bucket(auth, endpoint, bucket_name)

	file_name = f"{secrets.token_hex(9)}.png"
	with open(image_path, 'rb') as file:
		bucket.put_object(file_name, file)

	domain = endpoint[endpoint.find("http://") + 7:]
	return f'https://{bucket_name}.{domain}/{file_name}', file_name


def delete_image_from_oss(file_name, bucket_name='benverse', endpoint='http://oss-cn-hongkong.aliyuncs.com',
						  access_key_id='<your access key>', access_key_secret='<you access key secret>'):
	import oss2
	endpoint = endpoint
	auth = oss2.Auth(access_key_id, access_key_secret)
	bucket = oss2.Bucket(auth, endpoint, bucket_name)
	bucket.delete_object(file_name)


class QwenVLAPI(QAModelInstance):
	model_name = None

	def __init__(self, ckpt, *args, **kwargs):
		self.ckpt = ckpt[0]
		self.access_key_id = ckpt[1]
		self.access_key_secret = ckpt[2]

	def _get_response(self, image_path, base64_image, prompt):
		import dashscope
		dashscope.api_key = self.ckpt
		image_url, image_file_name = upload_image_to_oss(image_path, access_key_id=self.access_key_id, access_key_secret=self.access_key_secret)
		messages = [{
			'role'   : 'system',
			'content': [{
				'text': 'You are a helpful assistant.'
			}]
		}, {
			'role'   :
				'user',
			'content': [
				{
					'image': image_url
				},
				{
					'text': prompt
				},
			]
		}]
		while True:
			try:
				response = dashscope.MultiModalConversation.call(model=self.model_name, messages=messages)
				if response.code == 'DataInspectionFailed':
					response = ""
				elif response.code == 'Throttling.RateQuota':
					time.sleep(60)
					continue
				else:
					response = response["output"]["choices"][0]["message"]["content"][0]["text"]
			except:
				continue
			break
		delete_image_from_oss(image_file_name, access_key_id=self.access_key_id, access_key_secret=self.access_key_secret)
		return response

	def qa(self, image, prompt):
		if isinstance(image, str):
			with open(image, "rb") as image_file:
				base64_image = base64.b64encode(image_file.read()).decode('utf-8')
			response = self._get_response(image, base64_image, prompt)
		else:
			base64_image = image_to_base64(image)
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				image.save(tmp.name)
				image_path = tmp.name
				response = self._get_response(image_path, base64_image, prompt)
		return response
	
	def batch_qa(self, iamges, prompts):
		if len(images) != len(promts):
			raise ValueError("The length of the image list and the prompt list must be the same.")

		uploaded_images = []
		for imaga in images:
			if isinstance(image, str):
				image_path, image_file_name = self._upload_image(image)
			else:
				with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tep:
					image.save(tmp.name)
					image_path, image_file_name = self._upload_image(tmp.name)
			uploaded_images.append((image_path, image_file_name))

		response = []
		for idx, (image_info, prompt) in enumerate(zip(uploaded_images, prompts)):
			image_url, image_file_name = image_info
			response = self._get_response(image_url, prompt)
			responses.append(response)
            # 删除已上传的图像
			self._delete_image(image_file_name)

		return responses



class QwenVLPlus(QwenVLAPI):
	model_name = 'qwen-vl-plus'


class QwenVLMax(QwenVLAPI):
	model_name = 'qwen-vl-max'


class GeminiVisionAPI(QAModelInstance):
	model_name = None

	def __init__(self, ckpt, *args, **kwargs):
		import google.generativeai as genai
		GOOGLE_API_KEY = ckpt
		genai.configure(api_key=GOOGLE_API_KEY)
		self.model = genai.GenerativeModel(self.model_name)

	def _get_response(self, image_path, base64_image, prompt):
		import google
		import google.generativeai as genai
		img = Image.open(image_path)
		prompt = prompt
		while True:
			try:
				response = self.model.generate_content([prompt, img], stream=True)
				response.resolve()
				response = response.text
			except ValueError:
				response = ""
			except genai.types.generation_types.BlockedPromptException:
				response = ""
			except google.api_core.exceptions.DeadlineExceeded:
				time.sleep(60)
				continue
			except google.api_core.exceptions.InternalServerError:
				continue
			break
		return response

	def qa(self, image, prompt):
		if isinstance(image, str):
			with open(image, "rb") as image_file:
				base64_image = base64.b64encode(image_file.read()).decode('utf-8')
			response = self._get_response(image, base64_image, prompt)
		else:
			base64_image = image_to_base64(image)
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				image.save(tmp.name)
				image_path = tmp.name
				response = self._get_response(image_path, base64_image, prompt)
		return response


class GeminiVisionPro(GeminiVisionAPI):
	model_name = 'gemini-pro-vision'


class ReplicateAPI(QAModelInstance):
	model_name = None
	model_list = None

	def __init__(self, ckpt, *args, **kwargs):
		import replicate
		self.replicate_client = replicate.Client(api_token=ckpt)

	def _get_response(self, image_path, prompt):
		image = open(image_path, "rb")
		input = {
			"image" : image,
			"prompt": prompt
		}
		while True:
			try:
				output = self.replicate_client.run(
					self.model_list[self.model_name],
					input=input
				)
				response = "".join(output)
			except:
				time.sleep(60)
				continue
			break
		return response

	def qa(self, image, prompt):
		if isinstance(image, str):
			response = self._get_response(image, prompt)
		else:
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				image.save(tmp.name)
				image_path = tmp.name
				response = self._get_response(image_path, prompt)

		return response


class LLaVA34B(ReplicateAPI):
	model_name = 'llava-v1.6-34b'
	model_list = {
		"llava-v1.6-34b": "yorickvp/llava-v1.6-34b:41ecfbfb261e6c1adf3ad896c9066ca98346996d7c4045c5bc944a79d430f174",
	}