import tempfile
from typing import Union
import torch
from PIL import Image
from transformers import image_utils

from .base_qa_model import QAModelInstance, QAModel
from .utils import image_to_base64, load_image

imageqa_models = {

    "llavav1.5-7b": ("LLaVA", "llava-hf/llava-1.5-7b-hf"),
    "llavav1.5-13b": ("LLaVA", "llava-hf/llava-1.5-13b-hf"),
    "llavav1.6-7b": ("LLaVA", "llava-hf/llava-v1.6-vicuna-7b-hf"),
    "llavav1.6-13b": ("LLaVA", "llava-hf/llava-v1.6-vicuna-13b-hf"),
    "qwenvl": ("QwenVL", "Qwen/Qwen-VL"),
    "qwenvl-chat": ("QwenVLChat", "Qwen/Qwen-VL-Chat"),
    "internvl-chat-v1.5": ("InternVLChat", 'failspy/InternVL-Chat-V1-5-quantable'),
    "idefics2-8b": ("IDEFICS2", "HuggingFaceM4/idefics2-8b"),

    "llavav1.5-7b-100-templated": ('LLaVA', "shijianS01/llava-v1.5-7b-lora-100-templated"),
    "llavav1.5-7b-1k-templated": ('LLaVA', "shijianS01/llava-v1.5-7b-lora-1k-templated"),
    "llavav1.5-7b-5k-templated": ('LLaVA', "shijianS01/llava-v1.5-7b-lora-5k-templated"),
    "llavav1.5-7b-10k-templated": ('LLaVA', "shijianS01/llava-v1.5-7b-lora-10k-templated"),
    "llavav1.5-7b-15k-templated": ('LLaVA', "shijianS01/llava-v1.5-7b-lora-15k-templated"),

    "llavav1.5-13b-100-templated": ('LLaVA', "shijianS01/llava-v1.5-13b-lora-100-templated"),
    "llavav1.5-13b-1k-templated": ('LLaVA', "shijianS01/llava-v1.5-13b-lora-1k-templated"),
    "llavav1.5-13b-5k-templated": ('LLaVA', "shijianS01/llava-v1.5-13b-lora-5k-templated"),
    "llavav1.5-13b-10k-templated": ('LLaVA', "shijianS01/llava-v1.5-13b-lora-10k-templated"),
    "llavav1.5-13b-15k-templated": ('LLaVA', "shijianS01/llava-v1.5-13b-lora-15k-templated"),

}


def set_imageqa_model_key(model_name, key):
    imageqa_models[model_name] = (imageqa_models[model_name][0], key)


def list_imageqa_models():
    return list(imageqa_models.keys())


class ImageQAModel(QAModel):
    def __init__(
        self,
        model_name: str,
        model: QAModelInstance = None,
        torch_device: Union[int, str] = -1,
        precision=torch.bfloat16,
        choice_format='letter',
        enable_choice_search: bool = True,
        cache_path: str = None,

    ):
        super().__init__(model_name, choice_format, enable_choice_search, cache_path)

        if isinstance(torch_device, str):
            if torch_device != "auto":
                torch_device = torch.device(torch_device)
            else:
                pass
        else:
            if torch_device == -1:
                torch_device = torch.device(
                    "cuda") if torch.cuda.is_available() else "cpu"
            else:
                torch_device = torch.device(f"cuda:{torch_device}")

        if model is None:
            print(f"Loading {model_name}...")
            class_name, ckpt = imageqa_models[model_name]
            self.model_precision = precision
            self.model = eval(class_name)(
                ckpt, torch_device, self.model_precision)
            print(f"Finish loading {model_name}")
        else:
            print(f"Using provided model...")
            self.model = model

    def _data_to_str(self, data):
        if isinstance(data, str):
            return data
        else:
            return image_to_base64(data)


class LLaVA(QAModelInstance):
    def __init__(self, ckpt="llava-hf/llava-1.5-7b-hf", torch_device=torch.device("cuda"), model_precision=torch.float32):

        if ckpt == "llava-hf/llava-v1.6-vicuna-13b-hf" or ckpt == "llava-hf/llava-v1.6-vicuna-7b-hf":
            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                ckpt,
                torch_dtype=model_precision,
                low_cpu_mem_usage=True,
                device_map=torch_device
            ).eval()
            self.processor = LlavaNextProcessor.from_pretrained(
                ckpt, device_map=torch_device)
        elif ckpt in {
                "llava-hf/llava-1.5-7b-hf",
                "shijianS01/llava-v1.5-7b-lora-100-templated",
                "shijianS01/llava-v1.5-7b-lora-1k-templated",
                "shijianS01/llava-v1.5-7b-lora-5k-templated",
                "shijianS01/llava-v1.5-7b-lora-10k-templated",
                "shijianS01/llava-v1.5-7b-lora-15k-templated",
        }:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            self.model = LlavaForConditionalGeneration.from_pretrained(
                ckpt,
                torch_dtype=model_precision,
                low_cpu_mem_usage=True,
                device_map=torch_device,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                "llava-hf/llava-1.5-7b-hf", device_map=torch_device)
        elif ckpt in {
                "llava-hf/llava-1.5-13b-hf",
                "shijianS01/llava-v1.5-13b-lora-100-templated",
                "shijianS01/llava-v1.5-13b-lora-1k-templated",
                "shijianS01/llava-v1.5-13b-lora-5k-templated",
                "shijianS01/llava-v1.5-13b-lora-10k-templated",
                "shijianS01/llava-v1.5-13b-lora-15k-templated"
        }:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            self.model = LlavaForConditionalGeneration.from_pretrained(
                ckpt,
                torch_dtype=model_precision,
                low_cpu_mem_usage=True,
                device_map=torch_device,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                "llava-hf/llava-1.5-13b-hf", device_map=torch_device)
        else:
            raise ValueError("Not Implemented")

    def qa(self, image, prompt):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        prompt = "USER: <image>\n" + prompt + "\nASSISTANT:"
        if isinstance(self.model, torch.nn.DataParallel):
            inputs = self.processor(prompt, image, return_tensors='pt').to(
                next(self.model.parameters()).device)
            out = self.model.module.generate(
                **inputs, max_new_tokens=200, do_sample=False)
        else:
            inputs = self.processor(
                prompt, image, return_tensors='pt').to(self.model.device)
            out = self.model.generate(
                **inputs, max_new_tokens=200, do_sample=False)
        answer = self.processor.decode(
            out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

        return answer


class QwenVL(QAModelInstance):
    def __init__(self, ckpt="Qwen/Qwen-VL", torch_device=torch.device("cuda"), model_precision=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt, trust_remote_code=True)
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

                inputs = self.tokenizer(
                    query, return_tensors='pt').to(self.model.device)
                out = self.model.generate(**inputs)

        else:
            # If `image` is not a PIL.Image object, use it directly
            query = self.tokenizer.from_list_format([
                {'image': image},
                {'text': prompt},
            ])

            inputs = self.tokenizer(
                query, return_tensors='pt').to(self.model.device)
            out = self.model.generate(**inputs)

        answer = self.tokenizer.decode(
            out[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()

        return answer


class QwenVLChat(QAModelInstance):
    def __init__(self, ckpt="Qwen/Qwen-VL-Chat", torch_device=torch.device("cuda"), model_precision=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig

        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt, trust_remote_code=True)
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
        self.model.generation_config = GenerationConfig.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True)

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

                answer, history = self.model.chat(
                    self.tokenizer, query=query, history=None)
        else:
            # If `image` is not a PIL.Image object, use it directly
            query = self.tokenizer.from_list_format([
                {'image': image},
                {'text': prompt},
            ])

            answer, history = self.model.chat(
                self.tokenizer, query=query, history=None)

        return answer


class InternVLChat(QAModelInstance):
    def __init__(self, ckpt="OpenGVLab/InternVL-Chat-V1-5", torch_device=torch.device("cuda"), model_precision=torch.float32):
        from transformers import AutoTokenizer, AutoModel
        # Required a 80GB A100. current not support multi gpus now, internvl's bug.
        self.model = AutoModel.from_pretrained(
            ckpt,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto').eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt, trust_remote_code=True)

    def qa(self, image, prompt):
        if isinstance(image, Image.Image):
            # Check if the image is a PIL.Image object and save to a temporary file if so
            with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
                image.save(tmp.name)
                image_path = tmp.name
                pixel_values = load_image(
                    image_path, max_num=6).to(torch.bfloat16).cuda()
        else:
            pixel_values = load_image(
                image, max_num=6).to(torch.bfloat16).cuda()

        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )

        response = self.model.chat(
            self.tokenizer, pixel_values, prompt, generation_config)
        return response


class IDEFICS2(QAModelInstance):
    def __init__(self, ckpt="HuggingFaceM4/idefics2-8b", torch_device=torch.device("cuda"), model_precision=torch.float32):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = AutoModelForVision2Seq.from_pretrained(
            ckpt,
            torch_dtype=model_precision,
            _attn_implementation="flash_attention_2",
            device_map=torch_device
        )

    def _extract_assistant_content(self, text: str):
        parts = text.split('\nAssistant:', 1)
        if len(parts) > 1:
            return 'Assistant:' + parts[1]
        return text

    def qa(self, image, prompt):

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ]
            }
        ]

        input_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True)

        if isinstance(image, Image.Image):
            inputs = self.processor(text=input_prompt, images=[
                                    image], return_tensors="pt")
        else:
            inputs = self.processor(text=input_prompt, images=[
                                    image_utils.load_image(image)], return_tensors="pt")

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True)

        # print(generated_texts[0])
        return self._extract_assistant_content(generated_texts[0])