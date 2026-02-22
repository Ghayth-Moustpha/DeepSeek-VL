import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images


model_path = "deepseek-ai/deepseek-vl-1.3b-chat"

# Load processor
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# Load model with 4-bit quantization
vl_gpt = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype= torch.float16,    
    device_map="auto",
    trust_remote_code=True
)
vl_gpt.eval()

# Use a sample image – replace with your own image path
conversation = [
    {
        "role": "User",
        "content": "<image_placeholder> As an autonomous driving assistant, explain step by step why the vehicle has stopped. Consider traffic rules, road signs, and environment.",
        "images": ["./images/car_red_light.png"],
    },
    {"role": "Assistant", "content": ""},
]

# Load image and prepare inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True
).to(vl_gpt.device)          # Move the whole object to GPU

# Cast image tensor to model's dtype (float16)
prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(vl_gpt.dtype)

# Generate response
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=500,
    do_sample=False,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(answer)