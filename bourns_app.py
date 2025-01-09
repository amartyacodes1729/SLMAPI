from flask import Flask, request, jsonify, render_template
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import os
import ast

app = Flask(__name__)

# Model and processor setup
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=compute_dtype,
)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto", quantization_config=quant_config
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image temporarily
    temp_image_path = os.path.join("temp", image_file.filename)
    os.makedirs("temp", exist_ok=True)
    image_file.save(temp_image_path)

    # Prepare the input
    text = (
        "You are an expert in table reading task. First obtain the part number/codes present inside the table "
        "under electrical specifications. Then obtain the attributes/column names present in the table like Q ref. "
        "Now obtain the attribute values for each and every part number present in the table. "
        "Give the output in the form of a dictionary. Make the main key as 'attribute_vales'. "
        "Give only the dictionary with unique part numbers."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": temp_image_path, "resized_height": 1024, "resized_width": 1024},
                {"type": "text", "text": text},
            ],
        }
    ]

    # Prepare inputs for the model
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=5000)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    print("Generating Report....")
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    output_text = output_text.replace("```", "").replace("json", "").replace("±", "")

    try:
        data = ast.literal_eval(output_text)
    except Exception as e:
        return jsonify({"error": "Failed to parse model output", "details": str(e)}), 500

    # Cleanup temporary file
    os.remove(temp_image_path)

    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
