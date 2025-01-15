from flask import Flask, request, jsonify, render_template
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import ast
import torch
import pandas as pd
import os
import pdf2image
from PIL import Image
import requests
import shutil
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

@app.route("/", methods=["GET"])
def home():
    return render_template("index_new.html")

@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    pdf_url = request.json.get('pdf_url')
    if not pdf_url:
        return jsonify({"error": "No URL provided."}), 400

    try:
        # Download the PDF file from the provided URL
        response = requests.get(pdf_url, stream=True)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download the PDF from the URL."}), 400

        # Save the downloaded PDF to a temporary file
        temp_pdf_path = "temp_downloaded.pdf"
        with open(temp_pdf_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)

        # Convert PDF to image
        print("Converting PDF to image...")
        pdf_read = pdf2image.convert_from_path(temp_pdf_path)
        pdf_read[0].save("temp_image.png", "PNG")
        image = Image.open("temp_image.png")
        print("PDF converted to image successfully.")
        # Define the prompt
        text = """You are an expert in table reading task. First obtain the part number/codes present inside the table under electrical specifications.Then for each part number obtain the electrical attribute values which are the names of the columns. The attribute values that you need to obtain are Inductance (L), Inductance Tol., SRF (MHz), DCR  (mW) Typ., DCR  (mW) Max., Irms (A) typ., Isat1 (A) typ. and Isat2 (A) typ. Give the output in the form of a dictionary. Make the main key as 'attribute_values'. Give only the dictionary with unique part numbers."""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "temp_image.png",
                        "resized_height": 1024,
                        "resized_width": 1024,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        print("Generating output...")
        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=5000)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        output_text = output_text.replace("```", "").replace("json", "").replace("\u00b1", "").replace("python", "")
        print("Output generated successfully.")
        print("Output:", output_text)
        # Parse the output
        data = ast.literal_eval(output_text)
        result = {
            {"Part Number": key, "URL": pdf_url, **value} for key, value in data.items()
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8002)
