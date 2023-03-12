from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

import base64
from io import BytesIO

# Load model
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

# Start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def initial():
  return render_template('index.html')


@app.route('/submit-prompt-image', methods=['POST'])
def generate_image():
  prompt = request.form['prompt-input']
  print(f"Generating an image of {prompt}")

  image = pipe(prompt).images[0]
  print("Image generated! Converting image ...")
  
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue())
  img_str = "data:image/png;base64," + str(img_str)[2:-1]

  print("Sending image ...")
  return render_template('index.html', generated_image=img_str)

@app.route('/submit-prompt-text', methods=['POST'])
def generate_text():
  prompt = request.form['prompt-input']
  print(f"Generating text for prompt: {prompt}")

  input_ids = tokenizer(prompt, return_tensors='pt').input_ids
  generated_output = model.generate(input_ids, do_sample=True, temperature=0.5, max_length=512, num_return_sequences=1)
  generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

  print(f"Text generated: {generated_text}")
  return render_template('index.html', generated_text=generated_text)

if __name__ == '__main__':
    app.run()