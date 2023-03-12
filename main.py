from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

import torch
from diffusers import StableDiffusionPipeline
from transformers import T5Tokenizer, FlaxT5ForConditionalGeneration

import base64
from io import BytesIO

# Load model
model_name = 'google/flan-t5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = FlaxT5ForConditionalGeneration.from_pretrained(model_name)

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

# Start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def initial():
  return render_template('index.html')


@app.route('/submit-prompt', methods=['POST'])
def generate():
  prompt = request.form['prompt-input']
  print(f"Generating an image of {prompt}")

  image = pipe(prompt).images[0]
  print("Image generated! Converting image ...")
  
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue())
  img_str = "data:image/png;base64," + str(img_str)[2:-1]

  input_ids = tokenizer(prompt, return_tensors='pt').input_ids
  generated_output = model.generate(input_ids, do_sample=True, temperature=0.5, max_length=512, num_return_sequences=1)
  generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

  print("Sending image ...")
  return render_template('index.html', generated_image=img_str, generated_text=generated_text)

if __name__ == '__main__':
    app.run()