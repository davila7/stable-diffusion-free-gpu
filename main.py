from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

import torch
from diffusers import StableDiffusionPipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

import base64
from io import BytesIO

# Load model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl").to("cuda")

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
  #get the prompt input
  prompt = request.form['prompt-input']
  print(f"Generating an image of {prompt}")

  # generate image
  image = pipe(prompt).images[0]
  print("Image generated! Converting image ...")
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue())
  img_str = "data:image/png;base64," + str(img_str)[2:-1]

  #generate text
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
  generated_output = model.generate(input_ids, do_sample=True, temperature=1.0, max_length=2500, num_return_sequences=1)
  generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

  print("Sending image and text ...")
 
  return render_template('index.html', generated_image=img_str, generated_text=generated_text, prompt=prompt)

if __name__ == '__main__':
    app.run()