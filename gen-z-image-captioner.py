import streamlit as st
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
import re

st.set_page_config(layout="wide", page_title="Gen Z Image Captioner")

st.write("# Gen-Z Image Captioner")
st.write("### Add fun, zesty captions to your images! Use our captions as inspiration for social media, marketing, and more.")
st.write(
    "Try uploading an image via the left sidebar to generate a gen-z stylized caption."
)
st.write("Created by Grace Alwan, Christine Manegan, and Brennan Megregian. Special thanks to the CS324 class at Stanford University for motivating this project! :grin:")
st.write("Remember that this captioner is based off of AI - and it doesn’t actually know you! The AI may create captions that do not correctly represent you or your photo, and in that case, we encourage you to try again :smile:")
st.sidebar.write("## Upload an image to caption :gear:")

@st.cache_resource
def download_models():
    #download image captioner
    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

    #download tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("xzyao/HWNWZ5B9AM9PSXV09BAF24ADGP97LMCN2LFQTZJNRQF318SMFM") 
    model = AutoModelForCausalLM.from_pretrained("xzyao/HWNWZ5B9AM9PSXV09BAF24ADGP97LMCN2LFQTZJNRQF318SMFM")
    print("downloaded tokenizer + model")
    return image_to_text, tokenizer, model



col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader(" ", type=["png", "jpg", "jpeg"])



def our_awesome_model(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)
    col2.write("Caption :heart_decoration:")
    # insert our pipeline here
    
    image_to_text, tokenizer, model = download_models()

    #get descriptive caption
    plain_text_caption_obj = image_to_text(image)
    plain_text_caption = plain_text_caption_obj[0]['generated_text']
    print("plaintext caption: ", plain_text_caption)

    #set prompt
    with open("./prompt.txt") as f: 
        plain_prompt = "".join(f.readlines())
    prompt = plain_prompt.replace("<ADD-PLAIN-TEXT-CAPTION>", plain_text_caption)

    #tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    #generate from model
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=1.0,
        max_new_tokens=20,
        eos_token_id=198, #new line token
        pad_token_id=198,
        num_return_sequences=5,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)

    #extract captions
    captions = [re.findall(r"gen-z social media caption: (.*)\n?", text)[-1].replace("\n", "") for text in gen_text]
    print(captions)
    col2.write("1: " + captions[0])
    col2.write("2: " + captions[1])
    col2.write("3: " + captions[2])
    col2.write("4: " + captions[3])
    col2.write("5: " + captions[4])

if my_upload is not None:
    our_awesome_model(upload=my_upload)
else:
    image = Image.open('./Gradpic.png')
    col1.write("Original Image :camera:")
    col1.image(image)
    col2.write("Caption :heart_decoration:")
    col2.write("1: Last night was an Amazon original 💯😫\U0001faf6")
    col2.write("2: 🐻⭐️")
    col2.write("3: Just me")
    col2.write("4: girls are from Mars! boys are from Venus! ⚡️")
    col2.write("5: red carpet redux")
