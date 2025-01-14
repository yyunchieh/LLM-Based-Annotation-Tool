import streamlit as st
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import os
import openai
import json

# Import SAM model

sam_checkpoint = r"c:\Users\4019-tjyen\Downloads\sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Load API key
API_path = r"C:\Users\4019-tjyen\Desktop\API.txt"
with open(API_path,"r") as file:
    openapi_key = file.read().strip()

os.environ['OPENAI_API_KEY'] = openapi_key
openai.api_key = openapi_key

#image_path = r"C:\Users\4019-tjyen\Desktop\cat\cat1.jpg"

# Initialize LLM
llm = OpenAI(model="gpt-4o")
prompt_template = PromptTemplate(
    input_variables=["object_descriptions"],
    template="""
    圖片中的物件如下:
    {object_descriptions}
    請生成語義標註，並以 Json 格式輸出:
    [
      {{"object": "cat", "category": "animal", "attributes": ["black", "sitting"]}},
      {{"object": "chair", "category", "furniture", "attributes" : ["wooden"]}}
    ] 
    """
)

chain = LLMChain(llm=llm, prompt=prompt_template)

# Image Segmentation
def segment_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    predictor.set_image(image_np)
    #mask, _, _ = predictor.predict(box=None, point_coords=None, point_labels=None, multimask_output=True)
    #return image, masks
    
    try:
        print("Calling predictor.predict...")
        mask, _, _ = predictor.predict(box=None, point_coords=None, point_labels=None, multimask_output=True)
        print(f"Masks returned: {masks}")
    except Exception as e:
        print("error")
        mask = []
    
    return image, masks
# Descriptions Generation
def generate_descriptions(masks):
    descriptions = []
    for idx, mask in enumerate(masks):
        descriptions.append(f"Object {idx + 1} with unique shape and size.")
    return descriptions

# Streamlit APP
def main():
    st.title("SAM+LangChain: Tools for Annotation")

    # Upload image
    uploaded_file = st.file_uploader("upload files", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Run segmentation
        st.write("Segmenting image...")
        image, masks = segment_image(uploaded_file)

        # Display segmentation results
        st.write(f"Detected {len(masks)} regions.")

        # Generate descriptions
        descriptions = generate_descriptions(masks)
        st.write("Generated descriptions:", descriptions)

        # Generate annotations with LangChain
        st.write("Generating semantic annotations...")
        annotations = chain.run(object_descriptions=", ".join(descriptions))
        st.json(annotations)
    

if __name__ == "__main__":
    main()


