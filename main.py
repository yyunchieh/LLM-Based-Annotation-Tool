import streamlit as st
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import os
import openai

# Import SAM model

sam_checkpoint = r"c:\Users\4019-tjyen\Downloads\sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)


API_path = r"C:\Users\4019-tjyen\Desktop\API.txt"
with open(API_path,"r") as file:
    openapi_key = file.read().strip()

os.environ['OPENAI_API_KEY'] = openapi_key
openai.api_key = openapi_key

image_path = r"C:\Users\4019-tjyen\Desktop\cat\cat1.jpg"

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

# Image Segmentation
def segment_image(image_path):
    # Load images
    image = Image.open(image_path).convert("RGB")
    image = np.array(Image)
    predictor.set_image(image_np)
    mask, _, _ = predictor.predict(box=None, pount_coords=None, point_labels=None, multimask_output=True)

    return image, masks

# Descriptions Generation
def generation_descriptions(masks):
    descriptions = []
    for idx, mask in enumerate(masks):
        descriptions.append(f"Object {idx + 1} with unique shape and size.")
    return descriptions

# Streamlit APP
def main():
    st.title("SAM+LangChain: tools for annotation")

    # Upload Images
    uploaded_file = st.file_uploader("upload files", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="uploaded images", use_column_width=True)

        st.write("Segmenting images...")
        image, masks = segment_image(uploaded_file)

        st.write(f"After checking, there are {len(masks)} zones.")

        descriptions = generate_descriptions(masks)
        st.write("Generated descriptions:", descriptions)

        st.write("Generating the annotations...")
        annotations = chain.run(object_descriptions=", ".join(descriptions))
        st.json(annotations)
    

if __name__ == "__main__":
    main()


