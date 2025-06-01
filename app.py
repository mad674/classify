from fastapi import FastAPI,Request, Response,APIRouter,UploadFile, Depends, HTTPException, status, File, Form
from fastapi.responses import JSONResponse#, FileResponse
# from schemas import MaskingResponse#,QueryIn, GenerateOut, PDFQuestionResponse , ImageMaskingResponse, ErrorResponse
# from retriever import generate_predicted_gold_inds
# from evaluator import evaluate_program
# from generator import infer, build_vocab, PointerProgramGenerator
# from masking import predict_and_mask, run_final_pattern_check#, BERTForNER, entity_mapping,mask_predictions
# from model_retriever import BertRetriever
# from qa_pipeline.main import convert_pdf_to_json
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from typing import Dict#, List, Any, Union, Optional
# import tempfile
# from utils import extract_text_from_file
# import pdfplumber
import gdown
import torch
from transformers import AutoTokenizer#,BertModel,BertTokenizerFast , PegasusForConditionalGeneration, DetrForObjectDetection, DetrImageProcessor, DetrConfig
import numpy as np
import json
import os
import io
# import base64
# import shutil
# from pathlib import Path
# from safetensors.torch import safe_open
# import pytesseract
# import zipfile
from contextlib import asynccontextmanager
# from PIL import Image, ImageDraw, ImageFont
# from docx import Document
from fastapi.middleware.cors import CORSMiddleware
os.environ["WANDB_DISABLED"] = "true"

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if not os.path.exists("pegasus/pegasus_quantized.pt"):
#     with zipfile.ZipFile("pegasus/pegasus_model.zip", 'r') as zip_ref:
#         zip_ref.extractall("pegasus")
#     os.remove("pegasus/pegasus_model.zip")
# IMAGE_MODEL_DIR = "./detr_model"
# PROCESSOR_DIR = "./detr_processor"
# try:
#     # Load model and processor
#     img_config = DetrConfig.from_json_file(os.path.join(IMAGE_MODEL_DIR,"config.json"))
#     img_model = DetrForObjectDetection(img_config)
    
#     # Fix the device parameter in safe_open
#     with safe_open(os.path.join(IMAGE_MODEL_DIR, "model.safetensors"), framework="pt") as f:
#         img_state_dict = {k: f.get_tensor(k) for k in f.keys()}
#     # Move model to device after loading state dict
#     img_model.load_state_dict(img_state_dict)
#     img_model = img_model.to(device)
#     img_model.eval()
    
#     if os.path.exists(os.path.join(PROCESSOR_DIR, "preprocessor_config.json")):
#         img_processor = DetrImageProcessor.from_pretrained(
#             PROCESSOR_DIR,
#             local_files_only=True
#         )
#     else:
#         raise FileNotFoundError(
#             f"Processor files not found in {PROCESSOR_DIR}. "
#             "Please run download_processor.py first."
#         )
# except Exception as e:
#     print(f"Error loading DETR model: {str(e)}")
#     raise RuntimeError("Failed to load DETR model. Please ensure all dependencies are installed.")

def download_model(file_id, output_path):
    """
    Downloads a file from Google Drive using gdown.
    :param file_id: The Google Drive file ID.
    :param output_path: The local path to save the file.
    """
    output_dir = os.path.dirname(output_path) or "."  # Use current directory if no directory is specified
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        if file_id == "none":
            return
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{output_path} already exists.")
        
# MODEL_PATH ="quantized_ner_model.pt"
# download_model("1_bupFomoYtMq3WrexSsAv9DW7er-VdWD", MODEL_PATH)

# pegasus_tokenizer = PegasusTokenizer.from_pretrained("doc_summary_tok")
# pegasus_model = torch.load("pegasus_quantized.pt", map_location=device, weights_only=False)
# MODEL_NAME = "bert-base-uncased"

# # Load model and tokenizer
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_model, ner_tokenizer
    # logger.info("Initializing model and tokenizer...")
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # logger.info(f"Using device: {device}")
        
        # Load tokenizer
        ner_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Initialize mode
        # Check if model file exists
        # if os.path.exists(MODEL_PATH):
        #     # Load model weights
        #     ner_model=torch.load(MODEL_PATH, map_location=device,weights_only=False)
        #     # logger.info(f"Model loaded from {MODEL_PATH}")
        # else:
        #     print(f"Model file not found at {MODEL_PATH}, initializing with default weights")
        
        # ner_model.to(device)
        # ner_model.eval()
        # yield
        # logger.info("Model and tokenizer initialized successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your client app to make requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@app.head("/health")
async def health_check():
    return {"status": "ok"}
##check if the server is running
@app.get("/")
async def root():
    return {"message": "summary fin_gpt server is running"} 

@app.head("/")
async def health_check():
    return JSONResponse(content=None, status_code=200)

# # File upload and masking endpoint
# @app.post("/mask-file", response_model=MaskingResponse)
# async def mask_file(file: UploadFile = File(...)):
#     """
#     Upload a file and mask sensitive information in it.
#     """
#     try:
#         # Create a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
#             # Copy uploaded file to temporary file
#             shutil.copyfileobj(file.file, temp_file)
#             temp_path = temp_file.name
        
#         # Read the file content
#         with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
#             content = f.read()
        
#         # Process the content
#         original_content = content
#         masked_content = predict_and_mask(ner_tokenizer,ner_model,content,device)
        
#         # Run a final check to ensure correct masking
#         masked_content = run_final_pattern_check(masked_content, original_content)
        
#         # Clean up temporary file
#         os.unlink(temp_path)
        
#         # Return the masked content
#         return MaskingResponse(masked_text=masked_content)
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Text masking endpoint
# @app.post("/mask-text", response_model=MaskingResponse)
# async def mask_text(request: Dict[str, str]):
#     """
#     Mask sensitive information in provided text.
#     """
#     text = request.get("text", "")
    
#     if not text:
#         raise HTTPException(status_code=400, detail="No text provided")
    
#     try:
#         original_text = text
#         masked_text = predict_and_mask(ner_tokenizer,ner_model,text,device)
        
#         # Run a final check to ensure correct masking
#         masked_text = run_final_pattern_check(masked_text, original_text)
        
#         return MaskingResponse(masked_text=masked_text)
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


# @app.post("/mask-image", response_model=ImageMaskingResponse)
# async def mask_image(file: UploadFile = File(...)):
#     """
#     Mask sensitive information in an image.
#     """
#     # Validate file type
#     if not file.filename:
#         print("Error: No file provided")
#         raise HTTPException(status_code=400, detail="No file provided")
    
#     # Infer file type from extension if Content-Type is application/octet-stream
#     allowed_types = {"image/jpeg", "image/png", "image/bmp"}
#     file_extension = file.filename.split(".")[-1].lower()
#     inferred_type = {
#         "jpg": "image/jpeg",
#         "jpeg": "image/jpeg",
#         "png": "image/png",
#         "bmp": "image/bmp"
#     }.get(file_extension)

#     if file.content_type == "application/octet-stream" and inferred_type not in allowed_types:
#         print(f"Error: Unsupported file type inferred from extension: {file_extension}")
#         raise HTTPException(
#             status_code=400,
#             detail=f"File type not allowed. Must be one of: {', '.join(allowed_types)}"
#         )
#     if file.content_type not in allowed_types and inferred_type not in allowed_types:
#         print(f"Error: Unsupported file type {file.content_type} or inferred type {inferred_type}")
#         raise HTTPException(
#             status_code=400,
#             detail=f"File type not allowed. Must be one of: {', '.join(allowed_types)}"
#         )

#     try:
#         # Read and validate image
#         contents = await file.read()
#         if not contents:
#             print("Error: Empty file")
#             raise HTTPException(status_code=400, detail="Empty file")
            
#         # Open and convert image
#         try:
#             image = Image.open(io.BytesIO(contents)).convert("RGB")
#         except Exception as e:
#             print(f"Error: Invalid image file - {str(e)}")
#             raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

#         # Process with model
#         try:
#             inputs = img_processor(images=image, return_tensors="pt")
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             # print(inputs)
#             with torch.no_grad():
#                 outputs = img_model(**inputs)
#             # Postprocess outputs
#             target_sizes = torch.tensor([image.size[::-1]])
#             results = img_processor.post_process_object_detection(
#                 outputs, 
#                 target_sizes=target_sizes, 
#                 threshold=0.9
#             )[0]

#             # Check if any objects were detected
#             if len(results["boxes"]) == 0:
#                 print("No objects detected in the image")
#                 return ImageMaskingResponse(
#                     boxes=[],
#                     labels=[],
#                     scores=[],
#                     masked_image=""
#             )

#             # Generate masked image
#             masked_image = mask_predictions(image, results["boxes"])
            
#             # Convert to base64
#             buffered = io.BytesIO()
#             masked_image.save(buffered, format="PNG")
#             masked_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

#             # Prepare response
#             return ImageMaskingResponse(
#                 boxes=results["boxes"].tolist(),
#                 labels=[
#                     img_model.config.id2label.get(label.item(), f"Class {label.item()}")
#                     for label in results["labels"]
#                 ],
#                 scores=results["scores"].tolist(),
#                 masked_image=f"data:image/png;base64,{masked_image_base64}"
#             )
            
#         except Exception as e:
#             print(f"Model inference error: {str(e)}")
#             raise HTTPException(
#                 status_code=500,
#                 detail="Error processing image with model"
#             )

#     except HTTPException as http_exc:
#         print(f"HTTP Exception: {http_exc.detail}")
#         raise
#     except Exception as e:
#         print(f"Unexpected error in mask_image: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="An unexpected error occurred while processing the image"
#         )


# @app.post("/summarize")
# async def summarize(file: UploadFile = File(...)):
#     try:
#         filename = file.filename.lower()

#         # ✅ Check allowed extensions
#         allowed_extensions = (".pdf", ".docx", ".txt", ".jpeg", ".jpg", ".png")
#         if not filename.endswith(allowed_extensions):
#             raise HTTPException(status_code=400, detail="Only PDF, DOCX, TXT, JPG, JPEG, PNG files are allowed.")

#         # ✅ Read file contents
#         content = await file.read()

#         # ✅ Extract text
#         text = extract_text_from_file(file, content)
#         if not text.strip():
#             return {"error": "No readable text found in the file."}

#         # ✅ Tokenize input and generate summary using Pegasus
#         encoded = pegasus_tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding="longest"
#         ).to(device)

#         with torch.no_grad():
#             summary_ids = pegasus_model.generate(
#                 input_ids=encoded["input_ids"],
#                 attention_mask=encoded["attention_mask"],
#                 max_length=256,
#                 num_beams=5,
#                 early_stopping=True
#             )

#         summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         return {"summary": summary}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
