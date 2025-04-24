import gradio as gr
import os
import glob
from PIL import Image
import torch
import time
import io
import tempfile
import shutil
from pathlib import Path

# Import background removal models
from transformers import pipeline
from transparent_background import Remover
from rembg import remove as rembg_remove, new_session

# Import Carvekit models
from carvekit.ml.wrap.u2net import U2NET
from carvekit.ml.wrap.basnet import BASNET
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.deeplab_v3 import DeepLabV3
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.api.interface import Interface
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator

# Check if ormbg is available, if not provide fallback
try:
    from ormbg import ORMBGProcessor
    ORMBG_AVAILABLE = True
    ormbg_model_path = os.path.expanduser("~/.ormbg/ormbg.pth")
    if os.path.exists(ormbg_model_path):
        ormbg_processor = ORMBGProcessor(ormbg_model_path)
        if torch.cuda.is_available():
            ormbg_processor.to("cuda")
        else:
            ormbg_processor.to("cpu")
    else:
        ORMBG_AVAILABLE = False
except ImportError:
    ORMBG_AVAILABLE = False
    print("ORMBG model not available. This method will be disabled.")

# Initialize models
print("Initializing models...")

# BRIA model
bria_model = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device="cpu")

# InspyreNet model
inspyrenet_model = Remover()
inspyrenet_model.model.cpu()

# Rembg models
rembg_models = {
    'u2net': new_session('u2net'),
    'u2net_human_seg': new_session('u2net_human_seg'),
    'isnet-general-use': new_session('isnet-general-use'),
    'isnet-anime': new_session('isnet-anime')
}

# Initialize Carvekit models on CPU
def initialize_carvekit_model(seg_pipe_class, device='cpu'):
    model = Interface(
        pre_pipe=PreprocessingStub(),
        post_pipe=MattingMethod(
            matting_module=FBAMatting(device=device, input_tensor_size=2048, batch_size=1),
            trimap_generator=TrimapGenerator(),
            device=device
        ),
        seg_pipe=seg_pipe_class(device=device, batch_size=1)
    )
    return model

print("Loading Carvekit models...")
carvekit_models = {
    'u2net': initialize_carvekit_model(U2NET),
    'tracer': initialize_carvekit_model(TracerUniversalB7),
    'basnet': initialize_carvekit_model(BASNET),
    'deeplab': initialize_carvekit_model(DeepLabV3)
}
print("All models loaded!")

# Processing functions
def process_with_bria(image):
    result = bria_model(image, return_mask=True)
    if not isinstance(result, Image.Image):
        result = Image.fromarray((result * 255).astype('uint8'))
    no_bg_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    no_bg_image.paste(image, mask=result)
    return no_bg_image

def process_with_ormbg(image):
    if not ORMBG_AVAILABLE:
        raise ValueError("ORMBG model is not available")
    return ormbg_processor.process_image(image)

def process_with_inspyrenet(image):
    # Move to GPU if available for faster processing
    if torch.cuda.is_available():
        inspyrenet_model.model.to('cuda')
    result = inspyrenet_model.process(image, type='rgba')
    # Move back to CPU to save memory
    inspyrenet_model.model.to('cpu')
    return result

def process_with_rembg(image, model='u2net'):
    return rembg_remove(image, session=rembg_models[model])

def process_with_carvekit(image, model='u2net'):
    # Use the pre-initialized models
    if model in carvekit_models:
        interface = carvekit_models[model]
        return interface([image])[0]
    else:
        raise ValueError(f"Unsupported Carvekit model: {model}")

# Convert PIL image to numpy array
def pil_to_numpy(pil_img):
    import numpy as np
    return np.array(pil_img)

# Main processing function
def remove_background(input_image, method, output_format):
    if input_image is None:
        return None, "No image provided"
    
    start_time = time.time()
    
    try:
        # Convert to PIL Image if it's a path
        if isinstance(input_image, str):
            image = Image.open(input_image).convert('RGB')
        else:
            image = Image.fromarray(input_image).convert('RGB')
        
        # Process image with selected method
        if method == "bria":
            result = process_with_bria(image)
        elif method == "ormbg" and ORMBG_AVAILABLE:
            result = process_with_ormbg(image)
        elif method == "inspyrenet":
            result = process_with_inspyrenet(image)
        elif method in rembg_models:
            result = process_with_rembg(image, model=method)
        elif method in carvekit_models:
            result = process_with_carvekit(image, model=method)
        else:
            return None, f"Method {method} not available"
        
        # If GPU was used, clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        process_time = time.time() - start_time
        
        # Convert PIL image to numpy for Gradio
        output_img = pil_to_numpy(result)
        
        message = f"✅ Processed in {process_time:.2f} seconds using {method}"
        return output_img, message
    
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, f"❌ Error: {str(e)}"

def process_batch(input_dir, output_dir, method, output_format):
    if not os.path.exists(input_dir):
        return f"Input directory does not exist: {input_dir}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                 glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                 glob.glob(os.path.join(input_dir, "*.png"))
    
    if not image_files:
        return "No image files found in the input directory"
    
    total_images = len(image_files)
    processed = 0
    failed = 0
    
    for img_path in image_files:
        try:
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.{output_format}")
            
            # Process the image
            image = Image.open(img_path).convert('RGB')
            
            # Process image with selected method
            if method == "bria":
                result = process_with_bria(image)
            elif method == "ormbg" and ORMBG_AVAILABLE:
                result = process_with_ormbg(image)
            elif method == "inspyrenet":
                result = process_with_inspyrenet(image)
            elif method in rembg_models:
                result = process_with_rembg(image, model=method)
            elif method in carvekit_models:
                result = process_with_carvekit(image, model=method)
            else:
                failed += 1
                continue
                
            # Set correct output format
            if output_format == "png":
                save_format = "PNG"
            elif output_format == "webp":
                save_format = "WEBP"
            else:
                save_format = "PNG"
                
            # Save the result
            result.save(output_path, format=save_format)
            processed += 1
                
        except Exception as e:
            failed += 1
            print(f"Error processing {img_path}: {str(e)}")
    
    return f"✅ Processed {processed} images, failed {failed} out of {total_images}"

# Define available methods
available_methods = [
    "u2net",
    "u2net_human_seg",
    "isnet-general-use", 
    "isnet-anime",
    "bria",
    "inspyrenet",
    "basnet",
    "deeplab",
    "tracer"
]

# Add ORMBG if available
if ORMBG_AVAILABLE:
    available_methods.append("ormbg")

# Create Gradio interface
with gr.Blocks(title="BG Removal Tool") as app:
    gr.Markdown("""
    # Background Removal Tool
    
    Select a method, upload an image, and remove the background!
    """)
    
    with gr.Tabs():
        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Input Image", type="numpy")
                    method = gr.Dropdown(
                        choices=available_methods,
                        value="u2net",
                        label="Background Removal Method"
                    )
                    output_format = gr.Radio(
                        choices=["png", "webp"],
                        value="png",
                        label="Output Format"
                    )
                    process_btn = gr.Button("Remove Background", variant="primary")
                
                with gr.Column():
                    # Fixed: Changed type="auto" to type="numpy"
                    output_image = gr.Image(label="Output Image", type="numpy")
                    status = gr.Textbox(label="Status")
        
        with gr.Tab("Batch Processing"):
            with gr.Row():
                with gr.Column():
                    input_dir = gr.Textbox(
                        label="Input Directory",
                        placeholder="/path/to/input/folder"
                    )
                    output_dir = gr.Textbox(
                        label="Output Directory",
                        placeholder="/path/to/output/folder"
                    )
                    batch_method = gr.Dropdown(
                        choices=available_methods,
                        value="u2net",
                        label="Background Removal Method"
                    )
                    batch_output_format = gr.Radio(
                        choices=["png", "webp"],
                        value="png",
                        label="Output Format"
                    )
                    batch_process_btn = gr.Button("Process Batch", variant="primary")
                
                with gr.Column():
                    batch_status = gr.Textbox(label="Batch Processing Status")
    
    # Define method descriptions
    method_descriptions = {
        "u2net": "General-purpose background removal with U2NET",
        "u2net_human_seg": "Optimized for human subjects",
        "isnet-general-use": "General-purpose with good edge detection",
        "isnet-anime": "Specialized for anime and cartoon images",
        "bria": "Bria AI's model, good for complex backgrounds",
        "inspyrenet": "Good edge detection and preservation of details",
        "basnet": "Better for fine-grained details",
        "deeplab": "Good for clear subject-background separation",
        "tracer": "Excellent on complex backgrounds but slower"
    }
    
    if ORMBG_AVAILABLE:
        method_descriptions["ormbg"] = "Object-aware removal with detailed edge preservation"
    
    with gr.Accordion("Method Information", open=False):
        method_info = gr.Markdown(
            "\n".join([f"- **{m}**: {method_descriptions.get(m, 'No description available')}" 
                      for m in available_methods])
        )
    
    # Set up event handlers
    process_btn.click(
        fn=remove_background,
        inputs=[input_image, method, output_format],
        outputs=[output_image, status]
    )
    
    batch_process_btn.click(
        fn=process_batch,
        inputs=[input_dir, output_dir, batch_method, batch_output_format],
        outputs=[batch_status]
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Background Removal Gradio App")
    parser.add_argument("--share", action="store_true", help="Create a public sharable link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    
    args = parser.parse_args()
    
    app.launch(share=args.share, server_port=args.port)