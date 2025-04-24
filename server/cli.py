import argparse
from PIL import Image
import os

from server.ormbg.ormbg_processor import ORMBGProcessor
from transparent_background import Remover
from rembg import remove as rembg_remove, new_session
from transformers import pipeline

from carvekit.ml.wrap.u2net import U2NET
from carvekit.ml.wrap.basnet import BASNET
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.deeplab_v3 import DeepLabV3
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.api.interface import Interface
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator

import torch

# Inicializar modelos
bria_model = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device="cpu")
inspyrenet_model = Remover()
rembg_models = {
    'u2net': new_session('u2net'),
    'u2net_human_seg': new_session('u2net_human_seg'),
    'isnet-general-use': new_session('isnet-general-use'),
    'isnet-anime': new_session('isnet-anime')
}

ormbg_model_path = os.path.expanduser("~/.ormbg/ormbg.pth")
ormbg_processor = ORMBGProcessor(ormbg_model_path)

def process_with_bria(image):
    result = bria_model(image, return_mask=True)
    if not isinstance(result, Image.Image):
        result = Image.fromarray((result * 255).astype('uint8'))
    out = Image.new("RGBA", image.size, (0, 0, 0, 0))
    out.paste(image, mask=result)
    return out

def process_with_ormbg(image):
    return ormbg_processor.process_image(image)

def process_with_inspyrenet(image):
    return inspyrenet_model.process(image, type='rgba')

def process_with_rembg(image, model='u2net'):
    return rembg_remove(image, session=rembg_models[model])

def process_with_carvekit(image, model='u2net'):
    if model == 'u2net':
        seg_net = U2NET(device='cpu', batch_size=1)
    elif model == 'tracer':
        seg_net = TracerUniversalB7(device='cpu', batch_size=1)
    elif model == 'basnet':
        seg_net = BASNET(device='cpu', batch_size=1)
    elif model == 'deeplab':
        seg_net = DeepLabV3(device='cpu', batch_size=1)
    else:
        raise ValueError("Unsupported model type")

    fba = FBAMatting(device='cpu', input_tensor_size=2048, batch_size=1)
    trimap = TrimapGenerator()
    preprocessing = PreprocessingStub()
    postprocessing = MattingMethod(matting_module=fba, trimap_generator=trimap, device='cpu')

    interface = Interface(pre_pipe=preprocessing, post_pipe=postprocessing, seg_pipe=seg_net)
    return interface([image])[0]

def main():
    parser = argparse.ArgumentParser(description="Remove background from an image.")
    parser.add_argument("--input", required=True, help="Path to the input image")
    parser.add_argument("--output", required=True, help="Path to save the output image")
    parser.add_argument("--method", default="u2net", choices=[
        'u2net', 'u2net_human_seg', 'isnet-general-use', 'isnet-anime',
        'ormbg', 'bria', 'inspyrenet', 'basnet', 'deeplab', 'tracer'
    ], help="Background removal method to use")

    args = parser.parse_args()

    image = Image.open(args.input).convert("RGB")

    if args.method == "bria":
        result = process_with_bria(image)
    elif args.method == "ormbg":
        result = process_with_ormbg(image)
    elif args.method == "inspyrenet":
        result = process_with_inspyrenet(image)
    elif args.method in rembg_models:
        result = process_with_rembg(image, model=args.method)
    elif args.method in ["u2net", "tracer", "basnet", "deeplab"]:
        result = process_with_carvekit(image, model=args.method)
    else:
        raise ValueError("Método no soportado.")

    result.save(args.output)
    print(f"✅ Imagen procesada y guardada en: {args.output}")

if __name__ == "__main__":
    main()
