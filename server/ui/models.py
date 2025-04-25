import torch
import os
from PIL import Image
from transformers import pipeline
from transparent_background import Remover
from rembg import remove as rembg_remove, new_session

# Importar modelos Carvekit
from carvekit.ml.wrap.u2net import U2NET
from carvekit.ml.wrap.basnet import BASNET
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.deeplab_v3 import DeepLabV3
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.api.interface import Interface
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator

# Verificar disponibilidad de ORMBG
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
    print("Modelo ORMBG no disponible. Este método estará deshabilitado.")

class ModelManager:
    """Clase para gestionar la inicialización y uso de los modelos de eliminación de fondo"""
    
    def __init__(self, device='cpu'):
        """Inicializa todos los modelos de eliminación de fondo"""
        self.device = device
        print("Inicializando modelos...")
        
        # Modelo BRIA
        self.bria_model = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=device)
        
        # Modelo InspyreNet
        self.inspyrenet_model = Remover()
        self.inspyrenet_model.model.to(device)
        
        # Modelos Rembg
        self.rembg_models = {
            'u2net': new_session('u2net'),
            'u2net_human_seg': new_session('u2net_human_seg'),
            'isnet-general-use': new_session('isnet-general-use'),
            'isnet-anime': new_session('isnet-anime')
        }
        
        # Inicializar modelos Carvekit
        self.carvekit_models = {
            'u2net': self._initialize_carvekit_model(U2NET, device),
            'tracer': self._initialize_carvekit_model(TracerUniversalB7, device),
            'basnet': self._initialize_carvekit_model(BASNET, device),
            'deeplab': self._initialize_carvekit_model(DeepLabV3, device)
        }
        
        print("¡Todos los modelos cargados!")
    
    def _initialize_carvekit_model(self, seg_pipe_class, device):
        """Inicializa un modelo Carvekit"""
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
    
    def process_with_bria(self, image):
        """Procesa una imagen con el modelo BRIA"""
        result = self.bria_model(image, return_mask=True)
        if not isinstance(result, Image.Image):
            result = Image.fromarray((result * 255).astype('uint8'))
        no_bg_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        no_bg_image.paste(image, mask=result)
        return no_bg_image
    
    def process_with_ormbg(self, image):
        """Procesa una imagen con el modelo ORMBG"""
        if not ORMBG_AVAILABLE:
            raise ValueError("Modelo ORMBG no disponible")
        return ormbg_processor.process_image(image)
    
    def process_with_inspyrenet(self, image):
        """Procesa una imagen con el modelo InspyreNet"""
        # Movemos al GPU si está disponible para procesamiento más rápido
        if torch.cuda.is_available():
            self.inspyrenet_model.model.to('cuda')
        result = self.inspyrenet_model.process(image, type='rgba')
        # Devolvemos a CPU para ahorrar memoria
        self.inspyrenet_model.model.to('cpu')
        return result
    
    def process_with_rembg(self, image, model='u2net'):
        """Procesa una imagen con Rembg"""
        return rembg_remove(image, session=self.rembg_models[model])
    
    def process_with_carvekit(self, image, model='u2net'):
        """Procesa una imagen con Carvekit"""
        if model in self.carvekit_models:
            interface = self.carvekit_models[model]
            return interface([image])[0]
        else:
            raise ValueError(f"Modelo Carvekit no soportado: {model}")
    
    def process_image(self, image, method):
        """Procesa una imagen con el método especificado"""
        try:
            if method == "bria":
                return self.process_with_bria(image)
            elif method == "ormbg" and ORMBG_AVAILABLE:
                return self.process_with_ormbg(image)
            elif method == "inspyrenet":
                return self.process_with_inspyrenet(image)
            elif method in self.rembg_models:
                return self.process_with_rembg(image, model=method)
            elif method in self.carvekit_models:
                return self.process_with_carvekit(image, model=method)
            else:
                raise ValueError(f"Método {method} no disponible")
        finally:
            # Limpiar caché GPU si se usó
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Lista de métodos disponibles
AVAILABLE_METHODS = [
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

# Agregar ORMBG si está disponible
if ORMBG_AVAILABLE:
    AVAILABLE_METHODS.append("ormbg")

# Descripciones de los métodos
METHOD_DESCRIPTIONS = {
    "u2net": "Eliminación de fondo de propósito general con U2NET",
    "u2net_human_seg": "Optimizado para sujetos humanos",
    "isnet-general-use": "Propósito general con buena detección de bordes",
    "isnet-anime": "Especializado para imágenes de anime y dibujos animados",
    "bria": "Modelo de Bria AI, bueno para fondos complejos",
    "inspyrenet": "Buena detección de bordes y preservación de detalles",
    "basnet": "Mejor para detalles finos",
    "deeplab": "Bueno para separación clara entre sujeto y fondo",
    "tracer": "Excelente en fondos complejos pero más lento"
}

if ORMBG_AVAILABLE:
    METHOD_DESCRIPTIONS["ormbg"] = "Eliminación con conocimiento de objetos y preservación detallada de bordes" 