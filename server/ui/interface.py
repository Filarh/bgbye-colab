import gradio as gr
import numpy as np
from PIL import Image
import time
import torch

from server.ui.models import ModelManager, AVAILABLE_METHODS, METHOD_DESCRIPTIONS, ORMBG_AVAILABLE
from server.ui.batch_processor import BatchProcessor
from server.ui.theme import Theme

class GradioInterface:
    """Clase para gestionar la interfaz Gradio para eliminación de fondo"""
    
    def __init__(self):
        """Inicializa la interfaz Gradio"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_manager = ModelManager(device=device)
        self.batch_processor = BatchProcessor(self.model_manager)
        
    def process_image(self, input_image, method, output_format):
        """Procesa una imagen y elimina el fondo"""
        if input_image is None:
            return None, "No se proporcionó ninguna imagen"
        
        start_time = time.time()
        
        try:
            # Convertir a imagen PIL si es una ruta
            if isinstance(input_image, str):
                image = Image.open(input_image).convert('RGB')
            else:
                image = Image.fromarray(input_image).convert('RGB')
            
            # Procesar imagen con el método seleccionado
            result = self.model_manager.process_image(image, method)
            
            process_time = time.time() - start_time
            
            # Convertir imagen PIL a numpy para Gradio
            output_img = np.array(result)
            
            message = f"✅ Procesado en {process_time:.2f} segundos usando {method}"
            return output_img, message
        
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, f"❌ Error: {str(e)}"
    
    def create_interface(self):
        """Crea y configura la interfaz Gradio"""
        with gr.Blocks(
            title="BGBye - Herramienta de Eliminación de Fondo",
            theme=Theme.get_default_theme(),
            css=Theme.get_custom_css()
        ) as app:
            gr.Markdown("""
            # 🎭 BGBye - Herramienta Avanzada de Eliminación de Fondo
            
            Selecciona un método, sube una imagen y ¡elimina el fondo!
            """)
            
            with gr.Tabs():
                with gr.Tab("🖼️ Imagen Individual"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            input_image = gr.Image(label="Imagen de Entrada", type="numpy")
                            with gr.Row():
                                method = gr.Dropdown(
                                    choices=AVAILABLE_METHODS,
                                    value="u2net",
                                    label="Método de Eliminación",
                                    interactive=True
                                )
                                output_format = gr.Radio(
                                    choices=["png", "webp"],
                                    value="png",
                                    label="Formato de Salida",
                                    interactive=True
                                )
                            process_btn = gr.Button("Eliminar Fondo", variant="primary")
                        
                        with gr.Column(scale=1):
                            output_image = gr.Image(label="Imagen Procesada", type="numpy")
                            status = gr.Textbox(label="Estado")
                
                with gr.Tab("📚 Procesamiento por Lotes"):
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                input_dir = gr.Textbox(
                                    label="Directorio de Entrada",
                                    placeholder="/ruta/a/carpeta/entrada"
                                )
                                output_dir = gr.Textbox(
                                    label="Directorio de Salida",
                                    placeholder="/ruta/a/carpeta/salida"
                                )
                                with gr.Row():
                                    batch_method = gr.Dropdown(
                                        choices=AVAILABLE_METHODS,
                                        value="u2net",
                                        label="Método de Eliminación",
                                        interactive=True
                                    )
                                    batch_output_format = gr.Radio(
                                        choices=["png", "webp"],
                                        value="png",
                                        label="Formato de Salida",
                                        interactive=True
                                    )
                                batch_process_btn = gr.Button("Procesar Lote", variant="primary")
                        
                        with gr.Column():
                            batch_status = gr.Textbox(label="Estado del Procesamiento por Lotes", lines=5)
            
            with gr.Accordion("ℹ️ Información de Métodos", open=False):
                method_info_md = "| Método | Descripción |\n| --- | --- |\n"
                
                for method_name in AVAILABLE_METHODS:
                    if method_name in METHOD_DESCRIPTIONS:
                        desc = METHOD_DESCRIPTIONS[method_name]
                        if method_name == "ormbg" and ORMBG_AVAILABLE:
                            desc += " ⭐ *Recomendado para mejor calidad*"
                        if "human" in method_name:
                            desc += " 👤 *Especializado en figuras humanas*"
                        method_info_md += f"| **{method_name.upper()}** | {desc} |\n"
                
                gr.Markdown(method_info_md)
            
            gr.Markdown("### Desarrollado con ❤️ usando Gradio | BGBye - Eliminación de Fondos")
                        
            process_btn.click(
                fn=self.process_image,
                inputs=[input_image, method, output_format],
                outputs=[output_image, status]
            )
            
            batch_process_btn.click(
                fn=self.batch_processor.process_directory,
                inputs=[input_dir, output_dir, batch_method, batch_output_format],
                outputs=[batch_status]
            )
            
            return app 