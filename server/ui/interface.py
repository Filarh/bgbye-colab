import gradio as gr
import numpy as np
from PIL import Image
import time
import torch
import threading

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
        self.processing_lock = threading.Lock()
        
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
    
    def process_image_all_models(self, input_image, output_format):
        """Procesa una imagen con todos los modelos disponibles y actualiza la interfaz en tiempo real"""
        if input_image is None:
            return None, [], [], "No se proporcionó ninguna imagen", 0, "Inicializando...", gr.update(visible=True)
            
        # Inicializar estado
        gallery_images = []
        gallery_labels = []
        image_info = []  # Lista de info para cada imagen
        status_message = "Iniciando procesamiento con todos los modelos...\n"
        progress = 0
        processed_count = 0
        
        try:
            # Convertir a imagen PIL si es una ruta
            if isinstance(input_image, str):
                image = Image.open(input_image).convert('RGB')
            else:
                image = Image.fromarray(input_image).convert('RGB')
            
            # Obtener lista de métodos (excluyendo 'all')
            methods = [m for m in AVAILABLE_METHODS if m != 'all']
            total_methods = len(methods)
            
            # Inicializar resultados
            results_dict = {}
            
            # Función para procesar cada imagen y actualizar la interfaz
            def process_one_model(method_idx):
                nonlocal processed_count, progress, status_message, gallery_images, gallery_labels, image_info, results_dict
                
                method = methods[method_idx]
                try:
                    with self.processing_lock:
                        start_time = time.time()
                        
                        # Actualizar mensaje de estado
                        status_message += f"Procesando con {method}...\n"
                        current_status = f"Procesando {method}... ({processed_count+1}/{total_methods})"
                        
                        # Procesar imagen con el método actual
                        if method == "bria":
                            result = self.model_manager.process_with_bria(image)
                        elif method == "ormbg" and ORMBG_AVAILABLE:
                            result = self.model_manager.process_with_ormbg(image)
                        elif method == "inspyrenet":
                            result = self.model_manager.process_with_inspyrenet(image)
                        elif method in self.model_manager.rembg_models:
                            result = self.model_manager.process_with_rembg(image, model=method)
                        elif method in self.model_manager.carvekit_models:
                            result = self.model_manager.process_with_carvekit(image, model=method)
                        else:
                            raise ValueError(f"Método no soportado: {method}")
                        
                        process_time = time.time() - start_time
                        
                        # Actualizar resultados
                        result_np = np.array(result)
                        results_dict[method] = result
                        
                        # Crear información detallada para esta imagen
                        info = {
                            "method": method,
                            "time": process_time,
                            "desc": METHOD_DESCRIPTIONS.get(method, ""),
                            "size": f"{result.width}x{result.height}",
                            "output_format": output_format
                        }
                        
                        # Actualizar listas para la interfaz
                        gallery_images.append(result_np)
                        gallery_labels.append(method)
                        image_info.append(info)
                        
                        # Actualizar mensaje de estado
                        status_message += f"✅ {method}: Procesado en {process_time:.2f} segundos\n"
                        
                        # Actualizar progreso
                        processed_count += 1
                        progress = (processed_count / total_methods) * 100
                        
                except Exception as e:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    status_message += f"❌ {method}: Error: {str(e)}\n"
                    processed_count += 1
                    progress = (processed_count / total_methods) * 100
            
            # Procesar el primer modelo inmediatamente para mostrar resultados rápidos
            process_one_model(0)
            
            # Usar primer resultado como principal
            main_output = gallery_images[0] if gallery_images else None
            main_info = f"Método: {image_info[0]['method']} | Tiempo: {image_info[0]['time']:.2f}s" if image_info else ""
            
            # Devolver resultados del primer modelo y estado actual
            return (
                main_output,  # Imagen principal
                gallery_images,  # Galería parcial
                image_info,  # Info para cada imagen
                status_message,  # Mensaje actual
                progress,  # Progreso actual
                main_info,  # Info principal
                gr.update(visible=True)  # Mostrar spinner
            )
            
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return (
                None, [], [], f"❌ Error al iniciar el procesamiento: {str(e)}", 
                0, "Error", gr.update(visible=False)
            )
    
    def process_image_all_models_continue(self, state):
        """Continúa el procesamiento del resto de modelos después de mostrar el primero"""
        if not state or not isinstance(state, dict):
            return None, [], [], "Estado inválido", 0, "Error", gr.update(visible=False)
        
        # Extraer estado
        image = state.get("image")
        methods = state.get("methods", [])
        gallery_images = state.get("gallery_images", [])
        gallery_labels = state.get("gallery_labels", [])
        image_info = state.get("image_info", [])
        status_message = state.get("status_message", "")
        progress = state.get("progress", 0)
        processed_count = state.get("processed_count", 1)  # Ya procesamos el primero
        output_format = state.get("output_format", "png")
        
        if not image or not methods:
            return None, [], [], "Datos insuficientes", 0, "Error", gr.update(visible=False)
        
        total_methods = len(methods)
        results_dict = {}
        
        try:
            # Procesar los modelos restantes (desde el segundo)
            for i in range(1, len(methods)):
                method = methods[i]
                try:
                    start_time = time.time()
                    
                    # Actualizar mensaje de estado
                    status_message += f"Procesando con {method}...\n"
                    
                    # Procesar imagen con el método actual
                    if method == "bria":
                        result = self.model_manager.process_with_bria(image)
                    elif method == "ormbg" and ORMBG_AVAILABLE:
                        result = self.model_manager.process_with_ormbg(image)
                    elif method == "inspyrenet":
                        result = self.model_manager.process_with_inspyrenet(image)
                    elif method in self.model_manager.rembg_models:
                        result = self.model_manager.process_with_rembg(image, model=method)
                    elif method in self.model_manager.carvekit_models:
                        result = self.model_manager.process_with_carvekit(image, model=method)
                    else:
                        continue  # Saltar métodos no soportados
                    
                    process_time = time.time() - start_time
                    
                    # Actualizar resultados
                    result_np = np.array(result)
                    results_dict[method] = result
                    
                    # Crear información detallada para esta imagen
                    info = {
                        "method": method,
                        "time": process_time,
                        "desc": METHOD_DESCRIPTIONS.get(method, ""),
                        "size": f"{result.width}x{result.height}",
                        "output_format": output_format
                    }
                    
                    # Actualizar listas para la interfaz
                    gallery_images.append(result_np)
                    gallery_labels.append(method)
                    image_info.append(info)
                    
                    # Generar lista de imágenes con etiquetas para la galería
                    gallery_data = [(gallery_images[i], methods[i]) for i in range(len(gallery_images))]
                    
                    # Actualizar mensaje y progreso
                    status_message += f"✅ {method}: Procesado en {process_time:.2f} segundos\n"
                    processed_count += 1
                    progress = (processed_count / total_methods) * 100
                    
                    # Actualizar interfaz en tiempo real
                    yield (
                        gallery_images[0],  # Primera imagen como principal
                        gallery_data,       # Galería actualizada con etiquetas
                        status_message,     # Estado actualizado
                        progress,           # Progreso actualizado
                        f"Método: {image_info[0]['method']} | Tiempo: {image_info[0]['time']:.2f}s",  # Info imagen
                        gr.update(visible=True),  # Spinner visible
                        image_info          # Info actualizada
                    )
                    
                except Exception as e:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    status_message += f"❌ {method}: Error: {str(e)}\n"
                    processed_count += 1
                    progress = (processed_count / total_methods) * 100
            
            # Procesamiento completo
            status_message += f"\n✅ Procesamiento completo. {processed_count} de {total_methods} modelos procesados."
            
            # Resultado final
            return (
                gallery_images[0] if gallery_images else None,  # La primera imagen como principal
                gallery_images,     # Galería final
                image_info,         # Información final
                status_message,     # Estado final
                100,                # Progreso 100%
                f"Método: {image_info[0]['method']} | Tiempo: {image_info[0]['time']:.2f}s" if image_info else "",  # Info final
                gr.update(visible=False)  # Ocultar spinner
            )
            
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return (
                gallery_images[0] if gallery_images else None,
                gallery_images,
                image_info,
                status_message + f"\n❌ Error durante el procesamiento: {str(e)}",
                progress,
                "Error durante el procesamiento",
                gr.update(visible=False)
            )
    
    def show_image_details(self, evt, image_info_list):
        """Muestra detalles de la imagen seleccionada"""
        if evt is None or not image_info_list:
            return None, ""
        
        try:
            # En la versión actualizada de Gradio, evt es el índice directamente
            idx = evt if isinstance(evt, int) else 0
            
            if idx < len(image_info_list):
                info = image_info_list[idx]
                details = (
                    f"**Método:** {info['method']}\n"
                    f"**Descripción:** {info['desc']}\n"
                    f"**Tiempo de procesamiento:** {info['time']:.2f} segundos\n"
                    f"**Dimensiones:** {info['size']}\n"
                    f"**Formato de salida:** {info['output_format']}"
                )
                # Devolvemos la imagen correspondiente de la galería
                return None, details
        except Exception as e:
            return None, f"Error al mostrar detalles: {str(e)}"
        
        return None, ""
    
    def create_interface(self):
        """Crea y configura la interfaz Gradio"""
        with gr.Blocks(
            title="BGBye - Herramienta de Eliminación de Fondo",
            theme=Theme.get_default_theme(),
            css=Theme.get_custom_css()
        ) as app:
            # Estado para procesamiento asíncrono
            state = gr.State({})
            image_info_state = gr.State([])
            
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
                            # Componentes condicionales que se mostrarán según el método seleccionado
                            with gr.Group(visible=True) as single_output_group:
                                output_image = gr.Image(label="Imagen Procesada", type="numpy")
                                status = gr.Textbox(label="Estado")
                            
                            with gr.Group(visible=False) as all_models_output_group:
                                with gr.Row():
                                    output_image_all = gr.Image(label="Resultado", type="numpy", elem_classes="main-image")
                                    with gr.Column():
                                        image_details = gr.Markdown(label="Detalles de la imagen", value="")
                                        progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="Progreso", interactive=False)
                                        processing_spinner = gr.HTML("""
                                            <div class="processing-spinner">
                                                <div class="spinner"></div>
                                                <p>Procesando...</p>
                                            </div>
                                        """, visible=False)
                                
                                gallery = gr.Gallery(
                                    label="Todos los resultados (haz clic para ver en detalle)",
                                    columns=4,
                                    rows=3,
                                    height=320,
                                    show_label=True,
                                    object_fit="contain",
                                    elem_classes="gallery-container"
                                )
                                status_all = gr.Textbox(label="Estado de Procesamiento", lines=8)
                
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
                        if method_name == "all":
                            desc += " 🌟 *Procesa con todos los modelos a la vez*"
                        method_info_md += f"| **{method_name.upper()}** | {desc} |\n"
                
                gr.Markdown(method_info_md)
            
            gr.Markdown("### Desarrollado con ❤️ usando Gradio | BGBye - Eliminación de Fondos")
            
            # Función para alternar la visibilidad según el método seleccionado
            def toggle_output_groups(method_choice):
                if method_choice == "all":
                    return {
                        single_output_group: gr.update(visible=False),
                        all_models_output_group: gr.update(visible=True)
                    }
                else:
                    return {
                        single_output_group: gr.update(visible=True),
                        all_models_output_group: gr.update(visible=False)
                    }
            
            # Conectar la función para cambiar la visibilidad cuando se cambia el método
            method.change(
                fn=toggle_output_groups,
                inputs=method,
                outputs=[single_output_group, all_models_output_group]
            )
            
            # Función para preparar el procesamiento con todos los modelos
            def prepare_process_all(input_image, method, output_format):
                if method != "all":
                    # Procesar con un solo modelo directamente
                    output_img, status_msg = self.process_image(input_image, method, output_format)
                    return output_img, [], [], status_msg, 0, "", gr.update(visible=False), {}, [], False
                
                # Preparar procesamiento con todos los modelos
                try:
                    # Convertir imagen
                    if isinstance(input_image, str):
                        pil_image = Image.open(input_image).convert('RGB')
                    else:
                        pil_image = Image.fromarray(input_image).convert('RGB')
                    
                    # Obtener métodos (excluyendo 'all')
                    methods = [m for m in AVAILABLE_METHODS if m != 'all']
                    
                    # Crear estado inicial
                    state_data = {
                        "image": pil_image,
                        "methods": methods,
                        "gallery_images": [],
                        "gallery_labels": [],
                        "image_info": [],
                        "status_message": "Iniciando procesamiento con todos los modelos...\n",
                        "progress": 0,
                        "processed_count": 0,
                        "output_format": output_format
                    }
                    
                    # Indicar que debemos continuar con el procesamiento asíncrono
                    return None, [], [], "Preparando...", 0, "", gr.update(visible=True), state_data, [], True
                    
                except Exception as e:
                    return None, [], [], f"Error: {str(e)}", 0, "", gr.update(visible=False), {}, [], False
            
            # Función para iniciar procesamiento con todos los modelos
            def start_process_all(state_data, continue_process):
                if not continue_process or not state_data:
                    return None, [], [], "No se puede iniciar", 0, "", gr.update(visible=False), []
                
                # Iniciar procesamiento con el primer modelo
                try:
                    # Extraer datos
                    image = state_data["image"]
                    methods = state_data["methods"]
                    output_format = state_data["output_format"]
                    
                    if not methods:
                        return None, [], [], "No hay métodos disponibles", 0, "", gr.update(visible=False), []
                    
                    # Procesar primer modelo
                    method = methods[0]
                    start_time = time.time()
                    
                    # Procesar imagen con el método actual
                    if method == "bria":
                        result = self.model_manager.process_with_bria(image)
                    elif method == "ormbg" and ORMBG_AVAILABLE:
                        result = self.model_manager.process_with_ormbg(image)
                    elif method == "inspyrenet":
                        result = self.model_manager.process_with_inspyrenet(image)
                    elif method in self.model_manager.rembg_models:
                        result = self.model_manager.process_with_rembg(image, model=method)
                    elif method in self.model_manager.carvekit_models:
                        result = self.model_manager.process_with_carvekit(image, model=method)
                    else:
                        result = None
                    
                    if result is None:
                        return None, [], [], "Error al procesar el primer modelo", 0, "", gr.update(visible=False), []
                    
                    process_time = time.time() - start_time
                    
                    # Actualizar resultados
                    result_np = np.array(result)
                    
                    # Crear información detallada para esta imagen
                    info = {
                        "method": method,
                        "time": process_time,
                        "desc": METHOD_DESCRIPTIONS.get(method, ""),
                        "size": f"{result.width}x{result.height}",
                        "output_format": output_format
                    }
                    
                    # Actualizar listas para la interfaz
                    gallery_images = [result_np]
                    image_info = [info]
                    
                    # Actualizar mensaje de estado
                    status_message = f"✅ {method}: Procesado en {process_time:.2f} segundos\n"
                    
                    # Actualizar progreso
                    progress = (1 / len(methods)) * 100
                    
                    # Actualizar estado para el procesamiento continuo
                    state_data.update({
                        "gallery_images": gallery_images,
                        "image_info": image_info,
                        "status_message": status_message,
                        "progress": progress,
                        "processed_count": 1,
                    })
                    
                    # Generar lista de imágenes con etiquetas para la galería
                    gallery_data = [(result_np, method)]
                    
                    # Devolver primer resultado y continuar en segundo plano
                    return (
                        result_np,              # Imagen principal
                        gallery_data,         # Galería inicial 
                        status_message,         # Estado inicial
                        progress,               # Progreso inicial
                        f"Método: {method} | Tiempo: {process_time:.2f}s",  # Info imagen
                        gr.update(visible=True),  # Spinner
                        image_info               # Info para todas las imágenes
                    )
                    
                except Exception as e:
                    return None, [], [], f"Error al iniciar: {str(e)}", 0, "", gr.update(visible=False), []
            
            # Función para procesar el resto de modelos
            def process_remaining_models(state_data):
                if not state_data:
                    return None, [], [], "Estado no válido", 0, "", gr.update(visible=False), []
                
                # Extraer estado actual
                image = state_data["image"]
                methods = state_data["methods"]
                gallery_images = state_data["gallery_images"]
                image_info = state_data["image_info"]
                status_message = state_data["status_message"]
                progress = state_data["progress"]
                processed_count = state_data["processed_count"]
                output_format = state_data["output_format"]
                
                # Procesar los modelos restantes
                total_methods = len(methods)
                
                try:
                    for i in range(processed_count, total_methods):
                        try:
                            method = methods[i]
                            start_time = time.time()
                            
                            # Mostrar mensaje de procesamiento
                            status_message += f"Procesando con {method}...\n"
                            
                            # Procesar imagen
                            if method == "bria":
                                result = self.model_manager.process_with_bria(image)
                            elif method == "ormbg" and ORMBG_AVAILABLE:
                                result = self.model_manager.process_with_ormbg(image)
                            elif method == "inspyrenet":
                                result = self.model_manager.process_with_inspyrenet(image)
                            elif method in self.model_manager.rembg_models:
                                result = self.model_manager.process_with_rembg(image, model=method)
                            elif method in self.model_manager.carvekit_models:
                                result = self.model_manager.process_with_carvekit(image, model=method)
                            else:
                                continue
                            
                            process_time = time.time() - start_time
                            
                            # Actualizar resultados
                            result_np = np.array(result)
                            
                            # Crear información detallada
                            info = {
                                "method": method,
                                "time": process_time,
                                "desc": METHOD_DESCRIPTIONS.get(method, ""),
                                "size": f"{result.width}x{result.height}",
                                "output_format": output_format
                            }
                            
                            # Actualizar listas
                            gallery_images.append(result_np)
                            image_info.append(info)
                            
                            # Generar lista de imágenes con etiquetas para la galería
                            gallery_data = [(gallery_images[i], methods[i]) for i in range(len(gallery_images))]
                            
                            # Actualizar mensaje y progreso
                            status_message += f"✅ {method}: Procesado en {process_time:.2f} segundos\n"
                            processed_count += 1
                            progress = (processed_count / total_methods) * 100
                            
                            # Actualizar interfaz en tiempo real
                            yield (
                                gallery_images[0],  # Primera imagen como principal
                                gallery_data,       # Galería actualizada con etiquetas
                                status_message,     # Estado actualizado
                                progress,           # Progreso actualizado
                                f"Método: {image_info[0]['method']} | Tiempo: {image_info[0]['time']:.2f}s",  # Info imagen
                                gr.update(visible=True),  # Spinner visible
                                image_info          # Info actualizada
                            )
                            
                        except Exception as e:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            status_message += f"❌ {method}: Error: {str(e)}\n"
                            processed_count += 1
                            progress = (processed_count / total_methods) * 100
                    
                    # Finalizar
                    status_message += f"\n✅ Procesamiento completo. {processed_count} de {total_methods} modelos procesados."
                    
                    # Generar lista final de imágenes con etiquetas
                    gallery_data = [(gallery_images[i], methods[i]) for i in range(len(gallery_images))]
                    
                    # Resultado final - IMPORTANTE: Ocultar el spinner
                    yield (
                        gallery_images[0] if gallery_images else None,
                        gallery_data,
                        status_message,
                        100,
                        f"Método: {image_info[0]['method']} | Tiempo: {image_info[0]['time']:.2f}s" if image_info else "",
                        gr.update(visible=False),  # Ocultar spinner al finalizar
                        image_info
                    )
                    
                except Exception as e:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # En caso de error, asegurarnos de ocultar el spinner también
                    gallery_data = [(gallery_images[i], methods[i]) for i in range(len(gallery_images))] if gallery_images else []
                    yield (
                        gallery_images[0] if gallery_images else None,
                        gallery_data,
                        status_message + f"\n❌ Error durante el procesamiento: {str(e)}",
                        progress,
                        "Error durante el procesamiento",
                        gr.update(visible=False),  # Asegurarse de ocultar el spinner
                        image_info
                    )
            
            # Función para procesar según el método
            def process_based_on_method(input_image, method, output_format):
                if method == "all":
                    # Procesar con todos los modelos
                    main_img, gallery_imgs, info_list, status_msg, progress, img_info, spinner_update = self.process_image_all_models(input_image, output_format)
                    return main_img, gallery_imgs, status_msg, None, progress, img_info, spinner_update, info_list
                else:
                    # Procesar con un solo modelo
                    output_img, status_msg = self.process_image(input_image, method, output_format)
                    return output_img, [], None, status_msg, 0, "", gr.update(visible=False), []
            
            # Conectar botón de procesamiento
            process_btn.click(
                fn=prepare_process_all,
                inputs=[input_image, method, output_format],
                outputs=[
                    output_image, gallery, image_info_state, status_all, 
                    progress_bar, image_details, processing_spinner, 
                    state, output_image_all, gr.Checkbox(visible=False)
                ],
            ).then(
                fn=start_process_all,
                inputs=[state, gr.Checkbox(visible=False)],
                outputs=[
                    output_image_all, gallery, status_all, 
                    progress_bar, image_details, processing_spinner,
                    image_info_state
                ]
            ).then(
                fn=process_remaining_models,
                inputs=[state],
                outputs=[
                    output_image_all, gallery, status_all, 
                    progress_bar, image_details, processing_spinner,
                    image_info_state
                ]
            )
            
            # Función simple para mostrar detalles e imagen seleccionada
            def show_selected_image(evt, gallery_images, image_info_list):
                try:
                    # El índice del evento corresponde a la imagen seleccionada
                    selected_image = gallery_images[evt]
                    
                    # Generar información detallada
                    details = ""
                    if evt < len(image_info_list):
                        info = image_info_list[evt]
                        details = (
                            f"**Método:** {info['method']}\n"
                            f"**Descripción:** {info['desc']}\n"
                            f"**Tiempo de procesamiento:** {info['time']:.2f} segundos\n"
                            f"**Dimensiones:** {info['size']}\n"
                            f"**Formato de salida:** {info['output_format']}"
                        )
                    return selected_image, details
                except Exception as e:
                    return None, f"Error al mostrar imagen: {str(e)}"
            
            # Evento para mostrar detalles al hacer clic en una miniatura
            gallery.select(
                fn=show_selected_image,
                inputs=[gallery, gallery, image_info_state],
                outputs=[output_image_all, image_details]
            )
            
            # Conectar el botón de procesamiento por lotes
            batch_process_btn.click(
                fn=self.batch_processor.process_directory,
                inputs=[input_dir, output_dir, batch_method, batch_output_format],
                outputs=[batch_status]
            )
            
            return app 