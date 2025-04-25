import os
import glob
from PIL import Image
import time

class BatchProcessor:
    """Clase para gestionar el procesamiento por lotes de imágenes"""
    
    def __init__(self, model_manager):
        """Inicializa el procesador por lotes con el gestor de modelos"""
        self.model_manager = model_manager
    
    def process_directory(self, input_dir, output_dir, method, output_format="png"):
        """Procesa todas las imágenes en un directorio usando el método especificado"""
        if not os.path.exists(input_dir):
            return f"El directorio de entrada no existe: {input_dir}"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Obtener todos los archivos de imagen en el directorio
        image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                     glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(input_dir, "*.png"))
        
        if not image_files:
            return "No se encontraron archivos de imagen en el directorio de entrada"
        
        total_images = len(image_files)
        processed = 0
        failed = 0
        
        start_time = time.time()
        
        # Modo especial: procesar con todos los modelos disponibles
        if method == "all":
            # Para cada imagen, crear un subdirectorio con todos los resultados
            for img_path in image_files:
                try:
                    filename = os.path.basename(img_path)
                    base_name = os.path.splitext(filename)[0]
                    
                    # Crear un subdirectorio para esta imagen
                    img_output_dir = os.path.join(output_dir, base_name)
                    if not os.path.exists(img_output_dir):
                        os.makedirs(img_output_dir)
                    
                    # Cargar la imagen
                    image = Image.open(img_path).convert('RGB')
                    
                    # Establecer formato correcto de salida
                    if output_format == "png":
                        save_format = "PNG"
                    elif output_format == "webp":
                        save_format = "WEBP"
                    else:
                        save_format = "PNG"
                    
                    # Procesar con todos los modelos disponibles
                    results = self.model_manager.process_all_models(image)
                    
                    # Guardar cada resultado
                    for model_name, result in results.items():
                        output_path = os.path.join(img_output_dir, f"{base_name}_{model_name}.{output_format}")
                        result.save(output_path, format=save_format)
                    
                    processed += 1
                        
                except Exception as e:
                    failed += 1
                    print(f"Error al procesar {img_path} con todos los modelos: {str(e)}")
        else:
            # Procesar con un solo modelo
            for img_path in image_files:
                try:
                    filename = os.path.basename(img_path)
                    base_name = os.path.splitext(filename)[0]
                    output_path = os.path.join(output_dir, f"{base_name}.{output_format}")
                    
                    # Procesar la imagen
                    image = Image.open(img_path).convert('RGB')
                    result = self.model_manager.process_image(image, method)
                    
                    # Establecer formato correcto de salida
                    if output_format == "png":
                        save_format = "PNG"
                    elif output_format == "webp":
                        save_format = "WEBP"
                    else:
                        save_format = "PNG"
                        
                    # Guardar el resultado
                    result.save(output_path, format=save_format)
                    processed += 1
                        
                except Exception as e:
                    failed += 1
                    print(f"Error al procesar {img_path}: {str(e)}")
        
        total_time = time.time() - start_time
        
        return f"✅ Procesadas {processed} imágenes, fallidas {failed} de un total de {total_images} en {total_time:.2f} segundos" 