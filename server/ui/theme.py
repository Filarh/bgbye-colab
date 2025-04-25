import gradio as gr

class Theme:
    """Clase para gestionar el tema y apariencia de la interfaz"""
    
    @staticmethod
    def get_default_theme():
        """Devuelve el tema predeterminado con estilos mejorados"""
        return gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
            neutral_hue="slate"
        )
    
    @staticmethod
    def get_dark_theme():
        """Devuelve un tema oscuro con estilos mejorados"""
        return gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="emerald",
            neutral_hue="zinc"
        )
    
    @staticmethod
    def get_custom_css():
        """Devuelve CSS personalizado para la interfaz"""
        return """
        .gradio-container {
            font-family: 'Segoe UI', sans-serif;
        }
        
        button.primary {
            background-color: #2563eb !important;
            color: white !important;
        }
        
        button.primary:hover {
            background-color: #1d4ed8 !important;
        }
        
        h1, h2, h3 {
            font-weight: bold !important;
        }
        
        .gr-box, .gr-form, .gr-panel {
            border-radius: 8px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24) !important;
        }
        
        .footer {
            text-align: center;
            margin-top: 20px;
            font-size: 0.9em;
            color: #6b7280;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        table th, table td {
            padding: 8px;
            border: 1px solid #e2e8f0;
        }
        
        table th {
            background-color: #f8fafc;
            font-weight: bold;
        }
        
        /* Estilos para la galería de imágenes */
        .gallery-item {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            cursor: pointer;
            position: relative;
        }
        
        .gallery-item:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            z-index: 10;
        }
        
        /* Mejorar la visualización de la imagen principal */
        .main-image {
            border: 2px solid #2563eb;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Ajustes para galerías responsivas */
        @media (max-width: 640px) {
            .gallery-container {
                grid-template-columns: repeat(2, 1fr) !important;
            }
        }
        
        /* Spinner de carga */
        .processing-spinner {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 15px 0;
        }
        
        .processing-spinner .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-radius: 50%;
            border-top-color: #2563eb;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }
        
        .processing-spinner p {
            margin-top: 10px;
            font-size: 0.9rem;
            color: #4b5563;
        }
        
        /* Mejorar detalles de imagen */
        .gr-markdown {
            background-color: #f8fafc;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin-bottom: 12px;
        }
        
        .gr-markdown strong {
            color: #1e40af;
        }
        
        /* Etiquetas para las miniaturas */
        .gr-gallery .gr-gallery-item .caption-container,
        .gallery-container .gr-gallery-item .caption-container {
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 4px;
            font-size: 0.8rem;
            text-align: center;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
        }
        
        /* Asegurarse de que las etiquetas sean visibles */
        .gr-gallery-item img {
            border-radius: 8px 8px 0 0;
        }
        
        /* Estilo para la barra de progreso */
        .gr-slider.svelte-1cl284s {
            margin-top: 12px;
            margin-bottom: 12px;
        }
        
        .gr-slider .handle {
            background-color: #2563eb !important;
        }
        
        .gr-slider .track {
            background-color: #dbeafe !important;
        }
        
        .gr-slider .track-fill {
            background-color: #2563eb !important;
        }
        """ 