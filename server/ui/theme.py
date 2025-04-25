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
        """ 