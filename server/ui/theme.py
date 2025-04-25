import gradio as gr

class Theme:
    """Clase para gestionar el tema y apariencia de la interfaz"""
    
    @staticmethod
    def get_default_theme():
        """Devuelve el tema predeterminado con estilos mejorados"""
        return gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
            neutral_hue="slate",
            text_size=gr.themes.sizes.text_lg
        ).set(
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_hover="*primary_600",
            button_primary_text_color="white",
            button_secondary_background_fill="*neutral_100",
            button_secondary_background_fill_hover="*neutral_200",
            button_secondary_text_color="*neutral_800",
            block_title_text_weight="600",
            block_label_text_size="*text_md",
            block_label_text_weight="600",
            input_background_fill="*neutral_50",
            container_radius="*radius_lg",
            card_radius="*radius_lg"
        )
    
    @staticmethod
    def get_dark_theme():
        """Devuelve un tema oscuro con estilos mejorados"""
        return gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="emerald",
            neutral_hue="zinc",
            text_size=gr.themes.sizes.text_lg
        ).set(
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_hover="*primary_600",
            button_primary_text_color="white",
            button_secondary_background_fill="*neutral_700",
            button_secondary_background_fill_hover="*neutral_600",
            button_secondary_text_color="white",
            block_title_text_weight="600",
            block_label_text_size="*text_md",
            block_label_text_weight="600",
            input_background_fill="*neutral_800",
            container_radius="*radius_lg",
            card_radius="*radius_lg",
            background_fill_primary="*neutral_900",
            background_fill_secondary="*neutral_800",
            text_color="white",
            border_color_primary="*neutral_700"
        )
    
    @staticmethod
    def get_custom_css():
        """Devuelve CSS personalizado para la interfaz"""
        return """
        .gradio-container {
            font-family: 'Nunito', 'Segoe UI', sans-serif !important;
        }
        
        .gr-button-primary {
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        .gr-button-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        .gr-button.gr-button-lg {
            font-weight: 600;
        }
        
        .gr-form {
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .gr-box {
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            font-size: 0.875rem;
            color: var(--neutral-500);
        }
        
        .tab-nav * {
            font-weight: 600;
        }
        
        .gr-prose h1, .gr-prose h2, .gr-prose h3 {
            font-weight: 700;
        }
        
        .image-preview img {
            object-fit: contain;
            max-height: 500px;
        }
        """ 