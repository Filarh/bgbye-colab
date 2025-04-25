import argparse
from server.ui import GradioInterface

def main():
    """Función principal para ejecutar la aplicación Gradio"""
    parser = argparse.ArgumentParser(description="Ejecutar la aplicación Gradio de Eliminación de Fondo")
    parser.add_argument("--share", action="store_true", help="Crear un enlace público compartible")
    parser.add_argument("--port", type=int, default=7860, help="Puerto para ejecutar la aplicación")
    
    args = parser.parse_args()
    
    # Crear la interfaz Gradio
    interface = GradioInterface()
    app = interface.create_interface()
    
    # Lanzar la aplicación
    app.launch(share=args.share, server_port=args.port)

if __name__ == "__main__":
    main()