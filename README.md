# BGBye Gradio Notebook 🚀

> **Nota del autor:**
> Esta demo fue ensamblada principalmente con ayuda de **ChatGPT** y **Claude**. Yo aporté la idea y algunos ajustes mínimos, ¡el poder de la comunidad AI in action!

---

## 🔥 ¿Qué es BGBye?
BGBye es una interfaz de eliminación de fondo de imágenes y videos basada en múltiples modelos (U2NET, ORMBG, BriaAI, CarveKit, etc.), todo ejecutándose en Google Colab mediante **Gradio**. Sin instalaciones locales ni configuraciones complejas.

## 🚀 Inicia en Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Filarh/bgbye-colab/blob/main/BgBye_Gradio.ipynb)


---

## 📖 Instrucciones de uso
1. Haz clic en el botón **Open In Colab**.
2. En la notebook, ejecuta todas las celdas de instalación y configuración (pip, modelos, dependencias).
3. Una vez levantado el servidor interno de FastAPI y Gradio, aparecerá la interfaz:
   - **Imagen:** sube tu foto, elige modelo y formato, ¡listo!
   - **Video:** sube un `.mp4`, selecciona método y descarga tu video con fondo removido.
4. Experimenta con diferentes métodos para ver cuál funciona mejor según tu contenido.


---

## 🔧 Dependencias
La notebook instala automáticamente:

- Python 3.10+
- Gradio
- FastAPI + Uvicorn
- Transformers (BriaAI)
- `transparent_background` (InspyreNet)
- `rembg` (Rembg)
- CarveKit (U2NET, BASNet, DeepLabV3, Tracer)
- ORMBG (si está disponible)


---

## 🎉 Créditos
- Idea y ajustes mínimos: **Autor**
- Generación y ensamblaje de código: **ChatGPT** & **Claude**
- Modelos y frameworks: Marcas originales de cada proyecto (ver `requirements.txt` / celdas de instalación).

---

_Disfruta eliminando fondos desde la nube con un solo clic!_

