# SafeCity — Pasos para desplegar

## 1. Cambios en tu repo (haz esto en tu compu)

a) Renombrar el archivo principal:
   `04_dashboard_safecity_v6.py` → `streamlit_app.py`

b) En `streamlit_app.py`, reemplaza el bloque (líneas ~41-47):

   ```python
   def _ruta_proyecto() -> str:
       raiz = Path(__file__).resolve().parent
       for base in (raiz, Path(r"C:\Majo\SafeCity"),
                    Path(r"C:\Majo\SafeCityListo")):
           if (base / "safecity_model_reg.json").is_file():
               return str(base)
       return str(raiz)
   ```

   con:

   ```python
   def _ruta_proyecto() -> str:
       return str(Path(__file__).resolve().parent)
   ```

c) Añade un archivo `runtime.txt` en la raíz del repo con una sola línea:

   ```
   python-3.11
   ```

d) Sube los cambios:

   ```bash
   git add .
   git commit -m "Preparar para deploy en Streamlit Cloud"
   git push
   ```

## 2. Desplegar en Streamlit Community Cloud

1. Entra a https://share.streamlit.io e inicia sesión con tu cuenta de GitHub.
2. Autoriza el acceso a tus repos cuando te lo pida.
3. Click en **"New app"** (botón arriba a la derecha).
4. Llena el formulario:
   - **Repository**: `MajoAcevedo193/SafeCity`
   - **Branch**: `main` (o `master`, según tu repo)
   - **Main file path**: `streamlit_app.py`
   - **App URL** (opcional): elige un subdominio como `safecity-majo`
5. Click en **"Deploy!"**.

El primer build tarda 3-5 minutos porque xgboost y plotly pesan. Después, cada `git push` redespliega automáticamente.

## 3. URL pública

Tu app quedará en: `https://safecity-majo.streamlit.app` (o el nombre que hayas elegido).

## Notas

- Plan gratuito: 1 GB RAM. Tu app debería caber sin problema.
- Si la app duerme por inactividad, despierta en ~30 seg al primer visitante.
- Logs en tiempo real disponibles desde el dashboard de Streamlit Cloud (útil si algo falla).
