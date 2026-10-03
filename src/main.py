"""
Script para evaluar respuestas de estudiantes usando múltiples modelos LLM.
Realiza 10 iteraciones por cada combinación de pregunta-respuesta-modelo.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import uuid

from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

# Configuración
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
OUTPUT_DIR = DOCS_DIR /f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}" / "respuestas"
PROMPT_FILE = DOCS_DIR / "input_prompt_batch.txt"
PREGUNTAS_OOP_FILE = DOCS_DIR / "Preguntas_OOP.json"
PREGUNTAS_ARQ_FILE = DOCS_DIR / "Preguntas_PrincipiosArquitectura.json"
RESPUESTAS_FILE = DOCS_DIR / "Respuestas_prueba.json"

# Modelos a evaluar
MODELOS = [
    "minimax-m2.5",
    "mimo-v2.5",
    "qwen3.6-plus",
    "hy3"
]

# Número de iteraciones por cada combinación
NUM_ITERACIONES = 1

# Modo de calificación
MODO_CALIFICACION = "normal"


def cargar_prompt_base() -> str:
    """Carga el prompt base y reemplaza el placeholder del modo de calificación."""
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt.replace("%MODO_CALIFICACION%", MODO_CALIFICACION)


def cargar_preguntas() -> list[dict]:
    """Carga todas las preguntas de ambos archivos JSON."""
    preguntas = []
    
    with open(PREGUNTAS_OOP_FILE, "r", encoding="utf-8") as f:
        preguntas.extend(json.load(f))
    
    with open(PREGUNTAS_ARQ_FILE, "r", encoding="utf-8") as f:
        preguntas.extend(json.load(f))
    
    return preguntas


def cargar_respuestas() -> dict[str, dict]:
    """Carga las respuestas indexadas por pregunta_id."""
    with open(RESPUESTAS_FILE, "r", encoding="utf-8") as f:
        datos = json.load(f)
    
    return {item["pregunta_id"]: item["respuestas"] for item in datos}


def construir_user_prompt(pregunta: dict, nivel: str, respuesta: str) -> str:
    """Construye el prompt del usuario con la pregunta, rúbrica y respuesta del estudiante."""
    rubrica_text = "\n".join([
        f"  Puntaje {score}: {criterio}"
        for score, criterio in pregunta["criterios_rubrica"].items()
    ])
    
    prompt = f"""PREGUNTA:
{pregunta["text"]}

RÚBRICA DE EVALUACIÓN:
{rubrica_text}

RESPUESTA DEL ESTUDIANTE (answer_id: {pregunta["pregunta_id"]}_{nivel}):
{respuesta}

Evalúa la respuesta según la rúbrica proporcionada y devuelve el JSON con el resultado."""
    
    return prompt


def evaluar_respuesta(
    client: OpenAI,
    modelo: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """
    Envía una solicitud al LLM y mide el tiempo de respuesta.
    Retorna el resultado con metadata de timing.
    """
    inicio = time.time()
    
    try:
        response = client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        
        tiempo_respuesta = time.time() - inicio
        contenido = response.choices[0].message.content
        
        # Intentar parsear el JSON de respuesta
        try:
            # Limpiar posibles markdown code blocks
            contenido_limpio = contenido.strip()
            if contenido_limpio.startswith("```"):
                lineas = contenido_limpio.split("\n")
                contenido_limpio = "\n".join(lineas[1:-1])
            
            respuesta_json = json.loads(contenido_limpio)
        except json.JSONDecodeError:
            respuesta_json = None
        
        return {
            "success": True,
            "tiempo_segundos": round(tiempo_respuesta, 3),
            "respuesta_raw": contenido,
            "respuesta_json": respuesta_json,
            "error": None,
        }
    
    except Exception as e:
        tiempo_respuesta = time.time() - inicio
        return {
            "success": False,
            "tiempo_segundos": round(tiempo_respuesta, 3),
            "respuesta_raw": None,
            "respuesta_json": None,
            "error": str(e),
        }


def guardar_resultado(
    iteracion: int,
    modelo: str,
    resultados: list[dict],
) -> None:
    """Guarda los resultados de una iteración para un modelo específico."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sanitizar nombre del modelo para el archivo
    modelo_sanitizado = modelo.replace("/", "_").replace(".", "_")
    filename = f"iteracion_{iteracion:02d}_{modelo_sanitizado}.json"
    filepath = OUTPUT_DIR / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Guardado: {filename}")


def main():
    """Ejecuta el proceso completo de evaluación."""
    print("=" * 70)
    print("EVALUACIÓN DE RESPUESTAS CON MÚLTIPLES MODELOS LLM")
    print("=" * 70)
    
    # Verificar API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY no encontrada en .env")
        print("   Crea un archivo .env con: OPENAI_API_KEY=tu_api_key_aqui")
        return
    
    base_url = os.getenv("OPENAI_BASE_URL", "https://opencode.ai/zen/go/v1")
    
    # Inicializar cliente
    client = OpenAI(api_key=api_key, base_url=base_url, default_headers={"x-opencode-session": uuid.uuid4().hex})
    
    # Cargar datos
    print("\n📚 Cargando datos...")
    system_prompt = cargar_prompt_base()
    preguntas = cargar_preguntas()
    respuestas = cargar_respuestas()
    
    print(f"   • {len(preguntas)} preguntas cargadas")
    print(f"   • {len(respuestas)} conjuntos de respuestas cargados")
    print(f"   • Modo de calificación: {MODO_CALIFICACION}")
    print(f"   • Modelos: {', '.join(MODELOS)}")
    print(f"   • Iteraciones por combinación: {NUM_ITERACIONES}")
    
    total_combinaciones = len(preguntas) * 3 * NUM_ITERACIONES * len(MODELOS)
    print(f"   • Total de llamadas al LLM: {total_combinaciones}")
    
    print("\n" + "=" * 70)
    print("INICIANDO EVALUACIONES")
    print("=" * 70)
    
    # Iterar
    for iteracion in range(1, NUM_ITERACIONES + 1):
        print(f"\n🔄 ITERACIÓN {iteracion}/{NUM_ITERACIONES}")
        print("-" * 70)
        
        for modelo in MODELOS:
            print(f"\n🤖 Modelo: {modelo}")
            resultados_iteracion = []
            
            for pregunta in preguntas:
                pregunta_id = pregunta["pregunta_id"]
                
                if pregunta_id not in respuestas:
                    print(f"  ⚠️  No hay respuestas para {pregunta_id}")
                    continue
                
                respuestas_pregunta = respuestas[pregunta_id]
                
                # Evaluar cada nivel de respuesta
                for nivel, respuesta_texto in respuestas_pregunta.items():
                    user_prompt = construir_user_prompt(pregunta, nivel, respuesta_texto)
                    
                    resultado = evaluar_respuesta(
                        client=client,
                        modelo=modelo,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                    
                    # Agregar metadata al resultado
                    resultado_completo = {
                        "pregunta_id": pregunta_id,
                        "nivel": nivel,
                        "iteracion": iteracion,
                        "modelo": modelo,
                        "timestamp": datetime.now().isoformat(),
                        **resultado,
                    }
                    
                    resultados_iteracion.append(resultado_completo)
                    
                    status = "✓" if resultado["success"] else "✗"
                    tiempo = resultado["tiempo_segundos"]
                    print(f"  {status} {pregunta_id} [{nivel}] - {tiempo:.2f}s")
            
            # Guardar resultados de esta iteración y modelo
            guardar_resultado(iteracion, modelo, resultados_iteracion)
    
    print("\n" + "=" * 70)
    print("✅ EVALUACIÓN COMPLETADA")
    print("=" * 70)
    print(f"\n📁 Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
