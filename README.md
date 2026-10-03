# Comparador de Modelos LLM

Script para evaluar respuestas de estudiantes usando múltiples modelos LLM a través de una API compatible con OpenAI.

## Configuración

1. Copia `.env.example` a `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edita `.env` y agrega tu API key:
   ```
   OPENAI_API_KEY=tu_api_key_aqui
   OPENAI_BASE_URL=https://opencode.ai/zen/go/v1
   ```

## Uso

Ejecuta el script:
```bash
.venv\Scripts\python.exe src\main.py
```

## Configuración del script

En `src/main.py` puedes modificar:

- `MODELOS`: Lista de modelos a evaluar (default: minimax-m2.5, deepseek-v4.1-flash, mimo-v2.5)
- `NUM_ITERACIONES`: Número de veces que se evalúa cada combinación (default: 10)
- `MODO_CALIFICACION`: Modo de evaluación (default: normal, opciones: estricto, normal, flexible)

## Estructura de datos

### Entradas
- `docs/input_prompt_batch.txt`: Prompt base del evaluador
- `docs/Preguntas_OOP.json`: 5 preguntas de programación orientada a objetos
- `docs/Preguntas_PrincipiosArquitectura.json`: 5 preguntas de arquitectura de software
- `docs/Respuestas_prueba.json`: 3 niveles de respuesta por pregunta (bajo_conocimiento, conoce_pero_no_es_experto, conocimiento_experto)

### Salidas
- `docs/respuestas/iteracion_{N}_{modelo}.json`: Resultados de cada iteración por modelo

Cada archivo de salida contiene un array con:
- `pregunta_id`: Identificador de la pregunta
- `nivel`: Nivel de conocimiento de la respuesta evaluada
- `iteracion`: Número de iteración
- `modelo`: Modelo utilizado
- `timestamp`: Fecha y hora de la evaluación
- `tiempo_segundos`: Tiempo de respuesta del LLM
- `respuesta_raw`: Respuesta completa del LLM
- `respuesta_json`: Respuesta parseada como JSON (si fue posible)
- `success`: Si la llamada fue exitosa
- `error`: Mensaje de error (si hubo)

## Total de llamadas

Con la configuración por defecto:
- 10 preguntas × 3 niveles × 10 iteraciones × 3 modelos = **900 llamadas al LLM**
