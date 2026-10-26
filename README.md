# TAREA2
# Tarea 2 – OCR y ASR en documentos y audios peruanos

**Pregunta:** ¿Qué tan fielmente extraen texto los modelos abiertos de documentos y audios peruanos?

## Corpus
- 25 documentos: 8 impresos, 8 fotos con sombra o inclinación, 9 manuscritos.
- X clips de audio (Y minutos) de fuentes públicas (MIDIS / gob.pe) y grabaciones propias.
- Fuentes y licencias: `datos/fuentes.csv`. Fecha de extracción: 26/10/2026.

## Hallazgos
1. En audio, faster-whisper small obtuvo WER 0.054 frente a 0.148 de MMS, y es ~3.5 veces más rápido (RTF 0.41 frente a 1.46).
2. En documentos no hay un ganador único: EasyOCR es mucho mejor en fotos (CER 0.12 frente a 0.58), pero PaddleOCR es mejor en impresos y manuscritos.

## Limitaciones técnicas
1. Todo se ejecutó en CPU, lo que obligó a usar modelos livianos (PaddleOCR mobile, Whisper tiny en la extracción inicial).
2. Muestra de evaluación pequeña (10 documentos, 10 clips) y sin audios con ruido o música.

## Ejecución local
    pip install -r requirements.txt
    streamlit run app.py

## Dashboard
(https://tarea2flaviaromero.streamlit.app/)
