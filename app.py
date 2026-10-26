from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

DATOS = Path(__file__).parent / "datos"
st.set_page_config(page_title="OCR y ASR en documentos peruanos", layout="wide")

@st.cache_data
def cargar():
    leer = lambda n: pd.read_csv(DATOS / n)
    return (leer("documentos.csv"), leer("transcripciones.csv"), leer("extraccion_llm.csv"),
            leer("metricas.csv"), leer("metricas_detalle.csv"))

docs, trans, llm, met, det = cargar()

st.title("¿Qué tan fielmente extraen texto los modelos abiertos de documentos y audios peruanos?")
st.markdown(f"**Método:** {len(docs)} documentos (impresos, fotos y manuscritos) con PaddleOCR y {len(trans)} clips "
            "con faster-whisper + VAD. Textos seudonimizados. Evaluación CER/WER sobre 10 documentos y 10 clips "
            "transcritos a mano; extracción estructurada con LLM auditada en 20 casos.")

with st.sidebar:
    st.header("Filtros")
    q = st.text_input("Buscar palabra en OCR/ASR")
    tipos = st.multiselect("Tipo de documento", sorted(docs.tipo.unique()), default=sorted(docs.tipo.unique()))
    modelos = st.multiselect("Modelo evaluado", sorted(det.modelo.unique()), default=sorted(det.modelo.unique()))
    dmax = float(trans.duracion_seg.max())
    rango = st.slider("Duración del audio (s)", 0.0, dmax, (0.0, dmax))

d = docs[docs.tipo.isin(tipos)]
t = trans[trans.duracion_seg.between(*rango)]
if q:
    d = d[d.texto_ocr.fillna("").str.contains(q, case=False, regex=False)]
    t = t[t.texto_asr.fillna("").str.contains(q, case=False, regex=False)]
dt = det[det.modelo.isin(modelos)]
errores = met[(met.metrica != "acuerdo") & met.modelo.isin(modelos)]

c = st.columns(3)
c[0].metric("Documentos procesados", len(d))
c[1].metric("Minutos de audio transcritos", f"{t.duracion_seg.sum() / 60:.1f}")
acuerdo = met.loc[met.metrica == "acuerdo", "valor"]
c[2].metric("Concordancia LLM–humano", f"{acuerdo.iloc[0]:.0%}" if len(acuerdo) else "—")
for col, r in zip(st.columns(max(len(errores), 1)), errores.itertuples()):
    col.metric(f"{r.metrica} · {r.modelo}", f"{r.valor:.3f}")

def mostrar_llm(id_):
    fila = llm[llm.id == id_]
    if len(fila):
        st.json(json.loads(fila.campos_json.iloc[0]))
    else:
        st.caption("Sin extracción LLM.")

tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Documentos", "Audios", "Datos"])

with tab1:
    g = st.columns(2)
    g[0].plotly_chart(px.histogram(d, x="tipo", title="Documentos por tipo"), use_container_width=True)
    g[1].plotly_chart(px.histogram(t, x="duracion_seg", nbins=15, title="Duración de los clips (s)"),
                      use_container_width=True)
    st.plotly_chart(px.box(dt, x="modelo", y="error", color="modalidad", points="all",
                           title="Error por modelo (CER en documentos, WER en audio)"), use_container_width=True)

with tab2:
    if d.empty:
        st.info("Ningún documento coincide con los filtros.")
    else:
        sel = st.selectbox("Documento", d.id, format_func=lambda i: f"{i} – {d.loc[d.id == i, 'tipo'].iloc[0]}")
        fila = d[d.id == sel].iloc[0]
        a, b = st.columns(2)
        with a:
            st.subheader("Texto OCR")
            st.write(fila.texto_ocr)
        with b:
            st.subheader("Campos extraídos (LLM)")
            mostrar_llm(sel)
        if pd.notna(fila.texto_referencia):
            st.subheader("Referencia manual y CER")
            st.write(fila.texto_referencia)
            st.dataframe(dt[dt.id == sel][["modelo", "error", "hipotesis"]], use_container_width=True)
        st.caption("Imagen no publicada para proteger datos personales.")

with tab3:
    if t.empty:
        st.info("Ningún clip coincide con los filtros.")
    else:
        sel = st.selectbox("Clip", t.id, format_func=lambda i: f"{i} – {t.loc[t.id == i, 'archivo_audio'].iloc[0]}")
        fila = t[t.id == sel].iloc[0]
        st.write(f"**Duración:** {fila.duracion_seg:.0f} s · **Hablantes:** {fila.n_hablantes}")
        st.subheader("Transcripción ASR")
        st.write(fila.texto_asr)
        srt = DATOS / "ejemplos" / "asr" / f"{Path(fila.archivo_audio).stem}.srt"
        if srt.exists():
            with st.expander("Segmentos con marcas de hablante"):
                st.text(srt.read_text(encoding="utf-8"))
        audio = DATOS / "ejemplos" / "audio" / fila.archivo_audio
        if audio.exists():
            st.audio(str(audio))
        else:
            st.info("Audio no publicable: se muestra solo el texto y sus metadatos.")
        comp = dt[dt.id == sel]
        if len(comp):
            st.subheader("Comparación de modelos (WER)")
            st.dataframe(comp[["modelo", "error", "rtf", "hipotesis"]], use_container_width=True)
        st.subheader("Campos extraídos (LLM)")
        mostrar_llm(sel)

with tab4:
    st.dataframe(d[["id", "tipo", "fuente", "texto_ocr"]], use_container_width=True)
    st.download_button("Descargar documentos (CSV)", d.to_csv(index=False).encode("utf-8-sig"), "documentos.csv")
    st.dataframe(t[["id", "archivo_audio", "duracion_seg", "texto_asr"]], use_container_width=True)
    st.download_button("Descargar transcripciones (CSV)", t.to_csv(index=False).encode("utf-8-sig"), "transcripciones.csv")
    