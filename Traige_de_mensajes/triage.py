import re, random, numpy as np, pandas as pd
random.seed(42); np.random.seed(42)

ventas = [
    "Quiero saber el precio del plan premium",
    "¿Tienen descuentos por volumen para empresas?",
    "¿Cómo puedo pagar? ¿Tarjeta o transferencia?",
    "Estoy interesado en comprar 10 unidades",
    "¿Cuánto cuesta el plan anual y cómo se factura?"
]
soporte = [
    "No puedo iniciar sesión, sale error 403",
    "La app se cierra al abrir el carrito",
    "La impresora no conecta por wifi, ya reinicié",
    "Se perdió mi pedido en la app, ayuda",
    "No me llega el código de verificación"
]
queja = [
    "El pedido llegó incompleto y nadie responde",
    "Muy mala atención, llegó tarde y mal empacado",
    "Estoy inconforme, el producto vino dañado",
    "Demasiada demora, pésimo servicio",
    "Me trataron mal por WhatsApp, muy groseros"
]

def variar(s):
    extras = ["", "!", "!!", " por favor", " urgente", " de verdad", " gracias"]
    return s + random.choice(extras)

data = []
for _ in range(20):
    data += [(variar(x), "ventas") for x in ventas]
    data += [(variar(x), "soporte") for x in soporte]
    data += [(variar(x), "queja")   for x in queja]

df = pd.DataFrame(data, columns=["texto","etiqueta"]).sample(frac=1, random_state=42).reset_index(drop=True)
print("Muestras:", len(df), df["etiqueta"].value_counts().to_dict())

import re

def limpiar(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-záéíóúñü0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["texto_clean"] = df["texto"].apply(limpiar)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["texto_clean"], df["etiqueta"], test_size=0.2, random_state=42, stratify=df["etiqueta"]
)

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

pipe = make_pipeline(
    TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2),
    LinearSVC(class_weight="balanced", random_state=42)
)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

acc = accuracy_score(y_test, pred)
print(f"\nAccuracy test: {acc:.3f}\n")
print("Reporte por clase:\n", classification_report(y_test, pred, digits=3))

cm = confusion_matrix(y_test, pred, labels=["ventas","soporte","queja"])
print("\nMatriz de confusión (filas=real, cols=pred):\n", pd.DataFrame(cm,
      index=["real_ventas","real_soporte","real_queja"],
      columns=["pred_ventas","pred_soporte","pred_queja"]))


from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, df["texto_clean"], df["etiqueta"], cv=5, scoring="f1_macro")
print(f"\nCV 5-fold F1_macro: media={scores.mean():.3f} ±{scores.std():.3f}")


def enrutar_mensajes(textos):
    tx = [limpiar(t) for t in textos]
    etiquetas = pipe.predict(tx)
    area = {"ventas":"Equipo Ventas", "soporte":"Mesa Soporte", "queja":"Atención al Cliente"}
    rutas = [area[e] for e in etiquetas]
    return list(zip(textos, etiquetas, rutas))

nuevos = [
    "Se dañó el botón de encendido, necesito ayuda urgentemente",
    "¿Hacen descuento si compro 15 licencias?",
    "Estoy muy molesto: llegó tarde y la caja rota, pésimo servicio"
]
print("\nEnrutamiento de mensajes nuevos:")
for texto, etiqueta, ruta in enrutar_mensajes(nuevos):
    print(f"- '{texto}' -> clase: {etiqueta} | ruta: {ruta}")
    
    
import joblib

joblib.dump(pipe, "pipeline_triage.joblib")
print("\nPipeline guardado en pipeline_triage.joblib")

loaded = joblib.load("pipeline_triage.joblib")
print("Test carga:", loaded.predict(["No puedo entrar a mi cuenta, sale error 500"])[0])