import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import h5py
import time
from groq import Groq

try:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except ImportError:
    from keras.applications.mobilenet_v2 import preprocess_input

# --- 1. Page Config ---
st.set_page_config(page_title="IA Diagnostic Poumon", page_icon="🫁", layout="wide")

# --- 2. Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "score_final" not in st.session_state:
    st.session_state.score_final = 0.0

# --- 3. Groq Client ---
try:
    api_key = st.secrets["GROQ_KEY"]
    client = Groq(api_key=api_key)
    groq_available = True
except Exception:
    groq_available = False
    st.sidebar.error("⚠️ Clé API Groq manquante dans les secrets.")

# --- 4. Fonction de génération (Groq uniquement) ---
def generate_response(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content

# --- 5. H5 Diagnostic ---
with st.expander("🔧 Diagnostic fichier .h5", expanded=False):
    try:
        with h5py.File("mon_ia_cancer.h5", "r") as f:
            st.write("**Clés racine :**", list(f.keys()))
            if "model_config" in f.attrs:
                st.success("✅ Modèle complet détecté (architecture + poids)")
            else:
                st.warning("⚠️ Fichier poids uniquement")
            if "layer_names" in f.attrs:
                layers = [l.decode() if isinstance(l, bytes) else l
                          for l in f.attrs["layer_names"]]
                st.write(f"**Couches sauvegardées :** {len(layers)}")
                st.write(layers[:10])
    except Exception as e:
        st.error(f"Impossible de lire le fichier h5 : {e}")

# --- 6. Model Loading ---
@st.cache_resource
def load_my_model():
    # Attempt 1: Load complete model
    try:
        model = tf.keras.models.load_model("mon_ia_cancer.h5", compile=False)
        return model, "✅ Modèle complet chargé."
    except Exception:
        pass

    # Attempt 2: Rebuild full architecture + load all weights
    try:
        tf.keras.backend.clear_session()
        base = tf.keras.applications.MobileNetV2(
            weights=None, include_top=False, input_shape=(224, 224, 3)
        )
        out = base.layers[-1].output
        if isinstance(out, (list, tuple)):
            out = out[-1]
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(out)
        predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="dense")(x)
        model = tf.keras.Model(inputs=base.input, outputs=predictions)
        model.load_weights("mon_ia_cancer.h5")
        return model, "✅ Poids complets chargés (modèle reconstruit)."
    except Exception:
        pass

    # Attempt 3: ImageNet base + load only head layers by name
    try:
        tf.keras.backend.clear_session()
        inputs = tf.keras.Input(shape=(224, 224, 3))
        base = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, input_tensor=inputs
        )
        base.trainable = False
        out = base.layers[-1].output
        if isinstance(out, (list, tuple)):
            out = out[-1]
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(out)
        predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="dense")(x)
        model = tf.keras.Model(inputs=base.input, outputs=predictions)
        model.load_weights("mon_ia_cancer.h5", by_name=True, skip_mismatch=True)
        return model, "⚠️ Poids chargés partiellement (head only, by_name)."
    except Exception as e3:
        return None, f"❌ Échec total : {e3}"

model, model_status = load_my_model()

if "❌" in model_status:
    st.sidebar.error(model_status)

# --- 7. Sidebar ---
st.sidebar.header("📋 Profil Complet du Patient")

with st.sidebar.expander("👤 Informations générales", expanded=True):
    age = st.number_input("Âge", 1, 120, 45)
    sexe = st.selectbox("Sexe", ["Homme", "Femme"])
    ville = st.selectbox("Ville (Tunisie)", [
        "Tunis", "Sousse", "Sfax", "Bizerte", "Kairouan", "Gafsa", "Autre"
    ])

with st.sidebar.expander("🚬 Style de vie & Environnement", expanded=False):
    fumeur = st.selectbox("Tabagisme", ["Non-fumeur", "Ancien fumeur", "Fumeur actif"])
    paquets_annee = 0
    if fumeur != "Non-fumeur":
        paquets_annee = st.number_input("Nombre de paquets/an (Estimation)", 0, 200, 10)
    sport = st.select_slider(
        "Niveau d'activité physique",
        options=["Sédentaire", "Modérée", "Sportif"]
    )
    exposition = st.multiselect(
        "Exposition professionnelle/Air",
        ["Amiante", "Silice", "Gaz d'échappement", "Produits chimiques", "Pollution urbaine forte"]
    )

with st.sidebar.expander("🩺 Signes Cliniques", expanded=False):
    symptomes = st.multiselect(
        "Symptômes persistants (> 3 semaines)",
        [
            "Toux chronique", "Douleur thoracique", "Essoufflement",
            "Crachats de sang (Hémoptysie)", "Fatigue intense", "Perte de poids inexpliquée"
        ]
    )
    antecedents = st.checkbox("Antécédents familiaux de cancer du poumon")

# Contexte enrichi pour le chatbot
infos_complementaires = f"""
- Sexe : {sexe}
- Ville : {ville}
- Activité physique : {sport}
- Tabagisme : {fumeur}{f' ({paquets_annee} paquets/an)' if fumeur != 'Non-fumeur' else ''}
- Expositions : {', '.join(exposition) if exposition else 'Aucune'}
- Symptômes : {', '.join(symptomes) if symptomes else 'Aucun'}
- Antécédents familiaux : {'Oui' if antecedents else 'Non'}
"""

# --- 8. Main UI ---
st.title("🫁 Assistant Diagnostic : Cancer du Poumon")

tab1, tab2 = st.tabs(["🔍 Analyse du Scanner", "💬 Assistant & Conseils"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Téléchargez le scan CT (JPG, PNG)",
            type=["jpg", "png", "jpeg"]
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Scanner chargé", use_container_width=True)

    with col2:
        if uploaded_file and model:
            with st.spinner("Analyse du scan en cours..."):
                img_resized = image.resize((224, 224))
                img_array = np.expand_dims(np.array(img_resized), axis=0)
                img_preprocessed = preprocess_input(img_array.astype(np.float32))

                prediction = model.predict(img_preprocessed)
                prob_ia = float((1 - prediction[0][0]) * 100)
                st.session_state.score_final = prob_ia

                st.subheader("📊 Résultats")
                st.metric("Probabilité IA", f"{prob_ia:.2f}%")
                st.progress(int(prob_ia))

                if prob_ia > 70:
                    st.error("🔴 Résultat très suspect. Consultation médicale urgente recommandée.")
                elif prob_ia > 40:
                    st.warning("🟡 Résultat ambigu. Un suivi médical est conseillé.")
                else:
                    st.success("🟢 Aucun signe majeur détecté.")

                st.info("💡 Rendez-vous dans l'onglet **Assistant** pour interpréter ce résultat.")

        elif not model:
            st.error("❌ Le modèle n'a pas pu être chargé.")
        else:
            st.info("Veuillez charger un scan pour débuter l'analyse.")

with tab2:
    st.subheader("💬 Posez vos questions à l'IA")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ex: Expliquez-moi ce résultat..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not groq_available:
                st.error("❌ Chatbot non disponible (clé API Groq manquante).")
            else:
                try:
                    contexte = (
                        f"Tu es un assistant médical expert en oncologie pulmonaire. "
                        f"Voici le profil du patient :\n"
                        f"- Âge : {age} ans\n"
                        f"{infos_complementaires}"
                        f"- Score IA du scanner : {st.session_state.score_final:.1f}%\n\n"
                        f"Réponds de manière professionnelle,claire et bienveillante, "
                        f"sans poser de diagnostic définitif. "
                        f"Encourage toujours la consultation d'un médecin spécialiste."
                    )
                    full_prompt = f"{contexte}\n\nQuestion du patient : {prompt}"

                    with st.spinner("Génération de la réponse..."):
                        reply = generate_response(full_prompt)

                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                except Exception as e:
                    st.error(f"❌ Erreur chatbot : {e}")

st.divider()
st.caption("Projet Étudiant — Outil d'aide à la décision uniquement. Ne remplace pas un avis médical.")