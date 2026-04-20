import streamlit as st
import pandas as pd
import numpy as np
import sklearn.metrics.pairwise
import sklearn.decomposition
import sklearn.neighbors


# 1. CONFIGURATION DE LA PAGE

st.set_page_config(page_title="Recommandation de films", layout="centered")
st.title(" Système de recommandation de films")


def normaliser_genres(chaine_genres: str) -> str:
    """Normalise une chaîne de genres : minuscules, sans espaces, triée par ordre alphabétique.
    Cela évite que 'Action|Comedy' et 'comedy | action' soient considérés comme différents.
    """
    if not chaine_genres:
        return ""
    elements = [g.strip().lower() for g in chaine_genres.split("|") if g.strip()]
    return "|".join(sorted(set(elements)))


# 2. CHARGEMENT DU JEU DE DONNÉES

@st.cache_data
def charger_donnees():
    """Charge le fichier CSV contenant les notes, titres et genres des films.
    Le décorateur @st.cache_data met en cache le résultat pour accélérer les rechargements.
    """
    try:
        donnees = pd.read_csv("user_ratings_genres_mov.csv")
        return donnees
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return pd.DataFrame()


df = charger_donnees()

if df.empty:
    st.error("Le jeu de données est vide ou introuvable.")
    st.stop()


# 3. CRÉATION DU PROFIL UTILISATEUR (liste déroulante)

st.header("Création de votre profil utilisateur")
st.write("Veuillez sélectionner 3 films et leur attribuer une note.")

df_films = (
    df.drop_duplicates(subset="title")[["title", "genres"]]
    .sort_values("title")
    .reset_index(drop=True)
)
dict_film_genres = dict(zip(df_films["title"], df_films["genres"]))
liste_films = df_films["title"].tolist()


def selecteur_film(numero: int):
    """Affiche une liste déroulante de films, les genres associés et un curseur de note.
    Retourne un tuple (titre, genres, note).
    """
    titre = st.selectbox(
        f"Film {numero} - Choisissez un film",
        options=liste_films,
        key=f"film{numero}_titre",
    )
    genres = dict_film_genres.get(titre, "")
    st.markdown(f"**Genres :** {genres}")
    note = st.slider(
        f"Film {numero} - Note",
        min_value=0.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
        key=f"film{numero}_note",
    )
    return titre, genres, note


titre1, genres1, note1 = selecteur_film(1)
st.divider()
titre2, genres2, note2 = selecteur_film(2)
st.divider()
titre3, genres3, note3 = selecteur_film(3)
st.divider()

if st.button(" Valider mon profil", type="primary"):
    titres_choisis = [titre1, titre2, titre3]
    if len(set(titres_choisis)) < 3:
        st.error("Veuillez sélectionner 3 films différents.")
    else:
        profil = {
            "Film 1": {"titre": titre1, "note": note1, "genres": genres1},
            "Film 2": {"titre": titre2, "note": note2, "genres": genres2},
            "Film 3": {"titre": titre3, "note": note3, "genres": genres3},
        }
        st.session_state["profil"] = profil
        st.success("Profil créé avec succès !")

if "profil" in st.session_state:
    st.subheader("Votre profil")
    tableau_profil = pd.DataFrame(st.session_state["profil"]).T
    st.dataframe(tableau_profil, use_container_width=True)


# 4. VÉRIFICATIONS AVANT LE CALCUL DES RECOMMANDATIONS

if "profil" not in st.session_state or st.session_state["profil"] is None:
    st.info("Validez votre profil ci-dessus pour obtenir des recommandations.")
    st.stop()

profil = st.session_state["profil"]


# 5. PRÉPARATION DES DONNÉES

ID_NOUVEL_UTILISATEUR = "user_new"

df_unique = df.drop_duplicates(subset="title")[["title", "genres"]].copy()
df_unique["genres_norm"] = df_unique["genres"].fillna("").apply(normaliser_genres)

nouvel_utilisateur = [
    {
        "userId": ID_NOUVEL_UTILISATEUR,
        "title": film["titre"],
        "rating": film["note"],
        "genres": film["genres"],
    }
    for film in profil.values()
]
df_nouvel_utilisateur = pd.DataFrame(nouvel_utilisateur)

df_nettoye = df[df["userId"] != ID_NOUVEL_UTILISATEUR]
df_mis_a_jour = pd.concat([df_nettoye, df_nouvel_utilisateur], ignore_index=True)

matrice_notes = df_mis_a_jour.pivot_table(
    index="userId", columns="title", values="rating", aggfunc="mean"
)
matrice_notes_remplie = matrice_notes.fillna(0)

films_a_predire = matrice_notes.loc[ID_NOUVEL_UTILISATEUR][
    matrice_notes.loc[ID_NOUVEL_UTILISATEUR].isna()
].index

nb_composantes = min(20, min(matrice_notes_remplie.shape) - 1)
nb_composantes = max(nb_composantes, 1)


# 6. RECOMMANDATION BASÉE SUR LE CONTENU (Jaccard)

st.header(" Recommandation basée sur le contenu (Jaccard)")


def similarite_jaccard(g1: str, g2: str) -> float:
    """Calcule la similarité de Jaccard entre deux chaînes de genres.
    Formule : taille(intersection) / taille(union).
    """
    ensemble1 = set(g1.split("|")) if g1 else set()
    ensemble2 = set(g2.split("|")) if g2 else set()
    if not ensemble1 or not ensemble2:
        return 0.0
    return len(ensemble1 & ensemble2) / len(ensemble1 | ensemble2)


film_prefere = max(profil.values(), key=lambda f: f["note"], default=None)
genres_prefere_norm = normaliser_genres(film_prefere["genres"]) if film_prefere else ""

if film_prefere and genres_prefere_norm:
    st.write(
        f"Votre film préféré : **{film_prefere['titre']}** "
        f"(note : {film_prefere['note']})"
    )
    df_unique["similarite"] = df_unique["genres_norm"].apply(
        lambda g: similarite_jaccard(g, genres_prefere_norm)
    )
    titres_deja_vus = {f["titre"] for f in profil.values()}
    recommandations_contenu = (
        df_unique[~df_unique["title"].isin(titres_deja_vus)]
        .nlargest(5, "similarite")[["title", "genres", "similarite"]]
        .round({"similarite": 3})
        .reset_index(drop=True)
    )
    st.dataframe(recommandations_contenu, use_container_width=True)
else:
    st.error("Aucun film préféré détecté.")


# 7. RECOMMANDATION COLLABORATIVE – APPROCHE MÉMOIRE

st.header(" Recommandation basée sur la mémoire (Cosinus + top-K)")

matrice_sim_utilisateurs = sklearn.metrics.pairwise.cosine_similarity(
    matrice_notes_remplie
)
df_sim_utilisateurs = pd.DataFrame(
    matrice_sim_utilisateurs,
    index=matrice_notes_remplie.index,
    columns=matrice_notes_remplie.index,
)

sim_nouvel_utilisateur = df_sim_utilisateurs.loc[ID_NOUVEL_UTILISATEUR].drop(
    ID_NOUVEL_UTILISATEUR
)
K = min(20, len(sim_nouvel_utilisateur))
top_k_voisins = sim_nouvel_utilisateur.nlargest(K)

predictions_memoire = {}
for film in films_a_predire:
    notes_voisins = matrice_notes.loc[top_k_voisins.index, film].dropna()
    if notes_voisins.empty:
        continue
    sims_voisins = top_k_voisins.loc[notes_voisins.index]
    somme_sim = sims_voisins.sum()
    if somme_sim <= 0:
        continue
    predictions_memoire[film] = (
        np.dot(notes_voisins.values, sims_voisins.values) / somme_sim
    )

if predictions_memoire:
    reco_memoire = (
        pd.DataFrame(
            predictions_memoire.items(), columns=["title", "note_predite"]
        )
        .nlargest(5, "note_predite")
        .round({"note_predite": 2})
        .reset_index(drop=True)
    )
    st.dataframe(reco_memoire, use_container_width=True)
else:
    st.warning("Aucune recommandation basée sur la mémoire n'a pu être calculée.")


# 8. RECOMMANDATION COLLABORATIVE – APPROCHE NMF

st.header(" Recommandation basée sur NMF (factorisation non négative)")

try:
    modele_nmf = sklearn.decomposition.NMF(
        n_components=nb_composantes,
        init="nndsvd",
        random_state=42,
        max_iter=500,
    )
    W = modele_nmf.fit_transform(matrice_notes_remplie)
    H = modele_nmf.components_
    df_pred_nmf = pd.DataFrame(
        np.dot(W, H),
        index=matrice_notes_remplie.index,
        columns=matrice_notes_remplie.columns,
    )
    predictions_nmf = {
        film: df_pred_nmf.loc[ID_NOUVEL_UTILISATEUR, film]
        for film in films_a_predire
    }
    if predictions_nmf:
        reco_nmf = (
            pd.DataFrame(
                predictions_nmf.items(), columns=["title", "note_predite"]
            )
            .nlargest(5, "note_predite")
            .round({"note_predite": 2})
            .reset_index(drop=True)
        )
        st.dataframe(reco_nmf, use_container_width=True)
    else:
        st.warning("Aucune recommandation NMF disponible.")
except Exception as e:
    st.error(f"Erreur lors de l'exécution du modèle NMF : {e}")


# 9. RECOMMANDATION COLLABORATIVE – APPROCHE SVD

st.header(" Recommandation basée sur SVD (décomposition en valeurs singulières)")

try:
    modele_svd = sklearn.decomposition.TruncatedSVD(
        n_components=nb_composantes, random_state=42
    )
    U = modele_svd.fit_transform(matrice_notes_remplie)
    VT = modele_svd.components_
    df_pred_svd = pd.DataFrame(
        np.dot(U, VT),
        index=matrice_notes_remplie.index,
        columns=matrice_notes_remplie.columns,
    )
    predictions_svd = {
        film: df_pred_svd.loc[ID_NOUVEL_UTILISATEUR, film]
        for film in films_a_predire
    }
    if predictions_svd:
        reco_svd = (
            pd.DataFrame(
                predictions_svd.items(), columns=["title", "note_predite"]
            )
            .nlargest(5, "note_predite")
            .round({"note_predite": 2})
            .reset_index(drop=True)
        )
        st.dataframe(reco_svd, use_container_width=True)
    else:
        st.warning("Aucune recommandation SVD disponible.")
except Exception as e:
    st.error(f"Erreur lors de l'exécution du modèle SVD : {e}")


# 10. RECOMMANDATION COLLABORATIVE – APPROCHE KNN

st.header(" Recommandation basée sur KNN (k plus proches voisins)")

try:
    nb_voisins = min(10, matrice_notes_remplie.shape[0])
    modele_knn = sklearn.neighbors.NearestNeighbors(
        metric="cosine", algorithm="brute", n_neighbors=nb_voisins
    )
    modele_knn.fit(matrice_notes_remplie)

    distances, indices = modele_knn.kneighbors(
        matrice_notes_remplie.loc[[ID_NOUVEL_UTILISATEUR]], n_neighbors=nb_voisins
    )

    indices_voisins = indices[0]
    distances_voisins = distances[0]
    ids_voisins = matrice_notes_remplie.index[indices_voisins]
    masque_hors_soi = ids_voisins != ID_NOUVEL_UTILISATEUR
    ids_voisins = ids_voisins[masque_hors_soi]
    similarites = 1 - distances_voisins[masque_hors_soi]

    predictions_knn = {}
    for film in films_a_predire:
        notes_voisins = matrice_notes.loc[ids_voisins, film]
        masque = notes_voisins.notna()
        if not masque.any():
            continue
        somme_ponderee = np.dot(
            notes_voisins[masque].values, similarites[masque.values]
        )
        somme_similarites = similarites[masque.values].sum()
        if somme_similarites <= 0:
            continue
        predictions_knn[film] = somme_ponderee / somme_similarites

    if predictions_knn:
        reco_knn = (
            pd.DataFrame(
                predictions_knn.items(), columns=["title", "note_predite"]
            )
            .nlargest(5, "note_predite")
            .round({"note_predite": 2})
            .reset_index(drop=True)
        )
        st.dataframe(reco_knn, use_container_width=True)
    else:
        st.warning("Aucune recommandation KNN disponible.")
except Exception as e:
    st.error(f"Erreur lors de l'exécution du modèle KNN : {e}")