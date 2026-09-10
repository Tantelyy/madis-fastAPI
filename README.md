# MADIS — Service de prévision

Service Python de **MADIS (Ma Distribution)** consacré à la prévision de la
demande et à l'anticipation des ruptures de stock. Il fournit au
[backend NestJS](https://github.com/Tantelyy/madis-back) une API FastAPI pour
estimer la demande d'un produit et l'évolution de son stock sur **1 à 7 jours**.

## Fonctionnalités

- Lecture des données historiques dans PostgreSQL.
- Construction de la demande journalière et des variables temporelles.
- Prévision récursive de J+1 à J+7, avec prise en compte des promotions.
- Projection des stocks disponibles et des quantités expirées.
- Identification d'une éventuelle rupture dans l'horizon demandé.
- Prévisions pour un produit ou un lot de produits.
- Consultation de l'état du service et des informations du modèle.

La demande utilisée par le traitement correspond aux sorties de stock :

```text
Demande = ventes (SALE) + produits offerts (PROMOTION_GIFT)
```

## Technologies et modèle

Le projet utilise **Python, FastAPI, Uvicorn, pandas, NumPy, scikit-learn,
joblib et psycopg**. Les dépendances sont listées dans
[requirements.txt](requirements.txt).

Le modèle de déploiement actuel repose sur Extra Trees. Il est chargé au
démarrage ; les requêtes de prévision ne le réentraînent pas et ne modifient
pas les données métier dans PostgreSQL.

**Le modèle versionné a été entraîné sur des données synthétiques.** Son
fonctionnement technique ne garantit pas sa pertinence sur des données réelles.
Une base neuve ne possède pas les produits et l'historique nécessaires aux
prévisions. Avant une utilisation métier réelle, réentraîner et évaluer le
modèle sur les données correspondantes.

## Organisation

- `app/api/` : routes HTTP et schémas de requêtes/réponses.
- `app/data/` : extraction des historiques.
- `app/features/` : préparation des variables du modèle.
- `app/prediction/` : prévisions de demande et de rupture.
- `app/training/` et `app/evaluation/` : entraînement et évaluation.
- `artifacts/models/` : modèles sérialisés et métadonnées.
- `scripts/` : préparation de données, expériences et entraînement.
- `tests/` : tests unitaires et d'intégration API.

## Développement local

La CI et l'image Docker utilisent Python 3.12. Depuis la racine du dépôt,
créer un environnement virtuel :

```bash
python -m venv .venv
```

Sous Linux/macOS, l'activer avec `source .venv/bin/activate`.
Sous PowerShell, utiliser `.\.venv\Scripts\Activate.ps1`. Si l'activation
est bloquée, appeler directement `.\.venv\Scripts\python.exe` à la place
de `python` dans les commandes suivantes.

```bash
python -m pip install -r requirements.txt
```

Copier [.env.example](.env.example) vers `.env`
(`Copy-Item .env.example .env` sous PowerShell), sans écraser une configuration
existante. Renseigner `DATABASE_URL` avec une base utilisant le schéma MADIS.

Choisir `DEMAND_DATA_MODE=production` pour lire les données métier, ou
`synthetic` avec `DEMAND_SYNTHETIC_BATCH` pour un jeu synthétique existant.
Le fichier d'exemple utilise le second mode ; il ne crée pas les données.
Conserver les chemins du modèle et des métadonnées si les fichiers restent à
leur emplacement versionné.

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

L'API locale est disponible sur `http://localhost:8000` et sa documentation
interactive sur `http://localhost:8000/docs`.

## Routes principales

Toutes les routes ci-dessous sont relatives au service ML :

| Méthode | Route | Usage |
| --- | --- | --- |
| GET | /api/v1/health | État du service et indicateur de chargement du modèle |
| GET | /api/v1/demand/model-info | Métadonnées du modèle |
| GET | /api/v1/demand/forecast/{product_id} | Demande prévue d'un produit |
| POST | /api/v1/demand/forecast | Demande prévue de plusieurs produits |
| GET | /api/v1/stockout/forecast/{product_id} | Projection de stock d'un produit |
| POST | /api/v1/stockout/forecast | Projection de stock de plusieurs produits |

Consulter `/docs` pour les paramètres, dont l'horizon `days` et la date
de référence `as_of_date` des requêtes GET.

## Vérifications

```bash
python -m compileall -q app tests
python -m unittest discover -s tests -v
python -m pip check
```

Les tests d'intégration API nécessitent la base synthétique attendue par les
tests. Ils sont ignorés si `DATABASE_URL` n'est pas définie. Le seul chargement
du modèle ne suffit pas à valider les prévisions de bout en bout.

## CI/CD et déploiement

La CI vérifie la syntaxe, les tests et les dépendances sur les pull requests et
les branches `dev` et `main`. Elle construit aussi l'image et vérifie le
chargement du modèle. Un push sur `main` publie
`ghcr.io/tantelyy/madis-fastapi`, puis déploie le service sur Contabo lorsque
`DEPLOY_ENABLED=true`.

En production, FastAPI et PostgreSQL restent sur le réseau Docker privé.
Le [frontend React](https://github.com/Tantelyy/madis-front) reçoit les prévisions
par le backend ; les routes ML et `/docs` ne sont pas exposées par ngrok.
Le service ML utilise la base dont les migrations sont gérées par Nest.

Consulter le **[guide de déploiement Contabo](https://github.com/Tantelyy/madis-back/blob/main/deploy/README.md)**
du dépôt backend pour l'installation de la stack complète.
