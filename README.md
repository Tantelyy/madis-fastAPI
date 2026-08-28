# Madis Demand Forecast Service

Service de prédiction de demande basé sur FastAPI et un modèle de Machine Learning.

Le service permet de prévoir la demande future d'un produit sur un horizon de 1 à 7 jours.

Le modèle actuellement utilisé est un `ExtraTreesRegressor` entraîné sur les données historiques de mouvements de stock.

> Important : le modèle actuellement versionné a été entraîné et validé techniquement sur des données synthétiques. Il devra être réentraîné et réévalué avec les données réelles de production.

---

## Fonctionnalités

- Extraction des données depuis PostgreSQL
- Construction de la demande journalière
- Feature engineering temporel
- Prédiction récursive J+1 à J+7
- Prise en compte des promotions planifiées
- API REST avec FastAPI
- Prédiction d'un produit
- Prédiction batch de plusieurs produits
- Documentation Swagger automatique
- Tests automatisés

La demande est actuellement définie par :

```text
DemandQty = SALE + PROMOTION_GIFT