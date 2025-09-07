# Pipeline ML avec MLflow + Airflow + Grafana

Une stack complète de machine learning avec orchestration, tracking et monitoring de drift.

## 🏗️ Architecture

- **Apache Airflow**: Orchestration du pipeline ML
- **MLflow**: Tracking des expériences et model registry
- **Grafana**: Dashboards de monitoring et alerting
- **Prometheus**: Collecte des métriques custom
- **PostgreSQL**: Backend pour Airflow et MLflow

## 📁 Structure du projet

```
project/
├── docker-compose.yml          # Configuration Docker Compose
├── prometheus.yml              # Configuration Prometheus
├── init-db.sql                # Script d'initialisation DB
├── airflow/
│   ├── dags/
│   │   └── ml_pipeline.py      # DAG principal du pipeline ML
│   └── requirements.txt        # Dépendances Python Airflow
├── mlflow/
│   ├── Dockerfile              # Image MLflow custom
│   └── requirements.txt        # Dépendances Python MLflow
├── grafana/
│   ├── provisioning/
│   │   ├── dashboards/
│   │   │   └── dashboards.yml  # Configuration dashboards
│   │   └── datasources/
│   │       └── datasources.yml # Configuration datasources
│   └── dashboards/
│       └── ml-monitoring.json  # Dashboard ML monitoring
├── data/                       # Données (volume Docker)
└── monitoring/
    └── drift_detector.py       # Détecteur de drift
```

## 🚀 Démarrage rapide

### 1. Cloner et naviguer vers le projet

```bash
git clone <repository>
cd PART_4
```

### 2. Lancer la stack

```bash
docker-compose up -d
```

### 3. Vérifier le déploiement

Attendre que tous les services soient UP (peut prendre 2-3 minutes):

```bash
docker-compose ps
```

### 4. Accéder aux interfaces

- **Airflow**: http://localhost:8080
  - Username: `admin`
  - Password: `admin`

- **MLflow**: http://localhost:5001

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin`

- **Prometheus**: http://localhost:9090

## 📊 Pipeline ML

### Tâches du DAG Airflow

1. **load_data**: Chargement des données depuis `/data`
2. **preprocess_data**: Preprocessing et split train/val/test
3. **train_model**: Entraînement MLPClassifier
4. **evaluate_model**: Calcul des métriques de performance
5. **detect_drift**: Détection de drift vs modèle précédent
6. **log_metrics_to_prometheus**: Envoi métriques vers Prometheus
7. **register_model**: Enregistrement dans MLflow Model Registry

### Métriques trackées

#### 🚨 **Métriques FRAUD-FOCUSED (les vraies métriques qui comptent !)**

**Performance Classe Fraude (Classe 1):**
- `test_fraud_f1` : F1-Score spécifique à la détection de fraude (≈ 0.80)
- `test_fraud_precision` : Précision sur les fraudes détectées (≈ 0.84)
- `test_fraud_recall` : Rappel des fraudes (≈ 0.77) - **CRITIQUE** pour éviter les pertes

**Impact Business:**
- `fraud_detection_rate` : Taux de détection des fraudes (0.77 = 77% des fraudes détectées)
- `frauds_detected_count` : Nombre de fraudes détectées (76 ✅)
- `frauds_missed_count` : **Nombre de fraudes ratées** (23 ⚠️ - COÛT FINANCIER)
- `false_alarm_rate` : Taux de fausses alertes (0.02% - impact opérationnel)
- `false_alarms_count` : Nombre de transactions légitimes bloquées (14)

#### 📊 **Métriques Globales (pour comparaison)**

**Performance Générale:**
- F1 Score (macro, weighted) - moyennes qui diluent les performances fraud
- Précision/Recall globaux - dominés par la classe majoritaire (99.83%)
- ~~Accuracy~~ - **SUPPRIMÉE** car trompeuse sur données déséquilibrées

**Drift Detection:**
- `drift_score` : Score global de dérive des données (0-1)
- `n_drifted_features` : Nombre de features ayant dérivé
- `max_feature_drift` : Score maximum de dérive par feature

#### ⚠️ **POURQUOI CES MÉTRIQUES SONT CRUCIALES ?**

Le dataset creditcard.csv est **ultra-déséquilibré** :
- 99.83% transactions normales vs 0.17% fraudes
- Un modèle "naïf" qui prédit toujours "normal" obtient 99.83% d'accuracy !
- **Les vraies performances se mesurent sur la classe minoritaire (fraude)**

**Exemple concret avec vos résultats actuels :**
```
test_fraud_f1: 0.804 (80.4% F1-Score sur la fraude)
test_fraud_precision: 0.844 (84.4% des alertes sont de vraies fraudes)  
test_fraud_recall: 0.768 (76.8% des fraudes sont détectées)
fraud_detection_rate: 0.768 (76.8% taux de détection)
frauds_detected_count: 76 (fraudes attrapées ✅)
frauds_missed_count: 23 (fraudes ratées ⚠️ - PERTE FINANCIÈRE)
false_alarms_count: 14 (clients gênés inutilement)
false_alarm_rate: 0.0002 (0.02% de fausses alertes)
```

**Interprétation Business :**
- ✅ **76 fraudes détectées** = Pertes évitées (si fraude moyenne = 100€, économies = 7,600€)
- ⚠️ **23 fraudes ratées** = Pertes subies (2,300€ de pertes non évitées)  
- 📊 **14 fausses alertes** = Gêne client minime (0.02% des transactions)
- 🎯 **Performance globale** : Correcte mais **23% de fraudes ratées reste élevé**

## 🔍 Monitoring et alerting

### Dashboard Grafana

Le dashboard "ML Model Monitoring" inclut **6 panels fraud-focused** :

1. **🚨 FRAUD Detection Performance** : F1, Precision, Recall de la classe fraude
2. **💰 Business Impact** : Fraudes détectées vs manquées, fausses alertes
3. **🎯 Performance Over Time** : Évolution temporelle des métriques fraud
4. **Data Drift Metrics** : Détection de dérive des données
5. **Drift Score Over Time** : Évolution de la dérive
6. **Drifted Features Count** : Nombre de features ayant dérivé

### Seuils d'alerte BUSINESS-ORIENTED

**Métriques Fraude (seuils adaptés à l'impact business) :**
- 🔴 **Critique** : Fraud Recall < 60% (trop de fraudes ratées)
- 🟡 **Attention** : Fraud Recall 60-80% (performance acceptable)
- 🟢 **Bon** : Fraud Recall > 80% (excellente détection)

**Impact Business :**
- 🔴 **Critique** : > 25 fraudes manquées par run
- 🟡 **Attention** : 10-25 fraudes manquées
- 🟢 **Acceptable** : < 10 fraudes manquées

**Drift (impact sur la stabilité du modèle) :**
- 🟢 **Stable** : Drift score < 0.1
- 🟡 **Surveillance** : Drift score 0.1-0.3
- 🔴 **Re-entraînement** : Drift score > 0.3

## 📂 Données

### Utilisation de données existantes

1. Placer vos fichiers CSV dans le dossier `data/`
2. Le pipeline chargera automatiquement le premier CSV trouvé
3. Assurez-vous que la dernière colonne soit la target

### Données synthétiques

Si aucun CSV n'est trouvé, le pipeline génère automatiquement des données synthétiques pour la démonstration.

## ⚙️ Configuration

### Variables d'environnement

Les principales variables sont configurées dans `docker-compose.yml`:
- `MLFLOW_TRACKING_URI`
- `PROMETHEUS_PUSHGATEWAY_URL`
- Credentials PostgreSQL

### Personnalisation du modèle

Modifier les paramètres dans `airflow/dags/ml_pipeline.py`:
- Architecture du MLPClassifier
- Seuils de drift
- Critères d'enregistrement du modèle

## 🛠️ Commandes utiles

### Redémarrer un service

```bash
docker-compose restart <service_name>
```

### Voir les logs

```bash
docker-compose logs -f <service_name>
```

### Accéder à un container

```bash
docker-compose exec <service_name> bash
```

### Nettoyer et redémarrer

```bash
docker-compose down -v
docker-compose up -d
```

## 📋 Troubleshooting

### Services qui ne démarrent pas

1. Vérifier les logs: `docker-compose logs <service_name>`
2. S'assurer que les ports ne sont pas occupés
3. Vérifier l'espace disque disponible

### Pipeline Airflow qui échoue

1. Aller dans l'interface Airflow
2. Consulter les logs des tâches échouées
3. Vérifier les dépendances Python
4. S'assurer que MLflow est accessible

### Métriques manquantes dans Grafana

1. Vérifier que Prometheus scrape les métriques
2. S'assurer que le pushgateway reçoit les données
3. Vérifier la configuration des datasources Grafana

## 🔄 Workflow typique

1. **Développement**: Modifier le code dans `airflow/dags/` ou `monitoring/`
2. **Test**: Déclencher manuellement le DAG dans Airflow
3. **Monitoring**: Observer les métriques dans Grafana
4. **Itération**: Ajuster les seuils et paramètres selon les résultats

## 🚀 Améliorations pour optimiser la détection de fraude

### 💡 **Techniques pour réduire les fraudes ratées (23 → < 10)**

1. **Rééquilibrage des classes :**
   - **SMOTE** : Synthèse de nouvelles fraudes artificielles
   - **Undersampling** : Réduire la classe majoritaire  
   - **Class weights** : Pénaliser plus les erreurs sur la classe fraude

2. **Optimisation des hyperparamètres :**
   - **Seuil de décision** : Abaisser le seuil (favorise recall vs precision)
   - **Architecture réseau** : Plus de couches, dropout, regularization
   - **Fonction de coût** : Focal loss pour classes déséquilibrées

3. **Modèles spécialisés :**
   - **XGBoost/LightGBM** : Très efficaces sur données tabulaires déséquilibrées
   - **Isolation Forest** : Détection d'anomalies non supervisée
   - **Ensemble methods** : Combiner plusieurs modèles

4. **Feature engineering :**
   - **Dérivées temporelles** : Variations de comportement
   - **Agrégations** : Historique transactionnel par client
   - **Ratios** : Montant vs historique client

### 📊 **Métriques business avancées**

5. **Optimisation par coût-bénéfice :**
   - Définir le **coût d'une fraude ratée** vs **coût d'une fausse alerte**
   - Optimiser le **seuil de décision** selon l'équation business
   - Calculer le **ROI du modèle** (gains évités vs coûts opérationnels)

## 📈 Extensions possibles

- **Alerting temps réel**: Notifications Slack/email quand > 10 fraudes ratées
- **A/B Testing**: Comparer MLPClassifier vs XGBoost vs LightGBM
- **Feature Store**: Intégrer un feature store (Feast) pour l'historique client
- **Model Serving**: Ajouter un service de prédiction temps réel (FastAPI)
- **CI/CD**: Intégrer avec GitHub Actions pour déploiement automatique
- **AutoML**: Hyperparameter tuning automatisé (Optuna, Hyperopt)

## 🐛 Debugging

### Logs détaillés

```bash
# Logs Airflow scheduler
docker-compose logs -f airflow-scheduler

# Logs MLflow
docker-compose logs -f mlflow

# Logs PostgreSQL
docker-compose logs -f postgres
```

### Vérification des connexions

```bash
# Test connexion PostgreSQL
docker-compose exec postgres psql -U airflow -d airflow -c "\\l"

# Test API MLflow
curl http://localhost:5000/health

# Test Prometheus targets
curl http://localhost:9090/api/v1/targets
```