# 🎓 Chatbot RAG – Institut Langues et Cultures

## 📌 Présentation

Ce projet implémente un système **RAG (Retrieval-Augmented Generation)** prêt pour la production, conçu pour un institut de langues.

Il permet aux utilisateurs (parents, élèves, visiteurs) de poser des questions en langage naturel et d’obtenir des réponses **fiables, contextualisées et strictement basées sur les données internes de l’établissement**.

🎯 Objectifs principaux :

* Garantir **zéro hallucination**
* Fournir des réponses **précises et exploitables**
* Construire une architecture **simple, robuste et maintenable**

---

## 🧠 Architecture

Le système repose sur un pipeline RAG modulaire :

```
Question utilisateur
        ↓
Analyse d’intention
        ↓
Retriever (pgvector + reranking)
        ↓
Contrôle anti-hallucination (seuil de distance)
        ↓
Construction du prompt
        ↓
LLM (Mistral)
        ↓
Réponse finale
```

---

## ⚙️ Composants principaux

### 🔹 1. Retriever (Recherche vectorielle + reranking)

* Base de données : PostgreSQL + pgvector
* Génération d’embeddings via Mistral
* Reranking léger basé sur :

  * similarité sémantique
  * recouvrement de mots-clés
  * priorité aux FAQ
  * cohérence avec la catégorie

👉 Objectif : améliorer la pertinence sans complexifier le système

---

### 🔹 2. Mécanisme anti-hallucination (critique)

Un **seuil de distance** est utilisé pour contrôler la qualité des documents :

```python
if best_distance > threshold:
    return "Je n’ai pas cette information pour le moment."
```

👉 Si les documents ne sont pas suffisamment pertinents :

* aucune réponse n’est générée
* le système retourne un fallback sécurisé

💡 C’est un point clé pour une mise en production fiable

---

### 🔹 3. Prompt Engineering

Le prompt impose des règles strictes :

* ❌ aucune connaissance externe
* ❌ aucune supposition
* ❌ aucune invention
* ✅ réponses uniquement basées sur les données
* ✅ combinaison des extraits si nécessaire
* ✅ structuration claire (règle générale → exceptions)

---

### 🔹 4. Optimisation FAQ

* Détection des cas FAQ très pertinents
* Réponse directe sans appel complet au LLM

👉 Gain :

* performance ⚡
* précision 🎯

---

## 🗂️ Pipeline de données

### Sources supportées :

* Documents PDF / Word / TXT
* Données métier structurées (tarifs, niveaux…)

### Étapes :

* Nettoyage
* Normalisation
* Découpage (chunking)
* Enrichissement metadata
* Vectorisation
* Stockage dans PostgreSQL (pgvector)

---

## 🚀 Fonctionnalités

* 🔍 Recherche sémantique
* 🧠 Génération de réponses contextualisées
* 🛡️ Protection contre les hallucinations
* ⚡ Réponses rapides pour FAQ
* 📊 Filtrage par metadata
* 🧱 Architecture modulaire
* 🐳 Déployable via Docker

---

## 📈 Cas d’usage

* “Quels sont les tarifs ?”
* “Comment s’inscrire ?”
* “Les parents peuvent-ils récupérer leur enfant plus tôt ?”
* “Y a-t-il une cantine ?”

---

## 🛡️ Stratégie de fiabilité

| Situation            | Comportement      |
| -------------------- | ----------------- |
| Documents pertinents | Réponse générée   |
| Documents faibles    | Fallback          |
| Information absente  | Réponse explicite |

👉 Priorité : **fiabilité > complétude**

---

## ⚙️ Configuration

```python
max_retrieval_distance = 0.28
top_k = 5
max_context_chunks = 5
```

---

## 🧪 Lancement en local

```bash
uvicorn app.main:app --reload
```

Endpoint :

```
POST /api/v1/chat
```

Exemple :

```json
{
  "question": "Quels sont les tarifs ?",
  "top_k": 5
}
```

---

## 🐳 Déploiement

* Backend : FastAPI
* Base vectorielle : PostgreSQL + pgvector
* LLM : API Mistral
* Infrastructure : Docker + VPS

---

## 📊 Points forts du projet

* ✅ Pipeline RAG complet et robuste
* ✅ Gestion avancée des hallucinations
* ✅ Architecture claire et maintenable
* ✅ Reranking efficace et léger
* ✅ Cas métier réel

---

## 🔮 Roadmap

* Recherche hybride (BM25 + vector)
* Query rewriting
* Reranking avancé (cross-encoder)
* Cache (Redis)
* Monitoring & métriques

---

## 👩‍💻 Auteur

**Marwa El Allouchi**
Data Engineer | IA & RAG
Ex Java / PL-SQL Engineer

---

## 📬 Contact : 
linkedin: https://www.linkedin.com/in/marwa-el-allouchi-483a83114
Disponible pour opportunités en :

* Data Engineering
* IA / RAG
* Backend & Cloud

---
