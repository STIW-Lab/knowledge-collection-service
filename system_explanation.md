# Knowledge Collection Service: RAG System Explanation

## Overview

This system is designed to collect and synthesize actionable advice from Reddit discussions. It leverages crowdsourced knowledge to provide personalized, practical steps for users facing specific challenges or goals. The system combines NLP techniques, vector embeddings, clustering algorithms, and scoring mechanisms to distill high-quality, relevant advice from noisy social media data.

## Architecture

The system consists of several key components:

1. **Data Collection Layer**: Gathers Reddit submissions and comments
2. **Embedding and Storage Layer**: Uses PostgreSQL with pgvector for efficient similarity search
3. **Query Processing Layer**: Enhances user queries using HyDE (Hypothetical Document Embeddings) along with user stories
4. **Retrieval Layer**: Fetches semantically similar content
5. **Processing Layer**: Extracts actionable steps and clusters similar ideas
6. **Ranking and Output Layer**: Scores and ranks recommendations based on multiple criteria

## Design Decisions and Reasoning

### 1. Choice of Data Source: Reddit

**Decision**: Focus exclusively on Reddit for initial implementation.

**Why ?**:

- Reddit provides high-quality, crowdsourced advice with diverse perspectives
- Structured format (submissions + threaded comments) allows for better context preservation
- Active communities ensure fresh, relevant content
- API access is straightforward and well-documented
- Legal considerations: Public data with proper attribution

### 2. Database Choice: PostgreSQL with pgvector

**Decision**: Use PostgreSQL with the pgvector extension for vector storage and similarity search.

**Why ?**:

- pgvector provides efficient cosine distance calculations
- Allows for hybrid search (combining semantic similarity with other filters like score, domain)
- Scalable for growing datasets

### 3. Embedding Model: text-embedding-3-small

**Decision**: Use OpenAI's text-embedding-3-small for all text embeddings.

**Reasoning**:

- High-quality embeddings with good semantic understanding
- Cost-effective compared to larger models
- Consistent dimensionality (1536) across all embeddings
- Proven performance in retrieval tasks
- API-based solution allows for easy updates to newer models

### 4. Query Enhancement: HyDE (Hypothetical Document Embeddings)

**Decision**: Generate hypothetical Reddit posts to improve query embeddings.

**Reasoning**:

- Addresses the "vocabulary mismatch" problem in semantic search
- Creates more detailed, context-rich queries that better match how users express problems on Reddit
- Improves retrieval quality by generating embeddings that are closer to actual discussion content
- Balances specificity with generality to avoid overfitting to exact query terms

### 5. Actionable Step Extraction

**Decision**: Use GPT-4o-mini to extract explicit and implied actionable steps from comments.

**Reasoning**:

- LLMs excel at understanding nuanced language and inferring implicit advice
- Structured output format ensures consistency
- Temperature setting (0.1) provides reliable, low-variance extractions
- Focus on "strongly implied" steps captures valuable indirect advice

### 6. Clustering Algorithm: HDBSCAN

**Decision**: Use HDBSCAN for clustering similar actionable steps.

**Reasoning**:

- Handles variable density clusters better than K-means
- Automatically determines number of clusters
- Robust to noise (outlier detection)
- Cosine distance metric appropriate for high-dimensional embedding space
- Min cluster size (4) ensures meaningful groupings without fragmentation

### 7. Scoring and Ranking System

**Decision**: Multi-factor usefulness score combining relevance, representativeness, and crowd validation.

**Reasoning**:

- Single metric (pure relevance) insufficient for quality assessment
- **Relevance (50%)**: Semantic similarity to user query
- **Representativeness (30%)**: How well step represents its cluster (centroid similarity)
- **Crowd Validation (20%)**: Normalized Reddit score indicating community agreement
- Weighted combination provides balanced, high-confidence recommendations

## RAG Flow

1. **User Input**: User provides a query or goal
2. **Query Enhancement**: Generate HyDE hypothetical Reddit post
3. **Embedding**: Create vector representation of enhanced query
4. **Retrieval**: Find semantically similar submissions and comments
5. **Step Extraction**: Parse actionable advice from retrieved content
6. **Clustering**: Group similar steps using HDBSCAN
7. **Summarization**: Condense clusters into representative actions, representing a cluster
8. **Scoring**: Calculate usefulness scores for each representative step
9. **Ranking**: Sort by usefulness and present to user

## Technical Implementation Details

### Database Schema

- `submissions`: Stores Reddit posts with embeddings
- `comments`: Stores comments with embeddings and metadata
- Vector columns use pgvector for efficient similarity search

### Embedding Strategy

- Consistent use of text-embedding-3-small (1536 dimensions)
- Embeddings stored as JSON arrays in PostgreSQL
- Cosine similarity for all distance calculations

### Clustering Parameters

- HDBSCAN with min_cluster_size=4, min_samples=2
- Euclidean distance (equivalent to cosine for normalized vectors)
- Cluster selection epsilon=0.05 for fine-grained grouping

### Scoring Formula

```
usefulness = 0.5 × relevance + 0.3 × centroid_similarity + 0.2 × cluster_value

Where:
- relevance = 1 - cosine_distance(query_embedding, step_embedding)
- centroid_similarity = 1 - cosine_distance(step_embedding, cluster_centroid)
- cluster_value = mean(comment_scores) / max(comment_scores)
```

## Clustering Concepts: K-means vs HDBSCAN

### What is Clustering in This Context?

Clustering is the process of grouping similar actionable steps extracted from Reddit comments. Since multiple users might suggest similar advice in different ways, clustering helps identify consensus patterns and reduce redundancy. Each cluster represents a distinct category of advice, and the system selects the most representative step from each cluster for final recommendations.

### K-means Clustering

**How it works:**

- Requires you to specify the number of clusters (K) beforehand
- Iteratively assigns data points to K cluster centers and updates centers to minimize within-cluster variance
- Assumes clusters are spherical and of similar size
- Uses Euclidean distance by default

**Pros:**

- Fast and scalable
- Simple to understand and implement
- Deterministic results (with fixed random seed)

**Cons:**

- Requires knowing K in advance (often unknown in real data)
- Sensitive to initial cluster center placement
- Assumes spherical clusters (poor for irregular shapes)
- Can't handle noise or outliers well
- All points must be assigned to a cluster

### HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)

**How it works:**

- Density-based algorithm that finds clusters of varying densities
- Builds a hierarchy of clusters by connecting points based on density
- Automatically determines the number of clusters
- Can identify noise points that don't belong to any cluster
- Uses mutual reachability distance to handle varying densities

**Pros:**

- No need to specify number of clusters beforehand
- Can find clusters of arbitrary shapes and sizes
- Robust to noise and outliers
- Handles varying cluster densities well
- Provides cluster stability scores

**Cons:**

- More computationally expensive than K-means
- Requires tuning of density parameters (min_cluster_size, min_samples)
- Results can be less stable across different parameter settings

### Why HDBSCAN Was Chosen

**Primary Reasons:**

1. **Automatic Cluster Detection**: In advice synthesis, you don't know how many distinct types of solutions exist for a given problem. HDBSCAN discovers this automatically.

2. **Handles Irregular Cluster Shapes**: Actionable steps in embedding space don't form perfect spheres. Similar advice might cluster in complex, non-spherical patterns that K-means would struggle with.

3. **Noise Handling**: Not all extracted steps are meaningful. HDBSCAN can identify and exclude outlier/noise steps that don't fit well into any cluster.

4. **Variable Density Support**: Some advice themes might have many similar suggestions (dense clusters), while others have fewer (sparse clusters). HDBSCAN handles this naturally.

**Implementation Details:**

- `min_cluster_size=4`: Ensures clusters have at least 4 steps (meaningful consensus)
- `min_samples=2`: Controls how conservative the clustering is
- `cluster_selection_epsilon=0.05`: Fine-tunes cluster granularity in embedding space
- Euclidean distance metric (though cosine would be more appropriate for normalized embeddings)

**Comparison in Practice:**
For your Reddit advice system, K-means might force unrelated steps into artificial clusters or split related advice across multiple clusters. HDBSCAN allows the data to naturally determine cluster boundaries, leading to more meaningful groupings of similar advice.

This choice enables the system to adapt to different types of queries - some might yield 2-3 major advice categories, others might have 5-6 distinct approaches, all discovered automatically without manual tuning.
