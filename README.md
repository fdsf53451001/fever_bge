# fever_bge
use bge embedding and reranker to do sentence retrivel for fever dataset

# how to generate top-k evidences
1. load wiki pages into chroma db
2. generate top100 evidence by cosine similarity check for bge embedding
3. get the final top-k evidence by bge rerank
4. you can compute strict recall for devset data
