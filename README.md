# fever_bge
use bge embedding and reranker to do sentence retrivel for fever dataset

# architecture
![fact_check_arch drawio](https://github.com/fdsf53451001/fever_bge/assets/35889113/d5546547-d914-42ff-8921-13e596572569)

# how to generate top-k evidences
1. load wiki pages into chroma db
2. generate top100 evidence by cosine similarity check for bge embedding
3. get the final top-k evidence by bge rerank
4. you can compute strict recall for devset data
