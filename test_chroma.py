import chromadb, tempfile, os
d = os.path.join(tempfile.gettempdir(), "chroma_exe_test")
c = chromadb.PersistentClient(path=d)
col = c.get_or_create_collection("t1col")
col.add(ids=["a"], embeddings=[[0.1]*8], documents=["hello"])
print("add OK, count =", col.count())
r = col.query(query_embeddings=[[0.1]*8], n_results=1)
print("query OK:", r["ids"])
