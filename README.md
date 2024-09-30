# mekari-sakho

# RAG dengan Function Calling Tool
## Arsitektur sistem
![image](https://github.com/user-attachments/assets/44de1965-b314-4932-9a14-14d3e8c393f4)
### Mengapa menggunakan Function calling tool?
RAG (Retrieval-Augmented Generation) berbasis function calling tool memberikan solusi untuk menjawab pertanyaan dengan efisien.  
Bayangkan, jika pengguna bertanya tentang review Spotify, misalnya: <br> 
<img src="https://github.com/user-attachments/assets/ec3a1ae7-ac97-4430-a0a5-2b9f403859a5" alt="Review Spotify" width="500"/>

Sistem dapat mengambil data review dari sumber eksternal untuk memberikan jawaban yang relevan. Namun ketika pengguna melanjutkan dengan pertanyaan lebih spesifik, seperti "kartu Maestro itu apa dan dari mana?", Pendekatan RAG biasa tidaklah efektif.
Dengan pendekatan function calling RAG, sistem menawarkan pengalaman interaktif efisien dalam segi token input.<br>
<img src="https://github.com/user-attachments/assets/256af3e9-450e-4b72-9822-2c59889f58d2" alt="Review Spotify" width="500"/>
## Perbandingan LLM -> gpt-40-mini VS Mistral-Large

| Attempt | gpt-40-mini | Mistral-Large  |
| ------- | --- | --- |
| Cost | $0.150 / 1M input tokens & $0.600 / 1M output tokens | $2 /1M input tokens & $6 /1M output tokens |
| function call | very good, adaptive | Bad, always calling for function |
| Accuracy | overall good, depends on retrieved data | overall good, depends on retrieved data |
<br>
Pada task ini, saya menggunakan gpt-40-mini karena kemampuannya beradaptasi dengan function calling tool RAG.

## Perbandingan Model Embbeding -> text-embedding-3-small VS mistral-embed

| Attempt | text-embedding-3-small | mistral-embed  |
| ------- | --- | --- |
| Cost | $0.020 / 1M tokens | $0.1 /1M tokens |
| Size | 1536 | 1024 |
| Quality | good | good |
<br>
Pada task ini, saya menggunakan mistral-embed untuk embedding.

## TOOLS
tools yang digunakan
- Openai -> untuk LLM
- Mistral embedding -> untuk embedding
- Langchain 
- Chroma Database ->Penyimpanan embedding/vektor




## Cara Run
- Download folder chromadb di link ini (Tidak bisa ter-upload di github (2.7 GB):
  https://drive.google.com/drive/folders/13wTK9UZ_GNDoommmod-SnkHALXJBHKbL?usp=sharing
  letakkan folder tersebut di forlder projek (mekari-sakho)
- Buat API key di Openai (https://platform.openai.com/api-keys) (<strong>Berbayar</strong>) dan Mistral (https://console.mistral.ai/) (<strong>Gratis saat ini</strong>)
- Setelah menginstall semua library, download folder vektor database, dan membuat API, run "streamlit run app.py" di terminal




