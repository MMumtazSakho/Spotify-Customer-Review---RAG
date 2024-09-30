# mekari-sakho

# RAG dengan Function Calling Tool
## Arsitektur sistem
![image](https://github.com/user-attachments/assets/44de1965-b314-4932-9a14-14d3e8c393f4)
### Mengapa menggunakan Function calling tool?
RAG (Retrieval-Augmented Generation) berbasis function calling tool memberikan solusi untuk menjawab pertanyaan dengan efisien.  
Bayangkan, jika pengguna bertanya tentang review Spotify, misalnya: <br> 
<img src="https://github.com/user-attachments/assets/ec3a1ae7-ac97-4430-a0a5-2b9f403859a5" alt="Review Spotify" width="500"/>

Sistem dapat mengambil data review dari sumber eksternal untuk memberikan jawaban yang relevan. Namun ketika pengguna melanjutkan dengan pertanyaan lebih spesifik, seperti "kartu Maestro itu apa dan dari mana?", Pendekatan RAG biasa tidaklah efektif.
Dengan pendekatan function calling RAG, sistem menawarkan pengalaman interaktif efisien dalam segi token input.
<img src="https://github.com/user-attachments/assets/256af3e9-450e-4b72-9822-2c59889f58d2" alt="Review Spotify" width="500"/>

## TOOLS
tools yang digunakan
- Openai -> untuk LLM
- Mistral embedding -> untuk embedding
- Langchain 
- Chroma Database ->Penyimpanan embedding/vektor
