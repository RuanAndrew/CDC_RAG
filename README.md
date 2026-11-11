# CDC em Linguagem Simples — Assistente RAG

Aplicação local para consultar o Código de Defesa do Consumidor (Lei nº 8.078/1990), recuperar os dispositivos relevantes e gerar explicações em português simples com um LLM. O sistema foi organizado para que a extração, a indexação, a recuperação, a geração e a avaliação sejam módulos separados.

> **Aviso jurídico:** ferramenta educacional e experimental. Não substitui a legislação oficial, orientação jurídica profissional, Procon ou Defensoria Pública. A resposta depende da versão do documento efetivamente indexada.

## 1. O que foi implementado

- Extração de PDFs pesquisáveis com PyMuPDF, preservando página física.
- Identificação e isolamento da Lei nº 8.078 dentro de compilações que também incluem decretos e portarias.
- Extração de artigos, títulos, capítulos e seções; detecção de artigos vetados/revogados.
- Chunking por estrutura jurídica (caput, parágrafo, inciso e alínea), com offsets de tokenizador para não reescrever o texto original.
- Embeddings multilíngues E5 e armazenamento persistente no ChromaDB.
- Indexação versionada: uma nova tentativa não apaga o índice ativo se falhar antes da validação.
- Recuperação vetorial, BM25 e híbrida com Reciprocal Rank Fusion (RRF).
- Lookup direto quando a pergunta menciona um artigo, filtragem conservadora de trechos não marcados como vigentes e reranking opcional.
- Geração de respostas por uma camada de provedores configurável: Ollama local, Ollama Cloud (DeepSeek V4.1 Flash), APIs compatíveis com OpenAI e Anthropic. Perfis predefinidos podem ser editados na interface.
- Prompt com fundamentação estrita nos trechos recuperados, pedido de abstinência quando não há base suficiente e citações de artigo/página.
- Avaliação hit@k, recall@k e MRR em nível de artigo único, sem contar vários chunks do mesmo artigo como posições diferentes.
- Testes unitários e de integração usando o PDF histórico anexado.

## 2. Requisito de versão da linguagem

Recomendado: **Python 3.14 de 64 bits**. O usuário havia utilizado Python 3.14 em outro projeto; para este projeto de ML/NLP, utilize um ambiente Python 3.14 isolado para reduzir incompatibilidades de wheels de PyTorch, Transformers, Sentence Transformers e ChromaDB.

### Windows

1. Instale Python 3.14 (64-bit), marcando a opção de adicionar o Python Launcher.
2. Abra a pasta do projeto.
3. Rode o seguinte no termial para instalar as dependências:

```bash
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

4. Rode o seguinte comando para iniciar a aplicação:

```bash
streamlit run app.py
```

### Linux/macOS

```bash
py -3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

O primeiro carregamento baixa o modelo de embeddings e, portanto, requer acesso à internet. Depois, o modelo fica armazenado no cache local do Hugging Face.

## 3. Uso da aplicação

1. Abra a aplicação com `streamlit run app.py`.
2. Na seção **Preparar a base legal**, envie o PDF oficial atualizado.
3. Preencha a edição/data da fonte e, se disponível, a URL oficial.
4. Escolha o modelo de embeddings. Use `intfloat/multilingual-e5-small` para a primeira execução; o mesmo modelo será gravado no manifesto e usado nas consultas.
5. Clique em **Indexar PDF enviado**. O relatório mostra o intervalo de páginas isolado, dispositivos extraídos e quantidade de chunks.
6. Na barra lateral, escolha a estratégia de recuperação e o provedor de LLM.
7. Faça perguntas na caixa de chat. A resposta é exibida com os trechos recuperados em **Fontes recuperadas**.

## 4. Configurar o modelo de linguagem

A barra lateral oferece perfis predefinidos. A escolha do provedor fica isolada em `llm.py`; `retrieval.py`, ChromaDB e os embeddings não dependem do modelo gerador. Você pode trocar o modelo sem reindexar os documentos.

### Perfis disponíveis

- **Ollama Local — Qwen 2.5 7B:** usa `http://localhost:11434` e a API nativa `/api/chat`. Instale o Ollama e execute `ollama pull qwen2.5:7b` (ou informe outro modelo instalado).
- **Ollama Cloud — DeepSeek V4.1 Flash:** usa `https://ollama.com/v1/chat/completions`, no formato compatível com OpenAI. Informe uma chave em `OLLAMA_API_KEY` ou diretamente no campo de chave da interface.
- **API compatível com OpenAI — personalizada:** permite configurar endpoint, identificador do modelo e chave. Pode ser usada com OpenAI, NVIDIA NIM, OpenRouter, vLLM e outros serviços que implementem o endpoint Chat Completions.
- **Anthropic — Claude:** usa a API Messages da Anthropic e requer uma chave própria.

Os perfis são editáveis na interface: URL base, identificador do modelo e chave podem ser alterados. Para adicionar um perfil permanente, inclua uma entrada em `LLM_PRESETS` no `llm.py` e, se necessário, o mapeamento de variáveis de ambiente em `get_model_presets()`. Para um provedor com protocolo próprio, implemente seu adaptador no `generate_answer()` ou extraia-o para um módulo específico.

### Variáveis de ambiente

Copie `.env.example` para `.env` e preencha somente as chaves dos serviços que pretende utilizar:

```dotenv
# Ollama Cloud
OLLAMA_API_KEY=
OLLAMA_CLOUD_BASE_URL=https://ollama.com/v1
OLLAMA_CLOUD_MODEL=deepseek-v4.1-flash

# OpenAI ou outro endpoint compatível
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=

# Anthropic
ANTHROPIC_API_KEY=
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_MODEL=claude-haiku-5-5
```

As chaves não são gravadas no manifesto, nos chunks ou nos arquivos de índice. Não publique o arquivo `.env` no Git. Quando um provedor externo é selecionado, a pergunta e os trechos legais recuperados são enviados à API correspondente; não inclua dados pessoais desnecessários. O modelo gerador pode ser trocado sem alterar o modelo de embeddings.

## 5. Avaliação da recuperação

O conjunto `eval/questions.csv` mantém as perguntas de avaliação fornecidas, com os artigos esperados.

Depois de indexar a base, execute:

```bash
python evaluate.py
```

Exemplos de comparação:

```bash
python evaluate.py --configs vector,bm25,hybrid,hybrid+lookup
python evaluate.py --configs hybrid,hybrid+lookup,hybrid+lookup+rerank
```

O arquivo de saída padrão é `eval/results.csv`. As métricas são calculadas em **artigos únicos**: se três chunks do art. 18 aparecem antes do art. 19, o art. 19 ocupa a segunda posição do ranking de artigos, não a quarta.

A avaliação mede recuperação, não a correção jurídica da resposta gerada. Para uma avaliação completa do sistema, faça também avaliação humana de fidelidade, completude, citações, clareza e abstinência em perguntas fora do escopo.

## 6. Validar o PDF sem baixar os modelos de embeddings

Para conferir a extração estrutural:

```bash
python validate_pdf.py sample_data/cdc_2013_historico.pdf
```

Se o extrator não reconhecer automaticamente os marcadores formais, use `--single-law-pdf` somente depois de verificar que o arquivo contém exclusivamente a Lei 8.078/1990. A validação da aplicação não aceitará lacunas nos artigos-base 1 a 119, duplicações ou dispositivos fora de ordem.

O comando mostra o intervalo detectado, dispositivos, lacunas, duplicados e status. Ele usa PyMuPDF, mas não baixa Sentence Transformers nem inicializa o ChromaDB.

## 7. Testes

```bash
python -m pytest -q
```

Os testes incluem exemplos sintéticos e uma integração com o PDF histórico de 2013. A integração confirma que a seção da Lei 8.078 é isolada das normas posteriores e que os artigos numerados esperados são extraídos. **Isso valida o processamento daquele arquivo, não certifica a atualização jurídica de uma fonte.**

## 8. Estrutura do projeto

```text
cdc_rag_assistente/
├── app.py                 # Interface Streamlit e orquestração
├── extract.py             # Extração PDF, isolamento da Lei 8.078 e auditoria
├── chunking.py            # Chunking jurídico com offsets e metadados
├── indexing.py            # ChromaDB versionado e manifesto ativo
├── retrieval.py           # Vetorial, BM25, RRF, lookup e reranking
├── llm.py                 # Catálogo de perfis e adaptadores de LLM
├── evaluate.py            # Métricas de recuperação por artigo
├── validate_pdf.py        # Validação da extração sem embeddings
├── eval/questions.csv     # Perguntas e artigos esperados
├── sample_data/           # PDF histórico utilizado nos testes
├── tests/                 # Testes unitários e de integração
├── data/                  # Índice e chunks locais (não versionados)
├── requirements.txt
├── .env.example
└── README.md
```

## 9. Limitações e próximos passos de pesquisa

- A extração é baseada em texto selecionável e padrões de estrutura; PDFs escaneados exigem OCR externo e revisão humana.
- A detecção de status não substitui uma análise jurídica completa de vigência, eficácia, redação dada e controle de constitucionalidade. Trechos com status desconhecido ou misto são tratados conservadoramente pelo filtro padrão.
- O sistema registra o rótulo da versão informado pelo usuário, mas não consulta automaticamente alterações legislativas futuras.
- A similaridade vetorial e o reranker são heurísticas de ordenação, não provas de pertinência jurídica.
- A geração com LLM pode cometer erros. A resposta sempre deve ser conferida nas fontes oficiais indicadas.
- Para produção, acrescente avaliação jurídica por especialistas, trilha de auditoria, versionamento imutável das fontes e observabilidade das chamadas do modelo.
