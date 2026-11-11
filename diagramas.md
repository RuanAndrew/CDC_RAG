# Diagramas — CDC em Linguagem Simples

---

## 1. Visão geral da arquitetura

Apresenta os principais componentes do sistema, divididos entre ingestão da legislação, consulta e avaliação.

```mermaid
flowchart LR
    subgraph ING["Ingestão"]
        PDF["PDF do CDC"] --> EXT["Extração e validação"]
        EXT --> CHK["Chunking legal"]
        CHK --> IDX["Indexação"]
        IDX --> DB[("ChromaDB")]
    end

    subgraph QRY["Consulta"]
        USER["Usuário"] --> APP["Interface Streamlit"]
        APP --> RET["Recuperação de trechos"]
        RET --> LLM["Modelo de linguagem"]
        LLM --> RESP["Resposta com referências"]
    end

    DB --> RET
    MAN["Índice ativo"] -.-> RET

    EVAL["Avaliação"] -.-> RET
```

---

## 2. Pipeline de ingestão

Demonstra como o documento PDF é processado, validado e transformado em uma base de conhecimento para consultas.

```mermaid
flowchart TD
    A["PDF enviado"] --> B{"PDF legível?"}

    B -- Não --> X["Interromper ingestão"]
    B -- Sim --> C["Extrair texto das páginas"]

    C --> D{"Lei 8.078 identificada?"}

    D -- Sim --> E["Extrair artigos"]
    D -- Não --> F{"Usuário confirmou que o PDF contém apenas o CDC?"}

    F -- Não --> X
    F -- Sim --> E

    E --> G{"Estrutura dos artigos consistente?"}

    G -- Não --> Y["Reportar problemas e interromper"]

    G -- Sim --> H{"Possui arts. 54-A e 104-A?"}

    H -- Não --> I["Marcar como possivelmente desatualizada"]
    H -- Sim --> J["Preparar chunks"]

    I --> J
    J --> K["Indexar no ChromaDB"]
    K --> L["Base pronta para consultas"]
```

---

## 3. Fluxo de uma pergunta

Apresenta o caminho percorrido por uma pergunta do usuário até a geração de uma resposta fundamentada nos trechos recuperados.

```mermaid
sequenceDiagram
    actor U as Usuário
    participant A as Aplicação
    participant R as Recuperação
    participant L as Módulo LLM
    participant M as Modelo

    U->>A: Envia uma pergunta
    A->>R: Contextualiza e busca trechos

    Note over R: Busca vetorial, BM25 e lookup

    R-->>A: Trechos recuperados

    alt Nenhum trecho adequado
        A-->>U: Informa que não encontrou fontes suficientes
    else Existem trechos
        A->>L: Envia pergunta e fontes
        L->>M: Solicita geração da resposta
        M-->>L: Resposta com IDs das fontes
        L->>L: Finaliza e verifica citações
        L-->>A: Resposta e referências
        A-->>U: Exibe resposta e fontes
    end
```

---

## 4. Verificação de citações

Demonstra como o sistema verifica as referências utilizadas pelo modelo antes de apresentar a resposta ao usuário.

```mermaid
flowchart TD
    A["Resposta gerada pelo modelo"] --> B["Validar citações e referências"]

    B --> C{"Referências correspondem às fontes recuperadas?"}

    C -- Sim --> D["Converter IDs em referências oficiais"]

    C -- Não --> E["Remover ou sinalizar referências inválidas"]

    D --> F["Gerar relatório de validação"]
    E --> F

    F --> G{"Existem problemas que exigem revisão?"}

    G -- Sim --> H["Exibir aviso de revisão"]
    G -- Não --> I["Exibir resposta normalmente"]
```

---
