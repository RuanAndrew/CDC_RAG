import llm


def sample_hits():
    return [{
        "text": "Referência: Art. 49\n\nTrecho legal:\nO consumidor pode desistir...",
        "metadata": {
            "article": "Art. 49",
            "locator": "Art. 49",
            "page_start": 20,
            "page_end": 20,
            "source": "cdc.pdf",
            "source_version": "2026-11",
            "status": "vigente",
        },
    }]


def test_prompt_includes_exact_reference_and_version():
    prompt = llm.build_user_prompt("Posso desistir da compra?", sample_hits())
    assert "Art. 49" in prompt
    assert "2026-11" in prompt
    assert "Posso desistir da compra?" in prompt


def test_no_retrieved_context_abstains_without_calling_provider():
    answer = llm.generate_answer("Pergunta sem base", [], provider="ollama")
    assert "Não encontrei trechos legais" in answer


def test_ollama_request_uses_grounding_prompt(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {"message": {"content": "O trecho recuperado fundamenta a resposta [Art. 49, p. 20]."}}

    monkeypatch.setattr(llm, "_post_json", fake_post)
    answer = llm.generate_answer(
        "Posso desistir da compra?", sample_hits(), provider="ollama",
        model="fake-model", base_url="http://localhost:11434",
    )
    assert "Art. 49" in answer
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert "exclusivamente com base" in captured["payload"]["messages"][0]["content"]
    assert "Versão declarada" in captured["payload"]["messages"][1]["content"] or "2026-11" in captured["payload"]["messages"][1]["content"]


def test_previous_user_questions_are_context_not_legal_sources():
    prompt = llm.build_user_prompt(
        "E se a compra foi pela internet?",
        sample_hits(),
        conversation_history=["Posso desistir de uma compra?"],
    )
    assert "Posso desistir de uma compra?" in prompt
    assert "não são fonte jurídica" in prompt
    assert "E se a compra foi pela internet?" in prompt


def test_anthropic_request_uses_messages_api_without_temperature(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, payload):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {"content": [{"type": "text", "text": "Resposta fundamentada [Art. 49, p. 20]."}]}

    monkeypatch.setattr(llm, "_post_json", fake_post)
    answer = llm.generate_answer(
        "Posso desistir?", sample_hits(), provider="anthropic", model="claude-haiku-5-5",
        base_url="https://api.anthropic.com", api_key="test-key", temperature=0.2,
    )
    assert "Art. 49" in answer
    assert captured["url"].endswith("/v1/messages")
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert "temperature" not in captured["payload"]


def test_provider_defaults_use_current_model_ids():
    assert llm.provider_defaults("anthropic")["model"] == "claude-haiku-5-5"
    assert llm.provider_defaults("ollama")["model"] == "qwen2.5:7b"


def test_system_prompt_warns_about_inactive_legal_status():
    assert '"revogado"' in llm.SYSTEM_PROMPT
    assert '"vetado"' in llm.SYSTEM_PROMPT
    assert '"misto"' in llm.SYSTEM_PROMPT
