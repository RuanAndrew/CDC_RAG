import pytest

import llm


def sample_hits():
    return [{
        "text": "Referência: Art. 49\n\nTrecho legal:\nO consumidor pode desistir...",
        "metadata": {
            "article": "Art. 49", "locator": "Art. 49", "page_start": 20, "page_end": 20,
            "source": "cdc.pdf", "source_version": "2026-11", "status": "vigente",
        },
    }, {
        "text": "Referência: Art. 18, § 1º\n\nTrecho legal:\nNão sendo o vício sanado no prazo máximo de trinta dias...",
        "metadata": {
            "article": "Art. 18", "locator": "Art. 18, § 1º", "page_start": 8, "page_end": 9,
            "source": "cdc.pdf", "source_version": "2026-11", "status": "vigente",
        },
    }]


def test_prompt_includes_exact_reference_and_version():
    prompt = llm.build_user_prompt("Posso desistir da compra?", sample_hits())
    assert "Art. 49" in prompt and "2026-11" in prompt and "Posso desistir da compra?" in prompt
    assert '"source_id": "S1"' in prompt and '"source_id": "S2"' in prompt


def test_no_retrieved_context_abstains_without_calling_provider():
    answer = llm.generate_answer("Pergunta sem base", [], provider="ollama")
    assert "Não encontrei trechos legais" in answer


def test_ollama_request_uses_grounding_prompt_and_context_window(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {"message": {"content": "O trecho fundamenta a resposta [S1]."}}

    monkeypatch.setattr(llm, "_post_json", fake_post)
    answer = llm.generate_answer(
        "Posso desistir da compra?", sample_hits(), provider="ollama",
        model="fake-model", base_url="http://localhost:11434",
    )
    assert "[S1]" in answer
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert "exclusivamente com base" in captured["payload"]["messages"][0]["content"]
    assert captured["payload"]["options"]["num_ctx"] >= 4096


def test_previous_user_questions_are_context_not_legal_sources():
    prompt = llm.build_user_prompt(
        "E se a compra foi pela internet?", sample_hits(),
        conversation_history=["Posso desistir de uma compra?"],
    )
    assert "Posso desistir de uma compra?" in prompt
    assert "não são fonte jurídica" in prompt


def test_anthropic_request_uses_messages_api_without_temperature(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, payload):
        captured.update(url=url, headers=headers, payload=payload)
        return {"content": [{"type": "text", "text": "Resposta fundamentada [S1]."}]}

    monkeypatch.setattr(llm, "_post_json", fake_post)
    answer = llm.generate_answer(
        "Posso desistir?", sample_hits(), provider="anthropic", model="claude-haiku-5-5",
        base_url="https://api.anthropic.com", api_key="test-key", temperature=0.2,
    )
    assert "[S1]" in answer
    assert captured["url"].endswith("/v1/messages")
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert "temperature" not in captured["payload"]


def test_provider_defaults_use_current_model_ids():
    assert llm.provider_defaults("anthropic")["model"] == "claude-haiku-5-5"
    assert llm.provider_defaults("ollama")["model"] == "qwen2.5:1.5b"


def test_system_prompt_warns_about_inactive_legal_status():
    for word in ('"revogado"', '"vetado"', '"misto"'):
        assert word in llm.SYSTEM_PROMPT


def test_finalize_answer_resolves_source_ids_to_official_references():
    text, report = llm.finalize_answer("Pode desistir [S1]. Prazo de 30 dias [S2].", sample_hits())
    assert "[Art. 49, p. 20]" in text
    assert "[Art. 18, § 1º, pp. 8-9]" in text
    assert report["cited_sources"] == ["S1", "S2"] and not report["needs_review"]


def test_finalize_answer_merges_multiple_ids():
    text, _ = llm.finalize_answer("Regra geral [S1, S2].", sample_hits())
    assert "[Art. 49, p. 20; Art. 18, § 1º, pp. 8-9]" in text


def test_finalize_answer_flags_invented_ids_and_literal_references():
    text, report = llm.finalize_answer("Vale [S9] e também [Art. 99, p. 3].", sample_hits())
    assert "citação inválida removida" in text
    assert report["invalid_ids"] == ["S9"]
    assert report["unverified_references"] == ["Art. 99, p. 3"]
    assert report["needs_review"] and "Revise" in llm.describe_citation_issues(report)


def test_finalize_answer_flags_missing_citations_but_not_abstention():
    _, uncited = llm.finalize_answer("Pode desistir em sete dias.", sample_hits())
    assert uncited["needs_review"]
    _, abstained = llm.finalize_answer(
        "Não encontrei fundamento suficiente nos trechos recuperados para responder com segurança.",
        sample_hits(),
    )
    assert abstained["abstained"] and not abstained["needs_review"]


def test_empty_content_with_length_finish_gives_actionable_error(monkeypatch):
    monkeypatch.setattr(llm, "_post_json", lambda *a, **k: {
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
    with pytest.raises(RuntimeError, match="limite de tokens"):
        llm.generate_answer("q?", sample_hits(), provider="openai-compatible", model="m",
                            base_url="https://api.example.com/v1")


def test_truncated_answer_gets_notice(monkeypatch):
    monkeypatch.setattr(llm, "_post_json", lambda *a, **k: {
        "choices": [{"message": {"content": "Parcial [S1]"}, "finish_reason": "length"}]})
    answer = llm.generate_answer("q?", sample_hits(), provider="openai-compatible", model="m",
                                 base_url="https://api.example.com/v1")
    assert answer.endswith(llm.TRUNCATION_NOTE)


def test_max_tokens_is_forwarded(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm, "_post_json", lambda url, *, headers, payload: (
        captured.update(payload=payload) or {"choices": [{"message": {"content": "ok [S1]"}}]}))
    llm.generate_answer("q?", sample_hits(), provider="openai-compatible", model="m",
                        base_url="https://api.example.com/v1", max_tokens=3210)
    assert captured["payload"]["max_tokens"] == 3210


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest", "file:///etc/passwd", "ftp://x.com", "http://0.0.0.0:80",
    "http://metadata.google.internal/", "sem-esquema.com",
])
def test_validate_base_url_rejects_dangerous_targets(url):
    with pytest.raises(ValueError):
        llm.validate_base_url(url)


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:8000/v1", "https://api.openai.com/v1"])
def test_validate_base_url_accepts_normal_targets(url):
    assert llm.validate_base_url(url) == url
