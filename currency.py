"""Verificação estrutural de atualização do texto do CDC.

A Lei 14.181/2021 (superendividamento) incluiu os arts. 54-A a 54-G e 104-A a 104-C.
Um PDF sem nenhum desses dispositivos é anterior a ela. A checagem só detecta essa
ausência: NÃO prova que o texto esteja atualizado em relação a alterações posteriores.
"""
from __future__ import annotations

from collections.abc import Iterable

UPDATE_MARKER_ARTICLES = frozenset({"54-A", "104-A"})
OUTDATED_NOTE = (
    " [verificação automática: texto sem os arts. 54-A e 104-A da Lei 14.181/2021; "
    "possivelmente desatualizado]"
)


def possibly_outdated(article_keys: Iterable[str]) -> bool:
    return not (UPDATE_MARKER_ARTICLES & {str(key).upper() for key in article_keys})
