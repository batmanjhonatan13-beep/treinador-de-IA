"""Formato unico de prompt.

O modelo memoriza o fato junto com o texto que vem antes dele. Se o treino usa um
prefixo e a prova usa outro, o peso existe mas nao e acionado. Entao treino, prova
e chat sem caderno passam todos por aqui.
"""

from __future__ import annotations

SYSTEM = "Responda curto, so o fato. Se nao souber, diga que nao sabe."


def prompt_for(user: str, system: str = SYSTEM) -> str:
    """O texto que vai ate o ponto em que o modelo comeca a responder."""
    return f"system: {system}\nuser: {user}\nassistant:"


def completion_for(answer: str) -> str:
    """A resposta, como ela aparece logo depois do prompt."""
    return " " + answer.strip()


def as_text(user: str, answer: str, system: str = SYSTEM) -> str:
    return prompt_for(user, system) + completion_for(answer)
