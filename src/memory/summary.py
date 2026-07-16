from typing import Any


SUMMARY_PROMPT = """Сожми этот диалог в 3-5 предложений.
Сохрани тему разговора, важные факты, ограничения и намерение пользователя.
Не добавляй факты, которых нет в диалоге. Не пиши заголовок и пояснения.

ПРЕДЫДУЩЕЕ SUMMARY:
{previous_summary}

НОВЫЙ ФРАГМЕНТ ДИАЛОГА:
Пользователь: {user_message}
Ассистент: {assistant_message}

ОБНОВЛЕННОЕ SUMMARY:"""


class DialogueSummarizer:
    """Инкрементально сжимает диалог с помощью той же локальной LLM."""

    def __init__(self, llm: Any, max_summary_chars: int = 2000):
        self.llm = llm
        self.max_summary_chars = max_summary_chars

    def summarize(
        self,
        previous_summary: str,
        user_message: str,
        assistant_message: str,
    ) -> str:
        prompt = SUMMARY_PROMPT.format(
            previous_summary=previous_summary or "Нет — это начало диалога.",
            user_message=user_message,
            assistant_message=assistant_message,
        )
        response = self.llm.invoke(prompt)
        summary = getattr(response, "content", response)
        summary = str(summary).strip()
        if not summary:
            return previous_summary
        return summary[: self.max_summary_chars]
