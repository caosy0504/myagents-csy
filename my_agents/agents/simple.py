"""Direct conversational Agent."""

from collections.abc import Iterator

from ..core import Agent, Message


class SimpleAgent(Agent):
    """The smallest useful Agent: history + one model call."""

    def run(self, input_text: str, **kwargs: object) -> str:
        messages = self.build_messages(input_text)
        response = self.llm.invoke(messages, **self._model_kwargs(dict(kwargs)))
        self.save_turn(input_text, response)
        return response

    def stream_run(self, input_text: str, **kwargs: object) -> Iterator[str]:
        messages = self.build_messages(input_text)
        chunks: list[str] = []
        for chunk in self.llm.stream(messages, **self._model_kwargs(dict(kwargs))):
            chunks.append(chunk)
            yield chunk
        self.save_turn(input_text, "".join(chunks))
