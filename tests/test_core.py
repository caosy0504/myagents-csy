from my_agents import Config, Message, SimpleAgent


class FakeLLM:
    provider = "fake"

    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, messages, **kwargs):
        return next(self.responses)

    def stream(self, messages, **kwargs):
        yield from next(self.responses)


def test_message_has_api_contract():
    message = Message(content="你好", role="user", metadata={"source": "test"})
    assert message.to_api_dict() == {"role": "user", "content": "你好"}
    assert message.metadata["source"] == "test"


def test_agent_history_is_trimmed():
    config = Config(max_history_length=2)
    agent = SimpleAgent("test", FakeLLM(["一", "二"]), config=config)
    agent.run("问题一")
    agent.run("问题二")
    assert [message.content for message in agent.get_history()] == ["问题二", "二"]
