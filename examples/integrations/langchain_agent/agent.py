"""LangChain agent using hebbmem as memory backend.

hebbmem replaces LangChain's ConversationBufferMemory with bio-inspired memory.
Memories decay, associate, and activate — instead of just growing forever.

Usage:
    pip install hebbmem langchain langchain-community
    python agent.py
"""

from __future__ import annotations

from typing import Any

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

from hebbmem import HebbMem


class HebbMemLangChainMemory(ConversationBufferMemory):
    """Drop-in replacement for LangChain memory using hebbmem."""

    hebb: Any = None
    recall_top_k: int = 5

    def __init__(self, hebbmem: HebbMem | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hebb = hebbmem or HebbMem(encoder="hash")

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Recall relevant memories instead of returning full history."""
        query = inputs.get(self.input_key or "input", "")
        if not query:
            return {self.memory_key: ""}

        results = self.hebb.recall(str(query), top_k=self.recall_top_k)
        memory_text = "\n".join(r.content for r in results)
        self.hebb.step(1)
        return {self.memory_key: memory_text}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Store conversation turn as memory."""
        input_str = inputs.get(self.input_key or "input", "")
        output_str = outputs.get(self.output_key or "output", "")

        if input_str:
            self.hebb.store(f"Human: {input_str}", importance=0.5)
        if output_str:
            self.hebb.store(f"AI: {output_str[:200]}", importance=0.3)

    def clear(self) -> None:
        self.hebb = HebbMem(encoder="hash")


def main() -> None:
    llm = Ollama(model="qwen2.5:3b")
    memory = HebbMemLangChainMemory(hebbmem=HebbMem(encoder="hash"))
    chain = ConversationChain(llm=llm, memory=memory, verbose=True)

    print("LangChain + hebbmem | Type 'quit' to exit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input == "quit":
            break
        response = chain.predict(input=user_input)
        print(f"Bot: {response}\n")
        s = memory.hebb.stats()
        print(f"[Memory: {s['node_count']} nodes, {s['edge_count']} connections]\n")


if __name__ == "__main__":
    main()
