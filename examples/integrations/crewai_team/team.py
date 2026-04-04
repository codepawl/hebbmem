"""CrewAI multi-agent team with shared hebbmem memory.

Three agents share one hebbmem instance: researcher, writer, reviewer.
When the researcher finds info, the writer can recall it through
spreading activation — even topics the writer never directly saw.

Usage:
    pip install hebbmem crewai crewai-tools
    python team.py
"""

from __future__ import annotations

from crewai import Agent, Crew, Task

from hebbmem import HebbMem

# Shared memory — all agents read/write to the same hebbmem
shared_memory = HebbMem(encoder="hash")


class HebbMemTool:
    """CrewAI tool wrapper for hebbmem."""

    name = "memory"
    description = "Store and recall information from team memory."

    def store(self, content: str, importance: float = 0.5) -> str:
        memory_id = shared_memory.store(content, importance=importance)
        return f"Stored: {content[:50]}... (id: {memory_id})"

    def recall(self, query: str) -> str:
        results = shared_memory.recall(query, top_k=5)
        if not results:
            return "No relevant memories found."
        return "\n".join(f"- {r.content} (score: {r.score:.2f})" for r in results)


def main() -> None:
    researcher = Agent(
        role="Researcher",
        goal="Find and store key information",
        backstory="You research topics and save findings to team memory.",
    )

    writer = Agent(
        role="Writer",
        goal="Write content using team knowledge",
        backstory="You write based on what the team has learned.",
    )

    reviewer = Agent(
        role="Reviewer",
        goal="Review and fact-check using team memory",
        backstory="You verify claims against stored team knowledge.",
    )

    research_task = Task(
        description=(
            "Research the benefits of bio-inspired AI memory systems. "
            "Store 5 key findings in memory."
        ),
        expected_output="5 key findings stored in team memory",
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Write a short summary about bio-inspired memory for AI agents. "
            "Use team memory for facts."
        ),
        expected_output="A 200-word summary",
        agent=writer,
    )

    review_task = Task(
        description=(
            "Review the summary. Check claims against team memory. "
            "Flag anything not supported."
        ),
        expected_output="Review with corrections if needed",
        agent=reviewer,
    )

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, write_task, review_task],
        verbose=True,
    )

    result = crew.kickoff()
    print("\n=== Final Output ===")
    print(result)
    print(f"\n[Shared memory: {shared_memory.stats()}]")


if __name__ == "__main__":
    main()
