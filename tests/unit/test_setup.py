"""Tests for the setup wizard."""
import asyncio

from textual.widgets import OptionList

from whyframe.core.config import Config
from whyframe.setup import WhyframeSetupApp, WizardState, build_config


async def wait_for_step(app: WhyframeSetupApp, pilot, step: str) -> None:
    for _ in range(80):
        if app.current_step == step:
            return
        await pilot.pause(0.05)
    raise AssertionError(f"Timed out waiting for step {step!r}; current={app.current_step!r}")


def test_build_config_persists_compatible_values(tmp_path):
    state = WizardState(
        provider_choice="compatible",
        compatible_base_url="http://localhost:11434/v1",
        compatible_api_key="not-needed",
        compatible_model="qwen/qwen3-embedding-8b",
        vector_db_choice="skip",
        git_max_commits="5000",
        git_ignored_paths=".git,node_modules",
    )

    config = build_config(state)
    path = tmp_path / "whyframe-config.yaml"
    config.to_file(path)
    loaded = Config.from_file(path)

    assert loaded.embedding.provider == "openai"
    assert loaded.embedding.base_url == "http://localhost:11434/v1"
    assert loaded.embedding.api_key == "not-needed"
    assert loaded.embedding.model == "qwen/qwen3-embedding-8b"
    assert loaded.embedding.dimension == 1024
    assert loaded.vector_db.provider == "skip"
    assert loaded.git.max_commit_history == 5000
    assert loaded.git.ignored_paths == [".git", "node_modules"]


def test_openai_textual_flow_saves_config(tmp_path):
    async def scenario() -> None:
        app = WhyframeSetupApp(config_path=tmp_path / "whyframe-config.yaml")

        async with app.run_test() as pilot:
            await wait_for_step(app, pilot, "provider")
            option_list = app.query_one(OptionList)
            assert option_list.highlighted == 0

            await pilot.press("enter")
            await wait_for_step(app, pilot, "openai_model")

            await pilot.press("down", "enter")
            await wait_for_step(app, pilot, "openai_api_key")

            for key in "sk-test":
                await pilot.press(key)
            await pilot.press("enter")
            await wait_for_step(app, pilot, "vector_db_choice")

            await pilot.press("enter")
            await wait_for_step(app, pilot, "git_details")

            await pilot.press("enter")
            await pilot.press("enter")
            await wait_for_step(app, pilot, "review")

            await pilot.press("enter")
            await wait_for_step(app, pilot, "success")

            await pilot.press("enter")
            await pilot.pause()

        config = app.return_value
        assert config is not None
        assert config.embedding.provider == "openai"
        assert config.embedding.model == "text-embedding-3-large"
        assert config.embedding.dimension == 3072
        assert config.embedding.api_key == "sk-test"
        assert config.vector_db.provider == "skip"

        loaded = Config.from_file(tmp_path / "whyframe-config.yaml")
        assert loaded.embedding.api_key == "sk-test"
        assert loaded.vector_db.provider == "skip"

    asyncio.run(scenario())


def test_compatible_textual_flow_uses_fetched_models(tmp_path):
    async def scenario() -> None:
        app = WhyframeSetupApp(
            config_path=tmp_path / "whyframe-config.yaml",
            model_fetcher=lambda base_url: [
                "qwen/qwen3-embedding-8b",
                "mistralai/codestral-embed-2505",
            ],
        )

        async with app.run_test() as pilot:
            await wait_for_step(app, pilot, "provider")
            await pilot.press("down", "enter")
            await wait_for_step(app, pilot, "compatible_base_url")

            await pilot.press("enter")
            await wait_for_step(app, pilot, "compatible_api_key")

            await pilot.press("enter")
            await wait_for_step(app, pilot, "compatible_fetch_choice")

            await pilot.press("enter")
            await wait_for_step(app, pilot, "compatible_fetched_model")

            await pilot.press("down", "enter")
            await wait_for_step(app, pilot, "vector_db_choice")

            await pilot.press("down", "down", "enter")
            await wait_for_step(app, pilot, "weaviate_details")

            await pilot.press("enter")
            await pilot.press("enter")
            await wait_for_step(app, pilot, "git_details")

            await pilot.press("enter")
            await pilot.press("enter")
            await wait_for_step(app, pilot, "review")

            await pilot.press("enter")
            await wait_for_step(app, pilot, "success")

            await pilot.press("enter")
            await pilot.pause()

        config = app.return_value
        assert config is not None
        assert config.embedding.provider == "openai"
        assert config.embedding.base_url == "http://localhost:11434/v1"
        assert config.embedding.model == "mistralai/codestral-embed-2505"
        assert config.embedding.dimension == 1024
        assert config.vector_db.provider == "weaviate"
        assert config.vector_db.environment == "http://localhost:8080"

        loaded = Config.from_file(tmp_path / "whyframe-config.yaml")
        assert loaded.embedding.base_url == "http://localhost:11434/v1"
        assert loaded.embedding.model == "mistralai/codestral-embed-2505"
        assert loaded.vector_db.provider == "weaviate"

    asyncio.run(scenario())
