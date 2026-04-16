"""Interactive Textual setup wizard for Whyframe."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import requests
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.validation import Function, Integer, Validator
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    OptionList,
    Static,
)
from textual.worker import Worker, WorkerState

from whyframe.core.config import Config

REFERENCE_MODELS = [
    "qwen/qwen3-embedding-8b",
    "mistralai/codestral-embed-2505",
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "thenlper/gte-large",
    "intfloat/multilingual-e5-large",
]

NON_EMPTY = Function(lambda value: bool(value.strip()), "This field is required.")
ModelFetcher = Callable[[str], list[str]]


@dataclass(frozen=True)
class ChoiceItem:
    """A selectable option in the wizard."""

    value: str
    label: str
    description: str


@dataclass(frozen=True)
class FormField:
    """A single input field in a form pane."""

    name: str
    label: str
    value: str = ""
    placeholder: str = ""
    password: bool = False
    input_type: str = "text"
    validators: tuple[Validator, ...] = ()
    valid_empty: bool = False


@dataclass
class WizardState:
    """Mutable state collected across the wizard flow."""

    provider_choice: str = "openai"
    openai_model: str = "text-embedding-3-small"
    openai_api_key: str = ""
    compatible_base_url: str = "http://localhost:11434/v1"
    compatible_api_key: str = ""
    compatible_model: str = "text-embedding-3-small"
    compatible_model_source: str = "fetched"
    fetched_models: list[str] = field(default_factory=list)
    vector_db_choice: str = "skip"
    pinecone_api_key: str = ""
    pinecone_environment: str = "us-west1"
    pinecone_index_name: str = "whyframe"
    weaviate_api_key: str = ""
    weaviate_url: str = "http://localhost:8080"
    git_max_commits: str = "10000"
    git_ignored_paths: str = ".git,__pycache__,node_modules"


def fetch_models(base_url: str) -> list[str]:
    """Fetch available embedding models from an OpenAI-compatible API."""
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/v1/embeddings/models",
            timeout=5,
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return [m.get("id", m.get("model", "unknown")) for m in data["data"]]
            if isinstance(data, list):
                return [m.get("id", m.get("model", "unknown")) for m in data]
    except Exception:
        pass
    return []


def parse_ignored_paths(value: str) -> list[str]:
    """Parse ignored paths from a comma-separated string."""
    return [part.strip() for part in value.split(",") if part.strip()]


def mask_secret(value: str) -> str:
    """Return a human-friendly summary for secrets without printing them."""
    return "[green]set[/green]" if value else "[dim]not set[/dim]"


def build_config(state: WizardState) -> Config:
    """Build a Whyframe config from wizard state."""
    config = Config()

    if state.provider_choice == "compatible":
        from whyframe.pipeline.embeddings import EmbeddingPipeline

        config.embedding.provider = "openai"
        config.embedding.base_url = state.compatible_base_url
        config.embedding.api_key = state.compatible_api_key
        config.embedding.model = state.compatible_model
        config.embedding.dimension = EmbeddingPipeline.get_dimension(state.compatible_model)
    elif state.provider_choice == "openai":
        config.embedding.provider = "openai"
        config.embedding.model = state.openai_model
        config.embedding.dimension = 1536 if "small" in state.openai_model else 3072
        config.embedding.api_key = state.openai_api_key
    else:
        config.embedding.provider = "local"
        config.embedding.model = "sentence-transformers"
        config.embedding.dimension = 384

    if state.vector_db_choice == "skip":
        config.vector_db.provider = "skip"
        config.vector_db.api_key = ""
    elif state.vector_db_choice == "pinecone":
        config.vector_db.provider = "pinecone"
        config.vector_db.api_key = state.pinecone_api_key
        config.vector_db.environment = state.pinecone_environment
        config.vector_db.index_name = state.pinecone_index_name
    elif state.vector_db_choice == "weaviate":
        config.vector_db.provider = "weaviate"
        config.vector_db.api_key = state.weaviate_api_key
        config.vector_db.environment = state.weaviate_url
    elif state.vector_db_choice == "pgvector":
        config.vector_db.provider = "pgvector"
        config.vector_db.api_key = ""

    config.git.max_commit_history = int(state.git_max_commits)
    config.git.ignored_paths = parse_ignored_paths(state.git_ignored_paths)
    return config


def build_review_panel(state: WizardState, config_path: Path) -> Panel:
    """Render a review panel before saving."""
    lines = [
        "[bold cyan]Embedding[/bold cyan]",
        f"  Provider: [bold]{state.provider_choice}[/bold]",
    ]

    if state.provider_choice == "openai":
        lines.extend(
            [
                f"  Model: [cyan]{state.openai_model}[/cyan]",
                f"  API key: {mask_secret(state.openai_api_key)}",
            ]
        )
    elif state.provider_choice == "compatible":
        lines.extend(
            [
                f"  Base URL: [cyan]{state.compatible_base_url}[/cyan]",
                f"  Model: [cyan]{state.compatible_model}[/cyan]",
                f"  API key: {mask_secret(state.compatible_api_key)}",
            ]
        )
    else:
        lines.extend(
            [
                "  Model: [cyan]sentence-transformers[/cyan]",
                "  Dimension: [cyan]384[/cyan]",
            ]
        )

    lines.extend(["", "[bold cyan]Vector DB[/bold cyan]"])
    if state.vector_db_choice == "pinecone":
        lines.extend(
            [
                "  Provider: [bold]pinecone[/bold]",
                f"  Environment: [cyan]{state.pinecone_environment}[/cyan]",
                f"  Index: [cyan]{state.pinecone_index_name}[/cyan]",
                f"  API key: {mask_secret(state.pinecone_api_key)}",
            ]
        )
    elif state.vector_db_choice == "weaviate":
        lines.extend(
            [
                "  Provider: [bold]weaviate[/bold]",
                f"  URL: [cyan]{state.weaviate_url}[/cyan]",
                f"  API key: {mask_secret(state.weaviate_api_key)}",
            ]
        )
    else:
        lines.append(f"  Provider: [bold]{state.vector_db_choice}[/bold]")

    lines.extend(
        [
            "",
            "[bold cyan]Git[/bold cyan]",
            f"  Max commits: [cyan]{state.git_max_commits}[/cyan]",
            f"  Ignored paths: [cyan]{state.git_ignored_paths}[/cyan]",
            "",
            f"[dim]Config will be written to[/dim] [bold]{config_path}[/bold]",
        ]
    )
    return Panel(Text.from_markup("\n".join(lines)), title="[bold cyan]Review[/bold cyan]", border_style="cyan")


def build_success_panel(config_path: Path) -> Panel:
    """Render the final success panel."""
    return Panel(
        Text.from_markup(
            f"[bold green]Setup complete.[/bold green]\n"
            f"[dim]Config saved to[/dim] [bold]{config_path}[/bold]\n\n"
            f"[bold]Next steps:[/bold]\n"
            f"  [cyan]1.[/cyan] Review config:  [bold]nano {config_path}[/bold]\n"
            f"  [cyan]2.[/cyan] Index a repo:   "
            f"[bold]whyframe index /path/to/repo --config {config_path}[/bold]"
        ),
        title="[bold green]Success[/bold green]",
        border_style="green",
        padding=(1, 2),
    )


class ChoicePane(Vertical):
    """A pane that submits a selected option via arrow keys + Enter."""

    class Submitted(Message):
        """Choice submission event."""

        def __init__(self, pane: ChoicePane, value: str) -> None:
            self.pane = pane
            self.value = value
            super().__init__()

    def __init__(
        self,
        prompt: str,
        options: list[ChoiceItem],
        *,
        selected: str | None = None,
        hint: str = "[dim]Use ↑/↓ to move and Enter to continue.[/dim]",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.prompt = prompt
        self.options = options
        self.selected = selected or options[0].value
        self.hint = hint

    def compose(self) -> ComposeResult:
        yield Static(Text.from_markup(self.prompt), classes="prompt")
        yield Static(Text.from_markup(self.hint), classes="hint")
        yield OptionList(*[option.label for option in self.options], id="options")
        yield Static(id="description")

    def on_mount(self) -> None:
        option_list = self.query_one(OptionList)
        option_list.highlighted = self._selected_index()
        self._update_description(option_list.highlighted or 0)
        option_list.focus()

    def _selected_index(self) -> int:
        for index, option in enumerate(self.options):
            if option.value == self.selected:
                return index
        return 0

    def _update_description(self, index: int) -> None:
        description = Panel(
            Text.from_markup(self.options[index].description),
            border_style="cyan",
        )
        self.query_one("#description", Static).update(description)

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._update_description(event.option_index)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.post_message(self.Submitted(self, self.options[event.option_index].value))


class FormPane(Vertical):
    """A pane that collects one or more text inputs."""

    class Submitted(Message):
        """Form submission event."""

        def __init__(self, pane: FormPane, values: dict[str, str]) -> None:
            self.pane = pane
            self.values = values
            super().__init__()

    def __init__(
        self,
        prompt: str,
        fields: list[FormField],
        *,
        submit_label: str = "Continue",
        hint: str = "[dim]Tab between fields. Enter advances or submits.[/dim]",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.prompt = prompt
        self.fields = fields
        self.submit_label = submit_label
        self.hint = hint

    def compose(self) -> ComposeResult:
        yield Static(Text.from_markup(self.prompt), classes="prompt")
        yield Static(Text.from_markup(self.hint), classes="hint")
        for form_field in self.fields:
            yield Label(form_field.label, classes="field-label")
            yield Input(
                value=form_field.value,
                placeholder=form_field.placeholder,
                password=form_field.password,
                type=form_field.input_type,
                validators=form_field.validators or None,
                valid_empty=form_field.valid_empty,
                id=f"field-{form_field.name}",
            )
        yield Static("", id="error-message")
        with Horizontal(classes="button-row"):
            yield Button(self.submit_label, id="submit", variant="primary")

    def on_mount(self) -> None:
        self._input_widgets()[0].focus()

    def _input_widgets(self) -> list[Input]:
        return [self.query_one(f"#field-{form_field.name}", Input) for form_field in self.fields]

    def _values(self) -> dict[str, str]:
        return {form_field.name: self.query_one(f"#field-{form_field.name}", Input).value for form_field in self.fields}

    async def _submit(self) -> None:
        error_widget = self.query_one("#error-message", Static)
        for form_field in self.fields:
            widget = self.query_one(f"#field-{form_field.name}", Input)
            result = widget.validate(widget.value)
            if result is not None and not result.is_valid:
                failure = result.failure_descriptions[0] if result.failure_descriptions else "Invalid value."
                error_widget.update(f"[bold red]{form_field.label}[/bold red]: {failure}")
                widget.focus()
                return
        error_widget.update("")
        self.post_message(self.Submitted(self, self._values()))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            await self._submit()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        inputs = self._input_widgets()
        index = inputs.index(event.input)
        if index < len(inputs) - 1:
            inputs[index + 1].focus()
            return
        await self._submit()


class ActionPane(Vertical):
    """A pane with a panel and a single primary action."""

    class Submitted(Message):
        """Action submission event."""

        def __init__(self, pane: ActionPane, action: str) -> None:
            self.pane = pane
            self.action = action
            super().__init__()

    def __init__(
        self,
        content: Panel,
        button_label: str,
        *,
        button_id: str,
        hint: str = "[dim]Press Enter to continue.[/dim]",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.content = content
        self.button_label = button_label
        self.button_id = button_id
        self.hint = hint

    def compose(self) -> ComposeResult:
        yield Static(self.content, classes="panel-message")
        yield Static(Text.from_markup(self.hint), classes="hint")
        with Horizontal(classes="button-row"):
            yield Button(self.button_label, id=self.button_id, variant="primary")

    def on_mount(self) -> None:
        self.query_one(Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.post_message(self.Submitted(self, event.button.id or self.button_id))


class LoadingPane(Vertical):
    """A pane shown while models are being fetched."""

    def __init__(self, prompt: str, *, hint: str = "[dim]Contacting /v1/embeddings/models...[/dim]", id: str | None = None) -> None:
        super().__init__(id=id)
        self.prompt = prompt
        self.hint = hint

    def compose(self) -> ComposeResult:
        yield Static(Text.from_markup(self.prompt), classes="prompt")
        yield LoadingIndicator()
        yield Static(Text.from_markup(self.hint), classes="hint")


class WhyframeSetupApp(App[Config | None]):
    """Textual application for the interactive setup wizard."""

    CSS = """
    Screen {
        align: center top;
        background: $surface;
    }

    #shell {
        width: 96;
        height: auto;
        margin: 1 0;
        padding: 1 2;
        border: round $accent;
        background: $panel;
    }

    #hero {
        margin-bottom: 1;
    }

    #step-title {
        color: $accent;
        text-style: bold;
    }

    #step-subtitle {
        margin-bottom: 1;
    }

    #body {
        height: auto;
        max-height: 22;
        border: round $primary;
        padding: 1 2;
    }

    .prompt {
        text-style: bold;
        margin-bottom: 1;
    }

    .hint {
        margin-top: 1;
    }

    OptionList {
        height: auto;
        max-height: 10;
        margin: 1 0;
    }

    Input {
        width: 100%;
        margin-bottom: 1;
    }

    .field-label {
        margin-top: 1;
    }

    .button-row {
        align-horizontal: right;
        height: auto;
        margin-top: 1;
    }

    Button {
        min-width: 18;
    }

    #status {
        min-height: 1;
        margin-top: 1;
    }
    """

    TITLE = "Whyframe Setup Wizard"
    SUB_TITLE = "Arrow keys navigate lists · Tab moves between inputs"
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(
        self,
        *,
        config_path: Path | str = Path("whyframe-config.yaml"),
        model_fetcher: ModelFetcher = fetch_models,
    ) -> None:
        super().__init__()
        self.config_path = Path(config_path)
        self.model_fetcher = model_fetcher
        self.state = WizardState()
        self.current_step = ""
        self.history: list[str] = []
        self.saved_config: Config | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Container(id="shell"):
            yield Static(
                Panel(
                    Text("Whyframe Setup Wizard", justify="center", style="bold cyan"),
                    subtitle="[dim]configure embeddings, vector DB, and git parsing[/dim]",
                    border_style="cyan",
                    padding=(1, 4),
                ),
                id="hero",
            )
            yield Static("", id="step-title")
            yield Static("", id="step-subtitle")
            yield VerticalScroll(id="body")
            yield Static("", id="status")
        yield Footer()

    async def on_mount(self) -> None:
        await self.show_step("provider", record=False)

    def _step_meta(self, step: str) -> tuple[str, str]:
        meta = {
            "provider": (
                "[bold cyan]Step 1[/bold cyan] · Choose Embedding Provider",
                "Select how Whyframe should generate embeddings.",
            ),
            "openai_model": (
                "[bold cyan]Step 2[/bold cyan] · Choose Embedding Model",
                "Pick one of the hosted OpenAI embedding models.",
            ),
            "openai_api_key": (
                "[bold cyan]Step 2[/bold cyan] · OpenAI Credentials",
                "Provide the API key used for hosted embeddings.",
            ),
            "compatible_base_url": (
                "[bold cyan]Step 2[/bold cyan] · Compatible Endpoint",
                "Enter the base URL for your OpenAI-compatible API.",
            ),
            "compatible_api_key": (
                "[bold cyan]Step 2[/bold cyan] · Compatible Credentials",
                "Provide an API key if your endpoint requires one.",
            ),
            "fetching_models": (
                "[bold cyan]Step 2[/bold cyan] · Fetching Models",
                "Trying GET /v1/embeddings/models on your endpoint.",
            ),
            "compatible_fetch_choice": (
                "[bold cyan]Step 2[/bold cyan] · Choose Model Source",
                "Use the fetched model list or enter a model name manually.",
            ),
            "compatible_fetched_model": (
                "[bold cyan]Step 2[/bold cyan] · Select Fetched Model",
                "Choose one of the models returned by the API.",
            ),
            "compatible_manual_model": (
                "[bold cyan]Step 2[/bold cyan] · Enter Model Name",
                "Type the embedding model name manually.",
            ),
            "local_info": (
                "[bold cyan]Step 2[/bold cyan] · Local Embeddings",
                "Local mode uses sentence-transformers with dimension 384.",
            ),
            "vector_db_choice": (
                "[bold cyan]Step 3[/bold cyan] · Choose Vector Database",
                "Select a vector backend or skip persistence for now.",
            ),
            "pinecone_details": (
                "[bold cyan]Step 3[/bold cyan] · Pinecone Settings",
                "Configure the Pinecone environment and index.",
            ),
            "weaviate_details": (
                "[bold cyan]Step 3[/bold cyan] · Weaviate Settings",
                "Configure the Weaviate endpoint and optional API key.",
            ),
            "git_details": (
                "[bold cyan]Step 4[/bold cyan] · Git Configuration",
                "Tune commit history depth and ignored paths.",
            ),
            "review": (
                "[bold cyan]Step 5[/bold cyan] · Review",
                "Double-check the collected settings before writing the config.",
            ),
            "success": (
                "[bold green]Complete[/bold green]",
                "Your setup is ready.",
            ),
        }
        return meta[step]

    async def show_step(self, step: str, *, record: bool = True, status: str = "") -> None:
        if self.current_step and record:
            self.history.append(self.current_step)
        self.current_step = step
        title, subtitle = self._step_meta(step)
        self.query_one("#step-title", Static).update(Text.from_markup(title))
        self.query_one("#step-subtitle", Static).update(Text.from_markup(f"[dim]{subtitle}[/dim]"))
        self.query_one("#status", Static).update(Text.from_markup(status) if status else "")

        body = self.query_one("#body", VerticalScroll)
        await body.remove_children()
        await body.mount(self._build_step(step))

    def _build_step(self, step: str):
        if step == "provider":
            return ChoicePane(
                "[bold]Select an embedding provider.[/bold]",
                [
                    ChoiceItem(
                        "openai",
                        "OpenAI",
                        "Hosted embeddings via [cyan]api.openai.com[/cyan]. Choose between the small and large models.",
                    ),
                    ChoiceItem(
                        "compatible",
                        "OpenAI-Compatible API",
                        "Use any endpoint that implements the OpenAI embeddings API. Whyframe will try [cyan]/v1/embeddings/models[/cyan].",
                    ),
                    ChoiceItem(
                        "local",
                        "Local (sentence-transformers)",
                        "Use a local sentence-transformers pipeline with dimension [cyan]384[/cyan] and no remote API calls.",
                    ),
                ],
                selected=self.state.provider_choice,
            )

        if step == "openai_model":
            return ChoicePane(
                "[bold]Select the OpenAI embedding model.[/bold]",
                [
                    ChoiceItem(
                        "text-embedding-3-small",
                        "text-embedding-3-small",
                        "Fast and lower cost. Embedding dimension: [cyan]1536[/cyan].",
                    ),
                    ChoiceItem(
                        "text-embedding-3-large",
                        "text-embedding-3-large",
                        "Higher capacity model. Embedding dimension: [cyan]3072[/cyan].",
                    ),
                ],
                selected=self.state.openai_model,
            )

        if step == "openai_api_key":
            return FormPane(
                "[bold]Enter your OpenAI API key.[/bold]",
                [
                    FormField(
                        "api_key",
                        "OpenAI API key",
                        value=self.state.openai_api_key,
                        password=True,
                        placeholder="sk-...",
                        valid_empty=True,
                    )
                ],
            )

        if step == "compatible_base_url":
            return FormPane(
                "[bold]Enter the base URL for your compatible endpoint.[/bold]",
                [
                    FormField(
                        "base_url",
                        "Base URL",
                        value=self.state.compatible_base_url,
                        placeholder="http://localhost:11434/v1",
                        validators=(NON_EMPTY,),
                    )
                ],
            )

        if step == "compatible_api_key":
            return FormPane(
                "[bold]Enter the API key for the compatible endpoint.[/bold] [dim](optional)[/dim]",
                [
                    FormField(
                        "api_key",
                        "API key",
                        value=self.state.compatible_api_key,
                        password=True,
                        valid_empty=True,
                    )
                ],
            )

        if step == "fetching_models":
            return LoadingPane("[bold]Fetching available embedding models...[/bold]")

        if step == "compatible_fetch_choice":
            return ChoicePane(
                "[bold]How would you like to choose the embedding model?[/bold]",
                [
                    ChoiceItem(
                        "fetched",
                        "Pick from fetched models",
                        f"Use the [cyan]{len(self.state.fetched_models)}[/cyan] models returned by the API endpoint.",
                    ),
                    ChoiceItem(
                        "manual",
                        "Enter model manually",
                        "Type a model name yourself if you want a custom or unlisted model.",
                    ),
                ],
                selected=self.state.compatible_model_source,
            )

        if step == "compatible_fetched_model":
            return ChoicePane(
                "[bold]Select one of the fetched embedding models.[/bold]",
                [
                    ChoiceItem(model, model, "Model returned by [cyan]/v1/embeddings/models[/cyan].")
                    for model in self.state.fetched_models
                ],
                selected=self.state.compatible_model,
            )

        if step == "compatible_manual_model":
            reference = "\n".join(f"  [dim]-[/dim] [cyan]{model}[/cyan]" for model in REFERENCE_MODELS)
            return FormPane(
                "[bold]Enter the embedding model name manually.[/bold]\n\n"
                f"[bold]Reference models:[/bold]\n{reference}",
                [
                    FormField(
                        "model",
                        "Model name",
                        value=self.state.compatible_model,
                        placeholder="text-embedding-3-small",
                        validators=(NON_EMPTY,),
                    )
                ],
            )

        if step == "local_info":
            return ActionPane(
                Panel(
                    Text.from_markup(
                        "[bold green]Local sentence-transformers selected.[/bold green]\n"
                        "Whyframe will use the built-in local pipeline with model "
                        "[cyan]sentence-transformers[/cyan] and dimension [cyan]384[/cyan]."
                    ),
                    title="[bold cyan]Local embeddings[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                ),
                "Continue",
                button_id="continue",
            )

        if step == "vector_db_choice":
            return ChoicePane(
                "[bold]Select a vector database provider.[/bold]",
                [
                    ChoiceItem(
                        "skip",
                        "Skip for now (in-memory)",
                        "Keep setup lightweight for now. You can add a vector database later.",
                    ),
                    ChoiceItem(
                        "pinecone",
                        "Pinecone",
                        "Managed vector storage with configurable environment and index name.",
                    ),
                    ChoiceItem(
                        "weaviate",
                        "Weaviate",
                        "Use a Weaviate instance via URL, with an optional API key.",
                    ),
                    ChoiceItem(
                        "pgvector",
                        "PostgreSQL (pgvector)",
                        "Store embeddings in PostgreSQL with the pgvector extension.",
                    ),
                ],
                selected=self.state.vector_db_choice,
            )

        if step == "pinecone_details":
            return FormPane(
                "[bold]Configure Pinecone.[/bold]",
                [
                    FormField(
                        "api_key",
                        "Pinecone API key",
                        value=self.state.pinecone_api_key,
                        password=True,
                        valid_empty=True,
                    ),
                    FormField(
                        "environment",
                        "Environment",
                        value=self.state.pinecone_environment,
                    ),
                    FormField(
                        "index_name",
                        "Index name",
                        value=self.state.pinecone_index_name,
                    ),
                ],
            )

        if step == "weaviate_details":
            return FormPane(
                "[bold]Configure Weaviate.[/bold]",
                [
                    FormField(
                        "api_key",
                        "Weaviate API key",
                        value=self.state.weaviate_api_key,
                        password=True,
                        valid_empty=True,
                    ),
                    FormField(
                        "url",
                        "Weaviate URL",
                        value=self.state.weaviate_url,
                        validators=(NON_EMPTY,),
                    ),
                ],
            )

        if step == "git_details":
            return FormPane(
                "[bold]Configure git indexing limits.[/bold]",
                [
                    FormField(
                        "max_commits",
                        "Max commits to index",
                        value=self.state.git_max_commits,
                        input_type="integer",
                        validators=(Integer(minimum=1),),
                    ),
                    FormField(
                        "ignored_paths",
                        "Ignored paths (comma-separated)",
                        value=self.state.git_ignored_paths,
                        validators=(NON_EMPTY,),
                    ),
                ],
            )

        if step == "review":
            return ActionPane(
                build_review_panel(self.state, self.config_path),
                "Save config",
                button_id="save-config",
            )

        if step == "success":
            return ActionPane(
                build_success_panel(self.config_path),
                "Finish",
                button_id="finish",
            )

        raise ValueError(f"Unknown step: {step}")

    async def action_back(self) -> None:
        if self.current_step == "fetching_models" or not self.history:
            return
        previous = self.history.pop()
        await self.show_step(previous, record=False)

    def _fetch_models_worker(self) -> list[str]:
        return self.model_fetcher(self.state.compatible_base_url)

    async def on_choice_pane_submitted(self, message: ChoicePane.Submitted) -> None:
        if self.current_step == "provider":
            self.state.provider_choice = message.value
            next_step = {
                "openai": "openai_model",
                "compatible": "compatible_base_url",
                "local": "local_info",
            }[message.value]
            await self.show_step(next_step)
            return

        if self.current_step == "openai_model":
            self.state.openai_model = message.value
            await self.show_step("openai_api_key")
            return

        if self.current_step == "compatible_fetch_choice":
            self.state.compatible_model_source = message.value
            next_step = "compatible_fetched_model" if message.value == "fetched" else "compatible_manual_model"
            await self.show_step(next_step)
            return

        if self.current_step == "compatible_fetched_model":
            self.state.compatible_model = message.value
            await self.show_step("vector_db_choice")
            return

        if self.current_step == "vector_db_choice":
            self.state.vector_db_choice = message.value
            next_step = {
                "pinecone": "pinecone_details",
                "weaviate": "weaviate_details",
                "skip": "git_details",
                "pgvector": "git_details",
            }[message.value]
            await self.show_step(next_step)

    async def on_form_pane_submitted(self, message: FormPane.Submitted) -> None:
        values = message.values

        if self.current_step == "openai_api_key":
            self.state.openai_api_key = values["api_key"]
            await self.show_step("vector_db_choice")
            return

        if self.current_step == "compatible_base_url":
            self.state.compatible_base_url = values["base_url"]
            await self.show_step("compatible_api_key")
            return

        if self.current_step == "compatible_api_key":
            self.state.compatible_api_key = values["api_key"]
            await self.show_step(
                "fetching_models",
                status="[cyan]Fetching models from /v1/embeddings/models...[/cyan]",
            )
            self.run_worker(
                self._fetch_models_worker,
                name="fetch-models",
                group="setup",
                exclusive=True,
                thread=True,
                exit_on_error=False,
            )
            return

        if self.current_step == "compatible_manual_model":
            self.state.compatible_model = values["model"]
            await self.show_step("vector_db_choice")
            return

        if self.current_step == "pinecone_details":
            self.state.pinecone_api_key = values["api_key"]
            self.state.pinecone_environment = values["environment"]
            self.state.pinecone_index_name = values["index_name"]
            await self.show_step("git_details")
            return

        if self.current_step == "weaviate_details":
            self.state.weaviate_api_key = values["api_key"]
            self.state.weaviate_url = values["url"]
            await self.show_step("git_details")
            return

        if self.current_step == "git_details":
            self.state.git_max_commits = values["max_commits"]
            self.state.git_ignored_paths = values["ignored_paths"]
            await self.show_step("review")

    async def on_action_pane_submitted(self, message: ActionPane.Submitted) -> None:
        if self.current_step == "local_info":
            await self.show_step("vector_db_choice")
            return

        if self.current_step == "review":
            self.saved_config = build_config(self.state)
            self.saved_config.to_file(self.config_path)
            await self.show_step(
                "success",
                status=f"[green]Config saved to {self.config_path}.[/green]",
            )
            return

        if self.current_step == "success":
            self.exit(self.saved_config)

    async def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "fetch-models" or self.current_step != "fetching_models":
            return

        if event.state == WorkerState.SUCCESS:
            self.state.fetched_models = event.worker.result or []
            if self.state.fetched_models:
                await self.show_step(
                    "compatible_fetch_choice",
                    status=f"[green]Found {len(self.state.fetched_models)} models from the API.[/green]",
                )
            else:
                self.state.compatible_model_source = "manual"
                await self.show_step(
                    "compatible_manual_model",
                    status="[yellow]No models returned. Enter one manually.[/yellow]",
                )
            return

        if event.state == WorkerState.ERROR:
            self.state.fetched_models = []
            self.state.compatible_model_source = "manual"
            await self.show_step(
                "compatible_manual_model",
                status="[yellow]Could not fetch models. Enter one manually.[/yellow]",
            )


def run_setup() -> Config:
    """Run the interactive setup wizard."""
    app = WhyframeSetupApp()
    config = app.run()
    if config is None:
        raise SystemExit(1)
    return config


if __name__ == "__main__":
    run_setup()
