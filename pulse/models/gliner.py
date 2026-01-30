import logging

from gliner import GLiNER

logger = logging.getLogger(__name__)

MODEL_ID = "knowledgator/gliner-bi-edge-v2.0"


class CompanyScanner:
    """Bi-encoder entity scanner for detecting company mentions in text.

    Uses pre-computed label embeddings for fast inference against
    a large set of company name aliases.
    """

    def __init__(self):
        self._model: GLiNER | None = None
        self._alias_to_ticker: dict[str, str] = {}
        self._all_aliases: list[str] = []
        self._label_embeddings = None

    @property
    def ready(self) -> bool:
        return self._model is not None and self._label_embeddings is not None

    @property
    def all_aliases(self) -> list[str]:
        return self._all_aliases

    @property
    def alias_to_ticker(self) -> dict[str, str]:
        return self._alias_to_ticker

    def load(self):
        logger.info("Loading %s...", MODEL_ID)
        self._model = GLiNER.from_pretrained(MODEL_ID)
        logger.info("CompanyScanner loaded")

    def update_aliases(self, ticker_aliases: dict[str, list[str]]):
        """Rebuild alias-to-ticker map and pre-compute label embeddings.

        Args:
            ticker_aliases: {ticker: [alias1, alias2, ...]}
        """
        self._alias_to_ticker = {}
        self._all_aliases = []

        for ticker, aliases in ticker_aliases.items():
            for alias in aliases:
                alias = alias.strip()
                if not alias:
                    continue
                # Keep first mapping if duplicate aliases across tickers
                if alias not in self._alias_to_ticker:
                    self._alias_to_ticker[alias] = ticker
                    self._all_aliases.append(alias)

        if not self._all_aliases:
            self._label_embeddings = None
            logger.warning("No aliases to encode")
            return

        logger.info("Encoding %d aliases...", len(self._all_aliases))
        self._label_embeddings = self._model.encode_labels(self._all_aliases, batch_size=8)
        logger.info("Alias embeddings ready (%d labels)", len(self._all_aliases))

    def scan(self, text: str) -> list[str]:
        """Scan text for company mentions. Returns list of matched alias strings."""
        if not self.ready:
            return []

        entities = self._model.batch_predict_with_embeds(
            [text], self._label_embeddings, self._all_aliases,
        )

        matched = set()
        for entity in entities[0]:
            matched.add(entity["label"])

        return list(matched)
