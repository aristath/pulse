import logging

import spacy
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

SPACY_MODEL = "en_core_web_lg"
FUZZY_THRESHOLD = 80


class CompanyScanner:
    """Two-stage company scanner: spaCy NER extracts ORG entities,
    RapidFuzz matches them against the alias list.
    """

    def __init__(self):
        self._nlp = None
        self._alias_to_ticker: dict[str, str] = {}
        self._all_aliases: list[str] = []

    @property
    def ready(self) -> bool:
        return self._nlp is not None and len(self._all_aliases) > 0

    @property
    def all_aliases(self) -> list[str]:
        return self._all_aliases

    @property
    def alias_to_ticker(self) -> dict[str, str]:
        return self._alias_to_ticker

    def load(self):
        logger.info("Loading spaCy %s...", SPACY_MODEL)
        self._nlp = spacy.load(SPACY_MODEL)
        logger.info("spaCy loaded")

    def update_aliases(self, ticker_aliases: dict[str, list[str]]):
        """Rebuild alias-to-ticker map.

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
            logger.warning("No aliases provided")
            return

        logger.info("Alias map ready (%d aliases)", len(self._all_aliases))

    def scan(self, text: str, aliases: list[str] | None = None) -> list[str]:
        """Scan text for company mentions. Returns list of matched alias strings.

        If aliases is provided, fuzzy-match only against those.
        Otherwise match against all known aliases.
        """
        if not self.ready:
            return []

        candidates = aliases if aliases is not None else self._all_aliases
        if not candidates:
            return []

        doc = self._nlp(text)
        org_spans = {ent.text for ent in doc.ents if ent.label_ == "ORG"}

        if not org_spans:
            return []

        matched = set()
        for span in org_spans:
            result = process.extractOne(
                span,
                candidates,
                scorer=fuzz.token_set_ratio,
                score_cutoff=FUZZY_THRESHOLD,
            )
            if result is not None:
                matched.add(result[0])

        return list(matched)
