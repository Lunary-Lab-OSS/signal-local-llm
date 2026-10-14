"""
Linguistic Feature Extraction

Extracts explicit difficulty signals (grade level, length, polysyllables)
to help the router detect complexity that semantic embeddings might miss.
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)


def _get_textstat():
    """Import textstat lazily so this module works without it installed."""
    import textstat

    return textstat


class LinguisticFeatureExtractor:
    def __init__(self):
        # We normalize features based on typical question distributions
        # These constants should be calibrated on your specific dataset
        self.max_len = 50.0
        self.max_grade = 12.0

    def extract(self, text_list):
        """
        Extracts explicit difficulty signals:
        1. Flesch-Kincaid Grade Level
        2. Sentence Length (Word count)
        3. Polysyllable count
        """
        features = []
        textstat = _get_textstat()
        for text in text_list:
            # Readability metrics from the textstat library.
            try:
                grade = float(textstat.flesch_kincaid_grade(text))
                length = len(text.split())
                poly = textstat.polysyllabcount(text)

                # Normalize. A non-finite grade would survive min/max clamping
                # (``max(nan, 0)`` is ``nan``) and poison the head input.
                if not math.isfinite(grade):
                    grade = 0.0
                norm_grade = min(max(grade, 0.0), self.max_grade) / self.max_grade
                norm_len = min(length, self.max_len) / self.max_len
                norm_poly = min(poly, 10) / 10.0

                features.append([norm_grade, norm_len, norm_poly])
            except Exception as e:
                # Log metadata only, never the prompt content.
                logger.debug("feature extraction failed (%s)", type(e).__name__)
                features.append([0.0, 0.0, 0.0])  # Fallback

        # Empty input yields a defined (0, 3) shape instead of (0,).
        if not text_list:
            return torch.zeros((0, 3), dtype=torch.float32)
        return torch.tensor(features, dtype=torch.float32)
